import gc
import os
import json
import asyncio
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

# Import de llama_cpp
try:
    from llama_cpp import Llama, LlamaGrammar
    has_llama_cpp = True
except ImportError:
    has_llama_cpp = False
    print("llama_cpp n'est pas installé. Installation requise pour utiliser ce moteur d'inférence.")
    print("Installez-le avec: pip install llama-cpp-python")

from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
from exo import DEBUG

# Paramètres par défaut pour l'inférence
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
TOP_K = 40
TOP_P = 0.9
MODEL_PARAMS = {
    "1B": {"n_ctx": 8192},
    "3B": {"n_ctx": 8192},
    "8B": {"n_ctx": 8192},
    "70B": {"n_ctx": 8192},
}

# Singleton pour le partage de modèles entre instances
class LlamaModelRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LlamaModelRegistry, cls).__new__(cls)
            cls._instance.models = {}
            cls._instance.usage_count = {}
            cls._instance.use_count_threshold = 3  # Nombre d'utilisations après lesquelles un modèle peut être déchargé
        return cls._instance
    
    def get_model(self, model_id):
        return self.models.get(model_id)
    
    def register_model(self, model_id, model):
        if model_id not in self.models:
            self.models[model_id] = model
            self.usage_count[model_id] = 0
        self.usage_count[model_id] += 1
        
    def unregister_model(self, model_id):
        if model_id in self.models:
            # Déchargement explicite pour libérer la mémoire
            if self.models[model_id] is not None:
                try:
                    # LlamaCpp n'a pas de méthode explicite pour libérer la mémoire
                    # mais nous pouvons supprimer la référence et forcer le GC
                    del self.models[model_id]
                except Exception as e:
                    print(f"Erreur lors du déchargement du modèle {model_id}: {e}")
            self.models.pop(model_id, None)
            self.usage_count.pop(model_id, None)
            gc.collect()

def check_dependencies():
    """Vérifie si les dépendances nécessaires sont installées et suggère leur installation"""
    missing_packages = []
    
    if not has_llama_cpp:
        missing_packages.append("llama-cpp-python")
    
    if missing_packages:
        print("Des dépendances recommandées ne sont pas installées:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nInstallation recommandée:")
        print(f"pip install {' '.join(missing_packages)}")
        
        return False
    return True

def build_llama_model(model_path: Path, shard: Shard, model_size="8B", n_gpu_layers=None):
    """
    Charge un modèle LlamaCpp à partir du chemin spécifié avec gestion optimisée de la mémoire
    """
    print(f"Loading LlamaCpp model from {model_path}, exists: {model_path.exists()}, is directory: {model_path.is_dir() if model_path.exists() else 'N/A'}")
    
    if not has_llama_cpp:
        raise ImportError("llama-cpp-python n'est pas installé. Installez-le pour utiliser ce moteur d'inférence.")
    
    # Vérifier si le modèle est déjà chargé dans le registre
    model_registry = LlamaModelRegistry()
    model_id = shard.model_id
    existing_model = model_registry.get_model(model_id)
    
    if existing_model is not None:
        print(f"Model {model_id} already loaded, reusing instance")
        return existing_model
    
    # Vérifier l'espace disque et les permissions
    try:
        if model_path.exists():
            stat_info = os.stat(model_path)
            print(f"File permissions: {stat_info.st_mode}, size: {stat_info.st_size} bytes")
            
            # Vérifier l'espace disque disponible
            disk_stats = os.statvfs(model_path.parent)
            free_space = disk_stats.f_frsize * disk_stats.f_bavail
            print(f"Available disk space: {free_space / (1024**3):.2f} GB")
    except Exception as e:
        print(f"Error checking file stats: {e}")
    
    try:
        # Forcer le garbage collector
        gc.collect()
        
        # Détecter le modèle exact
        model_file = None
        if model_path.is_dir():
            # Chercher les fichiers .gguf ou .bin dans le répertoire
            gguf_files = list(model_path.glob("*.gguf"))
            bin_files = list(model_path.glob("*.bin"))
            
            if gguf_files:
                model_file = gguf_files[0]  # Utiliser le premier fichier .gguf trouvé
            elif bin_files:
                model_file = bin_files[0]  # Utiliser le premier fichier .bin trouvé
        else:
            model_file = model_path
            
        if not model_file or not model_file.exists():
            raise FileNotFoundError(f"Aucun fichier de modèle valide trouvé dans {model_path}")
            
        print(f"Using model file: {model_file}")
        
        # Déterminer les paramètres du modèle selon sa taille
        n_ctx = MODEL_PARAMS.get(model_size, {}).get("n_ctx", 8192)
        
        # Déterminer le nombre de couches GPU
        if n_gpu_layers is None:
            # Auto-détection basée sur la taille du modèle
            if model_size == "1B":
                n_gpu_layers = 32    # petits modèles - toutes les couches sur GPU
            elif model_size == "3B":
                n_gpu_layers = 32    # pour 3B, on peut charger toutes les couches sur GPU
            elif model_size == "8B":
                n_gpu_layers = 24    # pour 8B, chargement partiel sur GPU
            elif model_size == "70B":
                n_gpu_layers = 16    # pour les gros modèles, moins de couches sur GPU
            else:
                n_gpu_layers = -1    # auto (llama.cpp détermine automatiquement)
        
        # Configuration spécifique au modèle
        model_params = {
            "model_path": str(model_file),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": DEBUG >= 2
        }
        
        print(f"Loading model with parameters: {model_params}")
        
        # Charger le modèle
        model = Llama(**model_params)
        
        print(f"Model loaded successfully")
        
        # Enregistrer le modèle dans le registre
        model_registry.register_model(model_id, model)
            
        return model
    
    except Exception as e:
        print(f"Error in build_llama_model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

class PromptState:
    """État pour gérer le contexte du modèle pendant l'inférence"""
    def __init__(self, start_pos=0, kv_cache=None):
        self.start = start_pos
        self.cache = kv_cache or {}

def make_prompt_state():
    """Crée un nouvel état pour un prompt"""
    return PromptState(0, {})

_executor = ThreadPoolExecutor(max_workers=1)  # singleton pour exécutions séquentielles
class LlamaCppDynamicShardInferenceEngine(InferenceEngine):
    def __init__(self, shard_downloader: ShardDownloader):
        # Vérifier les dépendances recommandées
        if not check_dependencies():
            print("Avertissement: certaines dépendances ne sont pas installées.")
        
        self.shard = None
        self.shard_downloader = shard_downloader
        self.model = None
        self.states = OrderedDict()
        self.executor = _executor
        self.model_registry = LlamaModelRegistry()
        
        # Verrous pour éviter les chargements multiples simultanés
        self._loading_locks = {}
        self._loading_in_progress = set()
    
    def poll_state(self, request_id: str, max_states=2):
        """Gère l'état du modèle pour une requête spécifique"""
        if request_id not in self.states:
            if len(self.states) >= max_states:
                self.states.popitem(last=False)
            self.states[request_id] = make_prompt_state()
        else:
            self.states.move_to_end(request_id)
        
        return self.states[request_id]
    
    async def sample(self, x: np.ndarray, temp=TEMPERATURE, top_p: float = TOP_P) -> np.ndarray:
        """
        Échantillonne un token à partir des logits.
        Note: LlamaCpp gère son propre échantillonnage, cette méthode est
        principalement pour la compatibilité avec l'interface.
        """
        # Dans la pratique, avec llama.cpp, l'échantillonnage est fait lors de l'inférence
        # Cette méthode est gardée pour compatibilité 
        return x
    
    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        """Encode le texte d'entrée en tokens"""
        await self.ensure_shard(shard)
        
        def encode_wrapper():
            # Utilisez la méthode tokenize de llama_cpp
            tokens = self.model.tokenize(prompt.encode())
            return np.array(tokens)
            
        return await asyncio.get_running_loop().run_in_executor(self.executor, encode_wrapper)
    
    async def decode(self, shard: Shard, tokens) -> str:
        """Décode les tokens en texte"""
        await self.ensure_shard(shard)
        
        def decode_wrapper():
            # Si c'est une liste Python ou un tableau numpy, convertissez-le 
            # en format que llama_cpp peut comprendre
            if isinstance(tokens, list) or isinstance(tokens, np.ndarray):
                token_list = tokens.tolist() if isinstance(tokens, np.ndarray) else tokens
                # Utilisez la méthode detokenize de llama_cpp
                return self.model.detokenize(token_list).decode('utf-8')
            return ""
            
        return await asyncio.get_running_loop().run_in_executor(self.executor, decode_wrapper)
    
    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> Tuple[np.ndarray, Optional[dict]]:
        """Exécute l'inférence sur les données d'entrée"""
        await self.ensure_shard(shard)
        
        # Récupérer l'état de la requête
        state = self.poll_state(request_id)
        
        def infer_wrapper():
            # Convertir le numpy array en liste de tokens
            if len(input_data.shape) > 1:
                # Prendre la première séquence si plusieurs sont fournies
                tokens = input_data[0].tolist() 
            else:
                tokens = input_data.tolist()
                
            # Générer le token suivant
            generation = self.model.eval(tokens)
            
            # Mettre à jour l'état interne
            state.start += len(tokens)
            
            # Pour la compatibilité avec l'API, simuler un tableau de logits
            # LlamaCpp ne renvoie pas de logits, mais le token généré
            # Créer un tableau où seul l'indice du token généré a une valeur élevée
            vocab_size = self.model.n_vocab()
            logits = np.zeros((1, 1, vocab_size), dtype=np.float32)
            logits[0, 0, generation] = 100.0  # Valeur arbitraire élevée
            
            return logits
        
        output_data = await asyncio.get_running_loop().run_in_executor(self.executor, infer_wrapper)
        
        # Mettre à jour l'état pour l'interface
        updated_state = inference_state or {}
        
        return output_data, updated_state
    
    async def infer_prompt_complete(self, shard: Shard, prompt: str, 
                                   max_tokens: int = 256, 
                                   temperature: float = TEMPERATURE, 
                                   top_p: float = TOP_P,
                                   stop: List[str] = None) -> str:
        """
        Méthode complète d'inférence pour LlamaCpp qui utilise l'API native
        pour la génération de texte complète
        """
        await self.ensure_shard(shard)
        
        def complete_wrapper():
            # Paramètres de génération
            params = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop or [],
                "echo": False  # Ne pas répéter le prompt dans la sortie
            }
            
            # Appel à l'API de completion de llama_cpp
            completion = self.model.create_completion(prompt, **params)
            
            # Retourner le texte généré
            if isinstance(completion, dict):
                return completion.get("choices", [{}])[0].get("text", "")
            
            return ""
        
        return await asyncio.get_running_loop().run_in_executor(self.executor, complete_wrapper)
    
    async def load_checkpoint(self, shard: Shard, path: str):
        """Charge un checkpoint spécifique (non implémenté pour LlamaCpp)"""
        # LlamaCpp ne supporte pas vraiment le chargement/sauvegarde de checkpoints
        # de la même façon que PyTorch
        print("Load checkpoint n'est pas implémenté pour LlamaCpp")
        pass
            
    async def save_checkpoint(self, shard: Shard, path: str):
        """Sauvegarde l'état du modèle (non implémenté pour LlamaCpp)"""
        # LlamaCpp ne supporte pas vraiment le chargement/sauvegarde de checkpoints
        print("Save checkpoint n'est pas implémenté pour LlamaCpp")
        pass
    
    async def ensure_shard(self, shard: Shard):
        """S'assure que le bon shard du modèle est chargé, en utilisant le registre de modèles partagés"""
        # Vérifier si c'est déjà le modèle actuel
        if self.shard == shard and self.model is not None:
            if DEBUG >= 2: print(f"Model {shard.model_id} is already the current model, reusing")
            return
        
        # Vérifier si le modèle est déjà en cours de chargement
        model_id = shard.model_id
        if model_id in self._loading_in_progress:
            print(f"Loading of model {model_id} already in progress, waiting...")
            # Attendre que le chargement soit terminé
            if model_id not in self._loading_locks:
                self._loading_locks[model_id] = asyncio.Lock()
            
            async with self._loading_locks[model_id]:
                # Vérifier à nouveau si le modèle est disponible dans le registre
                cached_model = self.model_registry.get_model(model_id)
                
                if cached_model is not None:
                    print(f"Using shared model {model_id} that was loaded by another process")
                    self.model = cached_model
                    self.shard = shard
                return
        
        # Marquer le modèle comme étant en cours de chargement
        self._loading_in_progress.add(model_id)
        if model_id not in self._loading_locks:
            self._loading_locks[model_id] = asyncio.Lock()
        
        # Acquérir le verrou pour ce modèle
        async with self._loading_locks[model_id]:
            try:
                # Vérifier d'abord si le modèle est dans le registre
                cached_model = self.model_registry.get_model(model_id)
                
                if cached_model is not None:
                    print(f"Using cached model {model_id} from registry")
                    self.model = cached_model
                    self.shard = shard
                    return
                    
                model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
                print(f"Model path received: {model_path}")

                loop = asyncio.get_running_loop()
                
                # Déterminer les paramètres du modèle en fonction de sa taille
                parameters = "8B"
                if "1b" in shard.model_id.lower():
                    parameters = "1B"
                elif "3b" in shard.model_id.lower():
                    parameters = "3B"
                elif "8b" in shard.model_id.lower():
                    parameters = "8B"
                elif "70b" in shard.model_id.lower():
                    parameters = "70B"
                    
                print(f"Selected model parameters: {parameters}")
                
                try:
                    # Charger le modèle
                    n_gpu_layers = -1  # Auto-détection par défaut
                    model = await loop.run_in_executor(self.executor, build_llama_model, model_path, shard, parameters, n_gpu_layers)
                    
                    # Mettre à jour l'état
                    self.shard = shard
                    self.model = model
                    
                    # Enregistrer dans le registre si ce n'est pas déjà fait
                    if self.model_registry.get_model(shard.model_id) is None:
                        self.model_registry.register_model(shard.model_id, model)
                        
                    print(f"Successfully loaded model for {shard.model_id}")
                    
                except Exception as e:
                    print(f"Error loading model: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise
                    
            except Exception as e:
                print(f"Error in ensure_shard: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            finally:
                # Marquer le modèle comme n'étant plus en cours de chargement
                self._loading_in_progress.remove(model_id)
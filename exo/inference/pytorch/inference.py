import gc
import os
import torch
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
from exo.inference.pytorch import check_dependencies
from exo import DEBUG
# Paramètres par défaut
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.85))
TOP_K = 25
TOP_P = 0.9
MODEL_PARAMS = {
    "1B": {"max_context": 8192},
    "3B": {"max_context": 8192},
    "8B": {"max_context": 8192},
    "70B": {"max_context": 8192},
}

# Singleton pour le partage de modèles entre instances
class ModelRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance.models = {}
            cls._instance.tokenizers = {}
            cls._instance.usage_count = {}
            cls._instance.use_count_threshold = 3  # Nombre d'utilisations après lesquelles un modèle peut être déchargé
        return cls._instance
    
    def get_model(self, model_id):
        return self.models.get(model_id)
    
    def get_tokenizer(self, model_id):
        return self.tokenizers.get(model_id)
    
    def register_model(self, model_id, model, tokenizer):
        if model_id not in self.models:
            self.models[model_id] = model
            self.tokenizers[model_id] = tokenizer
            self.usage_count[model_id] = 0
        self.usage_count[model_id] += 1
        
    def unregister_model(self, model_id):
        if model_id in self.models:
            # Déchargement explicite pour libérer la mémoire
            if self.models[model_id] is not None:
                try:
                    self.models[model_id] = self.models[model_id].to("cpu")
                    del self.models[model_id]
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Erreur lors du déchargement du modèle {model_id}: {e}")
            self.models.pop(model_id, None)
            self.tokenizers.pop(model_id, None)
            self.usage_count.pop(model_id, None)
            gc.collect()

def build_transformer(model_path: Path, shard: Shard, model_size="8B", device=None):
    """
    Charge un modèle PyTorch à partir du chemin spécifié avec gestion optimisée de la mémoire
    """
    print(f"Loading model from {model_path}, exists: {model_path.exists()}, is directory: {model_path.is_dir() if model_path.exists() else 'N/A'}")
    
    # Vérifier si le modèle est déjà chargé dans le registre
    model_registry = ModelRegistry()
    model_id = shard.model_id
    existing_model = model_registry.get_model(model_id)
    existing_tokenizer = model_registry.get_tokenizer(model_id)
    
    if existing_model is not None:
        print(f"Model {model_id} already loaded, reusing instance")
        return existing_model
    
    # Analyser la taille du modèle en fonction du nom
    if "1b" in shard.model_id.lower():
        model_size_mb = 1000
    elif "3b" in shard.model_id.lower():
        model_size_mb = 3000
    elif "8b" in shard.model_id.lower():
        model_size_mb = 8000
    elif "70b" in shard.model_id.lower():
        model_size_mb = 70000
    else:
        # Taille par défaut si on ne peut pas déterminer
        model_size_mb = 3000
        
    # Estimer les besoins en mémoire GPU (règle approximative: 2x la taille du modèle en FP16)
    estimated_gpu_memory = model_size_mb * 2 / 1000  # En Go pour FP16
    print(f"Estimated GPU memory needed for model: {estimated_gpu_memory:.2f} GB (model size: {model_size})")
    
    try:
        # Libérer la mémoire GPU et forcer le garbage collector
        torch.cuda.empty_cache()
        gc.collect()
        
        # Déterminer le périphérique à utiliser
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Vérifier la mémoire GPU disponible
        free_mem = 0
        total_mem = 0
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
            free_mem = total_mem - allocated_mem
            print(f"GPU memory: Total {total_mem:.2f} GB, Used: {allocated_mem:.2f} GB, Free: {free_mem:.2f} GB")
        
        # Options de quantification en fonction de la mémoire disponible
        use_8bit = False
        use_4bit = False
        
        # Stratégie de chargement basée sur la mémoire disponible
        if device == "cuda" and torch.cuda.is_available():
            # Si le modèle est trop grand pour la mémoire disponible, utiliser la quantification
            if free_mem < estimated_gpu_memory * 1.2:  # Ajouter une marge de 20%
                # Adaptative memory saving strategy
                if model_size in ["1B", "3B"]:
                    # 8-bit est suffisant pour les petits modèles si la mémoire est limitée
                    use_8bit = True
                    print(f"Using 8-bit quantization for {model_size} model due to memory constraints")
                else:
                    # 4-bit pour les modèles plus grands
                    use_4bit = True
                    print(f"Using 4-bit quantization for {model_size} model due to memory constraints")
        
        # Vérifier les bibliothèques nécessaires
        has_accelerate = False
        has_bitsandbytes = False
        
        try:
            import accelerate
            has_accelerate = True
        except ImportError:
            print("Warning: Accelerate not found. This may impact performance.")
        
        try:
            import bitsandbytes as bnb
            has_bitsandbytes = True
        except ImportError:
            if use_8bit or use_4bit:
                print("Warning: BitsAndBytes not found but quantization requested. Disabling quantization.")
                use_8bit = False
                use_4bit = False
        
        # Configuration des options de chargement
        loading_options = {
            "device_map": "cpu" if device == "cpu" else "auto",
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32
        }
        
        # Options de quantification
        if has_bitsandbytes:
            if use_8bit:
                loading_options["load_in_8bit"] = True
                print("Loading model in 8-bit precision")
            elif use_4bit:
                loading_options["load_in_4bit"] = True
                loading_options["bnb_4bit_compute_dtype"] = torch.float16
                loading_options["bnb_4bit_use_double_quant"] = True
                loading_options["bnb_4bit_quant_type"] = "nf4"
                print("Loading model in 4-bit precision")
        
        # Configurer l'environnement pour éviter la fragmentation mémoire
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        print(f"Loading model from {model_path} with options: {loading_options}")
        
        # Charger le modèle
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **loading_options
        )
        
        # Vérifier la mémoire après chargement
        if torch.cuda.is_available():
            allocated_mem_after = torch.cuda.memory_allocated(0) / (1024**3)
            free_mem_after = total_mem - allocated_mem_after
            print(f"GPU memory after loading: Used: {allocated_mem_after:.2f} GB, Free: {free_mem_after:.2f} GB")
        
        # Passer en mode évaluation pour inférence
        model.eval()
        
        # Charger le tokenizer
        tokenizer_path = str(model_path if model_path.is_dir() else model_path.parent)
        print(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Enregistrer le modèle dans le registre
        model_registry.register_model(model_id, model, tokenizer)
            
        return model
    
    except Exception as e:
        print(f"Error in build_transformer: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

class PromptState:
    """État pour gérer le contexte du modèle pendant l'inférence"""
    def __init__(self, start_pos=0, kv_cache=None):
        self.start = start_pos
        self.cache = kv_cache or {}

def make_prompt_state(inputs, model):
    """Crée un nouvel état pour un prompt"""
    return PromptState(0, {})

_executor = ThreadPoolExecutor(max_workers=1)  # singleton pour que PyTorch s'exécute toujours sur le même thread
class PyTorchDynamicShardInferenceEngine(InferenceEngine):
    def __init__(self, shard_downloader: ShardDownloader):
        # Vérifier les dépendances recommandées
        check_dependencies()
        
        self.shard = None
        self.shard_downloader = shard_downloader
        self.model = None
        self.tokenizer = None
        self.states = OrderedDict()
        self.executor = _executor
        self.model_registry = ModelRegistry()
        # Verrous pour éviter les chargements multiples simultanés
        self._loading_locks = {}
        self._loading_in_progress = set()
    
    def poll_state(self, x, request_id: str, max_states=2):
        """Gère l'état du modèle pour une requête spécifique"""
        if request_id not in self.states:
            if len(self.states) >= max_states:
                self.states.popitem(last=False)
            self.states[request_id] = make_prompt_state(x, self.model)
        else:
            self.states.move_to_end(request_id)
        state = self.states[request_id]
        return {"start_pos": state.start, "past_key_values": state.cache.get("past_key_values")}
    
    async def sample(self, x: np.ndarray, temp=TEMPERATURE, top_p: float = TOP_P) -> np.ndarray:
        """Échantillonne une token à partir des logits"""
        def sample_wrapper():
            # Convertir numpy à torch
            logits = torch.tensor(x[:, -1, :]).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Appliquer température
            if temp > 0:
                logits = logits / temp
                
            # Appliquer top_p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Supprimer les tokens au-dessus du seuil
                sorted_indices_to_remove = cumulative_probs > top_p
                # Conserver au moins le token le plus probable
                sorted_indices_to_remove[..., 0] = 0
                
                # Appliquer le masque
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
            
            # Échantillonner
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).cpu().numpy().astype(int)
            return next_token
            
        return await asyncio.get_running_loop().run_in_executor(self.executor, sample_wrapper)
    
    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        """Encode le texte d'entrée en tokens"""
        await self.ensure_shard(shard)
        
        def encode_wrapper():
            encoded = self.tokenizer(prompt, return_tensors="np")
            return encoded["input_ids"]
            
        return await asyncio.get_running_loop().run_in_executor(self.executor, encode_wrapper)
    
    async def decode(self, shard: Shard, tokens) -> str:
        """Décode les tokens en texte"""
        await self.ensure_shard(shard)
        
        def decode_wrapper():
            if isinstance(tokens, list):
                tokens = np.array(tokens)
            return self.tokenizer.decode(tokens)
            
        return await asyncio.get_running_loop().run_in_executor(self.executor, decode_wrapper)
    
    async def load_checkpoint(self, shard: Shard, path: str):
        """Charge un checkpoint spécifique"""
        await self.ensure_shard(shard)
        
        def load_wrapper():
            state_dict = torch.load(path)
            self.model.load_state_dict(state_dict)
            
        await asyncio.get_running_loop().run_in_executor(self.executor, load_wrapper)
    
    async def save_checkpoint(self, shard: Shard, path: str):
        """Sauvegarde l'état du modèle dans un checkpoint"""
        await self.ensure_shard(shard)
        
        def save_wrapper():
            torch.save(self.model.state_dict(), path)
            
        await asyncio.get_running_loop().run_in_executor(self.executor, save_wrapper)
    
    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> Tuple[np.ndarray, Optional[dict]]:
        """Exécute l'inférence sur les données d'entrée avec une gestion optimisée de la mémoire"""
        await self.ensure_shard(shard)
        
        def infer_wrapper():
            # Déterminer la taille du modèle pour obtenir les limites de contexte appropriées
            model_size = "3B"  # Taille par défaut
            if "1b" in shard.model_id.lower():
                model_size = "1B"
            elif "3b" in shard.model_id.lower():
                model_size = "3B"
            elif "8b" in shard.model_id.lower():
                model_size = "8B"
            elif "70b" in shard.model_id.lower():
                model_size = "70B"
            
            # Obtenir la taille maximale du contexte pour ce modèle
            max_context = MODEL_PARAMS.get(model_size, {}).get("max_context", 2048)
            actual_context = max_context
            
            # Pour les petits modèles, utiliser une taille de contexte plus petite initialement pour économiser la VRAM
            if model_size in ["1B", "3B"]:
                # Commencer avec un contexte plus petit pour économiser la mémoire
                actual_context = min(2048, max_context)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Afficher les stats mémoire avant inférence
            if torch.cuda.is_available():
                before_mem = torch.cuda.memory_allocated() / (1024**3)
                before_max = torch.cuda.max_memory_allocated() / (1024**3)
                free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3)
                print(f"GPU memory before inference: {before_mem:.2f} GB (peak: {before_max:.2f} GB), free: {free_mem:.2f} GB")
            
            # Convertir numpy en torch tensor avec le type explicite pour les IDs de tokens
            input_tensor = torch.tensor(input_data, dtype=torch.long)
            
            # Limiter la taille du contexte au maximum supporté par le modèle
            if input_tensor.shape[1] > actual_context:
                print(f"Input size {input_tensor.shape[1]} exceeds context size {actual_context}, truncating to last {actual_context} tokens")
                input_tensor = input_tensor[:, -actual_context:]
            
            # Déplacer le tensor sur le bon périphérique seulement après avoir redimensionné
            input_tensor = input_tensor.to(device)
            
            # Récupérer l'état de la requête
            state = self.poll_state(input_tensor, request_id)
            
            # Calcul de l'utilisation estimée de mémoire pour le KV cache
            # Modèle simplifié : chaque token dans le contexte utilise approximativement:
            # - 2 (K+V) * nombre de couches * taille de la tête * nombre de têtes bytes
            token_memory_approx = {
                "1B": 0.0002,  # 200 KB par token environ
                "3B": 0.0005,  # 500 KB par token environ
                "8B": 0.0012,  # 1.2 MB par token environ
                "70B": 0.0095,  # 9.5 MB par token environ
            }
            
            # Mémoire estimée pour le KV cache
            kv_cache_memory_gb = input_tensor.shape[1] * token_memory_approx.get(model_size, 0.0005)
            print(f"Estimated KV cache memory usage: {kv_cache_memory_gb:.2f} GB for {input_tensor.shape[1]} tokens")
            
            # Paramètres d'inférence
            kwargs = {
                "use_cache": True  # Activer le cache KV pour accélérer la génération
            }
            
            # N'utiliser le KV cache précédent que si les tokens d'entrée sont peu nombreux
            # Cela évite l'accumulation excessive de mémoire
            if "past_key_values" in state and state["past_key_values"] is not None:
                # Si l'entrée est petite (ajout de quelques tokens seulement), utiliser le cache précédent
                if input_tensor.shape[1] < 8:  # Par exemple, pour une génération token par token
                    kwargs["past_key_values"] = state["past_key_values"]
                else:
                    # Sinon, on recommence avec un nouveau contexte complet
                    print("Input too large, not using previous KV cache")
                    # Ne pas utiliser le cache précédent
            
            try:
                # Exécuter le modèle avec garde-fou pour la mémoire
                with torch.no_grad():
                    # Configuration pour optimiser l'inférence
                    with torch.cuda.amp.autocast(enabled=device == "cuda"):
                        outputs = self.model(
                            input_ids=input_tensor,
                            **kwargs
                        )
                
                # Mettre à jour l'état du contexte pour les futures générations
                # Ne stocker le passé que si le modèle est en mode génération (peu de tokens en entrée)
                if input_tensor.shape[1] < 8:
                    if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                        self.states[request_id].cache["past_key_values"] = outputs.past_key_values
                else:
                    # Pour les longues entrées, on ne stocke pas le cache KV pour économiser la mémoire
                    if request_id in self.states:
                        self.states[request_id].cache["past_key_values"] = None
                
                self.states[request_id].start += input_tensor.shape[1]
                
                # Vérifier l'utilisation mémoire après inférence
                if torch.cuda.is_available():
                    after_mem = torch.cuda.memory_allocated() / (1024**3)
                    after_max = torch.cuda.max_memory_allocated() / (1024**3)
                    print(f"GPU memory after inference: {after_mem:.2f} GB (peak: {after_max:.2f} GB)")
                    print(f"Memory delta: {after_mem - before_mem:.2f} GB")
                
                # Renvoyer les logits
                return outputs.logits.cpu().numpy()
            
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA OOM error: {str(e)}")
                    print(f"Input shape: {input_tensor.shape}")
                    
                    # Libérer la mémoire
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Réinitialiser l'état pour cette requête et supprimer le KV cache
                    if request_id in self.states:
                        del self.states[request_id]
                    
                    # Essayer avec un contexte plus petit sans KV cache
                    # Réduction progressive: essayer d'abord avec 25% de réduction
                    reduced_context = max(128, int(input_tensor.shape[1] * 0.75))
                    print(f"Retrying with reduced context: {reduced_context} tokens")
                    
                    input_tensor = input_tensor[:, -reduced_context:].to(device)
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=device == "cuda"):
                            outputs = self.model(input_ids=input_tensor, use_cache=False)
                    
                    # Recréer un état propre avec un cache vide
                    self.states[request_id] = make_prompt_state(input_tensor, self.model)
                    self.states[request_id].start = reduced_context
                    
                    return outputs.logits.cpu().numpy()
                else:
                    raise
        
        output_data = await asyncio.get_running_loop().run_in_executor(self.executor, infer_wrapper)
        return output_data, inference_state
    
    async def evaluate(self, request_id: str, shard: Shard, inputs, targets, lengths, loss=None):
        """Évalue le modèle sur un ensemble d'entrées et de cibles"""
        await self.ensure_shard(shard)
        
        def eval_wrapper():
            input_tensor = torch.tensor(inputs).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            target_tensor = torch.tensor(targets).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_tensor, labels=target_tensor)
            
            return outputs.loss.cpu().numpy()
        
        return await asyncio.get_running_loop().run_in_executor(self.executor, eval_wrapper)
    
    async def train(self, request_id: str, shard: Shard, inputs, targets, lengths, loss=None, opt=None, lr=1e-5):
        """Entraîne le modèle sur un ensemble d'entrées et de cibles"""
        await self.ensure_shard(shard)
        
        def train_wrapper():
            # Créer un optimiseur si non spécifié
            optimizer = self.session.get('opt', None)
            if optimizer is None:
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
                self.session['opt'] = optimizer
            
            # Convertir en tenseurs PyTorch
            input_tensor = torch.tensor(inputs).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            target_tensor = torch.tensor(targets).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Entraînement
            optimizer.zero_grad()
            outputs = self.model(input_ids=input_tensor, labels=target_tensor)
            outputs.loss.backward()
            optimizer.step()
            
            return outputs.loss.cpu().numpy(), outputs.loss.cpu().numpy()
        
        return await asyncio.get_running_loop().run_in_executor(self.executor, train_wrapper)
    
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
                cached_tokenizer = self.model_registry.get_tokenizer(model_id)
                
                if cached_model is not None:
                    print(f"Using shared model {model_id} that was loaded by another process")
                    self.model = cached_model
                    self.tokenizer = cached_tokenizer
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
                cached_tokenizer = self.model_registry.get_tokenizer(model_id)
                
                if cached_model is not None:
                    print(f"Using cached model {model_id} from registry")
                    self.model = cached_model
                    self.tokenizer = cached_tokenizer
                    self.shard = shard
                    return
                    
                model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
                print(f"Model path received: {model_path}")

                loop = asyncio.get_running_loop()
                
                # Déterminer les paramètres du modèle en fonction de sa taille
                parameters = "1B"
                if "1b" in shard.model_id.lower():
                    parameters = "1B"
                elif "3b" in shard.model_id.lower():
                    parameters = "3B"
                elif "8b" in shard.model_id.lower():
                    parameters = "8B"
                else:
                    parameters = "70B"
                    
                print(f"Selected model parameters: {parameters}")
                
                try:
                    # Charger le modèle
                    model = await loop.run_in_executor(self.executor, build_transformer, model_path, shard, parameters)
                    
                    # Charger le tokenizer
                    tokenizer_path = str(model_path if model_path.is_dir() else model_path.parent)
                    print(f"Loading tokenizer from: {tokenizer_path}")
                    
                    def load_tokenizer():
                        return self.model_registry.get_tokenizer(shard.model_id) or AutoTokenizer.from_pretrained(tokenizer_path)
                    
                    tokenizer = await loop.run_in_executor(self.executor, load_tokenizer)
                    
                    # Mettre à jour l'état
                    self.shard = shard
                    self.model = model
                    self.tokenizer = tokenizer
                    
                    # Enregistrer dans le registre si ce n'est pas déjà fait
                    if self.model_registry.get_model(shard.model_id) is None:
                        self.model_registry.register_model(shard.model_id, model, tokenizer)
                        
                    print(f"Successfully loaded model and tokenizer for {shard.model_id}")
                    
                except Exception as e:
                    print(f"Error loading model or tokenizer: {str(e)}")
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
                # Note: On garde le verrou dans la liste au cas où d'autres appels arrivent plus tard
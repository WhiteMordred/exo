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
        # Libérer la mémoire GPU et forcer le garbage collector
        torch.cuda.empty_cache()
        gc.collect()
        
        if torch.cuda.is_available():
            # Afficher les statistiques de mémoire GPU
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
            free_mem = total_mem - allocated_mem
            print(f"GPU memory before loading: Total {total_mem:.2f} GB, Used: {allocated_mem:.2f} GB, Free: {free_mem:.2f} GB")
        
        # Déterminer le device à utiliser
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
        
        # Options d'optimisation mémoire basées sur la taille du modèle
        use_8bit = False
        use_4bit = False
        
        if device == "cuda" and torch.cuda.is_available():
            if model_size in ["1B", "3B"]:
                if free_mem < 5.0:
                    print("Memory limited for this model size. Using 8-bit quantization.")
                    use_8bit = True
            elif free_mem < 10.0:
                print("Memory limited for this model size. Using 4-bit quantization.")
                use_4bit = True
        
        # Vérifier si les bibliothèques nécessaires sont installées
        has_accelerate = False
        has_bitsandbytes = False
        
        try:
            import accelerate
            has_accelerate = True
            print("Accelerate is available, using it for efficient loading")
        except ImportError:
            print("Accelerate not found. Loading model with standard settings.")
        
        try:
            import bitsandbytes as bnb
            has_bitsandbytes = True
            print("BitsAndBytes is available, quantization options enabled")
        except ImportError:
            use_8bit = False
            use_4bit = False
            print("BitsAndBytes not found. Quantization options disabled.")
        
        # Configuration des options de chargement optimisées
        loading_options = {}
        
        # Utilisez toujours float16 qui est universellement supporté par les GPU CUDA
        # Évitez complètement BFloat16 qui cause l'erreur
        if device == "cuda":
            loading_options["torch_dtype"] = torch.float16
            print("Using Float16 for model loading")
        else:
            loading_options["torch_dtype"] = torch.float32
            print("Using Float32 for model loading on CPU")
        
        # Configurer l'environnement pour éviter la fragmentation mémoire
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            
        # Options basées sur les bibliothèques disponibles
        if has_accelerate:
            if device == "cpu":
                loading_options["device_map"] = "cpu"
            else:
                loading_options["device_map"] = "auto" if use_8bit or use_4bit else device
                loading_options["low_cpu_mem_usage"] = True
        
        # Options de quantification si BitsAndBytes est disponible
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
        
        # Charger le modèle avec HuggingFace Transformers
        print(f"Loading model from {model_path} with options: {loading_options}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **loading_options
        )
        
        print(f"Model loaded successfully")
        
        # Vérifier l'état de la mémoire après chargement
        if torch.cuda.is_available():
            allocated_mem_after = torch.cuda.memory_allocated(0) / (1024**3)
            free_mem_after = total_mem - allocated_mem_after
            print(f"GPU memory after loading: Used: {allocated_mem_after:.2f} GB, Free: {free_mem_after:.2f} GB")
        
        model.eval()  # Passer en mode évaluation
        
        # Si on n'est pas sur CUDA et qu'on n'a pas utilisé device_map, déplacer le modèle manuellement
        if device == "cuda" and not has_accelerate and torch.cuda.is_available():
            print("Moving model to CUDA manually")
            model = model.to("cuda")
        
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
        """Exécute l'inférence sur les données d'entrée"""
        await self.ensure_shard(shard)
        
        def infer_wrapper():
            # Convertir numpy en torch tensor
            input_tensor = torch.tensor(input_data).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Récupérer l'état de la requête
            state = self.poll_state(input_tensor, request_id)
            
            # Paramètres d'inférence
            kwargs = {}
            if "past_key_values" in state and state["past_key_values"] is not None:
                kwargs["past_key_values"] = state["past_key_values"]
                kwargs["use_cache"] = True
            
            # Exécuter le modèle
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_tensor,
                    **kwargs
                )
            
            # Mettre à jour l'état
            if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                self.states[request_id].cache["past_key_values"] = outputs.past_key_values
            self.states[request_id].start += input_tensor.shape[1]
            
            # Renvoyer les logits
            return outputs.logits.cpu().numpy()
        
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
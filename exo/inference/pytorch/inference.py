import os
import torch
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from typing import Optional, Tuple, Dict, List
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
from exo import DEBUG
import re

# Singleton pour gérer les modèles chargés
class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.current_shard = None
        return cls._instance
    
    def get_model_and_tokenizer(self, shard: Shard, model_path: Path):
        """Renvoie le modèle et le tokenizer, ne les charge qu'une seule fois pour un shard donné"""
        if self.current_shard == shard and self.model is not None and self.tokenizer is not None:
            if DEBUG >= 2: print(f"Model {shard.model_id} is already loaded, reusing")
            return self.model, self.tokenizer
        
        # Libérer la mémoire de l'ancien modèle si nécessaire
        if self.model is not None:
            if torch.cuda.is_available():
                try:
                    self.model = self.model.to("cpu")
                except Exception as e:
                    print(f"Error moving model to CPU: {e}")
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Charger le nouveau modèle de façon agnostique
        print(f"Loading model: {shard.model_id}")
        
        self.model, self.tokenizer, _ = build_transformer(model_path=model_path)
        
        # Mettre à jour le shard actuel
        self.current_shard = shard
        print(f"Successfully loaded model and tokenizer for {shard.model_id}")
        
        return self.model, self.tokenizer

# Paramètres par défaut
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.85))
TOP_K = 25
TOP_P = 0.9

def get_model_params(model_path=None, model_size_gb=None):
    """
    Détermine dynamiquement les paramètres du modèle basés sur sa taille estimée
    au lieu d'utiliser des tailles prédéfinies hardcodées.
    
    Args:
        model_path (str): Chemin vers le modèle
        model_size_gb (float): Taille estimée du modèle en GB
        
    Returns:
        dict: Paramètres adaptés à la taille du modèle
    """
    # Valeurs par défaut sécuritaires pour tout modèle
    params = {
        "max_context": 8192,  # Valeur par défaut
        "rope_scaling": None  # Pas de scaling RoPE par défaut
    }
    
    # Si nous ne pouvons pas estimer la taille, retourner les paramètres par défaut
    if model_size_gb is None:
        return params
    
    # Ajuster les paramètres en fonction de la taille du modèle
    # Les seuils sont approximatifs et peuvent être ajustés selon les besoins
    if model_size_gb <= 2.0:  # ~1B parameters
        params["max_context"] = 4096
    elif model_size_gb <= 6.0:  # ~3B parameters
        params["max_context"] = 4096
    elif model_size_gb <= 16.0:  # ~8B parameters
        params["max_context"] = 8192
    else:  # Grands modèles (>16GB)
        params["max_context"] = 8192
        # Activer le scaling RoPE pour les très grands modèles si nécessaire
        if model_size_gb > 32.0:
            params["rope_scaling"] = {"type": "dynamic", "factor": 2.0}
    
    # Vérifier si le nom du modèle contient des informations sur le contexte maximal
    if model_path:
        # Détecter les mentions de taille de contexte dans le nom du modèle
        context_indicators = re.findall(r'(\d+)k', os.path.basename(model_path).lower())
        if context_indicators:
            try:
                # Prendre le plus grand nombre mentionné suivi de 'k'
                largest_context = max(int(c) * 1024 for c in context_indicators)
                params["max_context"] = largest_context
            except ValueError:
                pass
    
    return params

class PromptState:
    """État pour gérer le contexte du modèle pendant l'inférence"""
    def __init__(self, start_pos=0, kv_cache=None):
        self.start = start_pos
        self.cache = kv_cache or {}

def make_prompt_state(inputs, model):
    """Crée un nouvel état pour un prompt"""
    return PromptState(0, {})

def build_transformer(model_name_or_path, model_class, config_class, tokenizer_class):
    """
    Construit un modèle transformeur avec le tokenizer associé.
    Cette version est complètement agnostique à la taille du modèle.
    """
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Estimer la taille du modèle dynamiquement
    model_size_gb = estimate_model_size(model_name_or_path)
    
    # Obtenir les paramètres dynamiquement en fonction de la taille estimée
    model_params = get_model_params(model_name_or_path, model_size_gb)
    
    config = config_class.from_pretrained(
        model_name_or_path,
        rope_scaling=model_params.get("rope_scaling"),
    )
    
    # Définir la valeur de max_context
    max_context = model_params.get("max_context", 4096)
    
    # Log des informations détectées pour aider au débogage
    logging.info(f"Modèle détecté de taille approximative: {model_size_gb:.2f} GB")
    logging.info(f"Paramètres utilisés: max_context={max_context}, rope_scaling={model_params.get('rope_scaling')}")
    
    # Charger le modèle avec les paramètres déterminés dynamiquement
    model = model_class.from_pretrained(
        model_name_or_path,
        config=config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    return model, tokenizer

def estimate_model_size(model_path: Path) -> float:
    """
    Estime la taille du modèle en GB en analysant les fichiers de poids
    ou le répertoire du modèle.
    """
    if not model_path.exists():
        return 0
        
    size_bytes = 0
    
    # Si c'est un fichier unique (comme un checkpoint)
    if model_path.is_file():
        return model_path.stat().st_size / 1e9  # Convertir en GB
    
    # Si c'est un répertoire (structure HF)
    for file_path in model_path.glob("**/*.bin"):
        size_bytes += file_path.stat().st_size
    
    # Vérifier s'il y a des safetensors (format alternatif de HF)
    safetensor_size = 0
    for file_path in model_path.glob("**/*.safetensors"):
        safetensor_size += file_path.stat().st_size
    
    # Utiliser le plus grand des deux formats (bin ou safetensors)
    size_bytes = max(size_bytes, safetensor_size)
    
    # Convertir en GB
    return size_bytes / 1e9

_executor = ThreadPoolExecutor(max_workers=1)  # singleton pour que PyTorch s'exécute toujours sur le même thread
class PyTorchDynamicShardInferenceEngine(InferenceEngine):
    def __init__(self, shard_downloader: ShardDownloader):
        self.shard = None
        self.shard_downloader = shard_downloader
        self.model = None
        self.tokenizer = None
        self.states = OrderedDict()
        self.executor = _executor
        self.session = {}
    
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
        """Échantillonne un token à partir des logits"""
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
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Convertir numpy en torch tensor
            input_tensor = torch.tensor(input_data, dtype=torch.long).to(device)
            
            # Récupérer l'état de la requête
            state = self.poll_state(input_tensor, request_id)
            
            # Paramètres d'inférence
            kwargs = {
                "use_cache": True  # Activer le cache KV pour accélérer la génération
            }
            
            # Utiliser le KV cache précédent si disponible
            if "past_key_values" in state and state["past_key_values"] is not None:
                kwargs["past_key_values"] = state["past_key_values"]
            
            try:
                # Exécuter le modèle
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_tensor,
                        **kwargs
                    )
                
                # Mettre à jour l'état du contexte pour les futures générations
                if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                    self.states[request_id].cache["past_key_values"] = outputs.past_key_values
                
                self.states[request_id].start += input_tensor.shape[1]
                
                # Renvoyer les logits
                return outputs.logits.cpu().numpy()
            
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA OOM error: {str(e)}")
                    
                    # Libérer la mémoire
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Réinitialiser l'état pour cette requête et supprimer le KV cache
                    if request_id in self.states:
                        del self.states[request_id]
                    
                    # Essayer avec un contexte plus petit
                    reduced_context = max(128, int(input_tensor.shape[1] * 0.75))
                    print(f"Retrying with reduced context: {reduced_context} tokens")
                    
                    input_tensor = input_tensor[:, -reduced_context:].to(device)
                    with torch.no_grad():
                        outputs = self.model(input_ids=input_tensor, use_cache=False)
                    
                    # Recréer un état propre
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
        """S'assure que le bon shard du modèle est chargé"""
        # Vérifier si c'est déjà le modèle actuel
        if self.shard == shard and self.model is not None:
            if DEBUG >= 2: print(f"Model {shard.model_id} is already the current model, reusing")
            return
        
        try:
            print(f"Ensuring shard: {shard.model_id}")
            model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
            print(f"Model path received: {model_path}")
            
            if self.shard != shard:
                model_manager = ModelManager()
                self.model, self.tokenizer = model_manager.get_model_and_tokenizer(shard, model_path)
                
                # Mettre à jour le shard actuel
                self.shard = shard
                print(f"Successfully loaded model and tokenizer for {shard.model_id}")
                
        except Exception as e:
            print(f"Error in ensure_shard: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
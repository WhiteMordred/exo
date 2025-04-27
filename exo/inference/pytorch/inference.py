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

class PromptState:
    """État pour gérer le contexte du modèle pendant l'inférence"""
    def __init__(self, start_pos=0, kv_cache=None):
        self.start = start_pos
        self.cache = kv_cache or {}

def make_prompt_state(inputs, model):
    """Crée un nouvel état pour un prompt"""
    return PromptState(0, {})

def build_transformer(model_path: Path, shard: Shard, model_size="8B", device=None):
    """
    Charge un modèle PyTorch à partir du chemin spécifié
    """
    print(f"Loading model from {model_path}, exists: {model_path.exists()}, is directory: {model_path.is_dir() if model_path.exists() else 'N/A'}")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Options de chargement pour une répartition optimale des ressources
    loading_options = {
        "device_map": "auto",       # Toujours utiliser "auto" pour permettre la répartition
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "low_cpu_mem_usage": True
    }
    
    # Pour les très grands modèles, ajouter l'option d'offload
    if "70b" in model_path.name.lower():
        loading_options["offload_folder"] = "offload_folder"
    
    print(f"Loading model from {model_path} with options: {loading_options}")
    
    try:
        # Charger le modèle
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **loading_options
        )
        
        # Passer en mode évaluation pour inférence
        model.eval()
        
        return model
    
    except Exception as e:
        print(f"Error in build_transformer: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

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
                # Libérer la mémoire de l'ancien modèle si nécessaire
                if self.model is not None:
                    if torch.cuda.is_available():
                        # Déplacer l'ancien modèle vers CPU puis le supprimer
                        try:
                            self.model = self.model.to("cpu")
                        except Exception as e:
                            print(f"Error moving model to CPU: {e}")
                    self.model = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                parameters = "1B" if "1b" in shard.model_id.lower() else "3B" if "3b" in shard.model_id.lower() else "8B" if "8b" in shard.model_id.lower() else "70B"
                print(f"Selected model parameters: {parameters}")
                
                # Charger le nouveau modèle
                loop = asyncio.get_running_loop()
                self.model = await loop.run_in_executor(self.executor, build_transformer, model_path, shard, parameters)
                
                # Charger le tokenizer
                tokenizer_path = str((model_path if model_path.is_dir() else model_path.parent))
                print(f"Loading tokenizer from: {tokenizer_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                
                # Mettre à jour le shard actuel
                self.shard = shard
                print(f"Successfully loaded model and tokenizer for {shard.model_id}")
                
        except Exception as e:
            print(f"Error in ensure_shard: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
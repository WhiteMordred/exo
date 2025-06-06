import numpy as np
import os
from exo.helpers import DEBUG  # Make sure to import DEBUG

from typing import Tuple, Optional
from abc import ABC, abstractmethod
from .shard import Shard
from exo.download.shard_download import ShardDownloader


class InferenceEngine(ABC):
  session = {}

  @abstractmethod
  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    pass

  @abstractmethod
  async def sample(self, x: np.ndarray) -> np.ndarray:
    pass

  @abstractmethod
  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    pass

  @abstractmethod
  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    pass

  @abstractmethod
  async def load_checkpoint(self, shard: Shard, path: str):
    pass

  async def save_checkpoint(self, shard: Shard, path: str):
    pass

  async def save_session(self, key, value):
    self.session[key] = value

  async def clear_session(self):
    self.session.empty()

  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    tokens = await self.encode(shard, prompt)
    if shard.model_id != 'stable-diffusion-2-1-base':
      x = tokens.reshape(1, -1)
    else:
      x = tokens
    output_data, inference_state = await self.infer_tensor(request_id, shard, x, inference_state)

    return output_data, inference_state


inference_engine_classes = {
  "mlx": "MLXDynamicShardInferenceEngine",
  "tinygrad": "TinygradDynamicShardInferenceEngine",
  "pytorch": "PyTorchDynamicShardInferenceEngine",
  "llama_cpp": "LlamaCppDynamicShardInferenceEngine",
  "dummy": "DummyInferenceEngine",
}

# Variable globale pour stocker le dernier moteur d'inférence sélectionné
_last_selected_engine = None

def get_inference_engine(inference_engine_name: str, shard_downloader: ShardDownloader):
  global _last_selected_engine
  
  # Vérifier si nous avons déjà un moteur d'inférence sélectionné
  if _last_selected_engine is not None and inference_engine_name != _last_selected_engine and inference_engine_name == "tinygrad":
    print(f"ATTENTION: Une tentative de changer le moteur d'inférence de {_last_selected_engine} à {inference_engine_name} a été bloquée.")
    inference_engine_name = _last_selected_engine
  else:
    _last_selected_engine = inference_engine_name
    
  if DEBUG >= 2:
    print(f"get_inference_engine called with: {inference_engine_name}")
  if inference_engine_name == "mlx":
    from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine

    return MLXDynamicShardInferenceEngine(shard_downloader)
  elif inference_engine_name == "tinygrad":
    from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
    import tinygrad.helpers
    tinygrad.helpers.DEBUG.value = int(os.getenv("TINYGRAD_DEBUG", default="0"))

    return TinygradDynamicShardInferenceEngine(shard_downloader)
  elif inference_engine_name == "pytorch":
    from exo.inference.pytorch.inference import PyTorchDynamicShardInferenceEngine
    
    # Vérifier si CUDA est disponible
    try:
      import torch
      cuda_available = torch.cuda.is_available()
      device = "cuda" if cuda_available else "cpu"
      print(f"PyTorch using device: {device}")
    except ImportError:
      print("PyTorch n'est pas installé. Installation requise pour utiliser ce moteur d'inférence.")
      raise
      
    return PyTorchDynamicShardInferenceEngine(shard_downloader)
  elif inference_engine_name == "llama_cpp":  # Support du moteur llama_cpp
    from exo.inference.llama_cpp.inference import LlamaCppDynamicShardInferenceEngine, has_llama_cpp
    
    # Vérifier si llama_cpp est disponible
    if not has_llama_cpp:
      print("llama_cpp n'est pas installé. Installation requise pour utiliser ce moteur d'inférence.")
      print("Installez-le avec: pip install llama-cpp-python")
      raise ImportError("llama-cpp-python est requis pour utiliser le moteur d'inférence llama_cpp")
      
    return LlamaCppDynamicShardInferenceEngine(shard_downloader)
  elif inference_engine_name == "dummy":
    from exo.inference.dummy_inference_engine import DummyInferenceEngine
    return DummyInferenceEngine()
  raise ValueError(f"Unsupported inference engine: {inference_engine_name}")

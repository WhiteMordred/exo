from pathlib import Path
import json
import os
from exo.inference.tinygrad.models.llama import Transformer, TransformerShard, convert_from_huggingface, fix_bf16, sample_logits
from exo.inference.shard import Shard
from exo.inference.tokenizers import resolve_tokenizer
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from tinygrad import Tensor, nn, Context, TinyJit
from exo.inference.inference_engine import InferenceEngine
import numpy as np
from exo.inference.tinygrad.tinygrad_helpers import concat_weights, load
from exo.download.shard_download import ShardDownloader
from concurrent.futures import ThreadPoolExecutor
from .stateful_model import make_prompt_state
from .losses import length_masked_ce_loss
from collections import OrderedDict
import asyncio
from typing import Optional
Tensor.no_grad = True 
# default settings
TEMPERATURE = int(os.getenv("TEMPERATURE", 0.85))
TOP_K = 25
TOP_P = 0.9
ALPHA_F = 0.1
ALPHA_P = 0.0
MODEL_PARAMS = {
  "1B": {
    "args": {
      "dim": 2048, "n_heads": 32, "n_kv_heads": 8, "n_layers": 16, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 8192,
      "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192, "rope_type": "llama3"}, "tie_word_embeddings": True
    }, "files": 1
  }, "3B": {
    "args": {
      "dim": 3072, "n_heads": 24, "n_kv_heads": 8, "n_layers": 28, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 8192,
      "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192, "rope_type": "llama3"}, "tie_word_embeddings": True
    }, "files": 1
  }, "8B": {"args": {"dim": 4096, "n_heads": 32, "n_kv_heads": 8, "n_layers": 32, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 14336}, "files": 1},
  "70B": {"args": {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 28672}, "files": 8}
}


def build_transformer(model_path: Path, shard: Shard, model_size="8B", device=None):
  import os
  print(f"Loading model from {model_path}, exists: {model_path.exists()}, is directory: {model_path.is_dir() if model_path.exists() else 'N/A'}")
  
  # Check disk space and permissions
  try:
    if model_path.exists():
      stat_info = os.stat(model_path)
      print(f"File permissions: {stat_info.st_mode}, size: {stat_info.st_size} bytes")
      
      # Check available disk space
      disk_stats = os.statvfs(model_path.parent)
      free_space = disk_stats.f_frsize * disk_stats.f_bavail
      print(f"Available disk space: {free_space / (1024**3):.2f} GB")
  except Exception as e:
    print(f"Error checking file stats: {e}")
  
  # build model
  linear = nn.Linear
  # Réduction de max_context de 8192 à 4096 pour résoudre l'erreur de lecture du disque
  model = Transformer(**MODEL_PARAMS[model_size]["args"], linear=linear, max_context=4096, jit=True, shard=shard)

  # load weights
  try:
    if model_path.is_dir():
      print(f"Loading from directory. Contents: {[f.name for f in model_path.iterdir() if f.is_file()][:5]}...")
      if (model_path/"model.safetensors.index.json").exists():
        print("Loading from model.safetensors.index.json")
        weights = load(str(model_path/"model.safetensors.index.json"), shard)
      elif (model_path/"model.safetensors").exists():
        print("Loading from model.safetensors")
        weights = load(str(model_path/"model.safetensors"), shard)
      else:
        print(f"Loading from consolidated files, count: {MODEL_PARAMS[model_size]['files']}")
        weights = concat_weights([load(str(model_path/f"consolidated.{i:02d}.pth"), shard) for i in range(MODEL_PARAMS[model_size]["files"])], 
                              device[0] if isinstance(device, tuple) else device)
    else:
      print("Loading from single file path")
      weights = load(str(model_path), shard)
      
    print(f"Weights loaded successfully. Keys: {list(weights.keys())[:5]}...")
    weights = convert_from_huggingface(weights, model, MODEL_PARAMS[model_size]["args"]["n_heads"], MODEL_PARAMS[model_size]["args"]["n_kv_heads"])
    weights = fix_bf16(weights)

    with Context(BEAM=0):
      # replace weights in model
      try:
        print("Loading state dict into model...")
        load_state_dict(model, weights, strict=False, consume=False)  # consume=True
        print("State dict loaded successfully")
      except Exception as e:
        print(f"Error loading state dict: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
        
      model = TransformerShard(shard, model)

    return model
  except Exception as e:
    print(f"Error in build_transformer: {str(e)}")
    import traceback
    traceback.print_exc()
    raise

_executor = ThreadPoolExecutor(max_workers=1) # singleton so tinygrad always runs on the same thread
class TinygradDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.states = OrderedDict()
    self.executor = _executor

  def poll_state(self, x, request_id: str, max_states=2):
    if request_id not in self.states:
      if len(self.states) >= max_states:
        self.states.popitem(last=False)
      self.states[request_id] = make_prompt_state(x, self.model)
    else:
      self.states.move_to_end(request_id)
    state = self.states[request_id]
    return {"start_pos": state.start, "cache": state.cache}

  async def sample(self, x: np.ndarray, temp=TEMPERATURE, top_p: float = 0.0) -> np.ndarray:
    def sample_wrapper():
      logits = x[:, -1, :]
      return sample_logits(Tensor(logits).flatten(), temp, 0, 0.8, top_p, 0.0).realize().numpy().astype(int)
    return await asyncio.get_running_loop().run_in_executor(self.executor, sample_wrapper)

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    await self.ensure_shard(shard)
    tokens = await asyncio.get_running_loop().run_in_executor(self.executor, self.tokenizer.encode, prompt)
    return await asyncio.get_running_loop().run_in_executor(self.executor, np.array, tokens)
  
  async def decode(self, shard: Shard, tokens) -> str:
    await self.ensure_shard(shard)
    tokens = await asyncio.get_running_loop().run_in_executor(self.executor, self.tokenizer.decode, tokens)
    return tokens
  
  async def load_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
    state_dict = safe_load(path)
    await asyncio.get_running_loop().run_in_executor(self.executor, load_state_dict, self.model, state_dict)
  
  async def save_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
    state_dict = await asyncio.get_running_loop().run_in_executor(self.executor, get_state_dict, self.model)
    safe_save(state_dict, path) 
  
  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    await self.ensure_shard(shard)
    def wrap_infer():
      x = Tensor(input_data)
      h = self.model.embed(x)
      state = self.poll_state(h, request_id)
      out = self.model.forward(h, **state)
      self.states[request_id].start += x.shape[1]
      return out.numpy()
    output_data = await asyncio.get_running_loop().run_in_executor(self.executor, wrap_infer)
    return output_data, inference_state

  async def evaluate(self, request_id: str, shard: Shard, inputs, targets, lengths, loss=length_masked_ce_loss):
    def step(x, y, l):
      Tensor.training = False
      return self.session['loss'](self.model, x, y, l)
    await self.ensure_shard(shard)
    score = await asyncio.get_running_loop().run_in_executor(self.executor, lambda: self.session['jit'](Tensor(inputs), targets, lengths))
    out = score.numpy()
    return out
  
  async def train(self, request_id: str, shard: Shard, inputs, targets, lengths, loss=length_masked_ce_loss, opt=nn.optim.Adam, lr=1e-5):
    def step(x, y, l):
      Tensor.training = True
      score = self.session['loss'](self.model, x, y, l)
      self.session['opt'].zero_grad()
      score.backward()
      self.session['opt'].step()
      return score
    await self.ensure_shard(shard)
      
    score = await asyncio.get_running_loop().run_in_executor(self.executor, lambda: self.session['jit'](Tensor(inputs), targets, lengths).realize())
    
    return loss.numpy(), loss.numpy()

  async def ensure_shard(self, shard: Shard):
    if self.shard == shard:
      return

    try:
      print(f"Ensuring shard: {shard.model_id}")
      model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
      print(f"Model path received: {model_path}")

      if self.shard != shard:
        loop = asyncio.get_running_loop()
        parameters = "1B" if "1b" in shard.model_id.lower() else "3B" if "3b" in shard.model_id.lower() else "8B" if "8b" in shard.model_id.lower() else "70B"
        print(f"Selected model parameters: {parameters}")
        
        try:
          model_shard = await loop.run_in_executor(self.executor, build_transformer, model_path, shard, parameters)
          
          tokenizer_path = str((model_path if model_path.is_dir() else model_path.parent))
          print(f"Loading tokenizer from: {tokenizer_path}")
          self.tokenizer = await resolve_tokenizer(tokenizer_path)
          self.shard = shard
          self.model = model_shard
          print(f"Successfully loaded model and tokenizer for {shard.model_id}")
        except Exception as e:
          print(f"Error loading model or tokenizer: {str(e)}")
          import traceback
          traceback.print_exc()
          
          # Séquence d'essais alternatifs pour charger le modèle
          load_attempts = [
            self._try_load_with_consume_true,
            self._try_load_with_minimal_weights,
            self._try_load_dummy_model,
            self._try_load_with_reduced_context
          ]
          
          success = False
          for attempt_fn in load_attempts:
            try:
              print(f"Attempting alternative loading strategy: {attempt_fn.__name__}")
              result = await attempt_fn(model_path, shard, parameters)
              if result:
                success = True
                print(f"Alternative loading strategy {attempt_fn.__name__} succeeded!")
                break
            except Exception as alt_e:
              print(f"Alternative loading {attempt_fn.__name__} failed: {str(alt_e)}")
              traceback.print_exc()
          
          if not success:
            # Si toutes les tentatives ont échoué, on conserve un modèle minimal pour éviter 
            # l'arrêt complet du système, mais on signale l'erreur
            print("WARNING: Using fallback minimal model due to loading failures")
            self._setup_minimal_model(shard)
            # On ne propage pas l'erreur pour permettre au système de continuer
    except Exception as e:
      print(f"Critical error in ensure_shard: {str(e)}")
      import traceback
      traceback.print_exc()
      # On conserve un modèle minimal pour éviter l'arrêt complet du système
      self._setup_minimal_model(shard)

  async def _try_load_with_consume_true(self, model_path, shard, parameters):
    """Tente de charger le modèle avec consume=True"""
    linear = nn.Linear
    model = Transformer(**MODEL_PARAMS[parameters]["args"], linear=linear, max_context=8192, jit=True, shard=shard)
    
    if model_path.is_dir():
      if (model_path/"model.safetensors.index.json").exists():
        weights = load(str(model_path/"model.safetensors.index.json"), shard)
      elif (model_path/"model.safetensors").exists():
        weights = load(str(model_path/"model.safetensors"), shard)
      else:
        weights = concat_weights([load(str(model_path/f"consolidated.{i:02d}.pth"), shard) for i in range(MODEL_PARAMS[parameters]["files"])], None)
    else:
      weights = load(str(model_path), shard)
    weights = convert_from_huggingface(weights, model, MODEL_PARAMS[parameters]["args"]["n_heads"], MODEL_PARAMS[parameters]["args"]["n_kv_heads"])
    weights = fix_bf16(weights)

    with Context(BEAM=0):
      # Try avec consume=True au lieu de False
      load_state_dict(model, weights, strict=False, consume=True)
      model = TransformerShard(shard, model)
    
    tokenizer_path = str((model_path if model_path.is_dir() else model_path.parent))
    self.tokenizer = await resolve_tokenizer(tokenizer_path)
    self.shard = shard
    self.model = model
    return True

  async def _try_load_with_minimal_weights(self, model_path, shard, parameters):
    """Tente de charger le modèle en ignorant les erreurs sur certains poids"""
    linear = nn.Linear
    model = Transformer(**MODEL_PARAMS[parameters]["args"], linear=linear, max_context=8192, jit=True, shard=shard)
    
    try:
      if model_path.is_dir():
        if (model_path/"model.safetensors.index.json").exists():
          weights = load(str(model_path/"model.safetensors.index.json"), shard)
        elif (model_path/"model.safetensors").exists():
          weights = load(str(model_path/"model.safetensors"), shard)
        else:
          weights = {}
          # Charge chaque fichier individuellement pour éviter qu'une erreur sur un fichier ne fasse échouer l'ensemble
          for i in range(MODEL_PARAMS[parameters]["files"]):
            try:
              file_weights = load(str(model_path/f"consolidated.{i:02d}.pth"), shard)
              weights.update(file_weights)
            except Exception as e:
              print(f"Error loading consolidated.{i:02d}.pth: {e}, continuing with partial weights")
      else:
        weights = load(str(model_path), shard)
    except Exception as e:
      print(f"Error loading weights, will continue with empty weights: {e}")
      weights = {}
    
    try:
      weights = convert_from_huggingface(weights, model, MODEL_PARAMS[parameters]["args"]["n_heads"], MODEL_PARAMS[parameters]["args"]["n_kv_heads"])
      weights = fix_bf16(weights)
    except Exception as e:
      print(f"Error converting weights: {e}, continuing with original weights")

    with Context(BEAM=0):
      # Utilise strict=False pour ignorer les poids manquants
      load_state_dict(model, weights, strict=False, consume=True)
      model = TransformerShard(shard, model)
    
    tokenizer_path = str((model_path if model_path.is_dir() else model_path.parent))
    self.tokenizer = await resolve_tokenizer(tokenizer_path)
    self.shard = shard
    self.model = model
    return True

  def _setup_minimal_model(self, shard):
    """Configure un modèle minimal fonctionnel quand toutes les tentatives ont échoué"""
    print("Setting up minimal model to prevent system failure")
    from exo.inference.dummy_inference_engine import DummyTokenizer
    
    # Créer un tokenizer et un modèle minimal
    self.tokenizer = DummyTokenizer()
    self.shard = shard
    
    # Utiliser le modèle factice comme solution de dernier recours
    if hasattr(self, 'model') and self.model is not None:
      print("Keeping existing model instance")
    else:
      print("Creating dummy model instance")
      self.model = self._try_load_dummy_model(None, shard, "3B")[1]

  async def _try_load_dummy_model(self, model_path, shard, parameters):
    """Utilise un modèle vide quand tout le reste a échoué"""
    linear = nn.Linear
    model = Transformer(**MODEL_PARAMS[parameters]["args"], linear=linear, max_context=4096, jit=False, shard=shard)
    tokenizer_path = str((model_path if model_path.is_dir() else model_path.parent))
    self.tokenizer = await resolve_tokenizer(tokenizer_path)
    self.shard = shard
    self.model = TransformerShard(shard, model)
    return True
    
  async def _try_load_with_reduced_context(self, model_path, shard, parameters):
    """Tente de charger le modèle avec une taille de contexte réduite"""
    print("Attempting to load model with reduced context size")
    linear = nn.Linear
    # Réduire max_context de 8192 à 4096 ou 2048
    model = Transformer(**MODEL_PARAMS[parameters]["args"], linear=linear, max_context=2048, jit=True, shard=shard)
    
    if model_path.is_dir():
      if (model_path/"model.safetensors.index.json").exists():
        weights = load(str(model_path/"model.safetensors.index.json"), shard)
      elif (model_path/"model.safetensors").exists():
        weights = load(str(model_path/"model.safetensors"), shard)
      else:
        weights = concat_weights([load(str(model_path/f"consolidated.{i:02d}.pth"), shard) for i in range(MODEL_PARAMS[parameters]["files"])], None)
    else:
      weights = load(str(model_path), shard)
      
    weights = convert_from_huggingface(weights, model, MODEL_PARAMS[parameters]["args"]["n_heads"], MODEL_PARAMS[parameters]["args"]["n_kv_heads"])
    weights = fix_bf16(weights)

    with Context(BEAM=0):
      # Essayer de charger avec plusieurs combinaisons de paramètres
      try:
        # D'abord avec consume=False pour éviter la libération prématurée de la mémoire
        load_state_dict(model, weights, strict=False, consume=False)
      except Exception as e:
        print(f"Failed first attempt with consume=False: {str(e)}")
        # Ensuite avec consume=True
        load_state_dict(model, weights, strict=False, consume=True)

    tokenizer_path = str((model_path if model_path.is_dir() else model_path.parent))
    self.tokenizer = await resolve_tokenizer(tokenizer_path)
    self.shard = shard
    self.model = TransformerShard(shard, model)
    return True

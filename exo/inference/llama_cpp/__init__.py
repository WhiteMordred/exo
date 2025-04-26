# Module d'inférence basé sur llama.cpp pour exo
# Ce module permet d'utiliser le moteur d'inférence llama.cpp pour les modèles LLM

from exo.inference.llama_cpp.inference import LlamaCppDynamicShardInferenceEngine

__all__ = ["LlamaCppDynamicShardInferenceEngine"]
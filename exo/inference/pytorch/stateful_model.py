import torch
from typing import Dict, Optional, Any
import numpy as np

class PromptState:
    """
    Classe pour stocker l'état d'un modèle pendant l'inférence
    """
    def __init__(self, start_pos: int = 0, cache: Optional[Dict[str, Any]] = None):
        """
        Initialise un nouvel état pour un prompt
        
        Args:
            start_pos: Position de départ dans la séquence
            cache: Cache des clés/valeurs d'attention pour l'inférence rapide
        """
        self.start = start_pos
        self.cache = cache or {}

def make_prompt_state(inputs, model):
    """
    Crée un nouvel état de prompt pour un modèle et des entrées données
    
    Args:
        inputs: Tenseur d'entrée (peut être torch.Tensor ou numpy.ndarray)
        model: Modèle PyTorch
        
    Returns:
        Un nouvel objet PromptState
    """
    return PromptState(0, {"past_key_values": None})

def update_prompt_state(state: PromptState, new_kv_cache, seq_len: int):
    """
    Met à jour l'état du prompt avec de nouvelles informations
    
    Args:
        state: L'état du prompt à mettre à jour
        new_kv_cache: Nouveau cache des clés/valeurs d'attention
        seq_len: Longueur de la séquence traitée
        
    Returns:
        L'état mis à jour
    """
    state.cache["past_key_values"] = new_kv_cache
    state.start += seq_len
    return state
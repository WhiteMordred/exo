import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

def length_masked_ce_loss(model, inputs, targets, lengths):
    """
    Function de perte cross-entropy avec masque de longueur
    
    Args:
        model: Le modèle PyTorch
        inputs: Tenseur d'entrée [batch_size, seq_len]
        targets: Tenseur de cibles [batch_size, seq_len]
        lengths: Longueurs des séquences
        
    Returns:
        Tenseur de perte scalaire
    """
    # Convertir en tenseurs PyTorch si nécessaire
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs).to(next(model.parameters()).device)
    
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets).to(next(model.parameters()).device)
    
    # Exécuter forward pass
    outputs = model(inputs)
    logits = outputs.logits
    
    # Préparer les masques de longueur
    batch_size, seq_len = inputs.shape
    mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=logits.device)
    
    # Appliquer le masque basé sur les longueurs
    if lengths is not None:
        for i, length in enumerate(lengths):
            if isinstance(length, np.ndarray):
                length = length.item()
            mask[i, length:] = False
    
    # Adapter la forme pour le calcul de la loss
    logits_view = logits.view(-1, logits.size(-1))
    targets_view = targets.view(-1)
    mask_view = mask.view(-1)
    
    # Calculer la loss uniquement sur les positions non masquées
    loss = F.cross_entropy(logits_view[mask_view], targets_view[mask_view], reduction='mean')
    
    return loss
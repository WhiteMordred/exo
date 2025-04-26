import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple
import math

class LossFunction:
    """
    Classe de base pour les fonctions de perte utilisées dans l'entraînement des modèles LLM.
    """
    def __init__(self):
        pass
        
    def compute(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Calcule la fonction de perte.
        
        Args:
            logits: Les logits prédits par le modèle (forme: [batch_size, seq_len, vocab_size])
            targets: Les tokens cibles (forme: [batch_size, seq_len])
            
        Returns:
            La valeur de la perte
        """
        raise NotImplementedError("Implémentez cette méthode dans une sous-classe")
        
    def backward(self, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Calcule le gradient de la perte par rapport aux logits.
        
        Args:
            logits: Les logits prédits par le modèle
            targets: Les tokens cibles
            
        Returns:
            Le gradient de la perte
        """
        raise NotImplementedError("Implémentez cette méthode dans une sous-classe")


class CrossEntropyLoss(LossFunction):
    """
    Fonction de perte d'entropie croisée standard pour l'entraînement des modèles de langage.
    """
    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        """
        Initialise la fonction de perte d'entropie croisée.
        
        Args:
            ignore_index: Les tokens avec cette valeur d'index ne contribuent pas à la perte
            label_smoothing: Facteur de lissage d'étiquette (0.0 = pas de lissage)
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        
    def compute(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Calcule la perte d'entropie croisée.
        
        Args:
            logits: Les logits prédits par le modèle (forme: [batch_size, seq_len, vocab_size])
            targets: Les tokens cibles (forme: [batch_size, seq_len])
            
        Returns:
            La valeur moyenne de la perte d'entropie croisée
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape pour faciliter les calculs
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        # Créer un masque pour ignorer certains tokens
        mask = (targets_flat != self.ignore_index)
        valid_targets = targets_flat[mask]
        valid_logits = logits_flat[mask]
        
        if len(valid_targets) == 0:
            return 0.0  # Pas de tokens valides à traiter
        
        # Appliquer le softmax pour obtenir les probabilités
        # Stabilité numérique : soustraire le max avant l'exponentiation
        logits_max = np.max(valid_logits, axis=1, keepdims=True)
        exp_logits = np.exp(valid_logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Calcul de la perte avec éventuel lissage d'étiquette
        n_classes = vocab_size
        if self.label_smoothing > 0:
            # Distribution lissée: (1-alpha) pour la vraie classe + alpha/K pour toutes les classes
            smooth_targets = np.full_like(probs, self.label_smoothing / (n_classes - 1))
            for i, t in enumerate(valid_targets):
                smooth_targets[i, t] = 1.0 - self.label_smoothing
            loss = -np.sum(smooth_targets * np.log(probs + 1e-10)) / len(valid_targets)
        else:
            # Entropie croisée standard: -log(p[class])
            target_probs = probs[np.arange(len(valid_targets)), valid_targets]
            loss = -np.mean(np.log(target_probs + 1e-10))  # epsilon pour stabilité numérique
            
        return loss
        
    def backward(self, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Calcule le gradient de la perte d'entropie croisée par rapport aux logits.
        
        Args:
            logits: Les logits prédits par le modèle
            targets: Les tokens cibles
            
        Returns:
            Le gradient de la perte
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Initialiser le gradient avec des zéros
        grad = np.zeros_like(logits)
        
        # Reshape pour faciliter les calculs
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        # Créer un masque pour ignorer certains tokens
        mask = (targets_flat != self.ignore_index)
        valid_indices = np.where(mask)[0]
        valid_targets = targets_flat[mask]
        
        if len(valid_targets) == 0:
            return grad  # Pas de tokens valides à traiter
        
        # Calculer les probabilités softmax
        logits_max = np.max(logits_flat[valid_indices], axis=1, keepdims=True)
        exp_logits = np.exp(logits_flat[valid_indices] - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Gradient avec lissage d'étiquette si activé
        if self.label_smoothing > 0:
            smooth_grad = probs.copy()
            for i, t in enumerate(valid_targets):
                # Distribution cible: (1-alpha) pour la vraie classe + alpha/K pour les autres
                smooth_grad[i] = self.label_smoothing / vocab_size
                smooth_grad[i, t] = 1.0 - self.label_smoothing + (self.label_smoothing / vocab_size)
            
            # Gradient = probs - smooth_targets
            smooth_grad = probs - smooth_grad
            for i, idx in enumerate(valid_indices):
                grad_idx = np.unravel_index(idx, (batch_size, seq_len))
                grad[grad_idx[0], grad_idx[1]] = smooth_grad[i] / len(valid_targets)
        else:
            # Gradient standard de l'entropie croisée: probs - one_hot(targets)
            for i, idx in enumerate(valid_indices):
                grad_idx = np.unravel_index(idx, (batch_size, seq_len))
                # Copier les probabilités
                grad[grad_idx[0], grad_idx[1]] = probs[i]
                # Soustraire 1 pour la classe cible
                grad[grad_idx[0], grad_idx[1], valid_targets[i]] -= 1.0
                # Normaliser par le nombre d'exemples valides
                grad[grad_idx[0], grad_idx[1]] /= len(valid_targets)
                
        return grad


class KLDivergenceLoss(LossFunction):
    """
    Fonction de perte de divergence KL pour l'entraînement avec distillation de connaissances.
    """
    def __init__(self, reduction: str = 'mean', temperature: float = 1.0):
        """
        Initialise la fonction de perte KL.
        
        Args:
            reduction: Mode de réduction ('mean', 'sum', 'none')
            temperature: Température pour la distillation (T>1 adoucit les distributions)
        """
        super().__init__()
        self.reduction = reduction
        self.temperature = temperature
        
    def compute(self, logits: np.ndarray, targets_probs: np.ndarray) -> float:
        """
        Calcule la divergence KL entre les distributions de logits et cible.
        
        Args:
            logits: Les logits prédits par le modèle (forme: [batch_size, seq_len, vocab_size])
            targets_probs: Les probabilités cibles (forme: [batch_size, seq_len, vocab_size])
            
        Returns:
            La valeur moyenne de la divergence KL
        """
        # Appliquer la température
        scaled_logits = logits / self.temperature
        
        # Calculer les probabilités softmax pour les logits
        logits_max = np.max(scaled_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(scaled_logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Calculer la divergence KL: sum(target_prob * log(target_prob / pred_prob))
        # Numériquement stable: sum(target_prob * (log(target_prob) - log(pred_prob)))
        kl_div = targets_probs * (np.log(targets_probs + 1e-10) - np.log(probs + 1e-10))
        kl_div = np.sum(kl_div, axis=-1)  # Somme sur la dimension du vocabulaire
        
        # Appliquer la réduction
        if self.reduction == 'mean':
            return np.mean(kl_div)
        elif self.reduction == 'sum':
            return np.sum(kl_div)
        else:  # 'none'
            return kl_div
        
    def backward(self, logits: np.ndarray, targets_probs: np.ndarray) -> np.ndarray:
        """
        Calcule le gradient de la divergence KL par rapport aux logits.
        
        Args:
            logits: Les logits prédits par le modèle
            targets_probs: Les probabilités cibles
            
        Returns:
            Le gradient de la perte
        """
        # Appliquer la température
        scaled_logits = logits / self.temperature
        
        # Calculer les probabilités softmax
        logits_max = np.max(scaled_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(scaled_logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Gradient de la divergence KL par rapport aux logits
        # grad = (probs - target_probs) / temperature
        grad = (probs - targets_probs) / self.temperature
        
        # Appliquer la normalisation selon la méthode de réduction
        if self.reduction == 'mean':
            batch_size, seq_len = logits.shape[:2]
            grad /= (batch_size * seq_len)
        
        return grad


class CombinedLoss(LossFunction):
    """
    Combinaison pondérée de plusieurs fonctions de perte.
    """
    def __init__(self, losses: List[Tuple[LossFunction, float]]):
        """
        Initialise la fonction de perte combinée.
        
        Args:
            losses: Liste de tuples (fonction_perte, poids)
        """
        super().__init__()
        self.losses = losses
        
    def compute(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Calcule la perte combinée.
        
        Args:
            logits: Les logits prédits par le modèle
            targets: Les tokens ou distributions cibles
            
        Returns:
            La valeur combinée des pertes
        """
        total_loss = 0.0
        for loss_fn, weight in self.losses:
            total_loss += weight * loss_fn.compute(logits, targets)
        return total_loss
        
    def backward(self, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Calcule le gradient combiné.
        
        Args:
            logits: Les logits prédits par le modèle
            targets: Les tokens ou distributions cibles
            
        Returns:
            Le gradient combiné des pertes
        """
        grad = np.zeros_like(logits)
        for loss_fn, weight in self.losses:
            grad += weight * loss_fn.backward(logits, targets)
        return grad
        
        
class PerplexityMetric:
    """
    Métrique de perplexité pour évaluer les modèles de langage.
    """
    def __init__(self, ignore_index: int = -100):
        """
        Initialise la métrique de perplexité.
        
        Args:
            ignore_index: Les tokens avec cette valeur d'index sont ignorés
        """
        self.ignore_index = ignore_index
        self.ce_loss = CrossEntropyLoss(ignore_index=ignore_index)
        
    def compute(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Calcule la perplexité.
        
        Args:
            logits: Les logits prédits par le modèle
            targets: Les tokens cibles
            
        Returns:
            La valeur de la perplexité
        """
        # La perplexité est exp(entropie croisée)
        ce_loss = self.ce_loss.compute(logits, targets)
        return math.exp(ce_loss)
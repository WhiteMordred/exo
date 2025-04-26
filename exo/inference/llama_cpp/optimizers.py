import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import math

class Optimizer:
    """
    Classe de base pour les optimiseurs utilisés dans l'entraînement des modèles.
    """
    def __init__(self, learning_rate: float = 0.001):
        """
        Initialise l'optimiseur.
        
        Args:
            learning_rate: Taux d'apprentissage initial
        """
        self.learning_rate = learning_rate
        self.steps = 0
        
    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """
        Met à jour les paramètres en fonction des gradients.
        
        Args:
            params: Liste des paramètres du modèle
            grads: Liste des gradients correspondants
            
        Returns:
            Liste des paramètres mis à jour
        """
        raise NotImplementedError("Implémentez cette méthode dans une sous-classe")
        
    def set_learning_rate(self, lr: float):
        """
        Modifie le taux d'apprentissage.
        
        Args:
            lr: Nouveau taux d'apprentissage
        """
        self.learning_rate = lr
        
    def get_state(self) -> Dict[str, Any]:
        """
        Récupère l'état interne de l'optimiseur pour sauvegarde.
        
        Returns:
            Dictionnaire d'état
        """
        return {"learning_rate": self.learning_rate, "steps": self.steps}
        
    def load_state(self, state: Dict[str, Any]):
        """
        Charge un état sauvegardé.
        
        Args:
            state: Dictionnaire d'état
        """
        self.learning_rate = state.get("learning_rate", self.learning_rate)
        self.steps = state.get("steps", 0)


class SGD(Optimizer):
    """
    Optimiseur de descente de gradient stochastique (SGD) avec momentum.
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0):
        """
        Initialise l'optimiseur SGD.
        
        Args:
            learning_rate: Taux d'apprentissage initial
            momentum: Facteur de momentum (0.0 = pas de momentum)
            weight_decay: Facteur de régularisation L2
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = []  # Pour stocker les vitesses avec momentum
        
    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """
        Met à jour les paramètres avec SGD.
        
        Args:
            params: Liste des paramètres du modèle
            grads: Liste des gradients correspondants
            
        Returns:
            Liste des paramètres mis à jour
        """
        # Initialiser les vitesses si nécessaire
        if not self.velocities:
            self.velocities = [np.zeros_like(param) for param in params]
            
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            if self.weight_decay > 0:
                # Appliquer la régularisation L2
                grad = grad + self.weight_decay * param
            
            if self.momentum > 0:
                # Mettre à jour avec momentum
                self.velocities[i] = self.momentum * self.velocities[i] - self.learning_rate * grad
                updated_param = param + self.velocities[i]
            else:
                # Mise à jour standard
                updated_param = param - self.learning_rate * grad
                
            updated_params.append(updated_param)
            
        self.steps += 1
        return updated_params
        
    def get_state(self) -> Dict[str, Any]:
        """Récupère l'état incluant les vitesses de momentum"""
        state = super().get_state()
        state["momentum"] = self.momentum
        state["weight_decay"] = self.weight_decay
        state["velocities"] = [v.tolist() for v in self.velocities] if self.velocities else []
        return state
        
    def load_state(self, state: Dict[str, Any]):
        """Charge un état sauvegardé incluant les vitesses"""
        super().load_state(state)
        self.momentum = state.get("momentum", self.momentum)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        velocities_list = state.get("velocities", [])
        if velocities_list:
            self.velocities = [np.array(v) for v in velocities_list]


class Adam(Optimizer):
    """
    Optimiseur Adam (Adaptive Moment Estimation).
    """
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, 
                 epsilon: float = 1e-8, weight_decay: float = 0.0):
        """
        Initialise l'optimiseur Adam.
        
        Args:
            learning_rate: Taux d'apprentissage initial
            beta1: Coefficient d'estimation du premier moment
            beta2: Coefficient d'estimation du second moment
            epsilon: Terme pour éviter la division par zéro
            weight_decay: Facteur de régularisation L2
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = []  # Premier moment (moyenne du gradient)
        self.v = []  # Second moment (variance du gradient)
        
    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """
        Met à jour les paramètres avec Adam.
        
        Args:
            params: Liste des paramètres du modèle
            grads: Liste des gradients correspondants
            
        Returns:
            Liste des paramètres mis à jour
        """
        # Initialiser les moments si nécessaire
        if not self.m:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]
            
        self.steps += 1
        
        # Calculer les facteurs de correction du biais
        bias_correction1 = 1.0 - (self.beta1 ** self.steps)
        bias_correction2 = 1.0 - (self.beta2 ** self.steps)
        corrected_lr = self.learning_rate * np.sqrt(bias_correction2) / bias_correction1
        
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            if self.weight_decay > 0:
                # Appliquer la régularisation L2
                grad = grad + self.weight_decay * param
            
            # Mettre à jour les estimations des moments
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)
            
            # Calculer la mise à jour
            update = corrected_lr * self.m[i] / (np.sqrt(self.v[i]) + self.epsilon)
            updated_param = param - update
            updated_params.append(updated_param)
            
        return updated_params
        
    def get_state(self) -> Dict[str, Any]:
        """Récupère l'état incluant les moments Adam"""
        state = super().get_state()
        state["beta1"] = self.beta1
        state["beta2"] = self.beta2
        state["epsilon"] = self.epsilon
        state["weight_decay"] = self.weight_decay
        state["m"] = [m.tolist() for m in self.m] if self.m else []
        state["v"] = [v.tolist() for v in self.v] if self.v else []
        return state
        
    def load_state(self, state: Dict[str, Any]):
        """Charge un état sauvegardé incluant les moments"""
        super().load_state(state)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2 = state.get("beta2", self.beta2)
        self.epsilon = state.get("epsilon", self.epsilon)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        m_list = state.get("m", [])
        v_list = state.get("v", [])
        if m_list:
            self.m = [np.array(m) for m in m_list]
        if v_list:
            self.v = [np.array(v) for v in v_list]


class AdamW(Adam):
    """
    Optimiseur AdamW (Adam avec décroissance de poids découplée).
    """
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8, weight_decay: float = 0.01):
        """
        Initialise l'optimiseur AdamW.
        
        Args:
            learning_rate: Taux d'apprentissage initial
            beta1: Coefficient d'estimation du premier moment
            beta2: Coefficient d'estimation du second moment
            epsilon: Terme pour éviter la division par zéro
            weight_decay: Facteur de régularisation L2 découplée
        """
        super().__init__(learning_rate, beta1, beta2, epsilon, weight_decay)
        
    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """
        Met à jour les paramètres avec AdamW.
        
        Args:
            params: Liste des paramètres du modèle
            grads: Liste des gradients correspondants
            
        Returns:
            Liste des paramètres mis à jour
        """
        # Initialiser les moments si nécessaire
        if not self.m:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]
            
        self.steps += 1
        
        # Calculer les facteurs de correction du biais
        bias_correction1 = 1.0 - (self.beta1 ** self.steps)
        bias_correction2 = 1.0 - (self.beta2 ** self.steps)
        corrected_lr = self.learning_rate * np.sqrt(bias_correction2) / bias_correction1
        
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Dans AdamW, la régularisation L2 est appliquée directement aux poids
            # plutôt qu'au gradient, ce qui découple la régularisation du taux d'apprentissage adaptatif
            
            # Mettre à jour les estimations des moments (sans régularisation)
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)
            
            # Calculer la mise à jour
            update = corrected_lr * self.m[i] / (np.sqrt(self.v[i]) + self.epsilon)
            
            # Appliquer la décroissance de poids découplée
            weight_decay_update = self.learning_rate * self.weight_decay * param
            
            # Mise à jour des paramètres
            updated_param = param - update - weight_decay_update
            updated_params.append(updated_param)
            
        return updated_params


class LRScheduler:
    """
    Classe de base pour les planificateurs de taux d'apprentissage.
    """
    def __init__(self, optimizer: Optimizer):
        """
        Initialise le planificateur.
        
        Args:
            optimizer: L'optimiseur dont on ajuste le taux d'apprentissage
        """
        self.optimizer = optimizer
        self.base_lr = optimizer.learning_rate
        self.last_lr = self.base_lr
        
    def step(self):
        """
        Met à jour le taux d'apprentissage selon la politique du planificateur.
        """
        raise NotImplementedError("Implémentez cette méthode dans une sous-classe")
        
    def get_lr(self) -> float:
        """
        Récupère le taux d'apprentissage actuel.
        
        Returns:
            Taux d'apprentissage
        """
        return self.last_lr
        
        
class StepLR(LRScheduler):
    """
    Planificateur qui réduit le taux d'apprentissage par un facteur gamma
    toutes les step_size époques.
    """
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        """
        Initialise le planificateur StepLR.
        
        Args:
            optimizer: L'optimiseur dont on ajuste le taux d'apprentissage
            step_size: Nombre d'époques entre deux réductions
            gamma: Facteur de réduction du taux d'apprentissage
        """
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0
        
    def step(self):
        """
        Met à jour le taux d'apprentissage selon la politique par étapes.
        """
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            new_lr = self.last_lr * self.gamma
            self.optimizer.set_learning_rate(new_lr)
            self.last_lr = new_lr
            
            
class CosineAnnealingLR(LRScheduler):
    """
    Planificateur qui ajuste le taux d'apprentissage selon un cycle cosinus.
    """
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0):
        """
        Initialise le planificateur CosineAnnealingLR.
        
        Args:
            optimizer: L'optimiseur dont on ajuste le taux d'apprentissage
            T_max: Nombre maximal d'itérations (demi-cycle)
            eta_min: Taux d'apprentissage minimal
        """
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        self.iteration = 0
        
    def step(self):
        """
        Met à jour le taux d'apprentissage selon la politique cosinus.
        """
        self.iteration += 1
        if self.iteration > self.T_max:
            self.iteration = 1
            
        # Calcul du taux d'apprentissage selon le cycle cosinus
        cos_factor = (1 + math.cos(math.pi * self.iteration / self.T_max)) / 2.0
        new_lr = self.eta_min + (self.base_lr - self.eta_min) * cos_factor
        
        self.optimizer.set_learning_rate(new_lr)
        self.last_lr = new_lr
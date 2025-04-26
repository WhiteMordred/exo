import asyncio
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path

try:
    from llama_cpp import Llama
    has_llama_cpp = True
except ImportError:
    has_llama_cpp = False

class StatefulModel:
    """
    Classe pour gérer l'état persistant d'un modèle LlamaCpp à travers plusieurs appels d'inférence.
    Cette classe permet de conserver le contexte KV-cache entre les appels et d'optimiser la génération.
    """
    def __init__(self, model: 'Llama'):
        """
        Initialise un modèle avec état à partir d'un modèle LlamaCpp existant.
        
        Args:
            model: Une instance de modèle LlamaCpp
        """
        if not has_llama_cpp:
            raise ImportError("llama-cpp-python n'est pas installé")
        
        self.model = model
        self.context_size = model.n_ctx()  # Taille maximale du contexte
        self.current_tokens = []           # Tokens actuellement dans le contexte
        self.full_history = []             # Historique complet des tokens
        self.generation_settings = {}      # Paramètres de génération par défaut
        self.kv_cache_active = True        # Si le cache KV est actif
        
    def reset(self):
        """Réinitialise l'état du modèle"""
        # Réinitialiser le contexte de llama.cpp
        if hasattr(self.model, 'reset'):
            self.model.reset()
        # Réinitialiser notre suivi interne
        self.current_tokens = []
        self.full_history = []
        
    def get_context_length(self) -> int:
        """Renvoie la longueur actuelle du contexte"""
        return len(self.current_tokens)
        
    def get_available_context_space(self) -> int:
        """Renvoie l'espace de contexte disponible"""
        return self.context_size - len(self.current_tokens)
    
    def add_to_history(self, tokens: List[int]):
        """
        Ajoute des tokens à l'historique et au contexte actuel.
        
        Args:
            tokens: Liste des tokens à ajouter
        """
        self.full_history.extend(tokens)
        self.current_tokens.extend(tokens)
        
        # Si le contexte dépasse la taille maximale, tronquer
        if len(self.current_tokens) > self.context_size:
            overflow = len(self.current_tokens) - self.context_size
            self.current_tokens = self.current_tokens[overflow:]
    
    async def evaluate_tokens(self, tokens: List[int]) -> Tuple[int, float]:
        """
        Évalue un ensemble de tokens et renvoie le token suivant et sa probabilité.
        Cette méthode maintient le KV-cache interne de llama.cpp.
        
        Args:
            tokens: Liste des tokens à évaluer
            
        Returns:
            Tuple (token suivant, log_prob)
        """
        # Ajouter les tokens à l'historique
        self.add_to_history(tokens)
        
        # Évaluer les tokens avec llama.cpp qui gère automatiquement le cache
        result = self.model.eval(tokens)
        
        # Llama.cpp renvoie directement le token généré
        generated_token = result
        
        # Dans llama.cpp, il faut appeler séparément les log-probs
        # Note: certaines versions n'ont pas cette fonctionnalité
        log_prob = -1.0  # Valeur par défaut
        if hasattr(self.model, 'get_logits'):
            logits = self.model.get_logits()
            if logits is not None:
                # Convertir en probabilités avec softmax
                probs = np.exp(logits - np.max(logits))
                probs = probs / np.sum(probs)
                log_prob = np.log(probs[generated_token]) if probs[generated_token] > 0 else -float('inf')
        
        return generated_token, log_prob
    
    async def generate(self, 
                       prompt: str, 
                       max_tokens: int = 100, 
                       temperature: float = 0.7,
                       top_p: float = 0.9,
                       stop_tokens: List[str] = None) -> str:
        """
        Génère du texte à partir d'un prompt en conservant l'état du modèle.
        
        Args:
            prompt: Texte d'entrée pour la génération
            max_tokens: Nombre maximum de tokens à générer
            temperature: Température pour l'échantillonnage (0=déterministe)
            top_p: Valeur pour le nucleus sampling
            stop_tokens: Liste de tokens qui arrêtent la génération
            
        Returns:
            Texte généré
        """
        # Encoder le prompt en tokens
        input_tokens = self.model.tokenize(prompt.encode())
        self.add_to_history(input_tokens)
        
        # Définir les paramètres de génération
        params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop_tokens if stop_tokens else []
        }
        
        # Générer le texte avec llama.cpp
        completion = self.model.create_completion(prompt, **params)
        
        # Extraire le texte généré
        generated_text = ""
        if isinstance(completion, dict):
            generated_text = completion.get("choices", [{}])[0].get("text", "")
        
        # Ajouter les tokens générés à l'historique
        if generated_text:
            generated_tokens = self.model.tokenize(generated_text.encode())
            self.add_to_history(generated_tokens)
        
        return generated_text
    
    def get_token_probability(self, token: int) -> float:
        """
        Récupère la probabilité d'un token spécifique basé sur l'état actuel.
        
        Args:
            token: ID du token à évaluer
            
        Returns:
            Probabilité du token
        """
        # Si le modèle a la capacité d'accéder aux logits
        if hasattr(self.model, 'get_logits'):
            logits = self.model.get_logits()
            if logits is not None:
                # Calculer les probabilités avec softmax
                probs = np.exp(logits - np.max(logits))
                probs = probs / np.sum(probs)
                return probs[token] if 0 <= token < len(probs) else 0.0
        
        return 0.0  # Par défaut
    
    def set_generation_settings(self, settings: Dict[str, Any]):
        """
        Définit les paramètres par défaut pour la génération.
        
        Args:
            settings: Dictionnaire de paramètres
        """
        self.generation_settings.update(settings)
    
    def get_vocab_size(self) -> int:
        """Renvoie la taille du vocabulaire du modèle"""
        return self.model.n_vocab()

    def detokenize(self, tokens: List[int]) -> str:
        """
        Convertit une liste de tokens en texte.
        
        Args:
            tokens: Liste des tokens à convertir
            
        Returns:
            Texte décodé
        """
        try:
            # Utiliser la méthode native de llama.cpp
            return self.model.detokenize(tokens).decode('utf-8')
        except Exception as e:
            print(f"Erreur lors de la détokenisation: {e}")
            return ""
    
    def tokenize(self, text: str) -> List[int]:
        """
        Convertit du texte en tokens.
        
        Args:
            text: Texte à convertir
            
        Returns:
            Liste des tokens
        """
        try:
            return self.model.tokenize(text.encode())
        except Exception as e:
            print(f"Erreur lors de la tokenisation: {e}")
            return []
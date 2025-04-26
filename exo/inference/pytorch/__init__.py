# PyTorch inference engine implementation
# This file marks the directory as a Python package

# Dépendances recommandées:
# - accelerate>=0.26.0: Pour le chargement efficace des modèles avec low_cpu_mem_usage et device_map
# - bitsandbytes: Pour la quantification 8-bit et 4-bit si nécessaire
# - optimum: Pour des optimisations supplémentaires

import sys

def check_dependencies():
    """Vérifie les dépendances recommandées et émet des avertissements si nécessaire"""
    missing_packages = []
    
    try:
        import accelerate
    except ImportError:
        missing_packages.append("accelerate>=0.26.0")
    
    if missing_packages:
        print("PyTorch inference engine: Les dépendances recommandées suivantes ne sont pas installées:")
        for pkg in missing_packages:
            print(f" - {pkg}")
        print("Vous pouvez les installer avec: pip install " + " ".join(missing_packages))
        print("L'engine fonctionnera mais avec des performances ou capacités réduites.")
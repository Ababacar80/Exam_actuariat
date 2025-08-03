"""
Package de prédiction de fraude à l'assurance.

Ce package contient des modules pour :
- Chargement des données (data_loading)
- Préprocessing des données (data_processing)
- Analyse exploratoire (exploration)
- Extraction de features (features)
- Visualisation (visualization)
- Modélisation (models)
- Évaluation (evaluation)
"""

from . import data_loading
from . import data_processing
from . import exploration
from . import features
from . import visualization
from . import models
from . import evaluation

__version__ = "0.1.0"
__author__ = "ababacar sagna"
__email__ = "ababacarsagna10@gmail.com"

__all__ = [
    "data_loading",
    "data_processing", 
    "exploration",
    "features",
    "visualization",
    "models",
    "evaluation"
]
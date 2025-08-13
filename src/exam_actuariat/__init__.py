"""
Package de prédiction de sinistres d'assurance.
"""

from . import data_loading
from . import data_processing
from . import exploration
from . import features
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
    "models",
    "evaluation"
]
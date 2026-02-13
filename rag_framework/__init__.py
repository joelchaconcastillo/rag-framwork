"""RAG Framework - A comprehensive framework for Retrieval-Augmented Generation."""

__version__ = "0.1.0"

from . import clients
from . import preprocessing
from . import processing
from . import retrieval
from . import evaluation

__all__ = [
    "clients",
    "preprocessing",
    "processing",
    "retrieval",
    "evaluation",
]

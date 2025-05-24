# src/ai/ultimate/__init__.py
"""
Ultimate HOI4 AI Package
DreamerV3 + RND + NEC = Autonomous Learning
"""

from .ultimate_ai import UltimateHOI4AI
from .world_model import WorldModel
from .curiosity import RNDCuriosity, CombinedCuriosity
from .episodic_memory import NeuralEpisodicControl, PersistentEpisodicMemory
from .train_ultimate import UltimateTrainer

__all__ = [
    'UltimateHOI4AI',
    'WorldModel',
    'RNDCuriosity',
    'CombinedCuriosity',
    'NeuralEpisodicControl',
    'PersistentEpisodicMemory',
    'UltimateTrainer'
]

__version__ = '1.0.0'
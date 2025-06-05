# src/config.py - Centralized configuration for HOI4 AI
from dataclasses import dataclass
import os


@dataclass
class HOI4Config:
    """Central configuration for all HOI4 AI components"""

    # Display settings
    game_resolution: tuple = (3840, 2160)
    ai_resolution: tuple = (1280, 720)

    # Performance settings
    ocr_cache_duration: float = 2.0  # seconds
    memory_query_interval: int = 80  # steps
    frame_skip: int = 3  # process every Nth frame

    # Paths
    checkpoint_dir: str = "checkpoints"
    memory_dir: str = "hoi4_persistent_memory"
    log_dir: str = "logs"

    # Training hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 32
    curiosity_weight: float = 0.6
    goal_weight: float = 0.4
    memory_size: int = 50000
    replay_buffer_size: int = 100000

    # Action space
    action_keys: list = None  # Set in __post_init__

    def __post_init__(self):
        # Create directories if they don't exist
        for dir_path in [self.checkpoint_dir, self.memory_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Default HOI4 keys
        if self.action_keys is None:
            self.action_keys = [
                'space', 'escape', 'q', 'w', 'e', 'r', 't', 'y',
                'b', 'v', 'n', 'm', '1', '2', '3', '4', '5',
                'tab', 'enter', 'delete'
            ]


# Global config instance
CONFIG = HOI4Config()
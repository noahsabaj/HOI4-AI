# src/config.py - Centralized configuration for HOI4 AI
from dataclasses import dataclass, field
import os


@dataclass
class PerformanceConfig:
    """Performance tuning for Ultimate AI"""
    # Threading
    learning_thread_priority: float = 0.3  # Lower priority for learning
    eval_thread_priority: float = 0.2  # Lowest for strategic eval

    # Timing targets (ms)
    target_frame_time: float = 50.0  # 20 FPS minimum
    target_decision_time: float = 20.0  # Fast policy budget
    target_action_time: float = 100.0  # Execution budget

    # Caching
    strategy_cache_duration: float = 60.0  # Strategic advice validity

    # Queue sizes
    learn_queue_size: int = 64
    eval_queue_size: int = 10


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

    # Performance configuration
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

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
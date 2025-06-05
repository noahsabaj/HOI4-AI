# src/ai/ultimate/ultimate_ai.py
"""
The Ultimate HOI4 AI
Combines DreamerV3 world model, RND curiosity, and Neural Episodic Control
"""

import os
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import numpy as np
import pyautogui
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange

# Add this line with the other imports at the top:
from src.config import CONFIG
from src.perception.dynamic_ocr import DynamicOCR

from src.utils.common import extract_number, detect_screen_type

# Import our components
from .fast_policy import FastPolicy
from .curiosity import CombinedCuriosity
from .episodic_memory import NeuralEpisodicControl
from .fast_memory import FastEpisodicMemory  # NEW: LMDB memory
from .world_model import WorldModel
from .simple_world_model import SimpleWorldModel, LATENT_SIZE, NUM_ACTIONS

# Import existing components
from ...perception.ocr import HOI4OCR
from ...strategy.evaluation import StrategicEvaluator


class ActionSpace:
    """Dynamic action space that grows as AI discovers UI elements"""

    def __init__(self, ocr_instance):
        self.ocr = ocr_instance
        self.base_keys = [
            'space',  # Pause/unpause
            'escape',  # Back/menu
            'q',  # Politics
            'w',  # Research
            'e',  # Diplomacy
            'r',  # Trade
            't',  # Production
            'y',  # Logistics
            'b',  # Build mode
            'v',  # Army
            'n',  # Navy
            'm',  # Air
            '1', '2', '3', '4', '5',  # Speed controls
            'tab',  # Cycle
            'enter',  # Confirm
            'delete',  # Delete unit
            'f1', 'f2', 'f3', 'f4', 'f5',  # Function keys for menus
        ]
        self.exploration_grid_size = 10

    def get_action_size(self):
        """Dynamic size based on discovered UI elements"""
        discovered_buttons = sum(1 for elem in self.ocr.ui_elements.values()
                                 if elem.get('type') == 'button' and elem['confidence'] > 0.7)
        # Keys + discovered buttons + exploration grid
        return len(self.base_keys) + discovered_buttons + (self.exploration_grid_size * self.exploration_grid_size)

    def decode_action(self, action_idx: int) -> Dict:
        """Convert action index to executable action"""

        # First N actions are keys
        if action_idx < len(self.base_keys):
            return {
                'type': 'key',
                'key': self.base_keys[action_idx],
                'description': f'Press {self.base_keys[action_idx]}'
            }

        # Next actions are discovered buttons
        button_actions = [
            (name, data) for name, data in self.ocr.ui_elements.items()
            if data.get('type') == 'button' and data['confidence'] > 0.7
        ]

        button_idx = action_idx - len(self.base_keys)
        if button_idx < len(button_actions):
            name, data = button_actions[button_idx]
            x, y = data['click_position']
            return {
                'type': 'click',
                'x': x,
                'y': y,
                'button': 'left',
                'description': f'Click learned {name} button'
            }

        # Remaining actions are exploration clicks
        import pyautogui
        width, height = pyautogui.size()
        exploration_idx = action_idx - len(self.base_keys) - len(button_actions)

        # Systematic exploration pattern
        grid_x = exploration_idx % self.exploration_grid_size
        grid_y = (exploration_idx // self.exploration_grid_size) % self.exploration_grid_size

        # Add some randomness to exact position within grid cell
        import random
        cell_width = width / self.exploration_grid_size
        cell_height = height / self.exploration_grid_size

        x = int((grid_x + random.uniform(0.2, 0.8)) * cell_width)
        y = int((grid_y + random.uniform(0.2, 0.8)) * cell_height)

        return {
            'type': 'click',
            'x': x,
            'y': y,
            'button': 'left',
            'description': f'Explore click at grid ({grid_x},{grid_y})'
        }

    def get_discovered_actions_info(self) -> Dict:
        """Get information about discovered actions for debugging"""
        discovered_buttons = [
            name for name, data in self.ocr.ui_elements.items()
            if data.get('type') == 'button' and data['confidence'] > 0.7
        ]

        return {
            'total_actions': self.get_action_size(),
            'key_actions': len(self.base_keys),
            'discovered_buttons': len(discovered_buttons),
            'button_names': discovered_buttons,
            'exploration_slots': self.exploration_grid_size * self.exploration_grid_size
        }


class UltimateHOI4AI:
    """
    The complete Ultimate HOI4 AI system

    Features:
    - DreamerV3 world model for planning
    - RND for curiosity-driven exploration
    - NEC for fast learning from experience
    - Persistent memory across games
    - Integration with existing OCR and understanding
    """

    def __init__(
            self,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            checkpoint_path: Optional[str] = None
    ):
        print("🚀 Initializing Ultimate HOI4 AI...")

        from src.utils.logger import get_logger
        self.log = get_logger("ai")
        self.log.info("Logger for UltimateHOI4AI ready")

        self.device = device

        # Frame skipping for performance
        self.frame_skip = CONFIG.frame_skip
        self.frame_counter = 0

        # Initialize OCR first
        from src.perception.dynamic_ocr import DynamicOCR
        self.ocr = DynamicOCR()

        # Initialize action space with OCR
        self.action_space = ActionSpace(self.ocr)
        # Fast decision policy
        print("  ⚡ Loading fast policy...")
        self.fast_policy = FastPolicy(self.action_space, device=self.device)
        self.action_size = self.action_space.get_action_size()

        print("  🔍 Loading curiosity module...")
        self.curiosity = CombinedCuriosity(
            observation_dim=LATENT_SIZE,  # From world model encoder
            rnd_weight=0.6,
            goal_weight=0.4
        )

        # Simple world model for training
        self.world_model = SimpleWorldModel().to(self.device)
        self.wm_opt = torch.optim.Adam(self.world_model.parameters(), lr=2e-4)
        self.global_step = 0

        # 🧠 Loading episodic memory with correct state_dim
        state_dim = LATENT_SIZE

        self.nec = NeuralEpisodicControl(
            state_dim=state_dim,
            key_dim=128,
            memory_size=50000
        )

        print("  💾 Loading persistent memory...")
        self.persistent_memory = FastEpisodicMemory()

        # Keep existing components
        print("  👁️ Loading perception...")
        self.ocr = DynamicOCR()
        self.evaluator = StrategicEvaluator()

        # Optimizers
        self.world_model_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=3e-4
        )

        # Replay buffer for world model training
        self.replay_buffer = ReplayBuffer(capacity=100000)

        # Current state
        self.current_rssm_state = None
        self.last_action = None
        self.episode_steps = 0
        self.total_steps = 0

        # Metrics
        self.metrics = {
            'total_reward': 0,
            'intrinsic_reward': 0,
            'episode_count': 0,
            'discoveries': [],
            'current_game_id': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)

        print("✅ Ultimate HOI4 AI ready!")

    def should_process_frame(self) -> bool:
        """Determine if current frame should be processed"""
        self.frame_counter += 1
        return self.frame_counter % self.frame_skip == 0

    def _extract_number(self, text: str) -> int:
        """Extract number from text"""
        import re
        match = re.search(r'(\d+)', text)
        return int(match.group(1)) if match else 0

    def observe(self, screenshot: Image.Image) -> Tuple[torch.Tensor, Dict]:
        """
        Process screenshot into state.

        * OCR (the slow part) runs only once every 2 seconds and is
          cached in self._ocr_cache.
        * Everything else (resize, encode) still happens every frame.
        """
        import time  # local import—avoids circulars

        # Skip frames for performance
        if not self.should_process_frame():
            # Return cached observation if skipping
            if hasattr(self, '_last_observation'):
                return self._last_observation

        # Resize for world model
        screenshot_resized = screenshot.resize((1280, 720))

        # Convert to tensor for the encoder
        obs_array = np.array(screenshot_resized)
        obs_tensor = (
                torch.tensor(obs_array, dtype=torch.float32)
                .permute(2, 0, 1) / 255.0
        ).unsqueeze(0).to(self.device)

        # ── OCR throttled to once every 2 s ───────────────────────────
        now = time.time()

        current_hash = hash(screenshot.tobytes()[::1000])  # Sample pixels for change detection
        screen_changed = not hasattr(self, '_last_screen_hash') or current_hash != self._last_screen_hash

        # run full OCR only every 5 seconds to save ~0.3 s/step
        if screen_changed or (not hasattr(self, "_last_ocr")) or (now - self._last_ocr > 5.0):
            self._ocr_cache = self.ocr.extract_all_text(screenshot)
            self._last_ocr = now
            self._last_screen_hash = current_hash
        ocr_data = self._ocr_cache

        # Encode observation
        with torch.no_grad():
            encoded_obs = self.world_model.encoder(obs_tensor)

        # Parse game state
        game_info = {
            "ocr_data": ocr_data,
            "screen_type": detect_screen_type(ocr_data),
            "game_date": ocr_data.get("date", "Unknown"),
            "political_power": self._extract_number(ocr_data.get("political_power", "0")),
            "factories": {
                "civilian": self._extract_factory_count(ocr_data.get("factories", ""), 0),
                "military": self._extract_factory_count(ocr_data.get("factories", ""), 1),
            },
        }

        # Cache observation
        self._last_observation = (encoded_obs, game_info)
        return encoded_obs, game_info

    def act(self, screenshot: Image.Image) -> Dict:
        """
        Fast decision function (<50ms total)

        Args:
            screenshot: Current game screenshot

        Returns:
            Action to execute
        """
        start_time = time.perf_counter()

        # Observe (cached OCR)
        encoded_obs, game_info = self.observe(screenshot)

        # Get cached strategic advice (instant)
        from ..ultimate.train_ultimate import UltimateTrainer
        trainer = getattr(self, '_trainer_ref', None)

        if trainer and hasattr(trainer, 'strategic_reasoner'):
            advice = trainer.strategic_reasoner.get_cached_advice()
            strategy = advice['strategy']
            suggestions = advice.get('suggestions', [])
        else:
            strategy = 'exploring'
            suggestions = []

        # Fast decision (<20ms)
        action_idx = self.fast_policy.act(
            encoded_obs,
            cached_strategy=strategy,
            cached_suggestions=suggestions
        )

        # Decode action
        action = self.action_space.decode_action(action_idx)

        # Update last action for learning
        self.last_action = F.one_hot(
            torch.tensor([action_idx], device=self.device),
            num_classes=self.action_size
        ).float()

        # Update metrics
        self.episode_steps += 1
        self.total_steps += 1

        # Log timing
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > 50:
            print(f"⚠️ Slow frame: {elapsed_ms:.1f}ms")

        return action

    def learn_from_transition(
            self,
            prev_screenshot: Image.Image,
            action: Dict,
            next_screenshot: Image.Image,
            reward: float
    ):
        """
        Learn from a single transition

        Args:
            prev_screenshot: Previous screenshot
            action: Action taken
            next_screenshot: Next screenshot
            reward: Reward received
        """
        # Process observations
        prev_obs, prev_info = self.observe(prev_screenshot)
        next_obs, next_info = self.observe(next_screenshot)

        # Store in replay buffer
        self.replay_buffer.add(
            prev_obs.cpu(),
            self.last_action.cpu(),
            torch.tensor([reward]),
            next_obs.cpu(),
            torch.tensor([False])  # Episode not done
        )

        # Initialize state_features before using it
        state_features = None

        # Update NEC
        if self.current_rssm_state is not None:
            state_features = self._get_state_features()
            self.nec.write(state_features, torch.tensor([reward], device=self.device))
        else:
            # If no RSSM state, use encoded observation as features
            state_features = prev_obs.squeeze(0)

        # Store significant experiences
        if abs(reward) > 5.0 or self.metrics["intrinsic_reward"] > 10.0:
            if state_features is not None:
                self._store_persistent_memory(
                    state_features,
                    action,
                    f"Reward: {reward:.1f}",
                    prev_info,
                )

        # Learn button locations from successful actions
        if action['type'] == 'click' and reward > 0:
            # Extract button name from description if available
            desc = action.get('description', '')
            if 'Click' in desc:
                button_name = desc.replace('Click ', '').lower()
                self.ocr.learn_button_location(
                    button_name,
                    (action['x'], action['y']),
                    success=True
                )

        # Save UI memory periodically
        if self.total_steps % 100 == 0:
            self.ocr.save_ui_memory()

        # Train curiosity every 50 env steps
        if self.total_steps % 50 == 0:
            self._train_curiosity()

        # Train world-model every 200 env steps
        if (
                self.total_steps % 200 == 0
                and len(self.replay_buffer) > 256
        ):
            self._train_world_model()

        # Update running totals
        self.metrics["total_reward"] += reward

    # ------------------------------------------------------------------
    # Imagination-based planner (Dreamer-style)
    # ------------------------------------------------------------------
    def _imagination_policy(self, state_features: torch.Tensor) -> int:
        """Simple random policy for now (no imagination with SimpleWorldModel)"""
        # Just return a random action until world model is trained
        return np.random.randint(0, self.action_size)

    def _exploitation_policy(self, state_features: torch.Tensor) -> int:
        """Exploit learned knowledge"""
        # For now, use imagination policy
        # TODO: Add explicit Q-learning
        return self._imagination_policy(state_features)

    def _get_state_features(self) -> torch.Tensor:
        """Get current state features for NEC"""
        if self.current_rssm_state is None:
            return torch.zeros(LATENT_SIZE, device=self.device)

        return self.current_rssm_state['latent'].squeeze(0)

    def _initialize_rssm_state(self) -> Dict:
        """Initialize state (for SimpleWorldModel)"""
        return {
            'latent': torch.zeros(1, LATENT_SIZE, device=self.device)
        }

    def _check_memory(self, state_features: torch.Tensor, game_info: Dict):
        """
        Query ChromaDB only once every 20 env steps.
        """
        # recall every 80 env steps instead of 20 → saves ~0.15 s/step
        if self.total_steps % 80 != 0:
            return []  # skip most frames

        state_np = state_features.detach().cpu().numpy()

        memories = self.persistent_memory.search_similar(
            state_np[:128], k=3
        )

        if memories:
            # print at most once every 100 steps to avoid spam
            if self.total_steps % 100 == 0:
                self.log.info(f"💭 Remembering: {memories[0]['description'][:100]}...")

        return memories

    def _store_persistent_memory(self, state_features: torch.Tensor, action: Dict, outcome: str, game_info: Dict):
        """Store important experience in persistent memory"""
        # Use NEC key encoder
        with torch.no_grad():
            key_encoding = self.nec.encode_key(state_features)

        # Extract reward from outcome
        reward = 10.0 if "success" in outcome.lower() else -1.0

        context = {
            'game_date': game_info.get('game_date', 'Unknown'),
            'political_power': game_info.get('political_power', 0),
            'civilian_factories': game_info.get('factories', {}).get('civilian', 0),
            'military_factories': game_info.get('factories', {}).get('military', 0),
            'screen_type': game_info.get('screen_type', 'unknown'),
            'outcome': outcome
        }

        # Use the new fast memory
        self.persistent_memory.store_experience(
            key_encoding.cpu().numpy(),
            action,
            reward,
            context
        )

    def _train_world_model(self):
        """Train SimpleWorldModel on replay buffer"""
        if len(self.replay_buffer) < 256:
            return

        # Sample batch
        batch = self.replay_buffer.sample(8)

        # The replay buffer stores observations as 256-d encoded vectors
        # We need to train the model to predict next states and rewards

        obs = batch['obs'].to(self.device)
        actions = batch['action'].to(self.device)
        rewards = batch['reward'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)

        # For SimpleWorldModel, we'll just train reward prediction for now
        with torch.no_grad():
            current_latent = obs  # Already encoded

        # Simple training: predict rewards from states
        pred_rewards = self.world_model.reward_head(current_latent).squeeze()
        loss = F.mse_loss(pred_rewards, rewards.float())

        self.wm_opt.zero_grad()
        loss.backward()
        self.wm_opt.step()

        if self.total_steps % 1000 == 0:
            self.log.info(f"🧠 World model loss: {loss.item():.4f}")

    def _train_curiosity(self):
        """Train RND predictor"""
        if len(self.replay_buffer) < 100:
            return

        # Sample batch (already stored as encoded 1024-d vectors)
        batch = self.replay_buffer.sample(32)
        encoded_obs = batch['obs'].to(self.device).view(-1, 1024)

        # One gradient step on the predictor
        _ = self.curiosity.rnd.train_predictor(encoded_obs)

    def _detect_screen_type(self, ocr_data: Dict) -> str:
        """Detect current game screen"""
        text_content = ' '.join(ocr_data.values()).lower()

        if 'production' in text_content and 'queue' in text_content:
            return 'production'
        elif 'construction' in text_content:
            return 'construction'
        elif 'research' in text_content:
            return 'research'
        elif 'focus' in text_content:
            return 'focus_tree'
        elif 'trade' in text_content:
            return 'trade'
        else:
            return 'main_map'

    def _extract_factory_count(self, text: str, index: int) -> int:
        """Extract factory count from text"""
        import re
        numbers = re.findall(r'(\d+)', text)
        return int(numbers[index]) if len(numbers) > index else 0

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'world_model': self.world_model.state_dict(),
            'nec': self.nec.state_dict(),
            'curiosity': self.curiosity.state_dict(),
            'metrics': self.metrics,
            'total_steps': self.total_steps
        }, path)
        print(f"💾 Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.world_model.load_state_dict(checkpoint['world_model'])
        self.nec.load_state_dict(checkpoint['nec'])
        self.curiosity.load_state_dict(checkpoint['curiosity'])
        self.metrics = checkpoint['metrics']
        self.total_steps = checkpoint['total_steps']

        print(f"📂 Loaded checkpoint from {path}")

    def get_statistics(self) -> Dict:
        """Get current statistics"""
        stats = {
            'total_steps': self.total_steps,
            'episode_steps': self.episode_steps,
            'total_reward': self.metrics['total_reward'],
            'avg_intrinsic_reward': self.metrics['intrinsic_reward'] / max(1, self.episode_steps),
            'nec_stats': self.nec.get_statistics(),
            'curiosity_stats': self.curiosity.rnd.get_statistics()
        }

        return stats


class ReplayBuffer:
    """Simple replay buffer for world model training"""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, obs, action, reward, next_obs, done):
        """Add transition to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = {
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done
        }

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict:
        """Sample batch from buffer"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        batch = {
            'obs': torch.stack([self.buffer[i]['obs'] for i in indices]),
            'action': torch.stack([self.buffer[i]['action'] for i in indices]),
            'reward': torch.stack([self.buffer[i]['reward'] for i in indices]),
            'next_obs': torch.stack([self.buffer[i]['next_obs'] for i in indices]),
            'done': torch.stack([self.buffer[i]['done'] for i in indices])
        }

        return batch

    def __len__(self):
        return len(self.buffer)

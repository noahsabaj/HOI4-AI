# src/ai/ultimate/ultimate_ai.py
"""
The Ultimate HOI4 AI
Combines DreamerV3 world model, RND curiosity, and Neural Episodic Control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageGrab
import pyautogui
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import time
import json
import os

# Import our components
from src.ai.ultimate.world_model import WorldModel
from src.ai.ultimate.curiosity import CombinedCuriosity
from src.ai.ultimate.episodic_memory import NeuralEpisodicControl, PersistentEpisodicMemory

# Import existing components we want to keep
from src.perception.ocr import HOI4OCR
from src.comprehension.engine import HOI4UnderstandingEngine
from src.strategy.evaluation import StrategicEvaluator


class ActionSpace:
    """Define HOI4 action space"""

    # Common HOI4 keys
    KEYS = [
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
    ]

    # Important screen regions (resolution independent)
    @staticmethod
    def get_click_regions():
        # Get current screen size
        import pyautogui
        width, height = pyautogui.size()

        # Scale positions based on screen size
        return {
            'top_bar': (width // 2, height * 0.03),  # Center of top bar
            'left_menu': (width * 0.05, height // 2),  # Left side menu
            'main_map': (width // 2, height // 2),  # Center of map
            'bottom_bar': (width // 2, height * 0.97),  # Bottom UI
            'production': (width * 0.05, height * 0.19),  # Production button
            'construction': (width * 0.05, height * 0.28),  # Construction button
            'research': (width * 0.05, height * 0.37),  # Research button
            'focus': (width * 0.05, height * 0.46),  # Focus tree button
        }

    CLICK_REGIONS = property(get_click_regions)

    @staticmethod
    def get_action_size():
        # Click positions (8 regions) + Keys (20) + Click types (2: right, middle)
        return 8 + len(ActionSpace.KEYS) + 2

    @staticmethod
    def decode_action(action_idx: int) -> Dict:
        """Convert action index to executable action"""
        click_regions = ActionSpace.get_click_regions()
        num_regions = len(click_regions)
        num_keys = len(ActionSpace.KEYS)

        if action_idx < num_regions:
            # Click action
            region_name = list(click_regions.keys())[action_idx]
            x, y = click_regions[region_name]
            return {
                'type': 'click',
                'x': int(x),
                'y': int(y),
                'button': 'left',
                'description': f'Click {region_name}'
            }
        elif action_idx < num_regions + num_keys:
            # Key action
            key_idx = action_idx - num_regions
            return {
                'type': 'key',
                'key': ActionSpace.KEYS[key_idx],
                'description': f'Press {ActionSpace.KEYS[key_idx]}'
            }
        else:
            # Right/middle click
            width, height = pyautogui.size()
            click_type = ['right', 'middle'][action_idx - num_regions - num_keys]
            return {
                'type': 'click',
                'x': width // 2,
                'y': height // 2,
                'button': click_type,
                'description': f'{click_type} click'
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
        self.device = device

        # Action space
        self.action_size = ActionSpace.get_action_size()

        # Core components
        print("  📦 Loading world model...")
        self.world_model = WorldModel(action_size=self.action_size).to(device)

        print("  🔍 Loading curiosity module...")
        self.curiosity = CombinedCuriosity(
            observation_dim=1024,  # From world model encoder
            rnd_weight=0.6,
            goal_weight=0.4
        )

        print("  🧠 Loading episodic memory...")
        self.nec = NeuralEpisodicControl(
            state_dim=544,  # RSSM state size
            key_dim=128,
            memory_size=50000
        )

        print("  💾 Loading persistent memory...")
        self.persistent_memory = PersistentEpisodicMemory()

        # Keep existing components
        print("  👁️ Loading perception...")
        self.ocr = HOI4OCR()
        self.understanding = HOI4UnderstandingEngine()
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

        # Start new game session
        self.persistent_memory.start_new_game(self.metrics['current_game_id'])

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)

        print("✅ Ultimate HOI4 AI ready!")

    def observe(self, screenshot: Image.Image) -> Tuple[torch.Tensor, Dict]:
        """
        Process screenshot into state

        Returns:
            (encoded_observation, game_info)
        """
        # Resize for world model
        screenshot_resized = screenshot.resize((1280, 720))

        # Convert to tensor
        obs_array = np.array(screenshot_resized)
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32)
        obs_tensor = obs_tensor.permute(2, 0, 1) / 255.0
        obs_tensor = obs_tensor.unsqueeze(0).to(self.device)

        # Get game info from OCR
        ocr_data = self.ocr.extract_all_text(screenshot)

        # Encode observation
        with torch.no_grad():
            encoded_obs = self.world_model.encoder(obs_tensor)

        # Parse game state
        game_info = {
            'ocr_data': ocr_data,
            'screen_type': self._detect_screen_type(ocr_data),
            'game_date': ocr_data.get('date', 'Unknown'),
            'political_power': self._extract_number(ocr_data.get('political_power', '0')),
            'factories': {
                'civilian': self._extract_factory_count(ocr_data.get('factories', ''), 0),
                'military': self._extract_factory_count(ocr_data.get('factories', ''), 1)
            }
        }

        return encoded_obs, game_info

    def act(self, screenshot: Image.Image) -> Dict:
        """
        Main decision function

        Args:
            screenshot: Current game screenshot

        Returns:
            Action to execute
        """
        # Observe
        encoded_obs, game_info = self.observe(screenshot)

        # Initialize RSSM state if needed
        if self.current_rssm_state is None:
            self.current_rssm_state = self._initialize_rssm_state()

        # Update RSSM state
        if self.last_action is not None:
            rssm_output = self.world_model.rssm(
                self.current_rssm_state['stoch'],
                self.current_rssm_state['deter'],
                self.last_action,
                encoded_obs
            )
            self.current_rssm_state = rssm_output

        # Get full state representation
        state_features = self._get_state_features()

        # Check persistent memory
        memories = self._check_memory(state_features, game_info)
        if memories:
            print(f"💭 Remembering: {memories[0]['description'][:100]}...")

        # Compute curiosity
        intrinsic_reward = self.curiosity.compute_reward(
            encoded_obs,
            game_info['screen_type'],
            achieved_goal=False
        )

        # Get Q-value from NEC
        q_value = self.nec.q_value(state_features)

        # Decide action
        if intrinsic_reward > 2.0:
            # High novelty - explore!
            print(f"🔍 High curiosity ({intrinsic_reward.item():.2f}) - exploring!")
            action_idx = self._exploration_policy(state_features)
        elif q_value > 0.5 and np.random.random() > 0.2:
            # Good Q-value - exploit
            print(f"💡 Good memory (Q={q_value.item():.2f}) - exploiting!")
            action_idx = self._exploitation_policy(state_features)
        else:
            # Imagine and plan
            action_idx = self._imagination_policy(state_features)

        # Decode action
        action = ActionSpace.decode_action(action_idx)

        # Store for next step
        self.last_action = F.one_hot(
            torch.tensor([action_idx], device=self.device),
            num_classes=self.action_size
        ).float()

        # Update metrics
        self.metrics['intrinsic_reward'] += intrinsic_reward.item()
        self.episode_steps += 1
        self.total_steps += 1

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

        # Update NEC
        if self.current_rssm_state is not None:
            state_features = self._get_state_features()
            self.nec.write(state_features, torch.tensor([reward], device=self.device))

        # Store significant experiences
        if abs(reward) > 5.0 or self.metrics['intrinsic_reward'] > 10.0:
            self._store_persistent_memory(
                state_features,
                action,
                f"Reward: {reward:.1f}",
                prev_info
            )

        # Train world model periodically
        if self.total_steps % 100 == 0 and len(self.replay_buffer) > 1000:
            self._train_world_model()

        # Train curiosity
        if self.total_steps % 50 == 0:
            self._train_curiosity()

        self.metrics['total_reward'] += reward

    def _imagination_policy(self, state_features: torch.Tensor) -> int:
        """
        Use imagination to plan ahead

        Args:
            state_features: Current state

        Returns:
            Best action according to imagination
        """
        print("🎯 Planning with imagination...")

        # Sample random action sequences
        horizon = 15
        num_samples = 50

        action_sequences = torch.randint(
            0, self.action_size,
            (num_samples, horizon),
            device=self.device
        )

        # Convert to one-hot
        action_sequences_onehot = F.one_hot(
            action_sequences,
            num_classes=self.action_size
        ).float()

        # Imagine outcomes
        best_return = -float('inf')
        best_action = 0

        with torch.no_grad():
            for i in range(num_samples):
                imagined = self.world_model.imagine_ahead(
                    self.world_model.encoder.net[0](state_features.unsqueeze(0)),
                    action_sequences_onehot[i:i + 1],
                    horizon
                )

                # Compute return
                returns = imagined['rewards'].sum().item()

                if returns > best_return:
                    best_return = returns
                    best_action = action_sequences[i, 0].item()

        return best_action

    def _exploration_policy(self, state_features: torch.Tensor) -> int:
        """Random exploration with bias toward untried actions"""
        # Prefer clicking on areas we haven't explored much
        if np.random.random() < 0.7:
            # Click somewhere
            return np.random.randint(0, 8)  # 8 click regions
        else:
            # Press a key
            return np.random.randint(8, 8 + len(ActionSpace.KEYS))

    def _exploitation_policy(self, state_features: torch.Tensor) -> int:
        """Exploit learned knowledge"""
        # For now, use imagination policy
        # TODO: Add explicit Q-learning
        return self._imagination_policy(state_features)

    def _get_state_features(self) -> torch.Tensor:
        """Get current state features for NEC"""
        if self.current_rssm_state is None:
            return torch.zeros(544, device=self.device)

        stoch_flat = self.current_rssm_state['stoch'].flatten()
        deter = self.current_rssm_state['deter'].squeeze(0)

        return torch.cat([stoch_flat, deter])

    def _initialize_rssm_state(self) -> Dict:
        """Initialize RSSM state"""
        return {
            'stoch': torch.zeros(
                1, self.world_model.rssm.stoch_size,
                self.world_model.rssm.num_categories,
                device=self.device
            ),
            'deter': torch.zeros(
                1, self.world_model.rssm.deter_size,
                device=self.device
            )
        }

    def _check_memory(self, state_features: torch.Tensor, game_info: Dict) -> List[Dict]:
        """Check persistent memory for similar situations"""
        # Convert state to numpy for ChromaDB
        state_np = state_features.detach().cpu().numpy()

        # Recall similar experiences
        memories = self.persistent_memory.recall_similar_experiences(
            state_np[:128],  # Use key encoding dimension
            n_results=3
        )

        return memories

    def _store_persistent_memory(
            self,
            state_features: torch.Tensor,
            action: Dict,
            outcome: str,
            game_info: Dict
    ):
        """Store important experience in persistent memory"""
        # Use NEC key encoder
        with torch.no_grad():
            key_encoding = self.nec.encode_key(state_features)

        context = {
            'game_date': game_info.get('game_date', 'Unknown'),
            'political_power': game_info.get('political_power', 0),
            'civilian_factories': game_info.get('factories', {}).get('civilian', 0),
            'military_factories': game_info.get('factories', {}).get('military', 0),
            'screen_type': game_info.get('screen_type', 'unknown')
        }

        self.persistent_memory.store_key_experience(
            key_encoding.cpu().numpy(),
            action,
            outcome,
            context
        )

    def _train_world_model(self):
        """Train world model on replay buffer"""
        if len(self.replay_buffer) < 1000:
            return

        # Sample batch
        batch = self.replay_buffer.sample(32)

        # TODO: Implement world model training
        # This would involve reconstructing observations and predicting rewards

    def _train_curiosity(self):
        """Train curiosity predictor network"""
        if len(self.replay_buffer) < 100:
            return

        # Sample batch
        batch = self.replay_buffer.sample(32)

        # Train RND predictor
        obs_batch = batch['obs']
        encoded_obs = self.world_model.encoder(obs_batch.to(self.device))

        loss = self.curiosity.rnd.train_predictor(encoded_obs)

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

    def _extract_number(self, text: str) -> int:
        """Extract number from text"""
        import re
        match = re.search(r'(\d+)', text)
        return int(match.group(1)) if match else 0

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
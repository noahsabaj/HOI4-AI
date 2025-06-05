# src/core/learner.py - The complete unified learning system
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import ImageGrab
import pyautogui
import keyboard
import time
import os
import sys
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

# Import our components
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.ai.brain import HOI4Brain
from src.ai.memory import StrategicMemory, GameMemory
from src.perception.ocr import HOI4OCR
from src.strategy.evaluation import StrategicEvaluator


@dataclass
class Experience:
    """Single learning experience"""
    state: torch.Tensor
    action: Dict[str, Any]
    reward: float
    next_state: torch.Tensor
    game_context: Dict[str, Any]
    strategic_evaluation: Dict[str, Any]


class UnifiedHOI4Learner:
    """
    The complete self-learning HOI4 AI.
    Discovers how to win through pure reinforcement learning.
    """

    def __init__(self, model_path: str = 'models/hoi4_unified.pth'):
        print("🧠 Initializing Unified HOI4 Learner")
        print("=" * 50)

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {self.device}")

        # Core components
        self.brain = HOI4Brain().to(self.device)
        self.ocr = HOI4OCR()
        self.memory = StrategicMemory()
        self.evaluator = StrategicEvaluator()

        # Learning setup
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()

        # Experience replay
        self.experience_buffer = deque(maxlen=50000)
        self.priority_buffer = deque(maxlen=10000)  # High-value experiences

        # Current game tracking
        self.current_game = GameMemory(
            game_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            start_date=datetime.now(),
            end_date=datetime.now(),
            final_outcome='ongoing'
        )

        # Learning parameters
        self.exploration_rate = 0.3  # Will adapt based on performance
        self.discount_factor = 0.99
        self.learning_rate = 0.0001

        # Performance tracking
        self.metrics = {
            'games_played': 0,
            'total_victories': 0,
            'current_streak': 0,
            'best_streak': 0,
            'actions_taken': 0,
            'strategic_discoveries': 0
        }

        # Load existing model if available
        self.load_model(model_path)

        # Speed settings
        pyautogui.PAUSE = 0.05
        pyautogui.FAILSAFE = True

        print("✅ Unified learner initialized!")

    def play_and_learn(self):
        """Main learning loop - plays HOI4 and learns from experience"""
        print("\n🎮 Starting HOI4 Learning Session")
        print("Controls:")
        print("  F5: Start/Resume")
        print("  F6: Pause")
        print("  F7: Save Progress")
        print("  ESC (hold): Stop")

        playing = False
        last_save = time.time()
        last_evaluation = time.time()

        print("\n⏸️ Press F5 to begin...")

        while True:
            # Check controls
            if keyboard.is_pressed('f5') and not playing:
                print("\n▶️ Learning resumed!")
                playing = True
                time.sleep(0.3)

            elif keyboard.is_pressed('f6') and playing:
                print("\n⏸️ Learning paused")
                playing = False
                time.sleep(0.3)

            elif keyboard.is_pressed('f7'):
                self.save_progress()
                time.sleep(0.3)

            elif keyboard.is_pressed('escape'):
                if not hasattr(self, '_esc_held'):
                    self._esc_held = time.time()
                elif time.time() - self._esc_held > 2:
                    print("\n🛑 Stopping...")
                    break
            else:
                self._esc_held = None

            # Main learning loop
            if playing:
                try:
                    # Take a learning step
                    self.learning_step()

                    # Periodic evaluation
                    if time.time() - last_evaluation > 30:
                        self.evaluate_progress()
                        last_evaluation = time.time()

                    # Auto-save
                    if time.time() - last_save > 300:
                        self.save_progress()
                        last_save = time.time()

                except Exception as e:
                    print(f"❌ Error: {e}")
                    time.sleep(1)

        # Final save
        self.finalize_game()
        self.save_progress()

    def learning_step(self):
        """Single learning step"""
        # 1. Capture current state
        screenshot = ImageGrab.grab()
        screenshot_resized = screenshot.resize((1280, 720))

        # 2. Extract game information
        ocr_data = self.ocr.extract_all_text(screenshot)

        # 3. Evaluate strategic situation
        game_state = self._build_game_state(ocr_data)
        strategic_eval = self.evaluator.evaluate_game_state(game_state, ocr_data)

        # 4. Convert screenshot to tensor
        state_tensor = self._screenshot_to_tensor(screenshot_resized)

        # 5. Decide action
        action = self.decide_action(state_tensor, strategic_eval)

        # 6. Execute action
        self.execute_action(action)

        # 8. Capture result
        next_screenshot = ImageGrab.grab()
        next_screenshot_resized = next_screenshot.resize((1280, 720))
        next_ocr = self.ocr.extract_all_text(next_screenshot)
        next_state = self._build_game_state(next_ocr)
        next_eval = self.evaluator.evaluate_game_state(next_state, next_ocr)
        next_tensor = self._screenshot_to_tensor(next_screenshot_resized)

        # 9. Calculate reward
        reward = self.calculate_strategic_reward(
            strategic_eval, next_eval, action, game_state, next_state
        )

        # 10. Store experience
        experience = Experience(
            state=state_tensor,
            action=action,
            reward=reward,
            next_state=next_tensor,
            game_context=game_state,
            strategic_evaluation=strategic_eval
        )

        self.experience_buffer.append(experience)

        # Priority experiences (high reward)
        if reward > 5.0:
            self.priority_buffer.append(experience)
            print(f"⭐ High-value experience! Reward: {reward:.1f}")

        # 11. Learn from batch
        if len(self.experience_buffer) >= 32:
            self.learn_from_batch()

        # 12. Update game memory
        self._update_game_memory(action, strategic_eval, reward)

        # 13. Adapt exploration
        self._adapt_exploration(reward)

        self.metrics['actions_taken'] += 1

    def decide_action(self, state: torch.Tensor, strategic_eval: Dict) -> Dict:
        """Decide what action to take"""
        # Check if we should explore or exploit
        if np.random.random() < self.exploration_rate:
            # Exploration - but guided by strategy
            return self._guided_exploration(strategic_eval)
        else:
            # Exploitation - use neural network
            with torch.no_grad():
                predictions = self.brain(state)

            # Convert predictions to action
            action = self._predictions_to_action(predictions)

            # Enhance with strategic guidance
            action = self._enhance_with_strategy(action, strategic_eval)

            return action

    def _guided_exploration(self, strategic_eval: Dict) -> Dict:
        """Exploration guided by strategic evaluation"""
        # Get strategic suggestion
        suggestion = self.evaluator.suggest_immediate_action(strategic_eval)

        # Convert to concrete action
        if suggestion['action'] == 'open_construction':
            return {
                'type': 'click',
                'x': 200,  # Construction button area
                'y': 250,
                'desc': 'Open construction view'
            }
        elif suggestion['action'] == 'open_recruitment':
            return {
                'type': 'click',
                'x': 200,  # Recruitment button area
                'y': 350,
                'desc': 'Open recruitment view'
            }
        else:
            # Random exploration
            return self._random_action()

    def _random_action(self) -> Dict:
        """Generate random action for exploration"""
        action_type = np.random.choice(['click', 'key'], p=[0.7, 0.3])

        if action_type == 'click':
            return {
                'type': 'click',
                'x': np.random.randint(100, 3700),
                'y': np.random.randint(100, 2000),
                'button': 'left',
                'desc': 'Exploration click'
            }
        else:
            keys = ['space', 'b', 'q', 'w', 'e', '1', '2', '3', '4', '5']
            return {
                'type': 'key',
                'key': np.random.choice(keys),
                'desc': 'Exploration key'
            }

    def _predictions_to_action(self, predictions: Dict) -> Dict:
        """Convert neural network output to action"""
        action_type = predictions['action_type'][0].argmax().item()

        if action_type == 0:  # Click
            x = int(predictions['click_position'][0][0].item() * 3840)
            y = int(predictions['click_position'][0][1].item() * 2160)
            button_idx = predictions['click_type'][0].argmax().item()
            buttons = ['left', 'right', 'middle']

            return {
                'type': 'click',
                'x': x,
                'y': y,
                'button': buttons[button_idx],
                'desc': 'Neural network click'
            }
        else:  # Key
            key_idx = predictions['key_press'][0].argmax().item()
            keys = ['space', 'esc', 'f1', 'f2', 'f3', 'enter', 's', 'w', 'shift', 'ctrl']

            return {
                'type': 'key',
                'key': keys[key_idx] if key_idx < len(keys) else 'space',
                'desc': 'Neural network key'
            }

    def _enhance_with_strategy(self, action: Dict, strategic_eval: Dict) -> Dict:
        """Enhance action with strategic understanding"""
        # If we have immediate goals, bias actions toward them
        if strategic_eval['immediate_goals']:
            top_goal = strategic_eval['immediate_goals'][0]

            # Adjust click targets based on goal
            if action['type'] == 'click' and top_goal['type'] == 'economic':
                # Bias toward construction area
                action['y'] = min(max(action['y'], 200), 400)
                action['desc'] += ' (economic focus)'

        return action

    def execute_action(self, action: Dict):
        """Execute the decided action"""
        try:
            if action['type'] == 'click':
                pyautogui.click(action['x'], action['y'], button=action.get('button', 'left'))
            elif action['type'] == 'key':
                pyautogui.press(action['key'])

            # Log important actions
            if action.get('desc') and 'economic' in action['desc']:
                print(f"🏭 {action['desc']}")
            elif action.get('desc') and 'military' in action['desc']:
                print(f"⚔️ {action['desc']}")

        except Exception as e:
            print(f"❌ Action failed: {e}")

    def calculate_strategic_reward(self, old_eval: Dict, new_eval: Dict,
                                   action: Dict, old_state: Dict, new_state: Dict) -> float:
        """Calculate reward based on strategic progress"""
        reward = 0.0

        # 1. Victory progress (most important)
        progress_delta = new_eval['victory_progress'] - old_eval['victory_progress']
        if progress_delta > 0:
            reward += progress_delta * 100
            print(f"🎯 Victory progress: {old_eval['victory_progress']:.1%} → {new_eval['victory_progress']:.1%}")

        # 2. Strategic health improvement
        health_delta = new_eval['strategic_health'] - old_eval['strategic_health']
        reward += health_delta * 20

        # 3. Goal completion
        if len(new_eval['immediate_goals']) < len(old_eval['immediate_goals']):
            reward += 10
            print(f"✅ Completed a strategic goal!")

        # 4. Phase-appropriate actions
        if self._action_matches_phase(action, new_eval['phase']):
            reward += 1

        # 5. Factory growth
        old_factories = old_state.get('factories', {})
        new_factories = new_state.get('factories', {})
        factory_growth = (new_factories.get('civilian', 0) - old_factories.get('civilian', 0) +
                          new_factories.get('military', 0) - old_factories.get('military', 0))
        if factory_growth > 0:
            reward += factory_growth * 5
            print(f"🏭 Factory growth: +{factory_growth}")

        # 6. Small penalty for being stuck
        if new_state.get('current_screen') == old_state.get('current_screen'):
            stuck_count = old_state.get('stuck_count', 0) + 1
            new_state['stuck_count'] = stuck_count
            if stuck_count > 10:
                reward -= 0.5
        else:
            new_state['stuck_count'] = 0

        return reward

    def _action_matches_phase(self, action: Dict, phase: str) -> bool:
        """Check if action aligns with current phase"""
        desc = action.get('desc', '').lower()

        if phase == 'EARLY':
            return any(word in desc for word in ['construction', 'civilian', 'economic'])
        elif phase == 'EXPANSION':
            return any(word in desc for word in ['focus', 'political'])
        elif phase == 'WAR':
            return any(word in desc for word in ['military', 'division', 'army'])

        return False

    def learn_from_batch(self, batch_size: int = 32):
        """Learn from a batch of experiences"""
        # Sample batch (mix of regular and priority)
        batch = []

        # 70% regular, 30% priority
        regular_size = int(batch_size * 0.7)
        priority_size = batch_size - regular_size

        if len(self.experience_buffer) >= regular_size:
            indices = np.random.choice(len(self.experience_buffer), regular_size, replace=False)
            batch.extend([self.experience_buffer[i] for i in indices])

        if len(self.priority_buffer) >= priority_size:
            indices = np.random.choice(len(self.priority_buffer), priority_size, replace=False)
            batch.extend([self.priority_buffer[i] for i in indices])
        elif len(self.experience_buffer) >= batch_size:
            # Fill with regular experiences
            remaining = batch_size - len(batch)
            indices = np.random.choice(len(self.experience_buffer), remaining, replace=False)
            batch.extend([self.experience_buffer[i] for i in indices])

        if len(batch) < 4:
            return

        # Prepare batch tensors
        states = torch.stack([exp.state.squeeze(0) for exp in batch]).to(self.device)
        next_states = torch.stack([exp.next_state.squeeze(0) for exp in batch]).to(self.device)
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32).to(self.device)

        # Forward pass
        current_predictions = self.brain(states)
        next_predictions = self.brain(next_states)

        # Calculate Q-values
        current_q_values = self._calculate_q_values(current_predictions, [exp.action for exp in batch])
        next_q_values = self._calculate_max_q_values(next_predictions)

        # Target Q-values (Bellman equation)
        target_q_values = rewards + self.discount_factor * next_q_values

        # Loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
        self.optimizer.step()

        # Track learning
        if hasattr(self, '_losses'):
            self._losses.append(loss.item())
        else:
            self._losses = deque(maxlen=1000)
            self._losses.append(loss.item())

    def _calculate_q_values(self, predictions: Dict, actions: List[Dict]) -> torch.Tensor:
        """Calculate Q-values for taken actions"""
        q_values = []

        for i, action in enumerate(actions):
            if action['type'] == 'click':
                # Q-value based on position accuracy
                pred_x = predictions['click_position'][i][0]
                pred_y = predictions['click_position'][i][1]
                actual_x = action['x'] / 3840
                actual_y = action['y'] / 2160

                q = 1.0 - ((pred_x - actual_x) ** 2 + (pred_y - actual_y) ** 2)
            else:
                # Q-value based on key prediction
                q = predictions['key_press'][i].max()

            q_values.append(q)

        return torch.tensor(q_values, device=self.device)

    def _calculate_max_q_values(self, predictions: Dict) -> torch.Tensor:
        """Calculate maximum Q-values for next states"""
        # Simplified - take max action value
        click_values = predictions['click_position'].mean(dim=1)
        key_values = predictions['key_press'].max(dim=1)[0]

        max_values = torch.maximum(click_values, key_values)
        return max_values

    def _build_game_state(self, ocr_data: Dict) -> Dict:
        """Build game state from OCR data"""
        state = {
            'year': 1936,
            'month': 1,
            'factories': {'civilian': 0, 'military': 0},
            'territories_controlled': [],
            'current_screen': 'unknown'
        }

        # Extract date
        date_text = ocr_data.get('date', '')
        # Parse date (simplified)
        if '1936' in date_text:
            state['year'] = 1936
        elif '1937' in date_text:
            state['year'] = 1937
        elif '1938' in date_text:
            state['year'] = 1938
        elif '1939' in date_text:
            state['year'] = 1939

        # Extract factories
        factory_text = ocr_data.get('factories', '')
        import re
        numbers = re.findall(r'(\d+)', factory_text)
        if len(numbers) >= 2:
            state['factories']['civilian'] = int(numbers[0])
            state['factories']['military'] = int(numbers[1])

        return state

    def _screenshot_to_tensor(self, screenshot) -> torch.Tensor:
        """Convert screenshot to tensor"""
        img_array = np.array(screenshot)
        img_tensor = torch.tensor(img_array, dtype=torch.float32)
        img_tensor = img_tensor.permute(2, 0, 1) / 255.0
        return img_tensor.unsqueeze(0).to(self.device)

    def _update_game_memory(self, action: Dict, evaluation: Dict, reward: float):
        """Update current game memory"""
        # Track key decisions
        if reward > 5.0:  # Significant positive reward
            self.current_game.key_decisions.append({
                'action': action,
                'evaluation': evaluation,
                'reward': reward,
                'timestamp': time.time()
            })

        # Track factory progression
        if hasattr(self, '_last_factory_update'):
            if time.time() - self._last_factory_update > 60:  # Every minute
                game_state = self._build_game_state(self.ocr.extract_all_text())
                month = (game_state['year'] - 1936) * 12 + game_state['month']
                total_factories = (game_state['factories']['civilian'] +
                                   game_state['factories']['military'])
                self.current_game.factory_curve.append((month, total_factories))
                self._last_factory_update = time.time()
        else:
            self._last_factory_update = time.time()

    def _adapt_exploration(self, reward: float):
        """Adapt exploration rate based on performance"""
        if not hasattr(self, '_recent_rewards'):
            self._recent_rewards = deque(maxlen=100)

        self._recent_rewards.append(reward)

        if len(self._recent_rewards) >= 50:
            avg_reward = np.mean(self._recent_rewards)

            if avg_reward > 2.0:  # Doing well
                self.exploration_rate = max(0.1, self.exploration_rate * 0.95)
            elif avg_reward < 0:  # Doing poorly
                self.exploration_rate = min(0.5, self.exploration_rate * 1.05)

    def evaluate_progress(self):
        """Evaluate and report learning progress"""
        insights = self.memory.get_discovery_insights()

        print(f"\n📊 Learning Progress Report")
        print(f"{'=' * 40}")
        print(f"Games Played: {self.metrics['games_played']}")
        print(f"Victories: {self.metrics['total_victories']}")
        print(f"Win Rate: {self.metrics['total_victories'] / max(1, self.metrics['games_played']):.1%}")
        print(f"Actions Taken: {self.metrics['actions_taken']:,}")
        print(f"Exploration Rate: {self.exploration_rate:.1%}")

        if insights['top_winning_patterns']:
            print(f"\n🎯 Discovered Winning Patterns:")
            for pattern in insights['top_winning_patterns'][:3]:
                print(f"  • {pattern['pattern']}: {pattern['success_rate']:.1%} success")

        if hasattr(self, '_losses') and self._losses:
            avg_loss = np.mean(list(self._losses)[-100:])
            print(f"\nLearning Loss: {avg_loss:.4f}")

    def save_progress(self):
        """Save all progress"""
        print("\n💾 Saving progress...")

        # Save neural network
        torch.save({
            'brain_state': self.brain.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'exploration_rate': self.exploration_rate
        }, 'models/hoi4_unified.pth')

        # Save memories
        self.memory.save_memories()

        # Save current game if ongoing
        if self.current_game.final_outcome == 'ongoing':
            self.current_game.end_date = datetime.now()

        print("✅ Progress saved!")

    def load_model(self, path: str):
        """Load existing model"""
        if os.path.exists(path):
            print(f"📂 Loading model from {path}")
            checkpoint = torch.load(path, map_location=self.device)
            self.brain.load_state_dict(checkpoint['brain_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.metrics = checkpoint.get('metrics', self.metrics)
            self.exploration_rate = checkpoint.get('exploration_rate', 0.3)
            print("✅ Model loaded!")
        else:
            # Try to load from other models
            fallback_models = [
                'models/hoi4_ai_ultra_v2.pth',
                'models/hoi4_ai_best.pth'
            ]

            for fallback in fallback_models:
                if os.path.exists(fallback):
                    print(f"📂 Loading fallback model from {fallback}")
                    state_dict = torch.load(fallback, map_location=self.device)
                    if isinstance(state_dict, dict) and 'brain_state' in state_dict:
                        self.brain.load_state_dict(state_dict['brain_state'])
                    else:
                        self.brain.load_state_dict(state_dict)
                    print("✅ Fallback model loaded!")
                    break

    def finalize_game(self):
        """Finalize current game and learn from it"""
        self.current_game.end_date = datetime.now()

        # Determine outcome (simplified - would check actual victory)
        game_state = self._build_game_state(self.ocr.extract_all_text())
        if game_state['year'] >= 1945:
            self.current_game.final_outcome = 'victory'
            self.metrics['total_victories'] += 1
        else:
            self.current_game.final_outcome = 'defeat'

        # Store in memory
        self.memory.remember_game(self.current_game)

        # Learn from this game
        self.evaluator.learn_from_game(self.current_game)

        self.metrics['games_played'] += 1

        # Start new game
        self.current_game = GameMemory(
            game_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            start_date=datetime.now(),
            end_date=datetime.now(),
            final_outcome='ongoing'
        )
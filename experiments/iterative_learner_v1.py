# src/training/iterative_learner.py - AI that learns from playing!
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import ImageGrab
import pyautogui
import keyboard
import time
import json
import os
import sys
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.legacy.ai.hoi4_brain import HOI4Brain


class GameStateAnalyzer:
    """Understands what's happening in the game"""

    def __init__(self):
        self.last_screenshot = None
        self.game_state = {
            'political_power': 0,
            'factories': 0,
            'manpower': 0,
            'has_focus': False,
            'research_slots': 0,
            'game_speed': 0,
            'current_screen': 'unknown'
        }

    def analyze_screenshot(self, screenshot):
        """Extract game information from screenshot"""
        # Convert to numpy array
        img = np.array(screenshot)

        # Detect which screen we're on
        self.game_state['current_screen'] = self._detect_screen(img)

        # Detect if game is paused (look for pause indicator)
        self.game_state['paused'] = self._is_paused(img)

        # Detect game date/progress
        self.game_state['game_progress'] = self._estimate_progress(img)

        return self.game_state

    def _detect_screen(self, img):
        """Detect which screen we're on"""
        # Simple color-based detection
        height, width = img.shape[:2]

        # Check top bar colors
        top_bar = img[0:100, :]

        # Main map has specific UI elements
        if np.mean(top_bar[:, :, 0]) > 100:  # Reddish hue
            return 'main_map'
        elif np.mean(img[height // 2 - 50:height // 2 + 50, :]) > 150:
            return 'menu'
        else:
            return 'unknown'

    def _is_paused(self, img):
        """Check if game is paused"""
        # Look for pause indicator in top right
        pause_region = img[10:40, -200:]
        # If there's a bright pause symbol
        return np.max(pause_region) > 200

    def _estimate_progress(self, img):
        """Estimate how far into the game we are"""
        # This is simplified - in reality you'd OCR the date
        return 0.0  # Placeholder


class RewardCalculator:
    """Calculates rewards for AI actions"""

    def __init__(self):
        self.last_state = None
        self.reward_history = deque(maxlen=100)

    def calculate_reward(self, before_state, after_state, action):
        """Calculate reward for an action"""
        reward = 0.0

        # Reward for unpausing the game
        if before_state['paused'] and not after_state['paused']:
            reward += 1.0

        # Penalty for being paused too long
        if after_state['paused']:
            reward -= 0.1

        # Reward for being on main map (not stuck in menus)
        if after_state['current_screen'] == 'main_map':
            reward += 0.1
        elif after_state['current_screen'] == 'menu':
            reward -= 0.2

        # Penalty for repetitive actions
        if action.get('repetitive', False):
            reward -= 0.5

        # Bonus for diverse actions
        if action.get('type') == 'exploration':
            reward += 0.2

        self.reward_history.append(reward)
        return reward


class IterativeLearningAI:
    """AI that learns from playing the game"""

    def __init__(self, model_path='models/hoi4_ai_best.pth'):
        print("🧠 Initializing Iterative Learning AI...")

        # Core components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.brain = HOI4Brain().to(self.device)
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.0001)

        # Load existing model if available
        if os.path.exists(model_path):
            self.brain.load_state_dict(torch.load(model_path, weights_only=False))
            print("✅ Loaded existing model")

        # Learning components
        self.state_analyzer = GameStateAnalyzer()
        self.reward_calculator = RewardCalculator()

        # Experience replay buffer
        self.experience_buffer = deque(maxlen=1000)

        # Action tracking
        self.action_history = deque(maxlen=20)
        self.learning_enabled = True
        self.exploration_rate = 0.3

        # Performance tracking
        self.episode_rewards = []
        self.successful_actions = 0
        self.total_actions = 0

        # Game control
        pyautogui.PAUSE = 0.1
        pyautogui.FAILSAFE = True

    def capture_and_analyze(self):
        """Capture screen and analyze game state"""
        screenshot = ImageGrab.grab()
        screenshot_small = screenshot.resize((1280, 720))

        # Analyze game state
        game_state = self.state_analyzer.analyze_screenshot(screenshot_small)

        # Convert to tensor for neural network
        img_array = np.array(screenshot_small)
        img_tensor = torch.tensor(img_array).float()
        img_tensor = img_tensor.permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        return img_tensor, game_state, screenshot_small

    def select_action(self, state_tensor, game_state):
        """Select action using epsilon-greedy strategy"""

        # Check if we should explore
        if np.random.random() < self.exploration_rate:
            return self.generate_exploration_action()

        # Use neural network
        with torch.no_grad():
            predictions = self.brain(state_tensor)

        # Convert predictions to action
        action = self.predictions_to_action(predictions)

        # Add context from game state
        action['game_state'] = game_state

        # Check if action is repetitive
        if self.is_repetitive_action(action):
            action['repetitive'] = True
            # Force exploration
            if np.random.random() < 0.7:
                return self.generate_exploration_action()

        return action

    def predictions_to_action(self, predictions):
        """Convert neural network output to action"""
        action_type = predictions['action_type'][0].argmax().item()

        if action_type == 0:  # Click
            click_x = int(predictions['click_position'][0][0].item() * 3840)
            click_y = int(predictions['click_position'][0][1].item() * 2160)
            click_type_idx = predictions['click_type'][0].argmax().item()
            click_types = ['left', 'right', 'middle']

            return {
                'type': 'click',
                'x': click_x,
                'y': click_y,
                'button': click_types[click_type_idx]
            }
        else:  # Key
            key_idx = predictions['key_press'][0].argmax().item()
            keys = ['space', 'esc', 'f1', 'f2', 'f3', 'enter', 's', 'w', 'shift', 'ctrl']

            return {
                'type': 'key',
                'key': keys[key_idx]
            }

    def generate_exploration_action(self):
        """Generate smart exploration action based on game state"""
        exploration_actions = [
            # Strategic clicks
            {'type': 'click', 'x': 100, 'y': 200, 'button': 'left', 'desc': 'Open production'},
            {'type': 'click', 'x': 100, 'y': 300, 'button': 'left', 'desc': 'Open research'},
            {'type': 'click', 'x': 1920, 'y': 500, 'button': 'left', 'desc': 'Select state'},
            {'type': 'click', 'x': 200, 'y': 100, 'button': 'left', 'desc': 'Political power'},

            # Useful keys
            {'type': 'key', 'key': 'space', 'desc': 'Pause/unpause'},
            {'type': 'key', 'key': 'f1', 'desc': 'Army screen'},
            {'type': 'key', 'key': 'q', 'desc': 'Close screen'},
            {'type': 'key', 'key': 'b', 'desc': 'Build mode'},

            # Map navigation
            {'type': 'middle_drag', 'desc': 'Pan map'},
            {'type': 'zoom', 'desc': 'Zoom map'}
        ]

        action = np.random.choice(exploration_actions)
        action['exploration'] = True
        return action

    def is_repetitive_action(self, action):
        """Check if action is too similar to recent actions"""
        if len(self.action_history) < 5:
            return False

        recent_actions = list(self.action_history)[-5:]

        if action['type'] == 'click':
            # Check if clicking same spot
            similar_clicks = sum(
                1 for a in recent_actions
                if a.get('type') == 'click' and
                abs(a.get('x', 0) - action['x']) < 50 and
                abs(a.get('y', 0) - action['y']) < 50
            )
            return similar_clicks >= 3
        else:
            # Check if same key pressed multiple times
            same_keys = sum(1 for a in recent_actions if a.get('key') == action.get('key'))
            return same_keys >= 4

    def execute_action(self, action):
        """Execute action in game"""
        try:
            if action['type'] == 'click':
                pyautogui.moveTo(action['x'], action['y'], duration=0.2)
                if action['button'] == 'left':
                    pyautogui.click()
                elif action['button'] == 'right':
                    pyautogui.rightClick()
                elif action['button'] == 'middle':
                    pyautogui.middleClick()

                desc = action.get('desc', f"{action['button']} click")
                print(f"🖱️ {desc} at ({action['x']}, {action['y']})")

            elif action['type'] == 'key':
                pyautogui.press(action['key'])
                desc = action.get('desc', f"Press {action['key']}")
                print(f"⌨️ {desc}")

            elif action['type'] == 'middle_drag':
                # Pan the map
                start_x, start_y = 1920, 1080
                pyautogui.moveTo(start_x, start_y)
                pyautogui.drag(np.random.randint(-300, 300), np.random.randint(-200, 200),
                               duration=0.5, button='middle')
                print("🗺️ Panning map")

            elif action['type'] == 'zoom':
                # Zoom in/out
                pyautogui.scroll(np.random.choice([-3, 3]))
                print("🔍 Zooming map")

            self.action_history.append(action)
            self.total_actions += 1

        except Exception as e:
            print(f"❌ Action failed: {e}")

    def learn_from_experience(self):
        """Train on recent experiences"""
        if len(self.experience_buffer) < 32:
            return

        # Sample batch from experience
        batch_size = min(32, len(self.experience_buffer))
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)

        batch = [self.experience_buffer[i] for i in indices]

        # Prepare batch tensors
        states = torch.stack([exp['state'] for exp in batch]).squeeze(1).to(self.device)

        # Prepare target actions
        click_positions = []
        click_types = []
        action_types = []
        key_presses = []
        rewards = []

        for exp in batch:
            action = exp['action']
            reward = exp['reward']

            # Convert action to target format
            if action['type'] == 'click':
                click_positions.append([action['x'] / 3840, action['y'] / 2160])
                click_types.append(['left', 'right', 'middle'].index(action.get('button', 'left')))
                action_types.append(0)
                key_presses.append(0)
            else:
                click_positions.append([0.5, 0.5])
                click_types.append(0)
                action_types.append(1)
                keys = ['space', 'esc', 'f1', 'f2', 'f3', 'enter', 's', 'w', 'shift', 'ctrl']
                key_idx = keys.index(action.get('key', 'space')) if action.get('key', 'space') in keys else 0
                key_presses.append(key_idx)

            rewards.append(reward)

        # Convert to tensors
        click_positions = torch.tensor(click_positions).float().to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)

        # Forward pass
        predictions = self.brain(states)

        # Calculate losses with reward weighting
        position_loss = nn.MSELoss()(predictions['click_position'], click_positions)

        # Weight losses by rewards (good actions are reinforced)
        weighted_loss = position_loss * (1 + rewards.mean())

        # Backward pass
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        return weighted_loss.item()

    def save_progress(self):
        """Save model and learning progress"""
        # Save model
        torch.save(self.brain.state_dict(), 'models/hoi4_ai_iterative.pth')

        # Save learning stats
        stats = {
            'episode_rewards': self.episode_rewards,
            'successful_actions': self.successful_actions,
            'total_actions': self.total_actions,
            'exploration_rate': self.exploration_rate
        }

        with open('models/learning_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n💾 Saved progress - Success rate: {self.successful_actions / max(1, self.total_actions) * 100:.1f}%")

    def play_and_learn(self):
        """Main loop - play and learn simultaneously"""
        print("\n🚀 Iterative Learning AI Started!")
        print("=" * 50)
        print("This AI learns while playing!")
        print("\nControls:")
        print("  F5: Start/Resume")
        print("  F6: Pause")
        print("  F7: Save progress")
        print("  ESC: Stop")
        print("\n📈 Learning Settings:")
        print(f"  Exploration: {self.exploration_rate * 100:.0f}%")
        print(f"  Experience buffer: {len(self.experience_buffer)}/{self.experience_buffer.maxlen}")

        playing = False
        episode_reward = 0
        last_save = time.time()

        print("\nPress F5 to start...")

        while True:
            # Check controls
            if keyboard.is_pressed('f5') and not playing:
                print("\n▶️ AI Learning Started!")
                playing = True
                episode_reward = 0
                time.sleep(0.5)

            elif keyboard.is_pressed('f6') and playing:
                print("\n⏸️ Paused - Learning from experience...")
                playing = False

                # Learn from buffer
                if self.learning_enabled:
                    for _ in range(10):  # Multiple learning iterations
                        loss = self.learn_from_experience()
                        if loss:
                            print(f"  Learning step - Loss: {loss:.4f}")

                time.sleep(0.5)

            elif keyboard.is_pressed('f7'):
                print("\n💾 Saving progress...")
                self.save_progress()
                time.sleep(0.5)

            elif keyboard.is_pressed('escape'):
                print("\n🛑 Stopping...")
                break

            # Play and learn
            if playing:
                try:
                    # Capture current state
                    state_tensor, game_state, screenshot = self.capture_and_analyze()

                    # Select action
                    action = self.select_action(state_tensor, game_state)

                    # Execute action
                    self.execute_action(action)

                    # Wait a bit
                    time.sleep(0.5)

                    # Capture result state
                    next_state_tensor, next_game_state, _ = self.capture_and_analyze()

                    # Calculate reward
                    reward = self.reward_calculator.calculate_reward(
                        game_state, next_game_state, action
                    )

                    # Store experience
                    self.experience_buffer.append({
                        'state': state_tensor,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state_tensor
                    })

                    episode_reward += reward

                    # Learn every 10 steps
                    if len(self.experience_buffer) > 0 and self.total_actions % 10 == 0:
                        self.learn_from_experience()

                    # Decay exploration
                    if self.total_actions % 100 == 0:
                        self.exploration_rate *= 0.99
                        self.exploration_rate = max(0.1, self.exploration_rate)
                        print(f"\n📊 Exploration rate: {self.exploration_rate * 100:.1f}%")

                    # Auto-save every 5 minutes
                    if time.time() - last_save > 300:
                        self.save_progress()
                        last_save = time.time()

                except pyautogui.FailSafeException:
                    print("\n⚠️ Emergency stop!")
                    playing = False
                except pyautogui.FailSafeException:
                    print("\n⚠️ Emergency stop!")
                    playing = False

        # Final save
        self.save_progress()
        print("\n✨ Thanks for training the AI!")
        print(f"Final stats:")
        print(f"  Total actions: {self.total_actions}")
        print(f"  Success rate: {self.successful_actions / max(1, self.total_actions) * 100:.1f}%")
        print(f"  Episodes: {len(self.episode_rewards)}")


def main():
    print("🧠 HOI4 Iterative Learning AI")
    print("=" * 50)
    print("This AI learns while playing!")

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Create learner
    ai = IterativeLearningAI()

    # Start learning
    ai.play_and_learn()


if __name__ == "__main__":
    main()
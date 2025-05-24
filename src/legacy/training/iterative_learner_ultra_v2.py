# src/training/iterative_learner_ultra_v2.py - FIXED VERSION with working detection!
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
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.legacy.ai.hoi4_brain import HOI4Brain


class DebugGameAnalyzer:
    """Game analyzer with debug info to see what's happening"""

    def __init__(self):
        self.session_start = time.time()
        self.stuck_counter = 0
        self.last_screens = deque(maxlen=5)
        self.last_debug_time = time.time()

    def analyze_screenshot(self, screenshot):
        """Analyze with debug info"""
        img = np.array(screenshot)
        height, width = img.shape[:2]

        # Get screen brightness
        avg_brightness = np.mean(img)

        # Simple but effective detection
        current_screen = 'main_map'  # Default to main map

        # Check if very dark (paused with overlay)
        if avg_brightness < 80:
            current_screen = 'paused_dark'

        # Check center area variation
        center = img[height // 3:2 * height // 3, width // 3:2 * width // 3]
        center_std = np.std(center)

        # High variation = likely game map
        if center_std > 50:
            current_screen = 'main_map'
        # Low variation = likely menu
        elif center_std < 20:
            current_screen = 'menu'

        # Track screens
        self.last_screens.append(current_screen)

        # Only stuck if same screen for 5 checks AND not main_map
        stuck = (len(set(self.last_screens)) == 1 and
                 current_screen != 'main_map' and
                 len(self.last_screens) == 5)

        # Debug print every 3 seconds
        if time.time() - self.last_debug_time > 3:
            print(f"\n🔍 DEBUG: Screen={current_screen}, Brightness={avg_brightness:.1f}, "
                  f"CenterStd={center_std:.1f}, Stuck={stuck}")
            self.last_debug_time = time.time()

        game_state = {
            'current_screen': current_screen,
            'paused': current_screen == 'paused_dark',
            'stuck': stuck,
            'brightness': avg_brightness,
            'game_time': time.time() - self.session_start,
            'strategic_phase': 'early_buildup',  # Simplified
            'debug_info': f"B:{avg_brightness:.0f} S:{center_std:.0f}"
        }

        return game_state


class SimpleRewardSystem:
    """Simplified rewards that actually work"""

    def __init__(self):
        self.action_history = deque(maxlen=50)
        self.reward_history = deque(maxlen=100)
        self.last_brightness = 0
        self.positive_streak = 0

    def calculate_reward(self, before_state, after_state, action):
        """Simple but effective rewards"""
        reward = 0.0

        # Screen change = good!
        if before_state['current_screen'] != after_state['current_screen']:
            reward += 3.0
            print(f"🎯 Screen changed! {before_state['current_screen']} → {after_state['current_screen']} +3.0")

        # Brightness change = something happened
        brightness_change = abs(after_state['brightness'] - before_state['brightness'])
        if brightness_change > 10:
            reward += 1.0
            print(f"✨ Visual change detected! +1.0")

        # Not stuck = good
        if not after_state['stuck']:
            reward += 0.2

        # Main map = best place to be
        if after_state['current_screen'] == 'main_map':
            reward += 0.5

        # Unpausing
        if before_state['paused'] and not after_state['paused']:
            reward += 5.0
            print("🎮 UNPAUSED! +5.0")

        # Action variety bonus
        recent_actions = list(self.action_history)[-10:]
        if len(recent_actions) >= 10:
            unique_actions = len(set(str(a) for a in recent_actions))
            if unique_actions >= 7:  # Good variety
                reward += 0.5

        # Penalty for spam
        if len(recent_actions) >= 5:
            last_5 = [str(a.get('type', '')) + str(a.get('key', '')) for a in recent_actions[-5:]]
            if len(set(last_5)) == 1:  # All same
                reward -= 2.0
                print("🚫 Stop spamming! -2.0")

        # Track action
        self.action_history.append(action)
        self.reward_history.append(reward)

        # Momentum bonus
        if reward > 0:
            self.positive_streak += 1
            if self.positive_streak > 3:
                reward += 0.5
                print("🔥 On a roll! +0.5")
        else:
            self.positive_streak = 0

        return reward


class UltraLearningV2:
    """Version 2 - Actually works!"""

    def __init__(self, model_path='models/hoi4_ai_ultra.pth'):
        print("🚀 HOI4 ULTRA LEARNER V2")
        print("=" * 50)
        print("✨ Features:")
        print("  • Fixed screen detection")
        print("  • Debug visibility")
        print("  • Simplified rewards")
        print("  • Better exploration")

        # Setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Device: {self.device}")

        self.brain = HOI4Brain().to(self.device)
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.0003)

        # Try to load best model
        if os.path.exists(model_path):
            try:
                self.brain.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ Loaded: {model_path}")
            except:
                # Fallback to other models
                alt_models = ['models/hoi4_ai_strategic.pth', 'models/hoi4_ai_best.pth']
                for alt in alt_models:
                    if os.path.exists(alt):
                        self.brain.load_state_dict(torch.load(alt, map_location=self.device))
                        print(f"✅ Loaded fallback: {alt}")
                        break

        # Components
        self.analyzer = DebugGameAnalyzer()
        self.reward_system = SimpleRewardSystem()

        # Learning setup
        self.experience_buffer = deque(maxlen=10000)
        self.good_experience_buffer = deque(maxlen=1000)  # Store positive rewards
        self.exploration_rate = 0.5  # Start high

        # Tracking
        self.metrics = {
            'start_time': time.time(),
            'total_actions': 0,
            'positive_rewards': 0,
            'total_reward': 0
        }

        # Speed settings
        pyautogui.PAUSE = 0.05
        pyautogui.FAILSAFE = True

        # Action weights (prefer useful actions)
        self.exploration_actions = [
            # High priority
            {'type': 'key', 'key': 'space', 'weight': 3.0, 'desc': 'Toggle pause'},
            {'type': 'key', 'key': 'b', 'weight': 2.0, 'desc': 'Build mode'},
            {'type': 'key', 'key': 'q', 'weight': 2.0, 'desc': 'Politics'},
            {'type': 'key', 'key': 'w', 'weight': 2.0, 'desc': 'Research'},

            # Speed controls
            {'type': 'key', 'key': '1', 'weight': 1.0, 'desc': 'Speed 1'},
            {'type': 'key', 'key': '2', 'weight': 1.0, 'desc': 'Speed 2'},
            {'type': 'key', 'key': '3', 'weight': 1.0, 'desc': 'Speed 3'},
            {'type': 'key', 'key': '4', 'weight': 1.0, 'desc': 'Speed 4'},
            {'type': 'key', 'key': '5', 'weight': 1.0, 'desc': 'Speed 5'},

            # Clicks on important areas
            {'type': 'click', 'x': 100, 'y': 100, 'weight': 2.0, 'desc': 'Political power area'},
            {'type': 'click', 'x': 50, 'y': 200, 'weight': 2.0, 'desc': 'Production button'},
            {'type': 'click', 'x': 50, 'y': 300, 'weight': 2.0, 'desc': 'Construction button'},
            {'type': 'click', 'x': 960, 'y': 540, 'weight': 1.5, 'desc': 'Center screen'},

            # Exploration
            {'type': 'random_click', 'weight': 1.0, 'desc': 'Explore'},
            {'type': 'middle_drag', 'weight': 0.5, 'desc': 'Pan map'},
            {'type': 'key', 'key': 'esc', 'weight': 0.5, 'desc': 'Back/Close'},
        ]

    def get_weighted_action(self):
        """Get exploration action with weights"""
        weights = [a['weight'] for a in self.exploration_actions]
        total = sum(weights)
        weights = [w / total for w in weights]

        action = np.random.choice(self.exploration_actions, p=weights)

        # Handle special types
        if action['type'] == 'random_click':
            return {
                'type': 'click',
                'x': np.random.randint(100, 3700),
                'y': np.random.randint(100, 2000),
                'button': 'left',
                'desc': 'Explore click'
            }
        elif action['type'] == 'middle_drag':
            return {
                'type': 'middle_drag',
                'desc': action['desc']
            }
        else:
            return dict(action)  # Copy the action

    def select_action(self, state_tensor, game_state):
        """Smart action selection"""
        # Always explore if stuck
        if game_state.get('stuck', False):
            return self.get_weighted_action()

        # Exploration vs exploitation
        if np.random.random() < self.exploration_rate:
            return self.get_weighted_action()

        # Use neural network
        with torch.no_grad():
            predictions = self.brain(state_tensor)

        action = self.predictions_to_action(predictions)

        # Add some noise to clicks
        if action['type'] == 'click':
            action['x'] += np.random.randint(-50, 50)
            action['y'] += np.random.randint(-50, 50)
            action['x'] = max(10, min(action['x'], 3830))
            action['y'] = max(10, min(action['y'], 2150))

        return action

    def predictions_to_action(self, predictions):
        """Convert NN to action"""
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
                'button': buttons[button_idx]
            }
        else:  # Key
            key_idx = predictions['key_press'][0].argmax().item()
            keys = ['space', 'esc', 'f1', 'f2', 'f3', 'enter', 's', 'w', 'shift', 'ctrl']
            return {
                'type': 'key',
                'key': keys[key_idx] if key_idx < len(keys) else 'space'
            }

    def execute_action(self, action):
        """Execute with logging"""
        try:
            if action['type'] == 'click':
                pyautogui.click(action['x'], action['y'], button=action.get('button', 'left'))
                print(f"🖱️ {action.get('desc', f'Click ({action['x']}, {action['y']})')}")

            elif action['type'] == 'key':
                pyautogui.press(action['key'])
                print(f"⌨️ {action.get('desc', f'Key: {action["key"]}')}")

            elif action['type'] == 'middle_drag':
                x, y = pyautogui.position()
                pyautogui.drag(np.random.randint(-400, 400),
                               np.random.randint(-300, 300),
                               duration=0.3, button='middle')
                print("🗺️ Pan map")

            self.metrics['total_actions'] += 1

        except Exception as e:
            print(f"❌ Action failed: {e}")

    def learn_from_experience(self):
        """Learn from both good and bad"""
        if len(self.experience_buffer) < 32:
            return None

        # Mix of recent and good experiences
        batch = []

        # 50% recent experiences
        if len(self.experience_buffer) >= 16:
            recent = list(self.experience_buffer)[-16:]
            batch.extend(recent)

        # 50% good experiences (if available)
        if len(self.good_experience_buffer) >= 16:
            good = list(self.good_experience_buffer)[-16:]
            batch.extend(good)
        else:
            # Fill with more recent
            recent = list(self.experience_buffer)[-32:]
            batch = recent[:32]

        # Train
        total_loss = 0
        self.optimizer.zero_grad()

        for exp in batch:
            state = exp['state'].to(self.device)
            predictions = self.brain(state)

            # Loss based on action and reward
            action = exp['action']
            reward = exp['reward']

            if action.get('type') == 'click':
                target_x = torch.tensor([action['x'] / 3840], device=self.device)
                target_y = torch.tensor([action['y'] / 2160], device=self.device)

                pred_x = predictions['click_position'][0][0]
                pred_y = predictions['click_position'][0][1]

                loss_x = nn.MSELoss()(pred_x, target_x)
                loss_y = nn.MSELoss()(pred_y, target_y)

                # Weight by reward
                if reward > 0:
                    weight = 1.0 + reward  # Learn more from good
                else:
                    weight = 0.1  # Learn less from bad

                total_loss += (loss_x + loss_y) * weight

        if total_loss > 0:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
            self.optimizer.step()

        return total_loss.item() if total_loss > 0 else 0

    def update_exploration(self):
        """Adaptive exploration"""
        recent = list(self.reward_system.reward_history)[-30:]
        if recent:
            avg = np.mean(recent)
            positive_ratio = sum(1 for r in recent if r > 0) / len(recent)

            if positive_ratio > 0.3:  # 30% positive = good!
                self.exploration_rate = max(0.2, self.exploration_rate * 0.9)
                print(f"📉 Exploration → {self.exploration_rate:.1%} (doing well!)")
            elif positive_ratio < 0.1:  # Less than 10% positive
                self.exploration_rate = min(0.8, self.exploration_rate * 1.1)
                print(f"📈 Exploration → {self.exploration_rate:.1%} (need variety!)")

    def capture_and_analyze(self):
        """Capture and analyze"""
        screenshot = ImageGrab.grab()
        screenshot_small = screenshot.resize((1280, 720))

        # Analyze
        game_state = self.analyzer.analyze_screenshot(screenshot_small)

        # To tensor
        img_array = np.array(screenshot_small)
        img_tensor = torch.tensor(img_array).float()
        img_tensor = img_tensor.permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        return img_tensor, game_state, screenshot_small

    def save_progress(self):
        """Save model and stats"""
        torch.save(self.brain.state_dict(), 'models/hoi4_ai_ultra_v2.pth')

        runtime = (time.time() - self.metrics['start_time']) / 60
        apm = self.metrics['total_actions'] / runtime
        avg_reward = self.metrics['total_reward'] / max(1, self.metrics['total_actions'])
        positive_pct = self.metrics['positive_rewards'] / max(1, self.metrics['total_actions']) * 100

        stats = {
            'timestamp': datetime.now().isoformat(),
            'runtime_minutes': runtime,
            'total_actions': self.metrics['total_actions'],
            'actions_per_minute': apm,
            'average_reward': avg_reward,
            'positive_reward_percentage': positive_pct,
            'exploration_rate': self.exploration_rate
        }

        with open('models/ultra_v2_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n💾 Saved V2 Model!")
        print(f"📊 {apm:.1f} APM | {avg_reward:.2f} avg | {positive_pct:.1f}% positive")

    def play_and_learn(self):
        """Main loop"""
        print("\n🎮 Controls:")
        print("  F5: Start")
        print("  F6: Pause")
        print("  F7: Save")
        print("  ESC (hold 2s): Stop")
        print("\n🎯 Goals:")
        print("  • Get positive rewards")
        print("  • Change screens")
        print("  • Unpause game")
        print("  • Avoid spam")

        playing = False
        last_save = time.time()
        last_report = time.time()
        esc_held = None

        print("\n⚡ Press F5 to start...")

        while True:
            # Controls
            if keyboard.is_pressed('escape'):
                if esc_held is None:
                    esc_held = time.time()
                elif time.time() - esc_held > 2:
                    print("\n🛑 Stopping...")
                    break
            else:
                esc_held = None

            if keyboard.is_pressed('f5') and not playing:
                print("\n⚡ V2 AI ENGAGED!")
                playing = True
                time.sleep(0.3)

            elif keyboard.is_pressed('f6'):
                print("\n⏸️ Paused")
                playing = False
                time.sleep(0.3)

            elif keyboard.is_pressed('f7'):
                self.save_progress()
                time.sleep(0.3)

            # Play
            if playing:
                try:
                    # Capture before
                    state, game_state, _ = self.capture_and_analyze()

                    # Act
                    action = self.select_action(state, game_state)
                    self.execute_action(action)

                    # Brief pause
                    time.sleep(0.1)

                    # Capture after
                    next_state, next_game_state, _ = self.capture_and_analyze()

                    # Reward
                    reward = self.reward_system.calculate_reward(
                        game_state, next_game_state, action
                    )

                    self.metrics['total_reward'] += reward
                    if reward > 0:
                        self.metrics['positive_rewards'] += 1

                    # Store
                    exp = {
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state
                    }
                    self.experience_buffer.append(exp)

                    # Store good experiences
                    if reward > 1.0:
                        self.good_experience_buffer.append(exp)

                    # Learn
                    if self.metrics['total_actions'] % 10 == 0:
                        self.learn_from_experience()

                    # Update exploration
                    if self.metrics['total_actions'] % 30 == 0:
                        self.update_exploration()

                    # Report
                    if time.time() - last_report > 20:
                        runtime = (time.time() - self.metrics['start_time']) / 60
                        apm = self.metrics['total_actions'] / runtime
                        avg = self.metrics['total_reward'] / max(1, self.metrics['total_actions'])
                        pos_pct = self.metrics['positive_rewards'] / max(1, self.metrics['total_actions']) * 100

                        print(f"\n📊 {self.metrics['total_actions']} actions | "
                              f"{apm:.1f} APM | "
                              f"Avg: {avg:.2f} | "
                              f"Positive: {pos_pct:.1f}% | "
                              f"Explore: {self.exploration_rate:.1%}")
                        last_report = time.time()

                    # Auto-save
                    if time.time() - last_save > 300:
                        self.save_progress()
                        last_save = time.time()

                except Exception as e:
                    print(f"❌ Error: {e}")
                    time.sleep(0.5)

        # Final save
        self.save_progress()
        print(f"\n🏁 Session complete!")
        print(f"📊 Total: {self.metrics['total_actions']} actions")
        print(f"💰 Reward: {self.metrics['total_reward']:.1f}")
        print(f"✅ Positive: {self.metrics['positive_rewards']} times")


def main():
    print("⚡ HOI4 ULTRA LEARNER V2")
    print("=" * 50)
    print("🔧 Fixed:")
    print("  • Screen detection works")
    print("  • Rewards make sense")
    print("  • No more stuck loops")
    print("  • Shows debug info")

    os.makedirs('models', exist_ok=True)

    ai = UltraLearningV2()
    ai.play_and_learn()


if __name__ == "__main__":
    main()
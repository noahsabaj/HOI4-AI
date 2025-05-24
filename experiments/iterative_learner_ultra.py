# src/training/iterative_learner_ultra.py - Ultra enhanced AI that actually plays HOI4!
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


class SmartGameAnalyzer:
    """Actually understands HOI4 game state"""

    def __init__(self):
        self.session_start = time.time()
        self.last_successful_action = time.time()
        self.stuck_counter = 0
        self.last_screens = deque(maxlen=10)

    def analyze_screenshot(self, screenshot):
        """Smart game state detection"""
        img = np.array(screenshot)
        height, width = img.shape[:2]

        # Detect current screen more accurately
        current_screen = self._detect_screen_smart(img)
        self.last_screens.append(current_screen)

        # Detect if we're stuck
        if len(self.last_screens) == 10 and len(set(self.last_screens)) == 1:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        game_state = {
            'current_screen': current_screen,
            'paused': self._is_paused(img),
            'stuck': self.stuck_counter > 3,
            'game_time': time.time() - self.session_start,
            'strategic_phase': self._get_actual_phase(),
            'needs_action': self._detect_needed_action(current_screen)
        }

        return game_state

    def _detect_screen_smart(self, img):
        """Better screen detection"""
        height, width = img.shape[:2]

        # Sample specific regions
        top_bar = img[0:100, :]
        center = img[height // 3:2 * height // 3, width // 3:2 * width // 3]
        left_panel = img[:, 0:400]

        # Check for pause overlay (gray tint)
        if np.mean(img) < 50:
            return 'paused_overlay'

        # Check for focus tree (has specific pattern)
        focus_region = img[200:800, 600:1400]
        if np.std(focus_region) > 40 and np.mean(focus_region) > 100:
            return 'focus_tree'

        # Check for construction (map visible with UI)
        if self._has_map_visible(img) and np.mean(left_panel) > 80:
            return 'construction'

        # Check for production (lists on left)
        if np.mean(left_panel[:, 0:200]) > 120:
            return 'production'

        # Default to main map
        if self._has_map_visible(img):
            return 'main_map'

        return 'unknown'

    def _has_map_visible(self, img):
        """Check if game map is visible"""
        # Map has varied colors, not uniform
        center_std = np.std(img[500:1000, 1000:1500])
        return center_std > 20

    def _is_paused(self, img):
        """Better pause detection"""
        # Check multiple indicators
        # 1. Gray overlay
        if np.mean(img) < 60:
            return True
        # 2. Pause icon region (top right)
        pause_region = img[20:80, -300:-100]
        if np.max(pause_region) > 230:  # Bright pause symbol
            return True
        return False

    def _get_actual_phase(self):
        """Correct phase calculation"""
        minutes = (time.time() - self.session_start) / 60
        if minutes < 20:
            return 'early_buildup'
        elif minutes < 40:
            return 'military_buildup'
        else:
            return 'expansion'

    def _detect_needed_action(self, screen):
        """What should we do on this screen?"""
        actions = {
            'main_map': ['build', 'unpause', 'open_focus'],
            'focus_tree': ['select_focus', 'close'],
            'construction': ['build_factory', 'close'],
            'production': ['queue_units', 'close'],
            'paused_overlay': ['unpause'],
            'unknown': ['explore', 'escape']
        }
        return actions.get(screen, ['explore'])


class IntelligentRewardSystem:
    """Rewards that actually teach HOI4 strategy"""

    def __init__(self):
        self.action_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        self.goals_completed = set()
        self.last_screen = None

    def calculate_reward(self, before_state, after_state, action):
        """Smart rewards based on actual game progress"""
        reward = 0.0

        # Big rewards for important achievements
        if before_state['paused'] and not after_state['paused']:
            reward += 5.0  # HUGE reward for unpausing
            print("🎯 UNPAUSED! +5.0")

        # Screen transition rewards
        before_screen = before_state['current_screen']
        after_screen = after_state['current_screen']

        if before_screen != after_screen:
            # Good transitions
            if before_screen == 'main_map' and after_screen == 'focus_tree':
                reward += 2.0
                print("📋 Opened focus tree! +2.0")
            elif before_screen == 'main_map' and after_screen == 'construction':
                reward += 2.0
                print("🏗️ Opened construction! +2.0")
            elif after_screen == 'main_map' and before_screen != 'main_map':
                reward += 1.0  # Back to game

        # Punish being stuck
        if after_state.get('stuck', False):
            reward -= 2.0
            print("🔄 Stuck! -2.0")

        # Smart speed management rewards
        if '1' in str(action.get('key', '')) or '2' in str(action.get('key', '')):
            # Slower speeds good when in menus
            if after_state['current_screen'] in ['focus_tree', 'construction', 'production']:
                reward += 0.5
                print("🐌 Good - slow speed in menu! +0.5")
        elif '4' in str(action.get('key', '')) or '5' in str(action.get('key', '')):
            # Faster speeds good on main map
            if after_state['current_screen'] == 'main_map' and not after_state['paused']:
                reward += 1.0
                print("⚡ Good - fast speed on map! +1.0")

        # Phase-appropriate actions
        phase = after_state['strategic_phase']
        if phase == 'early_buildup':
            if 'construction' in str(action.get('desc', '')).lower():
                reward += 1.0
            if 'civilian' in str(action.get('desc', '')).lower():
                reward += 2.0

        # Punish spam
        self.action_history.append(action)
        recent = list(self.action_history)[-10:]
        if len(recent) >= 5:
            # Count repetitions
            last_action = recent[-1]
            repetitions = sum(1 for a in recent[-5:]
                              if a.get('type') == last_action.get('type')
                              and a.get('key') == last_action.get('key'))
            if repetitions >= 4:
                reward -= 3.0
                print("🚫 Stop spamming! -3.0")

        self.reward_history.append(reward)
        return reward


class UltraLearningAI:
    """The ultimate HOI4 learning AI - fast, smart, and adaptive"""

    def __init__(self, model_path='models/hoi4_ai_strategic.pth'):
        print("🚀 Initializing ULTRA HOI4 AI...")
        print("=" * 50)

        # Core setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Device: {self.device}")

        self.brain = HOI4Brain().to(self.device)
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.0002)  # Higher learning rate

        # Load best model
        if os.path.exists(model_path):
            self.brain.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded: {model_path}")

        # Smart components
        self.analyzer = SmartGameAnalyzer()
        self.reward_system = IntelligentRewardSystem()

        # Enhanced learning
        self.experience_buffer = deque(maxlen=5000)  # Much larger buffer
        self.priority_buffer = deque(maxlen=500)  # High-reward experiences
        self.exploration_rate = 0.3
        self.action_cooldown = 0.1  # Faster actions!

        # Anti-stuck mechanisms
        self.last_action_time = time.time()
        self.stuck_escape_counter = 0
        self.consecutive_negatives = 0

        # Performance tracking
        self.metrics = {
            'start_time': time.time(),
            'total_actions': 0,
            'successful_actions': 0,
            'total_reward': 0,
            'learning_steps': 0
        }

        # Better controls
        pyautogui.PAUSE = 0.05  # Much faster!
        pyautogui.FAILSAFE = True

        # Action sequences for common tasks
        self.action_sequences = {
            'open_construction': [
                {'type': 'key', 'key': 'esc'},  # Clear any menu
                {'type': 'wait', 'duration': 0.2},
                {'type': 'key', 'key': 'b'},  # Build mode
            ],
            'build_civilian': [
                {'type': 'click', 'x': 1920, 'y': 540, 'button': 'left'},  # Select state
                {'type': 'wait', 'duration': 0.2},
                {'type': 'click', 'x': 300, 'y': 400, 'button': 'left'},  # Civilian factory
            ],
            'unpause': [
                {'type': 'key', 'key': 'space'},
            ],
            'escape_menu': [
                {'type': 'key', 'key': 'esc'},
                {'type': 'wait', 'duration': 0.1},
                {'type': 'key', 'key': 'esc'},
            ]
        }

    def get_smart_action(self, game_state):
        """Get intelligent action based on game state"""
        screen = game_state['current_screen']
        phase = game_state['strategic_phase']

        # If stuck, try different things
        if game_state.get('stuck', False):
            self.stuck_escape_counter += 1
            # Try different actions when stuck
            stuck_actions = [
                {'type': 'click', 'x': 1920, 'y': 1080, 'button': 'left', 'desc': 'Click center'},
                {'type': 'key', 'key': 'space', 'desc': 'Toggle pause'},
                {'type': 'key', 'key': 'b', 'desc': 'Build mode'},
                {'type': 'key', 'key': 'q', 'desc': 'Politics'},
                {'type': 'middle_drag', 'desc': 'Pan map'},
                {'type': 'click', 'x': np.random.randint(500, 3000),
                 'y': np.random.randint(300, 1800), 'button': 'left', 'desc': 'Random click'},
            ]
            # Don't spam escape!
            if self.stuck_escape_counter > 10:
                self.stuck_escape_counter = 0  # Reset
            return np.random.choice(stuck_actions)

        # Priority actions based on screen
        if game_state['paused']:
            return {'type': 'sequence', 'name': 'unpause'}

        if screen == 'main_map':
            if phase == 'early_buildup':
                # 30% chance to open construction
                if np.random.random() < 0.3:
                    return {'type': 'sequence', 'name': 'open_construction'}
            # Otherwise explore
            actions = [
                {'type': 'click', 'x': 100, 'y': 100, 'button': 'left', 'desc': 'Political power'},
                {'type': 'click', 'x': 50, 'y': 200, 'button': 'left', 'desc': 'Open production'},
                {'type': 'key', 'key': 'q', 'desc': 'Open politics'},
                {'type': 'middle_drag', 'desc': 'Pan map'},
            ]
            return np.random.choice(actions)

        elif screen == 'focus_tree':
            # Click on focuses
            focuses = [
                {'type': 'click', 'x': 960, 'y': 300, 'button': 'left', 'desc': 'Rhineland'},
                {'type': 'click', 'x': 960, 'y': 400, 'button': 'left', 'desc': 'Industry focus'},
                {'type': 'click', 'x': 960, 'y': 500, 'button': 'left', 'desc': 'Research slot'},
            ]
            return np.random.choice(focuses)

        elif screen == 'construction':
            return {'type': 'sequence', 'name': 'build_civilian'}

        # Default exploration
        return self._generate_exploration()

    def _generate_exploration(self):
        """Smart exploration actions"""
        actions = [
            {'type': 'key', 'key': 'space', 'desc': 'Toggle pause'},
            {'type': 'key', 'key': 'esc', 'desc': 'Back'},
            {'type': 'click', 'x': np.random.randint(100, 3700),
             'y': np.random.randint(100, 2000), 'button': 'left', 'desc': 'Explore click'},
            {'type': 'middle_drag', 'desc': 'Pan'},
        ]
        return np.random.choice(actions)

    def execute_action_fast(self, action):
        """Execute actions quickly and reliably"""
        try:
            if action.get('type') == 'sequence':
                # Execute a sequence of actions
                sequence = self.action_sequences.get(action['name'], [])
                for step in sequence:
                    self._execute_single_action(step)
                print(f"📍 Sequence: {action['name']}")
            else:
                self._execute_single_action(action)

            self.metrics['total_actions'] += 1
            self.last_action_time = time.time()

        except Exception as e:
            print(f"❌ Action error: {e}")

    def _execute_single_action(self, action):
        """Execute a single action"""
        if action['type'] == 'click':
            pyautogui.click(action['x'], action['y'], button=action.get('button', 'left'))
            desc = action.get('desc', f"Click {action['x']}, {action['y']}")
            print(f"🖱️ {desc}")

        elif action['type'] == 'key':
            pyautogui.press(action['key'])
            print(f"⌨️ {action.get('desc', f'Key: {action["key"]}')}")

        elif action['type'] == 'middle_drag':
            x, y = pyautogui.position()
            pyautogui.drag(np.random.randint(-500, 500),
                           np.random.randint(-300, 300),
                           duration=0.3, button='middle')
            print("🗺️ Pan")

        elif action['type'] == 'wait':
            time.sleep(action['duration'])

    def select_action(self, state_tensor, game_state):
        """Intelligent action selection"""
        # Smart exploration vs exploitation
        if np.random.random() < self.exploration_rate or game_state.get('stuck', False):
            return self.get_smart_action(game_state)

        # Use neural network
        with torch.no_grad():
            predictions = self.brain(state_tensor)

        action = self.predictions_to_action(predictions)

        # Validate action
        if action['type'] == 'click':
            # Ensure valid coordinates
            action['x'] = max(10, min(action['x'], 3830))
            action['y'] = max(10, min(action['y'], 2150))

        return action

    def predictions_to_action(self, predictions):
        """Convert NN output to action"""
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
            # Note: Speed keys (1-5) are in exploration actions
            return {
                'type': 'key',
                'key': keys[key_idx] if key_idx < len(keys) else 'space'
            }

    def learn_fast(self):
        """Accelerated learning from experience"""
        if len(self.experience_buffer) < 16:
            return None

        # Prioritize high-reward experiences
        if len(self.priority_buffer) > 8:
            # 50% chance to learn from high-reward experiences
            if np.random.random() < 0.5:
                batch = list(self.priority_buffer)[-16:]
            else:
                batch = list(self.experience_buffer)[-32:]
        else:
            batch = list(self.experience_buffer)[-32:]

        # Quick training
        total_loss = 0
        states = torch.stack([exp['state'] for exp in batch]).squeeze(1).to(self.device)

        for i, exp in enumerate(batch):
            # Learn from ALL experiences, not just positive ones!
            predictions = self.brain(states[i].unsqueeze(0))

            # Create target based on action
            action = exp['action']
            if action.get('type') == 'click':
                target_x = action['x'] / 3840
                target_y = action['y'] / 2160
                pos_loss = nn.MSELoss()(
                    predictions['click_position'][0],
                    torch.tensor([target_x, target_y]).to(self.device)
                )
                # Weight by absolute reward - learn to avoid bad actions!
                weight = max(0.1, abs(exp['reward']))
                if exp['reward'] < 0:
                    weight *= 2  # Learn even more from mistakes
                total_loss += pos_loss * weight

        if total_loss > 0:
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
            self.optimizer.step()

        self.metrics['learning_steps'] += 1
        return total_loss.item() if total_loss > 0 else 0

    def capture_and_analyze(self):
        """Fast screen capture and analysis"""
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

    def update_exploration(self):
        """Smart exploration adjustment"""
        recent_rewards = list(self.reward_system.reward_history)[-20:]
        if recent_rewards:
            avg_reward = np.mean(recent_rewards)
            if avg_reward < -2.0:
                # Terrible performance - reset and explore differently
                self.exploration_rate = 0.8  # High exploration
                self.stuck_escape_counter = 0
                print("🔄 Resetting strategy - too many failures!")
            elif avg_reward < -0.5:
                # Doing badly, explore more
                self.exploration_rate = min(0.5, self.exploration_rate * 1.2)
            elif avg_reward > 1.0:
                # Doing well, exploit more
                self.exploration_rate = max(0.1, self.exploration_rate * 0.9)
            else:
                # Normal decay
                self.exploration_rate = max(0.15, self.exploration_rate * 0.995)

    def save_ultra_progress(self):
        """Save enhanced model and stats"""
        torch.save(self.brain.state_dict(), 'models/hoi4_ai_ultra.pth')

        runtime = (time.time() - self.metrics['start_time']) / 60
        actions_per_min = self.metrics['total_actions'] / runtime
        avg_reward = self.metrics['total_reward'] / max(1, self.metrics['total_actions'])

        stats = {
            'timestamp': datetime.now().isoformat(),
            'runtime_minutes': runtime,
            'total_actions': self.metrics['total_actions'],
            'actions_per_minute': actions_per_min,
            'learning_steps': self.metrics['learning_steps'],
            'exploration_rate': self.exploration_rate,
            'average_reward': avg_reward,
            'total_reward': self.metrics['total_reward']
        }

        with open('models/ultra_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n💾 Ultra Save!")
        print(f"📊 {actions_per_min:.1f} actions/min | Avg reward: {avg_reward:.2f}")

    def play_and_learn_ultra(self):
        """Ultra fast learning loop"""
        print("\n🚀 ULTRA HOI4 AI ACTIVATED!")
        print("=" * 50)
        print("⚡ Features:")
        print("  • 10x faster actions")
        print("  • Smart anti-stuck system")
        print("  • Intelligent exploration")
        print("  • Priority learning buffer")
        print("\n🎮 Controls:")
        print("  F5: Start")
        print("  F6: Pause")
        print("  F7: Save")
        print("  F8: Toggle fast mode")
        print("  ESC: Stop (hold for 2 seconds)")

        playing = False
        fast_mode = True
        last_save = time.time()
        last_report = time.time()
        esc_pressed_time = None

        print("\n⚡ Press F5 to unleash the AI...")

        while True:
            # Better ESC detection
            if keyboard.is_pressed('escape'):
                if esc_pressed_time is None:
                    esc_pressed_time = time.time()
                elif time.time() - esc_pressed_time > 2.0:
                    print("\n🛑 ESC held - Stopping...")
                    break
            else:
                esc_pressed_time = None

            # Other controls
            if keyboard.is_pressed('f5') and not playing:
                print("\n⚡ ULTRA AI ENGAGED!")
                playing = True
                time.sleep(0.3)

            elif keyboard.is_pressed('f6'):
                print("\n⏸️ Paused")
                playing = False
                time.sleep(0.3)

            elif keyboard.is_pressed('f7'):
                self.save_ultra_progress()
                time.sleep(0.3)

            elif keyboard.is_pressed('f8'):
                fast_mode = not fast_mode
                print(f"\n⚡ Fast mode: {'ON' if fast_mode else 'OFF'}")
                time.sleep(0.3)

            # Main loop
            if playing:
                try:
                    # Capture
                    state_tensor, game_state, _ = self.capture_and_analyze()

                    # Act
                    action = self.select_action(state_tensor, game_state)
                    self.execute_action_fast(action)

                    # Brief pause
                    time.sleep(self.action_cooldown if fast_mode else 0.3)

                    # Result
                    next_state, next_game_state, _ = self.capture_and_analyze()

                    # Reward
                    reward = self.reward_system.calculate_reward(
                        game_state, next_game_state, action
                    )
                    self.metrics['total_reward'] += reward

                    # Store
                    experience = {
                        'state': state_tensor,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state
                    }
                    self.experience_buffer.append(experience)

                    # Priority experiences
                    if reward > 2.0:
                        self.priority_buffer.append(experience)
                        print(f"⭐ High value experience! Reward: {reward:.1f}")

                    # Learn frequently
                    if self.metrics['total_actions'] % 5 == 0:
                        self.learn_fast()

                    # Update exploration
                    if self.metrics['total_actions'] % 20 == 0:
                        self.update_exploration()

                    # Report
                    if time.time() - last_report > 30:
                        runtime = (time.time() - self.metrics['start_time']) / 60
                        apm = self.metrics['total_actions'] / runtime
                        avg_reward = self.metrics['total_reward'] / max(1, self.metrics['total_actions'])
                        print(f"\n📊 {self.metrics['total_actions']} actions | "
                              f"{apm:.1f} APM | "
                              f"Reward: {avg_reward:.2f} | "
                              f"Explore: {self.exploration_rate:.1%}")
                        last_report = time.time()

                    # Auto-save
                    if time.time() - last_save > 300:
                        self.save_ultra_progress()
                        last_save = time.time()

                except pyautogui.FailSafeException:
                    print("\n⚠️ Failsafe triggered!")
                    playing = False
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    time.sleep(0.5)

        # Final save
        self.save_ultra_progress()
        print("\n🏁 Ultra session complete!")
        print(f"📊 Total actions: {self.metrics['total_actions']}")
        print(f"💰 Total reward: {self.metrics['total_reward']:.1f}")


def main():
    print("⚡ HOI4 ULTRA LEARNING AI")
    print("=" * 50)

    os.makedirs('../src/legacy/training/models', exist_ok=True)

    # Create ultra AI
    ai = UltraLearningAI()

    # Start!
    ai.play_and_learn_ultra()


if __name__ == "__main__":
    main()
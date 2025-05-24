# src/training/iterative_learner_enhanced.py - Strategic AI that learns HOI4!
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


class StrategicGameAnalyzer:
    """Enhanced analyzer that understands HOI4 strategy"""

    def __init__(self):
        self.last_screenshot = None
        self.game_start_time = time.time()
        self.last_factory_count = 0
        self.last_division_count = 0
        self.strategic_goals = {
            'build_civs': True,  # Start with civilian factories
            'build_mils': False,  # Military factories later
            'train_infantry': True,
            'focus_rhineland': True,
            'research_industry': True
        }

    def analyze_screenshot(self, screenshot):
        """Extract strategic game information"""
        img = np.array(screenshot)

        game_state = {
            'current_screen': self._detect_screen(img),
            'paused': self._is_paused(img),
            'game_progress': self._estimate_progress(),
            'needs_focus': self._check_focus_available(img),
            'needs_research': self._check_research_slots(img),
            'construction_possible': self._check_construction(img),
            'strategic_phase': self._determine_phase()
        }

        return game_state

    def _detect_screen(self, img):
        """Enhanced screen detection for HOI4"""
        height, width = img.shape[:2]

        # Check for specific UI elements
        top_bar = img[0:100, :]
        avg_color = np.mean(top_bar)

        # Focus tree has specific layout
        if self._has_focus_tree_elements(img):
            return 'focus_tree'
        # Production screen
        elif self._has_production_elements(img):
            return 'production'
        # Construction screen
        elif self._has_construction_elements(img):
            return 'construction'
        # Research screen
        elif self._has_research_elements(img):
            return 'research'
        # Main map
        elif avg_color > 50 and avg_color < 150:
            return 'main_map'
        else:
            return 'unknown'

    def _has_focus_tree_elements(self, img):
        """Check for focus tree UI"""
        # Focus tree has a specific pattern of boxes
        center_region = img[200:800, 800:1200]
        # Look for box-like structures
        edges = np.std(center_region)
        return edges > 30

    def _has_production_elements(self, img):
        """Check for production UI"""
        # Production has lists on the left
        left_region = img[:, 0:300]
        return np.mean(left_region) > 100

    def _has_construction_elements(self, img):
        """Check for construction UI"""
        # Construction has map with slots
        return False  # Simplified

    def _has_research_elements(self, img):
        """Check for research UI"""
        # Research has grid layout
        return False  # Simplified

    def _is_paused(self, img):
        """Check if game is paused"""
        pause_region = img[10:50, -250:-50]
        brightness = np.mean(pause_region)
        return brightness > 200

    def _estimate_progress(self):
        """Estimate game progress"""
        elapsed = time.time() - self.game_start_time
        # Rough estimate: 1 hour = 1 year in game
        return min(elapsed / 3600, 1.0)

    def _check_focus_available(self, img):
        """Check if we need to select a focus"""
        # In real implementation, would check for focus notification
        return np.random.random() < 0.1  # Simplified

    def _check_research_slots(self, img):
        """Check if research slots available"""
        return np.random.random() < 0.1  # Simplified

    def _check_construction(self, img):
        """Check if we can build"""
        return True  # Always can build something

    def _determine_phase(self):
        """Determine strategic phase"""
        progress = self._estimate_progress()
        if progress < 0.2:
            return 'early_buildup'  # Focus on civilian factories
        elif progress < 0.5:
            return 'military_buildup'  # Switch to military
        else:
            return 'preparation'  # Prepare for war


class StrategicRewardSystem:
    """Sophisticated reward system for HOI4"""

    def __init__(self):
        self.action_sequence = deque(maxlen=50)
        self.last_rewards = deque(maxlen=100)
        self.goal_progress = {
            'factories_built': 0,
            'divisions_trained': 0,
            'focuses_completed': 0,
            'research_completed': 0
        }

    def calculate_reward(self, before_state, after_state, action):
        """Calculate strategic rewards"""
        reward = 0.0

        # Core gameplay rewards
        if before_state['paused'] and not after_state['paused']:
            reward += 2.0  # Big reward for unpausing

        if not after_state['paused'] and after_state['current_screen'] == 'main_map':
            reward += 0.5  # Good to be playing on main map

        # Strategic screen rewards
        screen_rewards = {
            'focus_tree': 1.0 if after_state['needs_focus'] else -0.5,
            'production': 0.5,
            'construction': 1.5 if after_state['strategic_phase'] == 'early_buildup' else 0.5,
            'research': 1.0 if after_state['needs_research'] else -0.5,
            'main_map': 0.2,
            'unknown': -1.0
        }

        current_screen = after_state.get('current_screen', 'unknown')
        reward += screen_rewards.get(current_screen, 0)

        # Phase-specific rewards
        phase = after_state.get('strategic_phase', 'early_buildup')
        if phase == 'early_buildup':
            # Reward construction and civilian factories
            if action.get('desc', '').lower().find('construction') >= 0:
                reward += 2.0
            if action.get('desc', '').lower().find('civilian') >= 0:
                reward += 3.0

        elif phase == 'military_buildup':
            # Reward military factories and division training
            if action.get('desc', '').lower().find('military') >= 0:
                reward += 2.0
            if action.get('desc', '').lower().find('train') >= 0:
                reward += 1.5

        # Penalty for repetitive/stuck behavior
        if action.get('repetitive', False):
            reward -= 1.0

        # Bonus for strategic exploration
        if action.get('strategic', False):
            reward += 0.5

        # Track learning progress
        self.last_rewards.append(reward)

        return reward


class EnhancedIterativeLearner:
    """Strategic AI that learns to play HOI4 effectively"""

    def __init__(self, model_path='models/hoi4_ai_best.pth'):
        print("🎯 Initializing Strategic HOI4 AI...")
        print("=" * 50)

        # Core components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {self.device}")

        self.brain = HOI4Brain().to(self.device)
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.0001)

        # Load model
        if os.path.exists(model_path):
            self.brain.load_state_dict(torch.load(model_path, map_location=self.device))
            print("✅ Loaded existing model")
        else:
            print("🆕 Starting with fresh model")

        # Strategic components
        self.analyzer = StrategicGameAnalyzer()
        self.reward_system = StrategicRewardSystem()

        # Learning systems
        self.experience_buffer = deque(maxlen=2000)  # Larger buffer
        self.action_history = deque(maxlen=30)
        self.exploration_rate = 0.4  # Higher initial exploration
        self.learning_rate = 0.0001

        # Strategic tracking
        self.current_strategy = 'exploration'
        self.strategic_goals = deque([
            'open_construction',
            'build_civilian_factory',
            'select_focus',
            'research_industry',
            'train_infantry'
        ])

        # Performance metrics
        self.metrics = {
            'total_actions': 0,
            'successful_sequences': 0,
            'average_reward': 0,
            'learning_iterations': 0,
            'time_started': time.time()
        }

        # Control settings
        pyautogui.PAUSE = 0.1
        pyautogui.FAILSAFE = True

    def get_strategic_actions(self, game_state):
        """Generate strategic actions based on game state"""
        phase = game_state.get('strategic_phase', 'early_buildup')
        screen = game_state.get('current_screen', 'unknown')

        actions = []

        # Phase-specific strategic actions
        if phase == 'early_buildup':
            actions.extend([
                {'type': 'key', 'key': 'b', 'desc': 'Open construction mode', 'strategic': True},
                {'type': 'click', 'x': 100, 'y': 150, 'button': 'left', 'desc': 'Construction tab', 'strategic': True},
                {'type': 'click', 'x': 1920, 'y': 540, 'button': 'left', 'desc': 'Select state for building',
                 'strategic': True},
                {'type': 'click', 'x': 300, 'y': 400, 'button': 'left', 'desc': 'Build civilian factory',
                 'strategic': True},
            ])

        # Screen-specific actions
        if screen == 'main_map':
            actions.extend([
                {'type': 'key', 'key': 'q', 'desc': 'Open politics', 'strategic': True},
                {'type': 'key', 'key': 'w', 'desc': 'Open research', 'strategic': True},
                {'type': 'key', 'key': 'b', 'desc': 'Open construction', 'strategic': True},
                {'type': 'click', 'x': 50, 'y': 100, 'button': 'left', 'desc': 'Click political power',
                 'strategic': True},
            ])

        elif screen == 'focus_tree':
            # Specific focuses for Germany 1936
            actions.extend([
                {'type': 'click', 'x': 960, 'y': 300, 'button': 'left', 'desc': 'Select Rhineland focus',
                 'strategic': True},
                {'type': 'click', 'x': 960, 'y': 400, 'button': 'left', 'desc': 'Select industry focus',
                 'strategic': True},
            ])

        # Always include some exploration
        actions.extend([
            {'type': 'key', 'key': 'space', 'desc': 'Toggle pause', 'strategic': True},
            {'type': 'key', 'key': 'esc', 'desc': 'Close/back', 'strategic': True},
            {'type': 'middle_drag', 'desc': 'Pan map', 'strategic': False},
        ])

        return actions

    def select_action(self, state_tensor, game_state):
        """Select action with strategic awareness"""

        # Strategic decision making
        if np.random.random() < self.exploration_rate:
            # Strategic exploration
            strategic_actions = self.get_strategic_actions(game_state)
            if strategic_actions:
                action = np.random.choice(strategic_actions)
                action['exploration'] = True
                return action

        # Use neural network
        with torch.no_grad():
            predictions = self.brain(state_tensor)

        action = self.predictions_to_action(predictions)

        # Add strategic context
        action['game_state'] = game_state
        action['phase'] = game_state.get('strategic_phase', 'unknown')

        # Avoid repetition with strategic alternatives
        if self.is_repetitive_action(action):
            action['repetitive'] = True
            # Try strategic action instead
            strategic_actions = self.get_strategic_actions(game_state)
            if strategic_actions and np.random.random() < 0.8:
                return np.random.choice(strategic_actions)

        return action

    def predictions_to_action(self, predictions):
        """Convert predictions to actions"""
        action_type = predictions['action_type'][0].argmax().item()

        if action_type == 0:  # Click
            # Scale to your resolution (3840x2160)
            click_x = int(predictions['click_position'][0][0].item() * 3840)
            click_y = int(predictions['click_position'][0][1].item() * 2160)
            click_type_idx = predictions['click_type'][0].argmax().item()
            click_types = ['left', 'right', 'middle']

            # Ensure clicks are within screen bounds
            click_x = max(10, min(click_x, 3830))
            click_y = max(10, min(click_y, 2150))

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
                'key': keys[key_idx] if key_idx < len(keys) else 'space'
            }

    def is_repetitive_action(self, action):
        """Check for repetitive behavior"""
        if len(self.action_history) < 5:
            return False

        recent = list(self.action_history)[-5:]

        if action['type'] == 'click':
            similar = sum(1 for a in recent
                          if a.get('type') == 'click' and
                          abs(a.get('x', 0) - action['x']) < 100 and
                          abs(a.get('y', 0) - action['y']) < 100)
            return similar >= 3
        else:
            same_key = sum(1 for a in recent if a.get('key') == action.get('key'))
            return same_key >= 4

    def execute_action(self, action):
        """Execute action in game"""
        try:
            if action['type'] == 'click':
                pyautogui.moveTo(action['x'], action['y'], duration=0.15)
                pyautogui.click(button=action.get('button', 'left'))
                print(f"🖱️ {action.get('desc', 'Click')} at ({action['x']}, {action['y']})")

            elif action['type'] == 'key':
                pyautogui.press(action['key'])
                print(f"⌨️ {action.get('desc', f'Press {action["key"]}')})")

            elif action['type'] == 'middle_drag':
                x, y = pyautogui.position()
                pyautogui.drag(np.random.randint(-400, 400),
                               np.random.randint(-300, 300),
                               duration=0.5, button='middle')
                print("🗺️ Panning map")

            self.action_history.append(action)
            self.metrics['total_actions'] += 1

        except Exception as e:
            print(f"❌ Action failed: {e}")

    def capture_and_analyze(self):
        """Capture and analyze game state"""
        screenshot = ImageGrab.grab()
        screenshot_small = screenshot.resize((1280, 720))

        # Analyze
        game_state = self.analyzer.analyze_screenshot(screenshot_small)

        # Convert to tensor
        img_array = np.array(screenshot_small)
        img_tensor = torch.tensor(img_array).float()
        img_tensor = img_tensor.permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        return img_tensor, game_state, screenshot_small

    def learn_from_experience(self):
        """Enhanced learning with strategic focus"""
        if len(self.experience_buffer) < 32:
            return None

        # Sample strategically (recent experiences weighted higher)
        weights = np.linspace(0.5, 1.0, len(self.experience_buffer))
        weights = weights / weights.sum()
        indices = np.random.choice(len(self.experience_buffer), 32, p=weights)

        batch = [self.experience_buffer[i] for i in indices]

        # Prepare batch
        states = torch.stack([exp['state'] for exp in batch]).squeeze(1).to(self.device)

        # Calculate targets based on rewards
        total_loss = 0

        for i, exp in enumerate(batch):
            state = states[i].unsqueeze(0)
            predictions = self.brain(state)

            # Create target based on actual action and reward
            action = exp['action']
            reward = exp['reward']

            # Weighted loss based on reward
            weight = max(0.1, reward + 1.0)  # Higher weight for good actions

            # Calculate specific losses
            if action['type'] == 'click':
                target_x = action['x'] / 3840
                target_y = action['y'] / 2160
                position_loss = nn.MSELoss()(
                    predictions['click_position'][0],
                    torch.tensor([target_x, target_y]).to(self.device)
                )
                total_loss += position_loss * weight

        # Optimize
        self.optimizer.zero_grad()
        if total_loss > 0:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
            self.optimizer.step()

        self.metrics['learning_iterations'] += 1

        return total_loss.item() if total_loss > 0 else 0

    def update_exploration_rate(self):
        """Adaptive exploration rate"""
        # Decay exploration but keep minimum
        self.exploration_rate *= 0.995
        self.exploration_rate = max(0.15, self.exploration_rate)

        # Increase if stuck
        if len(self.reward_system.last_rewards) > 20:
            recent_rewards = list(self.reward_system.last_rewards)[-20:]
            if np.mean(recent_rewards) < -0.5:  # Doing poorly
                self.exploration_rate = min(0.5, self.exploration_rate * 1.2)
                print(f"📈 Increased exploration to {self.exploration_rate:.1%} (poor performance)")

    def save_enhanced_progress(self):
        """Save model and detailed metrics"""
        # Save model
        torch.save(self.brain.state_dict(), 'models/hoi4_ai_strategic.pth')

        # Calculate statistics
        runtime = time.time() - self.metrics['time_started']
        actions_per_minute = self.metrics['total_actions'] / (runtime / 60)

        # Detailed stats
        stats = {
            'timestamp': datetime.now().isoformat(),
            'runtime_minutes': runtime / 60,
            'total_actions': self.metrics['total_actions'],
            'actions_per_minute': actions_per_minute,
            'learning_iterations': self.metrics['learning_iterations'],
            'exploration_rate': self.exploration_rate,
            'average_reward': np.mean(list(self.reward_system.last_rewards)) if self.reward_system.last_rewards else 0,
            'strategic_phase': self.analyzer._determine_phase()
        }

        with open('models/strategic_learning_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n💾 Strategic Save Complete!")
        print(f"📊 Performance: {actions_per_minute:.1f} actions/min")
        print(f"🎯 Avg Reward: {stats['average_reward']:.2f}")

    def play_and_learn(self):
        """Main strategic learning loop"""
        print("\n🚀 Strategic HOI4 AI Started!")
        print("=" * 50)
        print("🎯 Goal: Learn to play Germany effectively")
        print("\n📋 Strategic Phases:")
        print("  1. Early: Build civilian factories")
        print("  2. Mid: Switch to military production")
        print("  3. Late: Prepare for expansion")
        print("\n🎮 Controls:")
        print("  F5: Start/Resume")
        print("  F6: Pause & Learn")
        print("  F7: Save Progress")
        print("  ESC: Stop")

        playing = False
        last_save = time.time()
        last_stats = time.time()

        print("\n🎯 Press F5 to begin strategic learning...")

        while True:
            # Controls
            if keyboard.is_pressed('f5') and not playing:
                print("\n▶️ Strategic AI Activated!")
                playing = True
                time.sleep(0.5)

            elif keyboard.is_pressed('f6') and playing:
                print("\n⏸️ Paused - Intensive Learning...")
                playing = False

                # Intensive learning session
                for i in range(20):
                    loss = self.learn_from_experience()
                    if loss and i % 5 == 0:
                        print(f"  📚 Learning iteration {i + 1}: Loss = {loss:.4f}")

                self.update_exploration_rate()
                time.sleep(0.5)

            elif keyboard.is_pressed('f7'):
                print("\n💾 Manual save...")
                self.save_enhanced_progress()
                time.sleep(0.5)

            elif keyboard.is_pressed('escape'):
                print("\n🛑 Stopping strategic AI...")
                break

            # Main gameplay loop
            if playing:
                try:
                    # Capture state
                    state_tensor, game_state, screenshot = self.capture_and_analyze()

                    # Strategic decision
                    action = self.select_action(state_tensor, game_state)

                    # Execute
                    self.execute_action(action)

                    # Brief pause
                    time.sleep(0.3)

                    # Capture result
                    next_state, next_game_state, _ = self.capture_and_analyze()

                    # Calculate reward
                    reward = self.reward_system.calculate_reward(
                        game_state, next_game_state, action
                    )

                    # Store experience
                    self.experience_buffer.append({
                        'state': state_tensor,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state,
                        'game_state': game_state
                    })

                    # Periodic learning
                    if self.metrics['total_actions'] % 10 == 0:
                        self.learn_from_experience()

                    # Update exploration
                    if self.metrics['total_actions'] % 50 == 0:
                        self.update_exploration_rate()

                    # Show stats every minute
                    if time.time() - last_stats > 60:
                        avg_reward = np.mean(list(self.reward_system.last_rewards)[-50:])
                        print(f"\n📊 Status: {self.metrics['total_actions']} actions | "
                              f"Reward: {avg_reward:.2f} | "
                              f"Explore: {self.exploration_rate:.1%}")
                        last_stats = time.time()

                    # Auto-save every 5 minutes
                    if time.time() - last_save > 300:
                        print("\n🔄 Auto-saving progress...")
                        self.save_enhanced_progress()
                        last_save = time.time()

                except pyautogui.FailSafeException:
                    print("\n⚠️ Emergency stop!")
                    playing = False
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(1)

        # Final save
        self.save_enhanced_progress()
        print("\n✨ Strategic AI session complete!")
        print(f"📊 Final Statistics:")
        print(f"  Total Actions: {self.metrics['total_actions']}")
        print(f"  Learning Iterations: {self.metrics['learning_iterations']}")
        print(f"  Final Exploration: {self.exploration_rate:.1%}")
        print(f"  Runtime: {(time.time() - self.metrics['time_started']) / 60:.1f} minutes")


def main():
    """Launch the strategic AI"""
    print("🎯 HOI4 Strategic Learning AI")
    print("=" * 50)

    # Ensure directories exist
    os.makedirs('models', exist_ok=True)

    # Create and run strategic learner
    ai = EnhancedIterativeLearner()
    ai.play_and_learn()


if __name__ == "__main__":
    main()
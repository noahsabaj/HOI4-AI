# src/ai/ultimate/train_ultimate.py
"""
Training loop for Ultimate HOI4 AI
Integrates with your existing main.py
"""

import os
import sys
import time
import keyboard
import pyautogui
from PIL import ImageGrab
from datetime import datetime
import json
import re
from typing import Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import our Ultimate AI
from src.ai.ultimate.ultimate_ai import UltimateHOI4AI

# Import your existing evaluation system
from src.strategy.evaluation import StrategicEvaluator


class UltimateTrainer:
    """
    Manages the training loop for Ultimate AI
    Works alongside your existing code
    """

    def __init__(self):
        print("""
        ╔═══════════════════════════════════════════════════════════╗
        ║              Ultimate HOI4 AI Training System             ║
        ║                                                           ║
        ║  DreamerV3 + RND + NEC = Autonomous Learning             ║
        ╚═══════════════════════════════════════════════════════════╝
        """)

        # Initialize Ultimate AI
        self.ai = UltimateHOI4AI()

        # Your existing evaluator
        self.evaluator = StrategicEvaluator()

        # Training state
        self.is_training = False
        self.session_start = time.time()
        self.last_screenshot = None
        self.last_action = None

        # Metrics
        self.session_metrics = {
            'actions_taken': 0,
            'discoveries': [],
            'key_moments': [],
            'current_strategy': 'exploring'
        }

        # Initialize strategic health tracking
        self.last_strategic_health = 0.0

        # Auto-save interval
        self.last_save = time.time()
        self.save_interval = 300  # 5 minutes

        # Status display
        self.last_status_update = time.time()
        self.status_interval = 10  # 10 seconds

    def run(self):
        """Main training loop"""
        print("\n📋 Controls:")
        print("  F5: Start/Resume Training")
        print("  F6: Pause Training")
        print("  F7: Save Progress")
        print("  F8: Show Statistics")
        print("  ESC (hold 2s): Exit")
        print("\n🎮 Start HOI4 and press F5 when ready...")

        # Control flags
        esc_start_time = None

        while True:
            # Handle controls
            if keyboard.is_pressed('f5') and not self.is_training:
                self.start_training()
                time.sleep(0.3)

            elif keyboard.is_pressed('f6') and self.is_training:
                self.pause_training()
                time.sleep(0.3)

            elif keyboard.is_pressed('f7'):
                self.save_progress()
                time.sleep(0.3)

            elif keyboard.is_pressed('f8'):
                self.show_statistics()
                time.sleep(0.3)

            elif keyboard.is_pressed('escape'):
                if esc_start_time is None:
                    esc_start_time = time.time()
                elif time.time() - esc_start_time > 2.0:
                    print("\n🛑 Stopping Ultimate AI...")
                    self.cleanup()
                    break
            else:
                esc_start_time = None

            # Training step
            if self.is_training:
                self.training_step()

            # Auto-save
            if time.time() - self.last_save > self.save_interval:
                self.save_progress()
                self.last_save = time.time()

            # Status update
            if time.time() - self.last_status_update > self.status_interval:
                self.update_status()
                self.last_status_update = time.time()

            # Small delay to prevent CPU overuse
            time.sleep(0.05)

    def training_step(self):
        """Single training step"""
        try:
            # Capture screenshot
            screenshot = ImageGrab.grab()

            # Get AI decision
            action = self.ai.act(screenshot)

            # Execute action
            self.execute_action(action)

            # Brief pause for game to respond
            time.sleep(0.1)

            # Learn from transition if we have previous state
            if self.last_screenshot is not None:
                # Calculate reward (simplified - you can enhance this)
                reward = self.calculate_reward(self.last_screenshot, screenshot, action)

                # AI learns
                self.ai.learn_from_transition(
                    self.last_screenshot,
                    self.last_action,
                    screenshot,
                    reward
                )

                # Track significant moments
                if abs(reward) > 5.0:
                    self.session_metrics['key_moments'].append({
                        'time': time.time() - self.session_start,
                        'action': action,
                        'reward': reward,
                        'description': self.describe_moment(action, reward)
                    })

            # Update state
            self.last_screenshot = screenshot
            self.last_action = action
            self.session_metrics['actions_taken'] += 1

        except Exception as e:
            print(f"\n❌ Error in training step: {e}")
            self.pause_training()

    def execute_action(self, action: Dict):
        """Execute the AI's chosen action"""
        if action['type'] == 'click':
            pyautogui.click(
                action['x'],
                action['y'],
                button=action.get('button', 'left')
            )
            if self.session_metrics['actions_taken'] % 50 == 0:
                print(f"\n🖱️ {action['description']}")

        elif action['type'] == 'key':
            pyautogui.press(action['key'])
            if action['key'] in ['q', 'w', 'e', 'r', 't', 'b']:
                print(f"\n⌨️ {action['description']}")

    def calculate_reward(self, prev_screenshot, curr_screenshot, action) -> float:
        """
        Calculate reward for the transition
        You can make this more sophisticated!
        """
        # Basic reward structure
        reward = -0.01  # Small negative for time pressure

        # Use your existing OCR to detect changes
        prev_ocr = self.ai.ocr.extract_all_text(prev_screenshot)
        curr_ocr = self.ai.ocr.extract_all_text(curr_screenshot)

        # Reward for changing screens (exploration)
        prev_screen = self.detect_screen_type(prev_ocr)
        curr_screen = self.detect_screen_type(curr_ocr)
        if prev_screen != curr_screen:
            reward += 2.0

        # Reward for resource changes
        prev_pp = self.extract_number(prev_ocr.get('political_power', '0'))
        curr_pp = self.extract_number(curr_ocr.get('political_power', '0'))

        if curr_pp > prev_pp:
            reward += 0.5  # PP increased

        # Factory changes (very good!)
        prev_factories = prev_ocr.get('factories', '')
        curr_factories = curr_ocr.get('factories', '')

        if 'factory' in action.get('description', '').lower() and curr_factories != prev_factories:
            reward += 5.0  # Likely built a factory

        # Use your strategic evaluator
        game_state = self.build_game_state(curr_ocr)
        strategic_eval = self.evaluator.evaluate_game_state(game_state, curr_ocr)

        # Reward for improving strategic position
        if hasattr(self, 'last_strategic_health'):
            health_delta = strategic_eval['strategic_health'] - self.last_strategic_health
            reward += health_delta * 10

        self.last_strategic_health = strategic_eval['strategic_health']

        return reward

    def describe_moment(self, action: Dict, reward: float) -> str:
        """Describe a key moment for logging"""
        if reward > 5:
            return f"Great success! {action['description']} led to major progress"
        elif reward > 2:
            return f"Good move: {action['description']}"
        elif reward < -2:
            return f"Setback after {action['description']}"
        else:
            return f"Explored {action['description']}"

    def detect_screen_type(self, ocr_data: Dict) -> str:
        """Detect current game screen from OCR data"""
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

    def extract_number(self, text: str) -> int:
        """Extract number from text"""
        match = re.search(r'(\d+)', text)
        return int(match.group(1)) if match else 0

    def build_game_state(self, ocr_data: Dict) -> Dict:
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
        numbers = re.findall(r'(\d+)', factory_text)
        if len(numbers) >= 2:
            state['factories']['civilian'] = int(numbers[0])
            state['factories']['military'] = int(numbers[1])

        return state

    def start_training(self):
        """Start or resume training"""
        self.is_training = True
        print("\n▶️ Training active! AI is learning HOI4...")
        print("🧠 Watch as it discovers game mechanics through pure exploration!")

    def pause_training(self):
        """Pause training"""
        self.is_training = False
        print("\n⏸️ Training paused")

    def save_progress(self):
        """Save current progress"""
        # Save AI checkpoint
        checkpoint_path = f"checkpoints/ultimate_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        os.makedirs('checkpoints', exist_ok=True)
        self.ai.save_checkpoint(checkpoint_path)

        # Save session metrics
        metrics_path = f"checkpoints/session_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.session_metrics, f, indent=2)

        print(f"\n💾 Progress saved!")

    def show_statistics(self):
        """Display current statistics"""
        stats = self.ai.get_statistics()

        print("\n📊 Ultimate AI Statistics")
        print("=" * 50)
        print(f"Total Steps: {stats['total_steps']:,}")
        print(f"Episode Steps: {stats['episode_steps']:,}")
        print(f"Total Reward: {stats['total_reward']:.2f}")
        print(f"Avg Intrinsic Reward: {stats['avg_intrinsic_reward']:.3f}")

        print(f"\n🧠 Neural Episodic Control:")
        nec_stats = stats['nec_stats']
        print(f"  Memory Size: {nec_stats['memory_size']:,}/{50000:,}")
        print(f"  Lookups per Write: {nec_stats['lookups_per_write']:.1f}")

        print(f"\n🔍 Curiosity (RND):")
        curiosity_stats = stats['curiosity_stats']
        print(f"  Mean Intrinsic: {curiosity_stats['mean_intrinsic_reward']:.3f}")
        print(f"  Max Intrinsic: {curiosity_stats['max_intrinsic_reward']:.3f}")

        print(f"\n🎯 Key Moments: {len(self.session_metrics['key_moments'])}")
        if self.session_metrics['key_moments']:
            latest = self.session_metrics['key_moments'][-1]
            print(f"  Latest: {latest['description']}")

    def update_status(self):
        """Update status line"""
        if self.is_training:
            runtime = (time.time() - self.session_start) / 60
            apm = self.session_metrics['actions_taken'] / runtime

            status = (
                f"\r⚡ Actions: {self.session_metrics['actions_taken']} | "
                f"APM: {apm:.1f} | "
                f"Runtime: {runtime:.1f}m | "
                f"Key Moments: {len(self.session_metrics['key_moments'])}"
            )
            print(status, end='', flush=True)

    def cleanup(self):
        """Clean up before exit"""
        self.save_progress()

        # Final statistics
        print("\n\n📊 Final Session Summary")
        print("=" * 50)

        runtime = (time.time() - self.session_start) / 60
        print(f"Total Runtime: {runtime:.1f} minutes")
        print(f"Total Actions: {self.session_metrics['actions_taken']:,}")
        print(f"Actions per Minute: {self.session_metrics['actions_taken'] / runtime:.1f}")

        # Interesting discoveries
        if self.ai.curiosity.seen_screens:
            print(f"\n🗺️ Screens Discovered: {len(self.ai.curiosity.seen_screens)}")
            for screen in sorted(self.ai.curiosity.seen_screens):
                print(f"  • {screen}")

        # Key moments
        if self.session_metrics['key_moments']:
            print(f"\n⭐ Top Key Moments:")
            sorted_moments = sorted(
                self.session_metrics['key_moments'],
                key=lambda x: x['reward'],
                reverse=True
            )[:5]

            for moment in sorted_moments:
                print(f"  • {moment['description']} (reward: {moment['reward']:.1f})")

        # Get successful strategies from persistent memory
        strategies = self.ai.persistent_memory.get_successful_strategies()
        if strategies:
            print(f"\n💡 Learned Strategies:")
            for strategy in strategies[:5]:
                print(f"  • {strategy['description'][:100]}...")

        print("\n👋 Thanks for training the Ultimate HOI4 AI!")


def integrate_with_main():
    """
    Function to integrate with your existing main.py
    Add this to your main.py file
    """

    def run_ultimate_mode(args):
        """Run the Ultimate AI mode"""
        print(f"\n🚀 ULTIMATE AI MODE - DreamerV3 + RND + NEC")
        print(f"{'=' * 60}")
        print(f"This mode features:")
        print(f"  ✓ World model that learns game dynamics")
        print(f"  ✓ Curiosity-driven exploration (RND)")
        print(f"  ✓ Fast learning from experience (NEC)")
        print(f"  ✓ Persistent memory across games")
        print(f"  ✓ Pure self-play learning")

        # Create trainer instance
        ultimate_trainer = UltimateTrainer()

        # Run training loop
        ultimate_trainer.run()

    return run_ultimate_mode


# Standalone entry point
if __name__ == "__main__":
    trainer_instance = UltimateTrainer()
    trainer_instance.run()
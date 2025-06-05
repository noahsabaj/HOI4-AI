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
from collections import deque
from PIL import ImageGrab
import torch
import numpy as np
from datetime import datetime
from queue import Queue, Empty, Full
from threading import Thread, Event
import json
import re
from typing import Dict
from src.config import CONFIG
from .hf_strategic_evaluator import HuggingFaceStrategicEvaluator
from src.utils.common import extract_number, detect_screen_type
from src.utils.logger import get_logger
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
# Import our Ultimate AI
from .ultimate_ai import UltimateHOI4AI, ActionSpace
# Import your existing evaluation system
from src.strategy.evaluation import StrategicEvaluator

# ── window-focus helper ─────────────────────────────────────────
def _ensure_hoi4_window_foreground() -> None:
    """
    Bring the Hearts of Iron IV window to the foreground if it isn’t.
    Works on Windows; NO-OP on other OSes.
    """
    try:
        import pygetwindow as gw
        win = next(w for w in gw.getWindowsWithTitle('Hearts of Iron') if w.isVisible)
        if not win.isActive:
            win.activate()
    except Exception:
        # don’t crash if pygetwindow missing or window not found
        pass


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
        ║  DreamerV3 + RND + NEC = Autonomous Learning              ║
        ╚═══════════════════════════════════════════════════════════╝
        """)

        # ── colour logger ──────────────────────────────────────────
        self.log = get_logger("trainer")
        self.log.info("🚀 Trainer initialized")

        # Initialize Ultimate AI & evaluator
        self.ai = UltimateHOI4AI()
        self.evaluator = StrategicEvaluator()

        # New code
        print("  🤗 Initializing Async Phi-4 strategic evaluator...")
        from .async_strategic_evaluator import AsyncStrategicEvaluator
        self.strategic_reasoner = AsyncStrategicEvaluator(
            evaluation_interval=60.0  # Evaluate every 60 seconds
        )

        # Add trainer reference to AI for accessing cached advice
        self.ai._trainer_ref = self

        self.learn_queue = Queue(maxsize=64)  # holds (prev, act, next, r)
        self.shutdown_event = Event()
        self.learn_thread = Thread(
            target=self._learner_loop,
            name="learner-thread",
            daemon=True,
        )
        self.learn_thread.start()

        # Training state
        self.is_training = False
        self.session_start = time.time()
        self.last_screenshot = None
        self.last_action = None

        # Metrics
        self.session_metrics = {
            "actions_taken": 0,
            "discoveries": [],
            "key_moments": [],
            "current_strategy": "exploring",
        }

        # Strategic health tracking
        self.last_strategic_health = 0.0

        # Auto-save & status intervals
        self.last_save = time.time()
        self.save_interval = 300  # 5 min, could move to CONFIG later
        self.last_status_update = time.time()
        self.status_interval = 10  # 10 s

    def run(self):
        """Main training loop"""
        self.log.info("📋 Controls:")
        self.log.info("  F5: Start/Resume Training")
        self.log.info("  F6: Pause Training")
        self.log.info("  F7: Save Progress")
        self.log.info("  F8: Show Statistics")
        self.log.info("  ESC (hold 2s): Exit")
        self.log.info("🎮 Start HOI4 and press F5 when ready...")

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
            time.sleep(0.005)

    def training_step(self):
        """Single non-blocking env step with async learning"""
        try:
            # 1. Screenshot (target: <100ms)
            t0 = time.perf_counter()
            screenshot = ImageGrab.grab()
            t1 = time.perf_counter()

            # 2. Fast decision (target: <50ms)
            action = self.ai.act(screenshot)
            t2 = time.perf_counter()

            # Emergency intervention after too many repetitions
            if not hasattr(self, '_intervention_counter'):
                self._intervention_counter = 0

            self._intervention_counter += 1
            if self._intervention_counter > 500:  # Every 500 actions
                print("🚨 INTERVENTION: Forcing construction menu")
                action = {'type': 'key', 'key': 'b', 'description': 'Emergency: Open construction'}
                self._intervention_counter = 0

            # 3. Execute (target: <100ms)
            self.execute_action(action)
            t3 = time.perf_counter()

            # Track menu discoveries
            if action['type'] == 'key' and action['key'] in ['q', 'w', 'e', 'r', 't', 'b', 'v', 'n', 'm']:
                if not hasattr(self, 'discovered_menus'):
                    self.discovered_menus = set()
                if action['key'] not in self.discovered_menus:
                    self.discovered_menus.add(action['key'])
                    print(f"🎉 NEW MENU DISCOVERED: {action['key']} ({len(self.discovered_menus)}/9)")

                    # Pass to fast policy
                    if hasattr(self.ai.fast_policy, 'discovered_menus'):
                        self.ai.fast_policy.discovered_menus = self.discovered_menus.copy()

            # 4. Queue for learning (non-blocking)
            if self.last_screenshot is not None:
                # Fast reward calculation
                reward = self.calculate_reward_fast(
                    self.last_screenshot, screenshot, action
                )

                # Get real game state for strategic evaluator every 30 actions
                if self.session_metrics["actions_taken"] % 30 == 0:
                    ocr_data = self.ai.ocr.extract_all_text(screenshot)
                    game_state = self.build_game_state(ocr_data)
                    self.strategic_reasoner.record_action(action, game_state)

                try:
                    self.learn_queue.put_nowait(
                        (self.last_screenshot, self.last_action, screenshot, reward)
                    )
                except Full:
                    pass  # Drop if queue full

            # 5. Update state
            self.last_screenshot = screenshot
            self.last_action = action
            self.session_metrics["actions_taken"] += 1

            # 6. Log performance (only if slow)
            total_ms = (time.perf_counter() - t0) * 1000
            if total_ms > 200:  # Only log if slower than 200ms
                dt = lambda a, b: (b - a) * 1000
                self.log.info(
                    f"⏱ cap {dt(t0, t1):.0f}ms | "
                    f"dec {dt(t1, t2):.0f}ms | "
                    f"act {dt(t2, t3):.0f}ms | "
                    f"tot {total_ms:.0f}ms"
                )

        except Exception as e:
            self.log.error(f"❌ Step error: {e}")
            import traceback
            traceback.print_exc()

    def _learner_loop(self) -> None:
        """
        Runs in a daemon thread.
        Blocks on `learn_queue` and calls `self.ai.learn_from_transition`.
        """
        while not self.shutdown_event.is_set():
            try:
                prev_scr, prev_act, next_scr, reward = self.learn_queue.get(timeout=0.5)
            except Empty:
                continue  # nothing to do – loop again
            try:
                self.ai.learn_from_transition(prev_scr, prev_act, next_scr, reward)
            except Exception as e:
                self.log.error(f"♻️ learner thread error: {e}")

    def execute_action(self, action: Dict):
        """Execute the AI's chosen action"""
        if action['type'] == 'click':
            pyautogui.click(
                action['x'],
                action['y'],
                button=action.get('button', 'left')
            )
            # Log every 10th action or significant ones
            if (self.session_metrics['actions_taken'] % 10 == 0 or
                    any(keyword in action.get('description', '').lower()
                        for keyword in ['factory', 'build', 'research', 'focus', 'production'])):
                print(
                    f"\n🖱️ [{self.session_metrics['actions_taken']}] {action['description']} at ({action['x']}, {action['y']})")

        elif action['type'] == 'key':
            _ensure_hoi4_window_foreground()
            key = action['key']
            pyautogui.keyDown(key)
            time.sleep(0.05)
            pyautogui.keyUp(key)
            print(f"\n⌨️ [{self.session_metrics['actions_taken']}] Pressed {key} - {action.get('description', '')}")

    def log_game_state(self, ocr_data: Dict, screen_type: str):
        """Log important game state information"""
        # Only log every 20 actions to avoid spam
        if self.session_metrics['actions_taken'] % 20 == 0:
            date = ocr_data.get('date', 'Unknown')
            pp = self.extract_number(ocr_data.get('political_power', '0'))

            print(f"\n📊 Game State Check:")
            print(f"   Date: {date}")
            print(f"   Screen: {screen_type}")
            print(f"   Political Power: {pp}")

            # Log any factory info
            factory_text = ocr_data.get('factories', '')
            if factory_text:
                print(f"   Factories: {factory_text}")

            # Log first few OCR entries to debug
            print(f"   OCR Keys: {list(ocr_data.keys())[:5]}...")

    def calculate_reward(self, prev_screenshot, curr_screenshot, action) -> float:
        """
        Calculate reward for the transition including DeepSeek-R1 evaluation
        """
        # Basic reward structure
        reward = -0.01  # Small negative for time pressure

        # Use your existing OCR to detect changes
        prev_ocr = self.ai.ocr.extract_all_text(prev_screenshot)
        curr_ocr = self.ai.ocr.extract_all_text(curr_screenshot)

        # After OCR extraction
        curr_screen = self.detect_screen_type(curr_ocr)
        self.log_game_state(curr_ocr, curr_screen)

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

        # Pixel-difference novelty bonus
        prev_arr = np.asarray(prev_screenshot, dtype=np.float32) / 255.0
        curr_arr = np.asarray(curr_screenshot, dtype=np.float32) / 255.0
        diff_ratio = (np.abs(prev_arr - curr_arr) > 0.05).mean()
        reward += diff_ratio * 2.0

        # === DEEPSEEK-R1 STRATEGIC EVALUATION ===
        # Build comprehensive game state for R1
        enhanced_game_state = {
            **game_state,  # Your existing game state
            'game_date': curr_ocr.get('date', 'Unknown'),
            'political_power': curr_pp,
            'screen_type': curr_screen,
            'civilian_factories': game_state['factories'].get('civilian', 0),
            'military_factories': game_state['factories'].get('military', 0),
        }

        # New non-blocking code
        # Record action (non-blocking)
        self.strategic_reasoner.record_action(action, enhanced_game_state)

        # Get cached reward modifier (instant)
        strategic_modifier = self.strategic_reasoner.get_reward_modifier()
        reward += strategic_modifier

        # Log significant modifiers
        if abs(strategic_modifier) > 2.0:
            advice = self.strategic_reasoner.get_cached_advice()
            self.session_metrics['key_moments'].append({
                'step': self.session_metrics['actions_taken'],
                'description': f"Strategic insight: {advice.get('reasoning', 'N/A')}",
                'reward': strategic_modifier,
                'strategy': advice['strategy']
            })

        return reward

    def calculate_reward_fast(self, prev_screenshot, curr_screenshot, action) -> float:
        """
        Fast reward calculation (<10ms)
        Heavy computation done async in strategic evaluator
        """
        # Initialize action history if needed
        if not hasattr(self, 'action_history'):
            self.action_history = deque(maxlen=50)

        # Track actions
        self.action_history.append(action)

        # Base time penalty
        reward = -0.01

        # Quick pixel difference check (novelty)
        prev_small = np.array(prev_screenshot.resize((128, 72)))
        curr_small = np.array(curr_screenshot.resize((128, 72)))

        # Fast difference calculation
        diff = np.abs(prev_small.astype(np.float32) - curr_small.astype(np.float32))
        novelty = (diff > 20).mean()  # Threshold for change
        reward += novelty * 2.0

        # Penalty for too many repeated actions
        recent_actions = [a['key'] for a in self.action_history if a['type'] == 'key'][-20:]
        if len(recent_actions) > 0:
            w_t_count = recent_actions.count('w') + recent_actions.count('t')
            if w_t_count > 15:
                reward -= 5.0  # Heavy penalty
                print(f"⚠️ W/T spam detected: {w_t_count}/20 actions")

        # Bonus for certain action types (heuristic)
        if action['type'] == 'key' and action['key'] in ['b', 't', 'w', 'v']:
            reward += 0.5  # Likely useful keys

        # Penalize clicking in dead zones
        if action['type'] == 'click':
            x, y = action['x'], action['y']
            # Don't reward clicking on map (usually doesn't do anything useful early)
            if 500 < x < 3340 and 200 < y < 1800:
                reward -= 0.5  # Discourage aimless map clicking
            # Reward UI areas
            elif x < 500 or x > 3340 or y < 200 or y > 1800:
                reward += 0.5  # Encourage UI exploration

        # Get async strategic modifier
        strategic_modifier = self.strategic_reasoner.get_reward_modifier()
        reward += strategic_modifier

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

        # ── NEW: force one real menu switch so the agent gets +2 reward ──
        bootstrap_action = {"type": "key",
                            "key": "f1",  # F1 = Politics screen
                            "description": "Press F1 (politics bootstrap)"}
        self.execute_action(bootstrap_action)
        # ----------------------------------------------------------------

    def pause_training(self):
        """Pause training"""
        self.is_training = False
        print("\n⏸️ Training paused")

    def save_progress(self):
        """Save current progress"""
        # Save AI checkpoint
        checkpoint_path = f"{CONFIG.checkpoint_dir}/ultimate_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
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

            # Get current strategy from DeepSeek-R1
            advice = self.strategic_reasoner.get_cached_advice()
            current_strategy = advice['strategy']

            status = (
                f"\r⚡ Actions: {self.session_metrics['actions_taken']} | "
                f"APM: {apm:.1f} | "
                f"Runtime: {runtime:.1f}m | "
                f"Key Moments: {len(self.session_metrics['key_moments'])} | "
                f"Phase: {current_strategy}"
            )
            print(status, end='', flush=True)

    def cleanup(self):
        """Clean up before exit"""
        # Shutdown async components first
        self.strategic_reasoner.shutdown()

        # Then existing cleanup
        self.shutdown_event.set()
        self.learn_thread.join(timeout=2)

        self.save_progress()

        # Final statistics
        print("\n\n📊 Final Session Summary")
        print("=" * 50)

        runtime = (time.time() - self.session_start) / 60
        print(f"Total Runtime: {runtime:.1f} minutes")
        print(f"Total Actions: {self.session_metrics['actions_taken']:,}")
        print(f"Actions per Minute: {self.session_metrics['actions_taken'] / runtime:.1f}")

        # === ADD THIS: DEEPSEEK-R1 STRATEGIC ANALYSIS ===
        strategic_summary = self.strategic_reasoner.get_strategic_summary()

        print(f"\n🧠 DeepSeek-R1 Strategic Analysis:")
        print(f"  • Strategic evaluations performed: {strategic_summary['evaluations']}")
        print(f"  • Average strategic score: {strategic_summary['average_reward']:+.2f}")
        print(f"  • Current strategic phase: {strategic_summary['current_strategy']}")
        print(f"  • Strategic decisions made: {strategic_summary['total_decisions']}")

        if strategic_summary['recent_insights']:
            print(f"\n💡 Recent Strategic Insights from DeepSeek-R1:")
            for insight in strategic_summary['recent_insights'][-3:]:
                print(f"  • {insight.get('insight', 'No insight text')}")

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
        strategies = []
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
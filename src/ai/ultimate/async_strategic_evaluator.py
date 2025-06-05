# src/ai/ultimate/async_strategic_evaluator.py
"""
Async wrapper for HuggingFace Strategic Evaluator
Runs Phi-4 evaluations in background thread
"""

import time
import threading
from queue import Queue, Empty
from typing import Dict, Optional, Tuple
from collections import deque
import numpy as np

from .hf_strategic_evaluator import HuggingFaceStrategicEvaluator


class AsyncStrategicEvaluator:
    """Runs strategic evaluation in background, provides cached advice"""

    def __init__(self, evaluation_interval: float = 60.0):
        # Core evaluator
        self.evaluator = HuggingFaceStrategicEvaluator(
            evaluation_frequency=999999  # Disable internal frequency check
        )

        # Threading
        self.eval_queue = Queue(maxsize=10)
        self.shutdown_event = threading.Event()
        self.eval_thread = threading.Thread(
            target=self._evaluation_loop,
            name="strategic-evaluator",
            daemon=True
        )

        # Caching
        self.current_advice = {
            'strategy': 'exploring',
            'reward_modifier': 0.0,
            'suggestions': [],
            'timestamp': 0,
            'valid_until': 0
        }
        self.advice_lock = threading.Lock()

        # Settings
        self.evaluation_interval = evaluation_interval
        self.last_eval_time = 0

        # Start thread
        self.eval_thread.start()
        print(f"✅ Strategic evaluator thread started (will evaluate every {evaluation_interval}s)")

    def record_action(self, action: Dict, game_state: Dict):
        """Non-blocking action recording"""
        # Always record to history
        self.evaluator.record_action(action, game_state)

        # Check if time for new evaluation
        current_time = time.time()
        if current_time - self.last_eval_time > self.evaluation_interval:
            # Try to queue evaluation (non-blocking)
            try:
                self.eval_queue.put_nowait({
                    'game_state': game_state.copy(),
                    'timestamp': current_time
                })
                self.last_eval_time = current_time
            except:
                pass  # Queue full, skip this evaluation

    def get_cached_advice(self) -> Dict:
        """Get current strategic advice (instant)"""
        with self.advice_lock:
            return self.current_advice.copy()

    def get_reward_modifier(self) -> float:
        """Get current reward modifier based on strategic evaluation"""
        with self.advice_lock:
            # Check if advice is still valid
            if time.time() < self.current_advice['valid_until']:
                return self.current_advice['reward_modifier']
            return 0.0  # No modifier if expired

    def _evaluation_loop(self):
        """Background thread that runs Phi-4 evaluation"""
        print("🧵 Strategic evaluator thread running...")
        evaluations_done = 0

        while not self.shutdown_event.is_set():
            try:
                # Get evaluation request (blocks for 0.5s if empty)
                request = self.eval_queue.get(timeout=0.5)
                evaluations_done += 1
                print(f"🔍 Strategic evaluation #{evaluations_done} starting...")

                # Run expensive Phi-4 evaluation (2-5 seconds)
                game_state = request['game_state']
                context = self.evaluator._prepare_context()

                # Get evaluation from model
                reward, reasoning, insight = self.evaluator._model_evaluation(context)

                # Package advice
                new_advice = {
                    'strategy': self.evaluator.get_current_strategy(),
                    'reward_modifier': reward,
                    'suggestions': self._parse_suggestions(insight),
                    'reasoning': reasoning,
                    'timestamp': request['timestamp'],
                    'valid_until': request['timestamp'] + self.evaluation_interval * 1.5
                }

                # Update cached advice
                with self.advice_lock:
                    self.current_advice = new_advice

                print(f"\n🤖 Strategic Update: {new_advice['strategy']}")
                if abs(reward) > 2.0:
                    print(f"   💡 {insight}")

            except Empty:
                continue
            except Exception as e:
                print(f"❌ Strategic evaluator error: {e}")

    def _parse_suggestions(self, insight: str) -> list:
        """Convert insight text to actionable suggestions"""
        suggestions = []

        insight_lower = insight.lower()
        if "construction" in insight_lower or "factory" in insight_lower:
            suggestions.append({'action': 'open_construction', 'priority': 1.0})
        if "research" in insight_lower:
            suggestions.append({'action': 'open_research', 'priority': 0.8})
        if "production" in insight_lower:
            suggestions.append({'action': 'open_production', 'priority': 0.7})

        return suggestions

    def get_strategic_summary(self) -> Dict:
        """Get summary of strategic performance"""
        return self.evaluator.get_strategic_summary()

    def shutdown(self):
        """Clean shutdown"""
        self.shutdown_event.set()
        self.eval_thread.join(timeout=2.0)
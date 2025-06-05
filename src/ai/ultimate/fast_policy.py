# src/ai/ultimate/fast_policy.py
"""
Fast policy for real-time decision making
Makes decisions in <20ms using cached values
"""

import time
import torch
import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque


class FastPolicy:
    """Lightning-fast decision maker using cached strategic advice"""

    def __init__(self, action_space, device='cuda'):
        self.action_space = action_space
        self.device = device

        # Caches
        self.last_screen_hash = None
        self.last_action_idx = None
        self.action_history = deque(maxlen=20)

        # Simple state
        self.stuck_counter = 0
        self.exploration_boost = 0.3

        # Precomputed action weights for different strategies
        self.strategy_weights = {
            'exploring': self._get_exploration_weights(),
            'Economic Buildup': self._get_economic_weights(),
            'Military Transition': self._get_military_weights(),
            'War Preparation': self._get_war_weights(),
        }

        self.action_sequences = deque(maxlen=100)  # Track longer sequences
        self.loop_detector = {}  # Track repeating patterns

    def act(self,
            encoded_obs: torch.Tensor,
            cached_strategy: str = 'exploring',
            cached_suggestions: list = None) -> int:
        """
        Make decision in <20ms

        Args:
            encoded_obs: Pre-encoded observation from world model
            cached_strategy: Current strategy from async evaluator
            cached_suggestions: Specific suggestions from async evaluator

        Returns:
            action_idx: Index into action space
        """
        # Auto-unpause detection
        if not hasattr(self, 'frames_since_change'):
            self.frames_since_change = 0

        # Simple heuristic: if nothing changed, try spacebar
        if encoded_obs.std() < 0.1:  # Very low variance = static screen
            self.frames_since_change += 1
            if self.frames_since_change > 30:  # About 1 second of no change
                print("🎮 Game might be paused, trying spacebar...")
                self.frames_since_change = 0
                return self._get_key_idx('space')
        else:
            self.frames_since_change = 0

        start_time = time.perf_counter()

        # Get action weights for current strategy
        weights = self.strategy_weights.get(cached_strategy,
                                            self.strategy_weights['exploring'])

        # Boost weights for suggested actions
        if cached_suggestions:
            for suggestion in cached_suggestions:
                if suggestion['action'] == 'open_construction':
                    # Boost construction-related keys
                    weights[self._get_key_idx('b')] *= 2.0
                elif suggestion['action'] == 'open_research':
                    weights[self._get_key_idx('w')] *= 2.0
                elif suggestion['action'] == 'open_production':
                    weights[self._get_key_idx('t')] *= 2.0

        # Enhanced loop detection
        if self.last_action_idx is not None:
            self.action_sequences.append(self.last_action_idx)

            # Check for various loop patterns
            if len(self.action_sequences) >= 20:
                # Check 2-action loops (w,t,w,t...)
                last_20 = list(self.action_sequences)[-20:]
                pattern_2 = [last_20[i:i + 2] for i in range(0, 18, 2)]
                if len(set(map(tuple, pattern_2))) == 1:
                    print("🚨 2-ACTION LOOP DETECTED!")
                    return self._force_random_exploration()

                # Check 3-action loops
                pattern_3 = [last_20[i:i + 3] for i in range(0, 18, 3)]
                if len(set(map(tuple, pattern_3))) == 1:
                    print("🚨 3-ACTION LOOP DETECTED!")
                    return self._force_random_exploration()

                # Check if stuck on same 2-3 actions
                unique_in_20 = len(set(last_20))
                if unique_in_20 <= 2:
                    print(f"🚨 STUCK ON {unique_in_20} ACTIONS!")
                    return self._force_random_exploration()

        # Add exploration noise
        if np.random.random() < self.exploration_boost:
            weights += np.random.exponential(0.5, size=len(weights))

        # Normalize weights
        weights = weights / weights.sum()

        # Sample action
        action_idx = np.random.choice(len(weights), p=weights)
        self.last_action_idx = action_idx

        # Check timing
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > 20:
            print(f"⚠️ Slow decision: {elapsed_ms:.1f}ms")

        return action_idx

    def _get_exploration_weights(self) -> np.ndarray:
        """Weights for exploration phase"""
        weights = np.ones(self.action_space.get_action_size())

        # Prefer menu keys
        for key in ['f1', 'f2', 'f3', 'f4', 'f5', 'q', 'w', 'e', 'r', 't', 'b', 'v', 'n']:
            idx = self._get_key_idx(key)
            if idx is not None:
                # Lower weight for already discovered menus
                if hasattr(self, 'discovered_menus') and key in self.discovered_menus:
                    weights[idx] = 1.5  # Still some chance, but lower
                else:
                    weights[idx] = 5.0  # High priority for undiscovered

        return weights

    def _get_economic_weights(self) -> np.ndarray:
        """Weights for economic buildup (1936-1937)"""
        weights = np.ones(self.action_space.get_action_size()) * 0.5

        # Heavily prefer construction
        weights[self._get_key_idx('b')] = 10.0  # Build mode
        weights[self._get_key_idx('f1')] = 5.0  # Politics

        # Also boost clicks in construction area
        for i in range(len(self.action_space.base_keys),
                       len(self.action_space.base_keys) + 20):
            weights[i] = 2.0

        return weights

    def _force_random_exploration(self):
        """Force a completely different action"""
        # Clear history to reset
        self.action_sequences.clear()
        self.stuck_counter = 0

        # Pick something we haven't done recently
        all_actions = list(range(self.action_space.get_action_size()))

        # Get recent actions if we have any
        if len(self.action_sequences) > 0:
            recent = set(list(self.action_sequences)[-50:])
        else:
            recent = set()

        # Remove recent actions from choices
        choices = [a for a in all_actions if a not in recent]
        if not choices:
            choices = all_actions

        return np.random.choice(choices)

    def _get_military_weights(self) -> np.ndarray:
        """Weights for military transition (1938)"""
        weights = np.ones(self.action_space.get_action_size()) * 0.5

        weights[self._get_key_idx('t')] = 8.0  # Production
        weights[self._get_key_idx('y')] = 6.0  # Logistics
        weights[self._get_key_idx('w')] = 7.0  # Research
        weights[self._get_key_idx('v')] = 5.0  # Army

        return weights

    def _get_war_weights(self) -> np.ndarray:
        """Weights for war preparation (1939+)"""
        weights = np.ones(self.action_space.get_action_size()) * 0.5

        weights[self._get_key_idx('v')] = 10.0  # Army
        weights[self._get_key_idx('n')] = 7.0  # Navy
        weights[self._get_key_idx('m')] = 7.0  # Air
        weights[self._get_key_idx('t')] = 8.0  # Production

        return weights

    def _get_key_idx(self, key: str) -> Optional[int]:
        """Get action index for a specific key"""
        try:
            return self.action_space.base_keys.index(key)
        except ValueError:
            return None
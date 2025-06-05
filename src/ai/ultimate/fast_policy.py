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

        # Anti-stuck mechanism
        if self.last_action_idx is not None:
            self.action_history.append(self.last_action_idx)

            # Check if stuck (repeated actions)
            if len(self.action_history) >= 10:
                recent = list(self.action_history)[-10:]
                if len(set(recent)) < 3:  # Less than 3 unique actions
                    self.stuck_counter += 1
                    # Heavily penalize recent actions
                    for idx in set(recent):
                        weights[idx] *= 0.1
                else:
                    self.stuck_counter = max(0, self.stuck_counter - 1)

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
        for key in ['f1', 'f2', 'f3', 'f4', 'f5', 'q', 'w', 'e', 'r', 't']:
            idx = self._get_key_idx(key)
            if idx is not None:
                weights[idx] = 3.0

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
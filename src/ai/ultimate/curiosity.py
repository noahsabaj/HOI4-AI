# src/ai/ultimate/curiosity.py
"""
Random Network Distillation (RND) for curiosity-driven exploration
This gives the AI intrinsic motivation to explore new game states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Optional, Tuple


class RNDNetwork(nn.Module):
    """Neural network for RND target and predictor"""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 512,
            output_dim: int = 512,
            num_layers: int = 3
    ):
        super().__init__()

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation on last layer
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDCuriosity(nn.Module):
    """
    Random Network Distillation for HOI4

    How it works:
    1. Fixed random network (target) produces random features
    2. Learned network (predictor) tries to match target
    3. High prediction error = novel state = high curiosity bonus
    """

    def __init__(
            self,
            observation_dim: int = 1024,  # From encoder
            feature_dim: int = 512,
            learning_rate: float = 1e-4,
            update_proportion: float = 0.25,  # How often to update normalizers
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()

        self.device = device
        self.feature_dim = feature_dim
        self.update_proportion = update_proportion

        # Target network (random and fixed)
        self.target_network = RNDNetwork(
            input_dim=observation_dim,
            output_dim=feature_dim
        ).to(device)

        # Freeze target network
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Initialize with random weights
        for module in self.target_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        # Predictor network (trained)
        self.predictor_network = RNDNetwork(
            input_dim=observation_dim,
            output_dim=feature_dim
        ).to(device)

        # Optimizer for predictor
        self.optimizer = torch.optim.Adam(
            self.predictor_network.parameters(),
            lr=learning_rate
        )

        # Running statistics for normalization
        self.reward_normalizer = RunningMeanStd(device=device)
        self.obs_normalizer = RunningMeanStd(shape=(observation_dim,), device=device)

        # Intrinsic reward history
        self.intrinsic_rewards = deque(maxlen=1000)

    def compute_intrinsic_reward(
            self,
            obs: torch.Tensor,
            update_stats: bool = True
    ) -> torch.Tensor:
        """
        Compute curiosity bonus for given observation

        Args:
            obs: Encoded observation from world model
            update_stats: Whether to update running statistics

        Returns:
            Intrinsic reward (higher = more novel)
        """
        with torch.no_grad():
            # Normalize observation
            if update_stats and np.random.random() < self.update_proportion:
                self.obs_normalizer.update(obs)

            normalized_obs = self.obs_normalizer.normalize(obs)

            # Get target features
            target_features = self.target_network(normalized_obs)

            # Get predicted features
            predicted_features = self.predictor_network(normalized_obs)

            # Compute error (MSE)
            intrinsic_reward = F.mse_loss(
                predicted_features,
                target_features,
                reduction='none'
            ).mean(dim=-1)

            # Update reward statistics
            if update_stats:
                self.reward_normalizer.update(intrinsic_reward)

            # Normalize reward
            normalized_reward = self.reward_normalizer.normalize(intrinsic_reward)

            # Clip to reasonable range
            normalized_reward = torch.clamp(normalized_reward, -5, 5)

            # Store for analysis
            self.intrinsic_rewards.extend(normalized_reward.cpu().numpy().tolist())

        return normalized_reward

    def train_predictor(
            self,
            obs_batch: torch.Tensor,
            train_steps: int = 1
    ) -> float:
        """
        Train predictor network to match target network

        Args:
            obs_batch: Batch of observations
            train_steps: Number of gradient steps

        Returns:
            Average loss
        """
        losses = []

        for _ in range(train_steps):
            # Normalize observations
            normalized_obs = self.obs_normalizer.normalize(obs_batch)

            # Forward pass
            target_features = self.target_network(normalized_obs).detach()
            predicted_features = self.predictor_network(normalized_obs)

            # Compute loss
            loss = F.mse_loss(predicted_features, target_features)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.predictor_network.parameters(),
                max_norm=5.0
            )

            self.optimizer.step()

            losses.append(loss.item())

        return np.mean(losses)

    def get_statistics(self) -> dict:
        """Get curiosity statistics for monitoring"""
        if len(self.intrinsic_rewards) == 0:
            return {
                'mean_intrinsic_reward': 0.0,
                'std_intrinsic_reward': 0.0,
                'max_intrinsic_reward': 0.0,
                'min_intrinsic_reward': 0.0
            }

        rewards = np.array(self.intrinsic_rewards)

        return {
            'mean_intrinsic_reward': float(np.mean(rewards)),
            'std_intrinsic_reward': float(np.std(rewards)),
            'max_intrinsic_reward': float(np.max(rewards)),
            'min_intrinsic_reward': float(np.min(rewards))
        }


class RunningMeanStd:
    """
    Running statistics for normalization
    Uses Welford's online algorithm for numerical stability
    """

    def __init__(
            self,
            shape: Tuple[int, ...] = (),
            epsilon: float = 1e-8,
            device: str = 'cpu'
    ):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, x: torch.Tensor) -> None:
        """Update running statistics"""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
            self,
            batch_mean: torch.Tensor,
            batch_var: torch.Tensor,
            batch_count: int
    ) -> None:
        """Update from batch statistics"""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using running statistics"""
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize back to original scale"""
        return x * torch.sqrt(self.var + self.epsilon) + self.mean


class CombinedCuriosity(nn.Module):
    """
    Combines RND with your existing goal-based curiosity
    Best of both worlds!
    """

    def __init__(
            self,
            observation_dim: int = 1024,
            rnd_weight: float = 0.5,
            goal_weight: float = 0.5
    ):
        super().__init__()

        self.rnd = RNDCuriosity(observation_dim=observation_dim)
        self.rnd_weight = rnd_weight
        self.goal_weight = goal_weight

        # Track which menus/screens we've seen
        self.seen_screens = set()
        self.menu_discovery_bonus = 10.0

    def compute_reward(
            self,
            obs: torch.Tensor,
            screen_type: str,
            achieved_goal: bool = False
    ) -> torch.Tensor:
        """
        Combine RND curiosity with goal-based rewards

        Args:
            obs: Encoded observation
            screen_type: Current menu/screen type
            achieved_goal: Whether a specific goal was achieved

        Returns:
            Combined intrinsic reward
        """
        # RND curiosity
        rnd_reward = self.rnd.compute_intrinsic_reward(obs)

        # Goal-based bonus
        goal_reward = torch.zeros_like(rnd_reward)

        # Bonus for discovering new screens
        if screen_type not in self.seen_screens:
            self.seen_screens.add(screen_type)
            goal_reward += self.menu_discovery_bonus
            print(f"🎉 Discovered new screen: {screen_type}!")

        # Bonus for achieving specific goals
        if achieved_goal:
            goal_reward += 5.0

        # Combine
        total_reward = (
                self.rnd_weight * rnd_reward +
                self.goal_weight * goal_reward
        )

        return total_reward
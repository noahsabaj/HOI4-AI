# src/ai/ultimate/world_model.py
"""
DreamerV3-inspired World Model for HOI4
This learns how the game works without being told the rules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, Tuple, Optional


class RSSM(nn.Module):
    """Recurrent State-Space Model - The core of DreamerV3"""

    def __init__(
            self,
            stoch_size: int = 32,
            deter_size: int = 512,
            hidden_size: int = 512,
            num_categories: int = 32,
            action_size: int = 20
    ):
        super().__init__()

        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.num_categories = num_categories

        # Deterministic state (GRU)
        self.gru = nn.GRUCell(
            input_size=stoch_size * num_categories + action_size,
            hidden_size=deter_size
        )

        # Prior: P(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, stoch_size * num_categories)
        )

        # Posterior: Q(z_t | h_t, o_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_size + 1024, hidden_size),  # 1024 from encoder
            nn.ELU(),
            nn.Linear(hidden_size, stoch_size * num_categories)
        )

    def forward(
            self,
            prev_stoch: torch.Tensor,
            prev_deter: torch.Tensor,
            prev_action: torch.Tensor,
            embed: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """One step of the world model"""

        # Flatten stochastic state
        prev_stoch_flat = rearrange(prev_stoch, 'b s c -> b (s c)')

        # Update deterministic state with GRU
        deter = self.gru(
            torch.cat([prev_stoch_flat, prev_action], dim=-1),
            prev_deter
        )

        # Compute prior
        prior_logits = self.prior_net(deter)
        prior = self._logits_to_dist(prior_logits)

        # Compute posterior if observation embedding provided
        if embed is not None:
            posterior_logits = self.posterior_net(torch.cat([deter, embed], dim=-1))
            posterior = self._logits_to_dist(posterior_logits)
            # draw categorical samples and convert to one-hot
            stoch_idx = posterior.sample()  # shape: [batch, stoch_size]
            stoch = F.one_hot(stoch_idx, num_classes=self.num_categories).float()
        else:
            posterior = None
            stoch_idx = prior.sample()  # shape: [batch, stoch_size]
            stoch = F.one_hot(stoch_idx, num_classes=self.num_categories).float()

        return {
            'stoch': stoch,
            'deter': deter,
            'prior': prior,
            'posterior': posterior
        }

    def _logits_to_dist(self, logits: torch.Tensor):
        """Convert logits to categorical distribution"""
        logits = rearrange(logits, 'b (s c) -> b s c', s=self.stoch_size)
        return torch.distributions.Categorical(logits=logits)

    def imagine(self, initial_state: Dict, actions: torch.Tensor) -> Dict:
        """Imagine future trajectories given actions"""
        batch_size, horizon, _ = actions.shape

        states = {'stoch': [], 'deter': []}
        state = initial_state

        for t in range(horizon):
            state = self.forward(
                state['stoch'],
                state['deter'],
                actions[:, t],
                embed=None  # No observations during imagination
            )
            states['stoch'].append(state['stoch'])
            states['deter'].append(state['deter'])

        # Stack along time dimension
        states['stoch'] = torch.stack(states['stoch'], dim=1)
        states['deter'] = torch.stack(states['deter'], dim=1)

        return states


class WorldModel(nn.Module):
    """Complete World Model with encoder, RSSM, decoder, and reward predictor"""

    def __init__(self, action_size: int = 20):
        super().__init__()

        # Image encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 360x640
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 180x320
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 90x160
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 45x80
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 22x40
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),  # Fixed size
            nn.Flatten(),
            nn.Linear(512 * 4 * 8, 1024),
            nn.LayerNorm(1024)
        )

        # RSSM
        self.rssm = RSSM(action_size=action_size)

        # Decoders
        feature_size = self.rssm.stoch_size * self.rssm.num_categories + self.rssm.deter_size

        # Observation decoder (predicts next frame)
        self.decoder = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512 * 4 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (512, 4, 8)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )

        # Reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

        # Value predictor for planning
        self.value_head = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

        # Continue predictor (is episode over?)
        self.continue_head = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict:
        """Forward pass through the world model"""
        batch_size, seq_len = obs.shape[:2]

        # Encode observations
        obs_flat = rearrange(obs, 'b t c h w -> (b t) c h w')
        embeds = self.encoder(obs_flat)
        embeds = rearrange(embeds, '(b t) d -> b t d', b=batch_size)

        # Initial state
        init_stoch = torch.zeros(
            batch_size,
            self.rssm.stoch_size,
            self.rssm.num_categories,
            device=obs.device
        )
        init_deter = torch.zeros(
            batch_size,
            self.rssm.deter_size,
            device=obs.device
        )

        # Run RSSM
        states = {'stoch': [], 'deter': [], 'prior': [], 'posterior': []}
        state = {'stoch': init_stoch, 'deter': init_deter}

        for t in range(seq_len):
            state = self.rssm(
                state['stoch'],
                state['deter'],
                actions[:, t] if t > 0 else torch.zeros(batch_size, actions.shape[-1], device=obs.device),
                embeds[:, t]
            )
            for k, v in state.items():
                if k in states:
                    states[k].append(v)

        # Stack states
        for k in states:
            if states[k]:
                states[k] = torch.stack(states[k], dim=1)

        # Get features for decoding
        stoch_flat = rearrange(states['stoch'], 'b t s c -> b t (s c)')
        features = torch.cat([stoch_flat, states['deter']], dim=-1)
        features_flat = rearrange(features, 'b t d -> (b t) d')

        # Decode
        predictions = {
            'reward': self.reward_head(features_flat),
            'value': self.value_head(features_flat),
            'continue': torch.sigmoid(self.continue_head(features_flat))
        }

        # Reshape predictions
        for k, v in predictions.items():
            predictions[k] = rearrange(v, '(b t) d -> b t d', b=batch_size).squeeze(-1)

        # Decode observations (optional, for visualization)
        predictions['decoded_obs'] = self.decoder(features_flat)
        predictions['decoded_obs'] = rearrange(
            predictions['decoded_obs'],
            '(b t) c h w -> b t c h w',
            b=batch_size
        )

        return states, predictions

    def imagine_ahead(
            self,
            initial_obs: torch.Tensor,
            action_sequence: torch.Tensor,
            horizon: int = 15
    ) -> Dict:
        """Imagine future states and rewards given initial observation and actions"""

        # Encode initial observation
        embed = self.encoder(initial_obs)

        # Initial RSSM state
        init_action = torch.zeros(initial_obs.shape[0],
                                  self.rssm.gru.input_size - self.rssm.stoch_size * self.rssm.num_categories,
                                  device=initial_obs.device)
        init_stoch = torch.zeros(
            initial_obs.shape[0],
            self.rssm.stoch_size,
            self.rssm.num_categories,
            device=initial_obs.device
        )
        init_deter = torch.zeros(
            initial_obs.shape[0],
            self.rssm.deter_size,
            device=initial_obs.device
        )

        # Get initial state from observation
        initial_state = self.rssm(init_stoch, init_deter, init_action, embed)

        # Imagine future
        imagined_states = self.rssm.imagine(initial_state, action_sequence)

        # Predict rewards and values
        stoch_flat = rearrange(imagined_states['stoch'], 'b t s c -> (b t) (s c)')
        deter_flat = rearrange(imagined_states['deter'], 'b t d -> (b t) d')
        features = torch.cat([stoch_flat, deter_flat], dim=-1)

        imagined_rewards = self.reward_head(features)
        imagined_values = self.value_head(features)

        return {
            'states': imagined_states,
            'rewards': rearrange(imagined_rewards, '(b t) 1 -> b t', t=horizon),
            'values': rearrange(imagined_values, '(b t) 1 -> b t', t=horizon)
        }
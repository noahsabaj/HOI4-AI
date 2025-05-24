# src/ai/ultimate/simple_world_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_SIZE = 256
NUM_ACTIONS = 64  # expand if you ever have >64 discrete actions

class SimpleWorldModel(nn.Module):
    """
    Super-light Dreamer-lite:
      s_t   = encoder(obs_t)
      ŝ_{t+1} = dynamics([s_t, onehot(a_t)])
      r̂_t = reward_head(s_t)
    """
    def __init__(self):
        super().__init__()

        # ── visual encoder (RGB 1280×720 → 256-D) ────────────────────
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4),  # Big kernel for first layer
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # Force to 4x4 regardless of input
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, LATENT_SIZE),
        )

        # ── dynamics: (latent‖action) → next-latent ────────────────
        self.dynamics = nn.Sequential(
            nn.Linear(LATENT_SIZE + NUM_ACTIONS, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, LATENT_SIZE)
        )

        # ── reward predictor ───────────────────────────────────────
        self.reward_head = nn.Sequential(
            nn.Linear(LATENT_SIZE, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    # --------------------------------------------------------------
    def encode(self, obs_bchw: torch.Tensor) -> torch.Tensor:
        # SimpleWorldModel encoder outputs LATENT_SIZE directly
        return self.encoder(obs_bchw)

    def forward(self, s_t, a_onehot):
        """predict next latent + reward"""
        dynamics_in = torch.cat([s_t, a_onehot], dim=-1)
        s_tp1_pred  = self.dynamics(dynamics_in)
        r_pred      = self.reward_head(s_t).squeeze(-1)
        return s_tp1_pred, r_pred
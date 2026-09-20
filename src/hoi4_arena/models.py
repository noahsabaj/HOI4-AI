from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from .actions import GRID, SLOTS, VOCAB
from .benchmark import load_encoder, verify_model_source


class VideoEncoder(nn.Module):
    def __init__(self, model_path, variant="large", train_last=2):
        super().__init__()
        if variant == "large":
            self.model = load_encoder(model_path)
        elif variant == "tiny":
            from transformers import AutoConfig, AutoModel

            verify_model_source(model_path)
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
            config.embed_dim, config.depth, config.num_heads = 192, 12, 3
            self.model = AutoModel.from_config(config, trust_remote_code=True)
        else:
            raise ValueError(variant)
        self.dim = self.model.config.embed_dim
        self.variant = variant
        if variant == "large" and train_last >= 0:
            self.model.requires_grad_(False)
            if train_last:
                for block in self.model.encoder.blocks[-train_last:]:
                    block.requires_grad_(True)
                self.model.encoder.norm.requires_grad_(True)

    def forward(self, clip):
        # All input frames are <= current observation time. No token dropping for control.
        tokens = self.model(pixel_values=clip).last_hidden_state
        return tokens[:, 0]


class DetailEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2, 2),
            nn.GELU(),
            nn.Conv2d(24, 48, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(48, 64, 3, 2, 1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )

    def forward(self, tiles):
        b, n, c, h, w = tiles.shape
        return self.net(tiles.reshape(b * n, c, h, w)).reshape(b, n * 256)


class ActionHead(nn.Module):
    """Autoregressive event slots with an exact, replayable conditional likelihood."""

    def __init__(self, memory_dim=512, noise_dim=16):
        super().__init__()
        self.noise_dim = noise_dim
        self.init = nn.Linear(memory_dim + noise_dim, 256)
        self.embedding = nn.Embedding(len(VOCAB), 64)
        self.xy = nn.Linear(2, 64)
        self.cell = nn.GRUCell(64, 256)
        self.type_head = nn.Linear(256, len(VOCAB))
        self.x_head = nn.Linear(256, GRID)
        self.y_head = nn.Linear(256, GRID)

    def forward(self, memory, actions=None, noise=None, deterministic=False):
        b = memory.shape[0]
        if noise is None:
            noise = torch.zeros(b, self.noise_dim, device=memory.device, dtype=memory.dtype)
        state = torch.tanh(self.init(torch.cat([memory, noise], -1)))
        previous = torch.zeros(b, 64, device=memory.device, dtype=memory.dtype)
        result, logps, entropies = [], [], []
        for slot in range(SLOTS):
            state = self.cell(previous, state)
            distributions = [
                Categorical(logits=head(state).float())
                for head in (self.type_head, self.x_head, self.y_head)
            ]
            values = (
                actions[:, slot]
                if actions is not None
                else torch.stack(
                    [d.logits.argmax(-1) if deterministic else d.sample() for d in distributions],
                    -1,
                )
            )
            move = (values[:, 0] == 1).float()
            lp = distributions[0].log_prob(values[:, 0])
            lp = lp + move * sum(distributions[j].log_prob(values[:, j]) for j in (1, 2))
            # Expected entropy of xy conditional on a move, not the sampled move indicator.
            entropy = distributions[0].entropy() + distributions[0].probs[:, 1] * sum(
                distributions[j].entropy() for j in (1, 2)
            )
            values = values.clone()
            values[:, 1:] *= move.long()[:, None]
            previous = self.embedding(values[:, 0]) + self.xy(
                values[:, 1:].to(memory.dtype) / (GRID - 1)
            )
            result.append(values)
            logps.append(lp)
            entropies.append(entropy)
        return (
            torch.stack(result, 1),
            torch.stack(logps, 1).sum(1),
            torch.stack(entropies, 1).sum(1),
        )


class Policy(nn.Module):
    def __init__(self, encoder, memory_dim=512):
        super().__init__()
        self.encoder = encoder
        self.details = DetailEncoder()
        self.previous_action = nn.Linear(SLOTS * 3, 64)
        self.fusion = nn.Linear(encoder.dim + 4 * 256 + 64, memory_dim)
        self.memory = nn.GRUCell(memory_dim, memory_dim)
        self.actor = ActionHead(memory_dim)
        self.value = nn.Linear(memory_dim, 1)
        self.memory_dim = memory_dim

    def forward(self, clip, tiles, previous, hidden=None, reset=None):
        if hidden is None:
            hidden = clip.new_zeros(clip.shape[0], self.memory_dim)
        if reset is not None:
            hidden = hidden * (~reset.bool()).to(hidden.dtype)[:, None]
        scales = previous.new_tensor([len(VOCAB) - 1, GRID - 1, GRID - 1])
        prior = self.previous_action((previous / scales).flatten(1).to(clip.dtype))
        features = self.encoder(clip)
        merged = torch.cat([features, self.details(tiles), prior], -1)
        hidden = self.memory(F.gelu(self.fusion(merged)), hidden)
        return hidden, self.value(hidden).squeeze(-1), features


def reprelu(value):
    soft = F.gelu(value)
    return value.relu().detach() + soft - soft.detach()


def rdmreg(z, sparse: bool, projections=256):
    """LpWM distribution matching: sample axis 0, separate time axis, shared projections.

    Lower projection count is a hardware adaptation, shared by both experimental arms.
    """
    if z.shape[0] < 2:
        raise ValueError("RDMReg needs at least two independent sequences in the batch")
    z = z.float()
    directions = F.normalize(torch.randn(z.shape[-1], projections, device=z.device), dim=0)
    if sparse:
        target = (
            torch.distributions.Laplace(z.new_tensor(0), z.new_tensor(2**-0.5))
            .sample(z.shape)
            .relu()
        )
    else:
        target = torch.randn_like(z)
    actual = (z @ directions).sort(dim=0).values
    reference = (target @ directions).sort(dim=0).values
    return (actual - reference).square().mean()


class PredictiveAuxiliary(nn.Module):
    """Training-only action-conditioned prediction; dense/sparse arms share capacity."""

    def __init__(self, memory_dim=512, feature_dim=1024, latent_dim=384, mode="sparse"):
        super().__init__()
        if mode not in {"none", "dense", "sparse"}:
            raise ValueError(mode)
        self.mode = mode
        self.project = nn.Linear(feature_dim, latent_dim)
        self.context = nn.Linear(memory_dim, latent_dim)
        self.action = nn.Linear(SLOTS * 3, latent_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim)
        )

    def link(self, z):
        return reprelu(z) if self.mode == "sparse" else z

    def forward(self, memories, features, actions, valid):
        if self.mode == "none":
            return memories.sum() * 0
        z = self.link(self.project(features))
        scales = actions.new_tensor([len(VOCAB) - 1, GRID - 1, GRID - 1])
        action_features = self.action((actions / scales).flatten(-2).to(memories.dtype))
        prediction = self.link(
            self.predictor(torch.cat([self.context(memories[:, :-1]), action_features[:, :-1]], -1))
        )
        mask = valid[:, 1:] & valid[:, :-1]
        error = (prediction.float() - z[:, 1:].float()).square().mean(-1)
        prediction_loss = (error * mask).sum() / mask.sum().clamp_min(1)
        # Only regularize temporal positions with a fully valid independent batch.
        regular = z[:, valid.all(0)]
        reg = rdmreg(regular, self.mode == "sparse") if regular.shape[1] else z.sum() * 0
        return prediction_loss + 0.5 * reg


def xm_loss(actor, memories, actions, candidates=5):
    """XM-inspired best-of-K conditional BC, not a reproduction of continuous XM.

    Select candidate noise without a graph, re-evaluate its exact categorical likelihood.
    At RL time retain sampled noise in the rollout; fixed prior cancels in PPO ratios.
    """
    noises = torch.randn(
        candidates, memories.shape[0], actor.noise_dim, device=memories.device, dtype=memories.dtype
    )
    with torch.no_grad():
        losses = torch.stack([-actor(memories, actions, noise=z)[1] for z in noises])
        best = losses.argmin(0)
    selected = noises[best, torch.arange(memories.shape[0], device=memories.device)]
    return -actor(memories, actions, noise=selected)[1]

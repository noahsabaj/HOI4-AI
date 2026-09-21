from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

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


def gumbel_argmax(logits):
    """Draw a category with the same law as Categorical.sample, without the object.

    argmax(logits + Gumbel(0,1)) is exactly a categorical draw, and a standard Gumbel is
    minus the log of a unit exponential. The textbook -log(-log(U)) form is rejected on
    both of its logs: torch.rand returns an exact zero often enough to matter, which
    would make one category unreachable forever, and the inner log loses precision as U
    approaches one. exponential_ is drawn on the open interval, so one kernel and one log
    suffice and no clamp is needed.
    """
    return (logits - torch.empty_like(logits).exponential_().log_()).argmax(-1)


def categorical(logits):
    """Normalized log-probabilities, the way torch.distributions.Categorical does it.

    Subtracting a logsumexp and taking a log_softmax are the same function and not the
    same bits -- they disagree by up to 2e-6, which is enough to move a stored old_logp
    and with it a PPO ratio. So this is the slower two-kernel form deliberately, and
    `log_prob` and `entropy` below are bit-for-bit what Categorical returned.
    """
    return logits - logits.logsumexp(-1, keepdim=True)


def log_prob(normalized, values):
    return normalized.gather(-1, values[:, None]).squeeze(-1)


def entropy(normalized, probabilities):
    floor = torch.finfo(normalized.dtype).min
    return -(normalized.clamp(min=floor) * probabilities).sum(-1)


class ActionHead(nn.Module):
    """Autoregressive event slots with an exact, replayable conditional likelihood.

    The likelihood is computed directly rather than through torch.distributions. That is
    not a stylistic preference: Distribution validates its arguments whenever __debug__
    is set, and each validation ends in `Tensor.__bool__` on a CUDA tensor, which is a
    host-device synchronization. Three heads times eight slots times a construction and a
    log_prob is 48 of them per forward -- including on the teacher-forcing path, which
    never samples anything. Removing the objects removes the stalls; the arithmetic below
    reproduces Categorical's exactly, and a test pins it bit-for-bit.
    """

    def __init__(self, memory_dim=512, noise_dim=16):
        super().__init__()
        self.noise_dim = noise_dim
        self.init = nn.Linear(memory_dim + noise_dim, 256)
        self.embedding = nn.Embedding(len(VOCAB), 64)
        self.xy = nn.Linear(2, 64)
        self.cell = nn.GRUCell(64, 256)
        # One projection, split three ways. At batch one these are launch-bound rather
        # than arithmetic-bound, so three narrow GEMMs cost more than one wide one, and
        # concatenating the weights is the same arithmetic row by row: measured 0.51 ms
        # to 0.23 ms across the eight slots, with bit-identical output.
        self.widths = (len(VOCAB), GRID, GRID)
        self.heads = nn.Linear(256, sum(self.widths))

    def forward(self, memory, actions=None, noise=None, deterministic=False):
        b = memory.shape[0]
        if noise is None:
            noise = torch.zeros(b, self.noise_dim, device=memory.device, dtype=memory.dtype)
        state = torch.tanh(self.init(torch.cat([memory, noise], -1)))
        previous = torch.zeros(b, 64, device=memory.device, dtype=memory.dtype)
        result, logps, entropies = [], [], []
        for slot in range(SLOTS):
            state = self.cell(previous, state)
            # The float cast is load-bearing, not incidental: run in bfloat16 this same
            # normalization lands about 0.05 away, which is a large number to put inside
            # a ratio of likelihoods.
            heads = [categorical(z) for z in self.heads(state).float().split(self.widths, -1)]
            if actions is not None:
                values = actions[:, slot]
            elif deterministic:
                values = torch.stack([z.argmax(-1) for z in heads], -1)
            else:
                values = torch.stack([gumbel_argmax(z) for z in heads], -1)
            probabilities = [z.softmax(-1) for z in heads]
            chosen = [log_prob(z, values[:, j]) for j, z in enumerate(heads)]
            each = [entropy(z, p) for z, p in zip(heads, probabilities, strict=True)]
            move = (values[:, 0] == 1).float()
            lp = chosen[0] + move * sum(chosen[j] for j in (1, 2))
            # Expected entropy of xy conditional on a move, not the sampled move indicator.
            expected = each[0] + probabilities[0][:, 1] * sum(each[j] for j in (1, 2))
            values = values.clone()
            values[:, 1:] *= move.long()[:, None]
            previous = self.embedding(values[:, 0]) + self.xy(
                values[:, 1:].to(memory.dtype) / (GRID - 1)
            )
            result.append(values)
            logps.append(lp)
            entropies.append(expected)
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


def configure_precision(tf32: bool = False):
    """Say out loud what float32 matmuls are allowed to do, instead of inheriting it.

    Torch leaves float32 matmuls at full precision and warns that the tensor cores are
    idle, which reads like a free speedup being declined. Measured, it is not one here:
    every matmul in this project runs inside torch.autocast in bfloat16, and autocast
    lowers matmul, addmm, linear and GRUCell regardless of what dtype reaches them. The
    two candidates that look like exceptions are not -- ActionHead casts the head's
    *output* to float32, after the Linear has already run in bfloat16, and rdmreg's
    explicit .float() sits inside the same autocast block. There is no float32 GEMM left
    for TF32 to accelerate, and a synthetic one at these shapes measures between 1.01x
    and 1.18x, all of it launch overhead at batch one.

    So the default is off, and the reason is not caution about the speedup. TF32 keeps
    ten mantissa bits, and on the shapes this project does use it moved results by up to
    2.1e-2 -- against a PPO log-likelihood, that is a phantom ratio larger than anything
    the optimizer is being asked to find. Zero measured gain is not worth that.

    Use set_float32_matmul_precision and not the newer
    torch.backends.cuda.matmul.fp32_precision: on torch 2.11 assigning that attribute
    leaves get_float32_matmul_precision() raising RuntimeError.
    """
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    return torch.get_float32_matmul_precision()


def halve_frozen(module):
    """Keep every parameter that carries no gradient in bfloat16.

    The live path shipped 1170 MiB of float32 weights to the card and then ran every
    matmul through autocast in bfloat16 regardless, so most of that was precision
    nothing ever read. Halving it recovers 530 MiB against a headroom gate the
    loaded-game run fails: 710 MiB free where 1024 is required.

    Only the frozen parameters, and the restriction is the whole point. Casting the
    trainable ones as well saves a further 55 MiB and moves a stored old_logp by
    1.02e-2, which is a systematic 1.01x PPO ratio on every sample before a single
    gradient step, because collection would then compute the likelihood differently from
    the update that scores it. Frozen weights cannot diverge that way -- load_policy
    serves both sides, so both run the same function -- and no optimizer reads them.
    """
    for parameter in module.parameters():
        if not parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    return module


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

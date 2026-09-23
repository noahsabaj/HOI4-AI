"""Memory cells the policy can carry from one decision to the next, stepped one at a time.

The policy's memory has been a GRU cell, trained on windows of 16 decisions: 3.2 s. The
2026 literature (STATUS.md) says the window matters first -- cutting gradients even at
100 steps costs measurably (Memoroids, Morad et al., 2024) -- and that the newest
sequence layers have been compared only as language models, never as an agent's memory.
So these cells share one interface and are compared on the same cached features
(features.py), at the same budget:

- `GRUMemory`, the policy's cell. It subclasses nn.GRUCell, so its weights load into
  Policy.memory unchanged.
- `GatedDeltaNet2Memory`, Gated DeltaNet-2 (Hatamizadeh et al., 2026, arXiv 2605.22791):
  a matrix memory whose keys are written and erased by separate channel-wise gates,
  which suits facts that get overwritten (what is queued, where an army was last seen).
  Written from the MIT-licensed reference in flash-linear-attention
  (fla/ops/gdn2/naive.py, fla/layers/gdn2.py), not NVlabs' non-commercial repository.
- `Mamba3Memory`, Mamba-3 SISO (Lahoti et al., 2026, arXiv 2603.15569): exponential
  trapezoidal discretisation and a complex state, applied as data-dependent rotations.
  Written from the Apache-2.0 reference step in state-spaces/mamba
  (tests/ops/triton/test_mamba3_siso.py, mamba_ssm/modules/mamba3.py).
- `NoMemory`, an MLP: what the policy does without any history.

Each cell maps (B, dim) to (B, dim) and carries a tuple of tensors. The GRU's output is
its state; the others are pre-norm residual blocks, as they sit in a language model, and
their state is the recurrence's, kept in float32. They run one decision at a time, in
plain PyTorch, because the policy's input at a decision depends on its memory at the one
before (Policy.observe reads the screen where the memory points): a parallel scan could
not be used even where one exists. At one layer of 512 this is a few kernels a decision,
nothing next to the vision tower.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def reset(state, done):
    """Zero the state of every sample whose `done` is set: a new game starts there."""
    keep = (~done.bool()).float()
    return tuple(s * keep.view(-1, *[1] * (s.dim() - 1)).to(s.dtype) for s in state)


def detach(state):
    return tuple(s.detach() for s in state)


def state_size(state):
    """Floats one sample carries."""
    return sum(s[0].numel() for s in state)


class GRUMemory(nn.GRUCell):
    """The policy's GRU cell. State: (h,); the output is h."""

    def __init__(self, dim=512):
        super().__init__(dim, dim)
        self.dim = dim

    def initial(self, batch, device):
        return (torch.zeros(batch, self.dim, device=device),)

    def forward(self, x, state):
        h = super().forward(x, state[0].to(x.dtype))
        return h, (h,)


class NoMemory(nn.Module):
    """No history: a residual MLP of about the GRU's size."""

    def __init__(self, dim=512, hidden=1536):
        super().__init__()
        self.dim = dim
        self.norm = nn.RMSNorm(dim, eps=1e-5)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def initial(self, batch, device):
        return ()

    def forward(self, x, state):
        return x + self.mlp(self.norm(x)), state


def _short_conv(x, buffer, weight):
    """One step of a causal depthwise convolution with SiLU, as fla's ShortConvolution.

    `buffer` holds the previous kernel-1 inputs, oldest first: (B, C, kernel-1).
    """
    window = torch.cat([buffer, x.float()[..., None]], -1)
    y = F.silu((window * weight.float()).sum(-1))
    return y.to(x.dtype), window[..., 1:]


class GatedDeltaNet2Memory(nn.Module):
    """One Gated DeltaNet-2 layer, pre-norm and residual, stepped one token at a time.

    The recurrence on the per-head state S (K x V), as fla writes it:

        S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T
        o_t = S_t^T (q_t / sqrt(K))

    b is the erase gate on the key axis, w the write gate on the value axis, g a
    channel-wise log-decay. q and k pass a short causal convolution and are
    L2-normalised; the output is RMS-normalised per head and gated. With 8 heads of 64
    this has 1.7M parameters, the GRU cell 1.6M, and it carries 32K floats to the GRU's
    512: a matrix memory is large by design.
    """

    def __init__(self, dim=512, heads=8, head_dim=64, conv=4, eps=1e-5):
        super().__init__()
        self.dim, self.heads, self.head_dim, self.conv = dim, heads, head_dim, conv
        width = heads * head_dim
        self.norm = nn.RMSNorm(dim, eps=eps)
        self.q_proj = nn.Linear(dim, width, bias=False)
        self.k_proj = nn.Linear(dim, width, bias=False)
        self.v_proj = nn.Linear(dim, width, bias=False)
        # Depthwise kernels, (channels, kernel), initialised as nn.Conv1d would be.
        bound = 1 / math.sqrt(conv)
        self.q_conv = nn.Parameter(torch.empty(width, conv).uniform_(-bound, bound))
        self.k_conv = nn.Parameter(torch.empty(width, conv).uniform_(-bound, bound))
        self.v_conv = nn.Parameter(torch.empty(width, conv).uniform_(-bound, bound))
        self.f_proj = nn.Sequential(
            nn.Linear(dim, head_dim, bias=False), nn.Linear(head_dim, width, bias=False)
        )
        self.b_proj = nn.Linear(dim, width, bias=False)
        self.w_proj = nn.Linear(dim, width, bias=False)
        self.A_log = nn.Parameter(torch.empty(heads).uniform_(1, 16).log())
        dt = (
            (torch.rand(width) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            .exp()
            .clamp(min=1e-4)
        )
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.g_proj = nn.Sequential(
            nn.Linear(dim, head_dim, bias=False), nn.Linear(head_dim, width, bias=True)
        )
        self.o_weight = nn.Parameter(torch.ones(head_dim))
        self.o_proj = nn.Linear(width, dim, bias=False)
        self.eps = eps

    def initial(self, batch, device):
        width = self.heads * self.head_dim
        return (
            torch.zeros(batch, self.heads, self.head_dim, self.head_dim, device=device),
            torch.zeros(batch, width, self.conv - 1, device=device),
            torch.zeros(batch, width, self.conv - 1, device=device),
            torch.zeros(batch, width, self.conv - 1, device=device),
        )

    def forward(self, x, state):
        S, cq, ck, cv = state
        b, heads, d = x.shape[0], self.heads, self.head_dim
        h = self.norm(x)
        q, cq = _short_conv(self.q_proj(h), cq, self.q_conv)
        k, ck = _short_conv(self.k_proj(h), ck, self.k_conv)
        v, cv = _short_conv(self.v_proj(h), cv, self.v_conv)
        q = F.normalize(q.float().view(b, heads, d), dim=-1) * d**-0.5
        k = F.normalize(k.float().view(b, heads, d), dim=-1)
        v = v.float().view(b, heads, d)
        g = -self.A_log.float().exp()[:, None] * F.softplus(
            self.f_proj(h).float() + self.dt_bias
        ).view(b, heads, d)
        erase_gate = self.b_proj(h).float().sigmoid().view(b, heads, d)
        write_gate = self.w_proj(h).float().sigmoid().view(b, heads, d)
        S = S * g.exp()[..., None]
        erased = ((erase_gate * k)[..., None] * S).sum(-2)
        S = S + k[..., None] * (write_gate * v - erased)[..., None, :]
        o = (q[..., None] * S).sum(-2)
        # Gated RMSNorm per head: norm(o) * weight * sigmoid(gate), fla's FusedRMSNormGated.
        o = o * torch.rsqrt(o.square().mean(-1, keepdim=True) + self.eps) * self.o_weight
        o = o * self.g_proj(h).float().view(b, heads, d).sigmoid()
        return x + self.o_proj(o.reshape(b, -1).to(x.dtype)), (S, cq, ck, cv)


def _heavy_tail(x):
    """Mamba-3's positive activation for the data-dependent A: 1 + x, or 1 / (1 - x)."""
    return x.clamp_min(0) + torch.reciprocal(1 - x.clamp_max(0))


def _rotate(x, cos, sin):
    """Rotate consecutive pairs of `x`'s last axis; pairs past len(cos) stay as they are."""
    pairs = x.view(*x.shape[:-1], -1, 2)
    n = cos.shape[-1]
    a, b = pairs[..., :n, 0], pairs[..., :n, 1]
    turned = torch.stack([a * cos - b * sin, a * sin + b * cos], -1)
    return torch.cat([turned, pairs[..., n:, :]], -2).view_as(x)


class Mamba3Memory(nn.Module):
    """One Mamba-3 (SISO) layer, pre-norm and residual, stepped one token at a time.

    Per head, with state S (P x N), input v (P), key k and query q (N):

        S_t = alpha_t S_{t-1} + beta_t v_{t-1} k_{t-1}^T + gamma_t v_t k_t^T
        y_t = S_t q_t + D v_t, gated by silu(z_t)

    alpha = exp(A dt), beta = (1 - lambda) dt alpha and gamma = lambda dt: the
    exponential-trapezoidal rule, with lambda learned per step. k and q are rotated by
    angles that accumulate with dt, which is the complex-valued state written as rotary
    embeddings. With d_state 64, expand 2 and heads of 64 it has 1.7M parameters and
    carries 66K floats.
    """

    def __init__(
        self,
        dim=512,
        d_state=64,
        expand=2,
        headdim=64,
        rope_fraction=0.5,
        dt_min=0.001,
        dt_max=0.1,
        dt_floor=1e-4,
        a_floor=1e-4,
    ):
        super().__init__()
        self.dim, self.d_state, self.headdim = dim, d_state, headdim
        self.inner = expand * dim
        self.heads = self.inner // headdim
        rotated = int(d_state * rope_fraction)
        self.angles = (rotated - rotated % 2) // 2
        self.a_floor = a_floor
        self.norm = nn.RMSNorm(dim, eps=1e-5)
        self.in_proj = nn.Linear(
            dim, 2 * self.inner + 2 * d_state + 3 * self.heads + self.angles, bias=False
        )
        dt = (
            (torch.rand(self.heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
            .exp()
            .clamp(min=dt_floor)
        )
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.B_bias = nn.Parameter(torch.ones(self.heads, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.heads, d_state))
        self.B_norm = nn.RMSNorm(d_state, eps=1e-5)
        self.C_norm = nn.RMSNorm(d_state, eps=1e-5)
        self.D = nn.Parameter(torch.ones(self.heads))
        self.out_proj = nn.Linear(self.inner, dim, bias=False)

    def initial(self, batch, device):
        return (
            torch.zeros(batch, self.heads, self.angles, device=device),
            torch.zeros(batch, self.heads, self.headdim, self.d_state, device=device),
            torch.zeros(batch, self.heads, self.d_state, device=device),
            torch.zeros(batch, self.heads, self.headdim, device=device),
        )

    def forward(self, x, state):
        angle, S, k_prev, v_prev = state
        b = x.shape[0]
        parts = (
            self.in_proj(self.norm(x))
            .float()
            .split(
                [self.inner, self.inner, self.d_state, self.d_state]
                + [self.heads] * 3
                + [self.angles],
                -1,
            )
        )
        z, v, key, query, dd_dt, dd_a, trap, angles = parts
        z = z.view(b, self.heads, self.headdim)
        v = v.view(b, self.heads, self.headdim)
        a = (-_heavy_tail(dd_a)).clamp(max=-self.a_floor)
        dt = F.softplus(dd_dt + self.dt_bias)
        k = self.B_norm(key)[:, None] + self.B_bias
        q = self.C_norm(query)[:, None] + self.C_bias
        angle = angle + (torch.tanh(angles) * math.pi)[:, None] * dt[..., None]
        angle = angle - 2 * math.pi * torch.floor(angle / (2 * math.pi))
        cos, sin = angle.cos(), angle.sin()
        k, q = _rotate(k, cos, sin), _rotate(q, cos, sin)
        lam = trap.sigmoid()
        alpha = (a * dt).exp()
        beta = (1 - lam) * dt * alpha
        gamma = lam * dt
        S = (
            alpha[..., None, None] * S
            + beta[..., None, None] * (v_prev[..., None] * k_prev[..., None, :])
            + gamma[..., None, None] * (v[..., None] * k[..., None, :])
        )
        y = torch.einsum("bhpn,bhn->bhp", S, q) + self.D[:, None] * v
        y = y * F.silu(z)
        return x + self.out_proj(y.reshape(b, -1).to(x.dtype)), (angle, S, k, v)


KINDS = {
    "gru": GRUMemory,
    "gdn2": GatedDeltaNet2Memory,
    "mamba3": Mamba3Memory,
    "none": NoMemory,
}


def build_memory(kind, dim=512):
    return KINDS[kind](dim)

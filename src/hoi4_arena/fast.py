"""Faster live decisions: the vision tower's per-call waste removed (exactly), and opt-in
numeric shortcuts ("fast mode") that change the numbers a little for a lot less time.

Everything here is for inference on a policy loaded by runner.load_policy, never for
training: the patches assume frozen weights and fixed input sizes.
"""

from __future__ import annotations

import contextlib
import types

import torch
import torch.nn.functional as F
from torch import nn


def _lean_attention(self, x, rope=None, attn_mask=None, is_causal=False):
    """timm's AttentionRope.forward for a fused qkv, no prefix tokens, no gate and no grouped
    heads (the Qwen3.5 towers), without its two largest copies: timm applies RoPE to q and
    k and then concatenates each with an empty prefix, which copies the whole tensor. The
    same kernels otherwise, so the same numbers."""
    from timm.layers import apply_rot_embed_cat

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    if rope is not None:
        half = getattr(self, "rotate_half", False)
        q = apply_rot_embed_cat(q, rope, half=half).type_as(v)
        k = apply_rot_embed_cat(k, rope, half=half).type_as(v)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    x = self.norm(x.transpose(1, 2).reshape(B, N, self.attn_dim))
    return self.proj_drop(self.proj(x))


def _cached_pos_embed(self, x):
    """Qwen3VitEncoder._pos_embed with the resampled position table and the RoPE table made
    once per grid size: both depend only on it, and remaking them was a dozen kernels a call."""
    B, H, W, C = x.shape
    key = (H, W, x.dtype, x.device)
    cache = self.__dict__.setdefault("_tables", {})
    if key not in cache:
        from timm.models.qwen3_vit import resample_pos_embed_grid

        pos = resample_pos_embed_grid(self.pos_embed, self.pos_embed_grid_size, (H, W))
        cache[key] = (pos.to(x.dtype), self.rope.get_embed(shape=(H, W)))
    pos, rope = cache[key]
    return x.reshape(B, H * W, C) + pos, rope


def lean_tower(encoder):
    """Remove the Qwen3.5 tower's per-call waste, keeping every number the same: the copies
    in its attention (_lean_attention), the position tables it rebuilt each call
    (_cached_pos_embed), and the float32 weights of its last blocks, which autocast cast to
    bfloat16 at every call (cast once here, with the same rounding). Returns whether it
    applied: only to a tower of that exact shape."""
    model = getattr(encoder, "model", None)
    blocks = getattr(model, "blocks", None)
    if blocks is None or not hasattr(model, "_pos_embed") or not hasattr(model, "rope"):
        return False
    for block in blocks:
        attn = getattr(block, "attn", None)
        if (
            type(attn).__name__ != "AttentionRope"
            or attn.qkv is None
            or attn.gate is not None
            or attn.num_prefix_tokens
            or attn.num_kv_groups != 1
            or not attn.fused_attn
        ):
            return False
    for block in blocks:
        attn = block.attn
        attn.forward = types.MethodType(_lean_attention, attn)
        for module in block.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.to(torch.bfloat16)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.bfloat16)
    model._pos_embed = types.MethodType(_cached_pos_embed, model)
    return True


# --- fast mode ------------------------------------------------------------------------

# What `--fast` turns on (runner.Actor): the tower in float16 with float16 accumulation,
# compiled by inductor. Measured on the second PC beside its game (bench/decide-ledger.tsv):
# today's tower 26 ms of GPU a decision from 38, the Qwen3.5-4B model's paced p50 72 ms
# from 135, every sampled action on the benchmark's decisions the same, each head's
# log-probability within 0.13. Float8 moved actions (4% of slots) and was no faster; cuDNN
# attention was no faster.
FAST = {"half": True, "compile": True}


class Float8Linear(nn.Module):
    """A frozen Linear in float8 (e4m3): the weight quantized once with one scale, the input
    at each call with its own (its largest magnitude), multiplied on the card's float8
    tensor cores (sm_89 and later) into bfloat16."""

    def __init__(self, linear):
        super().__init__()
        weight = linear.weight.detach().float()
        scale = weight.abs().amax().clamp_min(1e-12) / 448
        self.register_buffer("weight", (weight / scale).to(torch.float8_e4m3fn).t(), False)
        self.register_buffer("scale", scale.reshape(()), False)
        bias = linear.bias
        self.register_buffer(
            "bias", None if bias is None else bias.detach().to(torch.bfloat16), False
        )
        self.out_features = linear.out_features

    def forward(self, x):
        lead = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        amax = x.abs().amax().float().clamp_min(1e-12)
        scale = amax / 448
        q = (x.float() / scale).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            q, self.weight, scale_a=scale, scale_b=self.scale, bias=self.bias,
            out_dtype=torch.bfloat16,
        )  # fmt: skip
        return out.reshape(*lead, self.out_features)


def float8_tower(encoder):
    """Every Linear of the tower's blocks in float8 (Float8Linear). Needs lean_tower first."""
    for block in encoder.model.blocks:
        attn = block.attn
        attn.qkv = Float8Linear(attn.qkv)
        attn.proj = Float8Linear(attn.proj)
        block.mlp.fc1 = Float8Linear(block.mlp.fc1)
        block.mlp.fc2 = Float8Linear(block.mlp.fc2)


class FastTower(nn.Module):
    """A tower run in its own precision and attention kernel. `half` runs it in float16
    with float16 accumulation in the matmuls (twice bfloat16's rate on this card's tensor
    cores), `attention` names the SDPA backend to prefer ("cudnn")."""

    reads_clip = False

    def __init__(self, inner, half=False, attention=None):
        super().__init__()
        self.inner = inner
        self.dim, self.variant = inner.dim, getattr(inner, "variant", "screen")
        self.half = half
        self.attention = attention
        if half:
            # The parameters only: the buffers (normalization statistics, RoPE bands) stay
            # in float32.
            for parameter in inner.parameters():
                parameter.data = parameter.data.half()
            torch.backends.cuda.matmul.allow_fp16_accumulation = True

    def forward(self, clip, quadrants):
        with contextlib.ExitStack() as stack:
            if self.half:
                stack.enter_context(torch.autocast("cuda", dtype=torch.float16))
            if self.attention:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                chosen = {"cudnn": SDPBackend.CUDNN_ATTENTION}[self.attention]
                stack.enter_context(
                    sdpa_kernel([chosen, SDPBackend.EFFICIENT_ATTENTION], set_priority=True)
                )
            summary, grid = self.inner(clip, quadrants)
        return summary, grid


def apply_fast(policy, options):
    """Fast mode on a loaded policy: `options` is a dict of `half`, `attention` and
    `float8` (see FastTower and float8_tower). Returns what applied."""
    applied = {}
    encoder = policy.encoder
    if options.get("float8"):
        lean_tower(encoder)
        float8_tower(encoder)
        applied["float8"] = True
    if options.get("half") or options.get("attention"):
        policy.encoder = FastTower(
            encoder, half=bool(options.get("half")), attention=options.get("attention")
        )
        applied.update(half=bool(options.get("half")), attention=options.get("attention"))
    return applied

"""Attention for `cache-tower --fast`: a Triton flash-attention forward whose two matmuls
accumulate in float16, and the switch that routes the vision tower's float16 attention
to it.

The Qwen3.5-4B model's tower attends over 2880 patches in 24 layers. In float16 with the
matmuls accumulating in float16 and the rest compiled, attention was 40% of the tower's
time (2026-09-26): PyTorch's memory-efficient kernel (flash attention is not built into
torch on Windows) ran at 32 TFLOP/s on the 4060 Ti. This kernel runs the same shapes at
64 TFLOP/s, twice as fast, with float16 accumulation, which the GeForce cards run at twice
the rate of float32. Its error against float32 attention is ~0.3% RMS, below the bfloat16
rounding the plain build carries (tested in test_fast_attention).

Only the forward pass, without a mask or dropout, is provided: what a frozen tower needs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs: the fast mode needs a GPU anyway.
    triton = None


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=w, num_stages=s)
            for m, n, w, s in [
                (128, 64, 4, 3),
                (128, 64, 8, 3),
                (128, 128, 8, 3),
                (64, 64, 4, 3),
                (128, 32, 4, 4),
                (64, 128, 4, 3),
                (128, 64, 4, 4),
                (256, 64, 8, 3),
            ]
        ],  # fmt: skip
        key=["N", "D"],
    )
    @triton.jit
    def _forward(
        Q, K, V, Out, N, H,
        sqz, sqh, sqn, skz, skh, skn, svz, svh, svn, soz, soh, son,
        D: tl.constexpr, QK_SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):  # fmt: skip
        start = tl.program_id(0)
        zh = tl.program_id(1)
        z, h = zh // H, zh % H
        rows = start * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        dims = tl.arange(0, D)
        q = tl.load(
            Q + z * sqz + h * sqh + rows[:, None] * sqn + dims[None, :],
            mask=rows[:, None] < N, other=0.0,
        )  # fmt: skip
        keys, values = K + z * skz + h * skh, V + z * svz + h * svh
        top = tl.full([BLOCK_M], float("-inf"), tl.float32)
        total = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)
        for first in range(0, N, BLOCK_N):
            at = first + cols
            k = tl.load(keys + at[None, :] * skn + dims[:, None], mask=at[None, :] < N, other=0.0)
            qk = tl.dot(q, k, out_dtype=tl.float16).to(tl.float32)
            qk = tl.where(at[None, :] < N, qk * QK_SCALE, float("-inf"))
            new_top = tl.maximum(top, tl.max(qk, 1))
            p = tl.math.exp2(qk - new_top[:, None])
            alpha = tl.math.exp2(top - new_top)
            total = total * alpha + tl.sum(p, 1)
            v = tl.load(values + at[:, None] * svn + dims[None, :], mask=at[:, None] < N, other=0.0)
            pv = tl.dot(p.to(tl.float16), v, out_dtype=tl.float16).to(tl.float32)
            acc = acc * alpha[:, None] + pv
            top = new_top
        acc = acc / total[:, None]
        tl.store(
            Out + z * soz + h * soh + rows[:, None] * son + dims[None, :],
            acc.to(tl.float16), mask=rows[:, None] < N,
        )  # fmt: skip

    @torch.library.triton_op("hoi4_arena::fast_attention", mutates_args=())
    def fast_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """softmax(q k^T / sqrt(d)) v of float16 (B, H, N, D) CUDA tensors."""
        batch, heads, n, d = q.shape
        out = torch.empty((batch, heads, n, d), dtype=q.dtype, device=q.device)

        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_M"]), batch * heads)

        torch.library.wrap_triton(_forward)[grid](
            q, k, v, out, n, heads,
            q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2), out.stride(0), out.stride(1), out.stride(2),
            # Logits in log2 units for exp2, the usual flash-attention trick; a constant, as
            # a float argument would reach the kernel as float64 under torch.compile.
            D=d, QK_SCALE=d**-0.5 * 1.4426950408889634,
        )  # fmt: skip
        return out


def handles(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **rest):
    """Whether fast_attention computes this scaled_dot_product_attention call."""
    return (
        triton is not None
        and q.is_cuda
        and q.dtype == k.dtype == v.dtype == torch.float16
        and attn_mask is None
        and not dropout_p
        and not is_causal
        and scale is None
        and not rest
        and q.shape[-1] in (32, 64, 128)
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    )


def scaled_dot_product_attention(q, k, v, *args, **kwargs):
    """F.scaled_dot_product_attention, by fast_attention for the calls it handles."""
    if not args and handles(q, k, v, **kwargs):
        return fast_attention(q, k, v)
    return F.scaled_dot_product_attention(q, k, v, *args, **kwargs)


def install():
    """Route the timm tower's attention through fast_attention when it runs in float16 on
    the GPU. Every other call, the bfloat16 build's included, goes to PyTorch's own kernel
    unchanged. The Qwen towers' blocks attend with timm.layers.attention.AttentionRope
    (EVA's own attention is routed too); each of those modules gets a copy of
    torch.nn.functional with that one function replaced, and torch itself is left as it
    is."""
    import importlib
    import types

    for name in TIMM_ATTENTION:
        module = importlib.import_module(name)
        if getattr(module.F, "__name__", "") != _ROUTED:
            routed = types.ModuleType(_ROUTED)
            routed.__dict__.update({k: v for k, v in vars(F).items() if not k.startswith("__")})
            routed.scaled_dot_product_attention = scaled_dot_product_attention
            module.F = routed


# The timm modules whose attention calls F.scaled_dot_product_attention.
TIMM_ATTENTION = ("timm.layers.attention", "timm.models.eva")
_ROUTED = "hoi4_arena.fast_attention.functional"

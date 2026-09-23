"""LeVJEPA one frame at a time: cached keys and values, checked against the whole-clip pass.

LeVJEPA attends block-causally (Kuhn et al., 2026): a frame's tokens see only it and
earlier frames, so earlier frames never need encoding again. Each step here embeds the
new frame, lets its tokens attend to the cached keys and values of the frames before it
in every layer, and reads the summary token over the whole window, which, as in the
model, no patch attends to. Keys are cached before their rotary positions are applied, so
positions stay 0..window-1 however long the stream runs; the released code rotates in
the activation dtype, which in bfloat16 is exact only for small positions.

    python scripts/stream_levjepa.py --check     # streamed == whole clip, 8 frames
    python scripts/stream_levjepa.py --time      # per-step cost; run on an idle GPU
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def vit_of(model):
    for module in model.modules():
        if type(module).__name__ == "VisionTransformer":
            return module
    raise RuntimeError("no VisionTransformer")


class StreamingLeVJEPA:
    """Feed frames one by one; each step returns (summary, last-frame patch grid)."""

    def __init__(self, vit, window=8):
        self.vit, self.window = vit, window
        self.cache = [deque(maxlen=window - 1) for _ in vit.blocks]

    def reset(self):
        for layer in self.cache:
            layer.clear()

    @torch.inference_mode()
    def step(self, frame):
        """`frame`: (B, 3, H, W), ImageNet-normalized."""
        vit = self.vit
        B, _, H, W = frame.shape
        h, w = H // vit.patch_size, W // vit.patch_size
        n = h * w
        x = vit.patch_embed(frame[:, :, None])  # (B, n, C)
        cls = vit.cls_token.expand(B, -1, -1)
        for block, cache in zip(vit.blocks, self.cache, strict=True):
            attn = block.attn
            heads, head_dim = attn.num_heads, attn.head_dim
            q, k, v = (
                attn.qkv(block.norm1(x)).reshape(B, n, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
            )
            keys = [c[0] for c in cache] + [k]
            values = [c[1] for c in cache] + [v]
            frames = len(keys)
            ids = torch.arange(frames * n, device=x.device)
            d, hh, ww = attn.separate_positions(ids, h, w)
            # Rotate the whole window from position 0 each step.
            k_all = torch.cat(keys, dim=2)
            _, k_all = attn._apply_rope(k_all, k_all, (d, hh, ww))
            q_new, _ = attn._apply_rope(q, q, (d[-n:], hh[-n:], ww[-n:]))
            v_all = torch.cat(values, dim=2)
            out = F.scaled_dot_product_attention(q_new, k_all, v_all)
            x = x + attn.proj(out.transpose(1, 2).reshape(B, n, -1))
            x = x + block.mlp(block.norm2(x))
            # The summary token reads every patch of the window and itself.
            cq, ck, cv = (
                attn.qkv(block.norm1(cls)).reshape(B, 1, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
            )
            c_out = F.scaled_dot_product_attention(
                cq, torch.cat([ck, k_all], dim=2), torch.cat([cv, v_all], dim=2)
            )
            cls = cls + attn.proj(c_out.transpose(1, 2).reshape(B, 1, -1))
            cls = cls + block.mlp(block.norm2(cls))
            cache.append((k, v))
        x, cls = vit.norm(x), vit.norm(cls)
        return cls[:, 0], x.transpose(1, 2).reshape(B, -1, h, w)


def load():
    from hoi4_arena.benchmark import load_encoder

    return vit_of(load_encoder("models/levjepa-large").cuda().eval())


def check(size=(256, 448), frames=8):
    vit = load()
    torch.manual_seed(0)
    clip = torch.randn(1, 3, frames, *size, device="cuda")
    with torch.inference_mode():
        whole = vit(clip)
    h, w = size[0] // 16, size[1] // 16
    stream = StreamingLeVJEPA(vit, window=frames)
    for t in range(frames):
        summary, grid = stream.step(clip[:, :, t])
    last = whole[0, -h * w :].T.reshape(-1, h, w)
    print(
        "float32, max |streamed - whole| over the last frame:",
        f"{(grid[0] - last).abs().max().item():.2e};",
        f"summary: {(summary[0] - whole[0, 0]).abs().max().item():.2e};",
        f"scale {last.abs().mean().item():.2f}",
    )


def timing(sizes=((256, 448), (352, 640), (496, 896)), window=8, steps=40):
    vit = load().to(torch.bfloat16)
    for size in sizes:
        stream = StreamingLeVJEPA(vit, window=window)
        frame = torch.randn(1, 3, *size, device="cuda", dtype=torch.bfloat16)
        clip = torch.randn(1, 3, window, *size, device="cuda", dtype=torch.bfloat16)
        times, whole = [], []
        for i in range(steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            stream.step(frame)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            if i % 4 == 0:
                start = time.perf_counter()
                with torch.inference_mode():
                    vit(clip)
                torch.cuda.synchronize()
                whole.append(time.perf_counter() - start)
        steady = sorted(times[window:])
        print(
            f"{size[1]}x{size[0]}: streamed step {steady[len(steady) // 2] * 1000:.1f} ms p50,"
            f" whole {window}-frame clip {sorted(whole[2:])[len(whole[2:]) // 2] * 1000:.1f} ms,"
            f" peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB",
            flush=True,
        )


def timing_qwen(sizes=((896, 896), (640, 1152)), steps=40):
    """The Qwen3.5 screen tower on one frame, the same way, for comparison."""
    import timm

    model = timm.create_model(
        "qwen3_vit_88m_enc",
        pretrained=True,
        pretrained_cfg_overlay={"file": "models/qwen3-vit-88m/model.safetensors"},
    )
    model = model.cuda().eval().to(torch.bfloat16)
    for size in sizes:
        frame = torch.randn(1, 3, *size, device="cuda", dtype=torch.bfloat16)
        times = []
        for _ in range(steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                model.forward_features(frame)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        steady = sorted(times[5:])
        print(f"Qwen {size[1]}x{size[0]}: {steady[len(steady) // 2] * 1000:.1f} ms p50", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
    if args.time:
        timing()
        timing_qwen()

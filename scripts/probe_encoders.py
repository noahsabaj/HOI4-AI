"""Which frozen image encoder reads a HOI4 screen best? Linear probes on real recorded frames.

Trains on the frames of one recording and tests on another. Each frame is resized to the
encoder's input, as the policy would give it, and two things are asked of each patch:

- text: 8 random characters (A-Z, 0-9) are drawn per frame at 10-12 px (1080p scale) on
  the game's own pixels; a 36-way linear read-out names each from the feature of the
  patch it sits in. Chance is 2.8%.
- pointer: a linear read-out scores every patch, and a hit is the top patch within one of
  the patch under the pointer's hotspot. Needs recordings made since the worker draws the
  pointer (manifest `pointer_drawn`).

    python scripts/probe_encoders.py artifacts/ai-games-1080p/<game-a> artifacts/ai-games-1080p/<game-b> \
        qwen3_vit_88m_enc.qwen3_5_0_8b vit_base_patch16_lingbot.robbyant@672 qwen3_vit_88m_enc.qwen3_5_0_8b@1152x640

Models are timm names with pretrained weights, optionally @SIZE or @WIDTHxHEIGHT (default
896 square). Measured on 2026-09-23 (STATUS.md, "Choosing the screen encoder").
"""

import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
FONTS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
]


def frames(recording, every=40, limit=160):
    """Every `every`-th recorded frame at 1080p, with the pointer position."""
    rows = [json.loads(line) for line in (Path(recording) / "frames.jsonl").open()]
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(Path(recording) / "screen.mkv"), "-vf",
         f"select=not(mod(n\\,{every}))", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "pipe:1"],
        stdout=subprocess.PIPE,
    )  # fmt: skip
    out = []
    for k in range(limit):
        buf = proc.stdout.read(1920 * 1080 * 3)
        if len(buf) < 1920 * 1080 * 3:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3).copy()
        out.append((frame, rows[k * every]["cursor"]))
    proc.kill()
    return out


def draw_text(frame, rng, positions):
    """Draw one character centred at each 1080p position; returns their labels."""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    labels = []
    for x, y in positions:
        char = rng.choice(CHARS)
        font = ImageFont.truetype(rng.choice(FONTS), rng.choice([10, 11, 12]))
        colour = rng.choice([(255, 255, 255), (255, 220, 120), (230, 230, 230), (20, 20, 20)])
        draw.text((x, y), char, font=font, fill=colour, anchor="mm")
        labels.append(CHARS.index(char))
    return np.asarray(image), labels


def grid(features):
    """(C, h, w) per-patch features from a timm forward_features output."""
    if features.ndim == 4:
        if features.shape[1] > features.shape[-1] and features.shape[1] > 64:
            return features[0]  # NCHW
        return features[0].permute(2, 0, 1)  # NHWC
    tokens = features.shape[1]
    side = int(tokens**0.5)
    # Class and register tokens come first.
    return features[0, tokens - side * side :].T.reshape(-1, side, side)


def fit(features, targets, classes):
    """A linear read-out on standardised features."""
    mean, scale = features.mean(0), features.std(0) + 1e-6
    head = torch.nn.Linear(features.shape[1], classes)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(600):
        optimizer.zero_grad()
        logits = head((features - mean) / scale)
        if classes == 1:
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets)
        else:
            loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
    return lambda x: head((x - mean) / scale)


def probe(name, size, train, test):
    extra = {"img_size": size} if "tipsv2" in name else {}
    model = timm.create_model(name, pretrained=True, **extra).cuda().eval()
    mean = torch.tensor(model.pretrained_cfg.get("mean", (0.5,) * 3), device="cuda")[:, None, None]
    std = torch.tensor(model.pretrained_cfg.get("std", (0.5,) * 3), device="cuda")[:, None, None]
    rng = random.Random(0)

    def collect(recording):
        text_x, text_y, pointer, seconds = [], [], [], []
        for frame, cursor in frames(recording):
            positions = [(rng.randrange(60, 1860), rng.randrange(90, 990)) for _ in range(8)]
            drawn, labels = draw_text(frame, rng, positions)
            torch.cuda.synchronize()
            start = time.perf_counter()
            x = torch.from_numpy(drawn).cuda().permute(2, 0, 1)[None].float()
            x = (F.interpolate(x, size, mode="area") / 255 - mean) / std
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                g = grid(model.forward_features(x)).float()
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - start)
            _, h, w = g.shape
            for (px, py), label in zip(positions, labels, strict=True):
                text_x.append(g[:, int(py / 1080 * h), int(px / 1920 * w)].cpu())
                text_y.append(label)
            pointer.append((g.cpu(), int(cursor[1] / 1080 * h), int(cursor[0] / 1920 * w)))
        return torch.stack(text_x), torch.tensor(text_y), pointer, np.median(seconds[3:]) * 1000

    tx, ty, tp, ms = collect(train)
    vx, vy, vp, _ = collect(test)
    read = fit(tx, ty, len(CHARS))
    text = (read(vx).argmax(1) == vy).float().mean().item()
    xs, ys = [], []
    for g, py, px in tp:
        flat, w = g.flatten(1).T, g.shape[2]
        positive = py * w + px
        xs.append(flat[positive])
        ys.append(1.0)
        for k in random.Random(positive).sample(range(len(flat)), 40):
            if k != positive:
                xs.append(flat[k])
                ys.append(0.0)
    find = fit(torch.stack(xs), torch.tensor(ys), 1)
    hits = 0
    for g, py, px in vp:
        w = g.shape[2]
        best = find(g.flatten(1).T).squeeze(-1).argmax().item()
        hits += abs(best // w - py) <= 1 and abs(best % w - px) <= 1
    print(
        f"{name:45s} {size[1]}x{size[0]} grid {tuple(tp[0][0].shape[1:])} {ms:5.1f} ms"
        f" | text {text:5.1%} (chance 2.8%) | pointer {hits / len(vp):5.1%}",
        flush=True,
    )


def main():
    train, test, *specs = sys.argv[1:]
    for spec in specs:
        name, _, given = spec.partition("@")
        if "x" in given:
            width, height = (int(v) for v in given.split("x"))
            size = (height, width)
        else:
            side = int(given) if given else 896
            size = (side, side)
        probe(name, size, train, test)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

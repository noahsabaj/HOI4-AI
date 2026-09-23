"""LeVJEPA against the Qwen3.5 screen tower, probed frozen on recorded HOI4 games.

A second, fuller look at LeVJEPA (Kuhn et al., 2026, arXiv:2608.27395) after it was set
aside on speed alone. Its paper evaluates frozen features with an attentive probe because
they need not be linearly separable, reads video at any size through 3D rotary positions,
and attends block-causally, so a frame's tokens depend only on it and earlier frames. So
here every encoder gets the same frames, and each is read out both linearly and by a
small MLP; LeVJEPA reads 16:9 frames at several sizes, with 1, 4 or 8 frames of context,
from three depths.

Three things are asked of each encoder, training on one game and testing on another:

- text: 8 random characters (A-Z, 0-9) are drawn at 10-12 px (1080p scale) on the last
  frame; the read-out names each from the feature of the patch it sits in. Chance 2.8%.
- pointer: the read-out scores every patch of the last frame; a hit is the top patch
  within one of the patch under the pointer's hotspot.
- motion: what the camera did over the last 200 ms, from the recorder's own inputs:
  still, a pan (left, right, up, down) or a zoom (in, out). Read from the mean of the
  last frame's patches, and, for an encoder that sees one frame, from that and the
  frame before, side by side. Balanced accuracy, chance 1/7.

    python scripts/probe_video_encoders.py artifacts/ai-games-1080p/<game-a> artifacts/ai-games-1080p/<game-b>

Times are measured separately (see --time), on an idle GPU.
"""

import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
FONTS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
]
MOTIONS = ["still", "left", "right", "up", "down", "zoom in", "zoom out"]
ARROWS = {0x25: 1, 0x27: 2, 0x26: 3, 0x28: 4}
CONTEXT = 8  # Frames kept before each sampled frame (200 ms apart).
MOTION_FRAMES = 4
# Pans are brief, so each direction has only a handful of examples: also scored as four
# classes, still / pan / zoom in / zoom out.
COARSE = np.array([0, 1, 1, 1, 1, 2, 3])
IMAGENET = (torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))


def motion_labels(rows):
    """The camera's motion over the interval ending at each frame, from its own inputs."""
    held, labels = set(), []
    for row in rows:
        label = 0
        for item in row.get("scripted_events") or []:
            event = item["event"]
            if event.get("kind") == "wheel":
                label = 5 if event["delta"] > 0 else 6
            elif event.get("kind") == "key" and event.get("vk") in ARROWS:
                (held.add if event["down"] else held.discard)(event["vk"])
                if event["down"]:
                    label = ARROWS[event["vk"]]
        if label == 0 and len(held) == 1:
            label = ARROWS[next(iter(held))]
        labels.append(label)
    return labels


def choose(labels, count, seed):
    """Sampled frame indices: half still, half moving, each with CONTEXT frames before it."""
    rng = random.Random(seed)
    still = [i for i, m in enumerate(labels) if m == 0 and i >= CONTEXT]
    moving = [i for i, m in enumerate(labels) if m and i >= CONTEXT]
    picks = rng.sample(still, min(len(still), count // 2))
    picks += rng.sample(moving, min(len(moving), count - len(picks)))
    return sorted(picks)


def draw_text(frame, rng):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    positions = [(rng.randrange(60, 1860), rng.randrange(90, 990)) for _ in range(8)]
    labels = []
    for x, y in positions:
        char = rng.choice(CHARS)
        font = ImageFont.truetype(rng.choice(FONTS), rng.choice([10, 11, 12]))
        colour = rng.choice([(255, 255, 255), (255, 220, 120), (230, 230, 230), (20, 20, 20)])
        draw.text((x, y), char, font=font, fill=colour, anchor="mm")
        labels.append(CHARS.index(char))
    return np.asarray(image), positions, labels


STORE = (640, 360)  # Context frames are kept at this size; a single frame at 1080p.


def samples(recording, count, seed):
    """One dict per sampled frame, decoded in one pass: the frame with text drawn on it and
    the frame before, at 1080p; the CONTEXT+1 frames ending with it at STORE size (the
    text drawn frame last); the text positions and labels, the cursor and the motion."""
    import cv2

    root = Path(recording)
    rows = [json.loads(line) for line in (root / "frames.jsonl").open()]
    labels = motion_labels(rows)
    wanted = choose(labels, count, seed)
    keep = {j for i in wanted for j in range(i - CONTEXT, i + 1)}
    full_keep = {j for i in wanted for j in (i - 1, i)}
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(root / "screen.mkv"), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE,
    )  # fmt: skip
    frames, small, size, rng = {}, {}, 1920 * 1080 * 3, random.Random(seed)
    for index in range(max(wanted) + 1):
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        if index in keep:
            frame = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3)
            small[index] = cv2.resize(frame, STORE, interpolation=cv2.INTER_AREA)
            if index in full_keep:
                frames[index] = frame
    proc.kill()
    out = []
    for i in wanted:
        last, positions, text = draw_text(frames[i], rng)
        clip = [small[j] for j in range(i - CONTEXT, i)]
        clip.append(cv2.resize(last, STORE, interpolation=cv2.INTER_AREA))
        out.append(
            {"last": last, "before": frames[i - 1], "clip": clip, "positions": positions,
             "text": text, "cursor": rows[i]["cursor"], "motion": labels[i]}
        )  # fmt: skip
    return out


def resize(frames, size, mean, std):
    """(T, 3, h, w) normalized, area-averaged like the worker's views."""
    x = torch.from_numpy(np.stack(frames)).cuda().permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size, mode="area") / 255
    return (x - mean.cuda()[:, None, None]) / std.cuda()[:, None, None]


class LeVJEPA:
    """The released ViT-L, reading `frames` frames of 16:9 video at `size` (h, w)."""

    model = None

    def __init__(self, size, frames, layers=(11, 17, 23)):
        from hoi4_arena.benchmark import load_encoder

        if LeVJEPA.model is None:
            LeVJEPA.model = load_encoder("models/levjepa-large").cuda().eval().to(torch.bfloat16)
        self.size, self.frames, self.layers = size, frames, layers
        self.name = f"LeVJEPA {size[1]}x{size[0]} T={frames}"

    def features(self, sample, frame=None):
        """{layer: (C, h, w) last-frame grid}, one forward over the clip.

        One frame is read from 1080p (or `frame`, given); a clip from the stored context
        frames, all resized alike so nothing but the content differs between its frames.
        """
        if frame is not None:
            frames = [frame]
        elif self.frames == 1:
            frames = [sample["last"]]
        else:
            frames = sample["clip"][-self.frames :]
        x = resize(frames, self.size, *IMAGENET).permute(1, 0, 2, 3)[None]
        vit = find_vit(LeVJEPA.model)
        vit.out_layers = list(self.layers)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outs = vit(x.to(torch.bfloat16))
        vit.out_layers = None
        h, w = self.size[0] // 16, self.size[1] // 16
        return {
            layer: out[0, -h * w :].float().T.reshape(-1, h, w)
            for layer, out in zip(self.layers, outs, strict=True)
        }


def find_vit(model):
    for module in model.modules():
        if type(module).__name__ == "VisionTransformer":
            return module
    raise RuntimeError("no VisionTransformer inside the LeVJEPA model")


class Qwen:
    """The Qwen3.5-0.8B vision tower, one frame at `size` (h, w), as the policy uses it."""

    model = None

    def __init__(self, size, layers=(5, 11)):
        import timm

        if Qwen.model is None:
            Qwen.model = (
                timm.create_model(
                    "qwen3_vit_88m_enc",
                    pretrained=True,
                    pretrained_cfg_overlay={"file": "models/qwen3-vit-88m/model.safetensors"},
                )
                .cuda()
                .eval()
            )
        self.size, self.frames, self.layers = size, 1, layers
        self.name = f"Qwen3.5 tower {size[1]}x{size[0]}"

    def features(self, sample, frame=None):
        half = torch.tensor([0.5, 0.5, 0.5])
        x = resize([sample["last"] if frame is None else frame], self.size, half, half)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outs = Qwen.model.forward_intermediates(
                x, indices=list(self.layers), intermediates_only=True, output_fmt="NCHW"
            )
        return {layer: out[0].float() for layer, out in zip(self.layers, outs, strict=True)}


def collect(encoder, data):
    """Per layer: text features, pointer grids, motion features."""
    result = {}
    for sample in data:
        positions, text, cursor, motion = (
            sample[k] for k in ("positions", "text", "cursor", "motion")
        )
        now = encoder.features(sample)
        # An encoder that reads one frame gets the same four frames for motion, side by
        # side, as the clip encoder reads in sequence.
        context = (
            [encoder.features(sample, frame=f) for f in sample["clip"][-MOTION_FRAMES:]]
            if encoder.frames == 1
            else None
        )
        for layer, g in now.items():
            _, h, w = g.shape
            entry = result.setdefault(layer, {"tx": [], "ty": [], "grids": [], "mx": [], "my": []})
            for (px, py), label in zip(positions, text, strict=True):
                entry["tx"].append(g[:, int(py / 1080 * h), int(px / 1920 * w)].cpu())
                entry["ty"].append(label)
            entry["grids"].append(
                (g.half().cpu(), int(cursor[1] / 1080 * h), int(cursor[0] / 1920 * w))
            )
            pooled = g.mean((1, 2))
            if context is not None:
                pooled = torch.cat([c[layer].mean((1, 2)) for c in context])
            entry["mx"].append(pooled.cpu())
            entry["my"].append(motion)
    return result


def fit(features, targets, classes, mlp, steps=600):
    torch.manual_seed(0)
    mean, scale = features.mean(0), features.std(0) + 1e-6
    dim = features.shape[1]
    head = (
        torch.nn.Sequential(
            torch.nn.Linear(dim, 512), torch.nn.GELU(), torch.nn.Linear(512, classes)
        )
        if mlp
        else torch.nn.Linear(dim, classes)
    )
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(steps):
        optimizer.zero_grad()
        logits = head((features - mean) / scale)
        if classes == 1:
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets)
        else:
            loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
    head.eval()
    return lambda x: head((x - mean) / scale)


def score(train, test, mlp):
    read = fit(torch.stack(train["tx"]), torch.tensor(train["ty"]), len(CHARS), mlp)
    with torch.no_grad():
        text = (read(torch.stack(test["tx"])).argmax(1) == torch.tensor(test["ty"])).float().mean()
    xs, ys = [], []
    for g, py, px in train["grids"]:
        flat, w = g.float().flatten(1).T, g.shape[2]
        positive = py * w + px
        xs.append(flat[positive])
        ys.append(1.0)
        for k in random.Random(positive).sample(range(len(flat)), 40):
            if k != positive:
                xs.append(flat[k])
                ys.append(0.0)
    find = fit(torch.stack(xs), torch.tensor(ys), 1, mlp)
    hits = 0
    with torch.no_grad():
        for g, py, px in test["grids"]:
            w = g.shape[2]
            best = find(g.float().flatten(1).T).squeeze(-1).argmax().item()
            hits += abs(best // w - py) <= 1 and abs(best % w - px) <= 1
    move = fit(torch.stack(train["mx"]), torch.tensor(train["my"]), len(MOTIONS), mlp)
    with torch.no_grad():
        guess = move(torch.stack(test["mx"])).argmax(1).numpy()
    truth = np.array(test["my"])

    def balanced(guess, truth):
        return float(np.mean([(guess[truth == c] == c).mean() for c in np.unique(truth)]))

    coarse = balanced(COARSE[guess], COARSE[truth])
    return float(text), hits / len(test["grids"]), balanced(guess, truth), coarse


def spec_encoder(spec):
    """ "qwen:HxW" or "levjepa:HxW:frames"."""
    kind, size, *rest = spec.split(":")
    size = tuple(int(v) for v in size.split("x"))
    return Qwen(size) if kind == "qwen" else LeVJEPA(size, int(rest[0]))


def main():
    train_dir, test_dir = sys.argv[1:3]
    torch.cuda.set_per_process_memory_fraction(0.35)  # Leave the running game its memory.
    started = time.perf_counter()
    train, test = samples(train_dir, 160, 0), samples(test_dir, 160, 1)
    print(f"decoded {len(train)} + {len(test)} samples in {time.perf_counter() - started:.0f} s")
    counts = np.bincount([s["motion"] for s in test], minlength=len(MOTIONS))
    print("test motions:", dict(zip(MOTIONS, counts.tolist(), strict=True)), flush=True)
    specs = sys.argv[3:]
    encoders = [spec_encoder(spec) for spec in specs] or [
        Qwen((896, 896)),
        Qwen((640, 1152)),
        Qwen((256, 448)),
        LeVJEPA((224, 224), 8),
        LeVJEPA((256, 448), 1),
        LeVJEPA((256, 448), 4),
        LeVJEPA((352, 640), 1),
        LeVJEPA((352, 640), 4),
        LeVJEPA((496, 896), 1),
    ]
    print(
        f"{'encoder':34s} {'layer':>5s} | text lin / mlp | pointer lin / mlp |"
        " motion lin / mlp | 4-class lin / mlp"
    )
    for encoder in encoders:
        a, b = collect(encoder, train), collect(encoder, test)
        for layer in a:
            lin, mlp = score(a[layer], b[layer], False), score(a[layer], b[layer], True)
            print(
                f"{encoder.name:34s} {layer:5d} | {lin[0]:5.1%} / {mlp[0]:5.1%} |"
                f" {lin[1]:5.1%} / {mlp[1]:5.1%} | {lin[2]:5.1%} / {mlp[2]:5.1%} |"
                f" {lin[3]:5.1%} / {mlp[3]:5.1%}",
                flush=True,
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

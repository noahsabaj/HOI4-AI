"""Offline capacity check on a real saved screenshot, with explicitly untrained heads."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hoi4_arena.benchmark import gpu_memory
from hoi4_arena.dataset import normalize, views
from hoi4_arena.models import Policy, PredictiveAuxiliary, VideoEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=["large", "tiny"], default="tiny")
parser.add_argument("--image", default="artifacts/benchmark-loaded/screen.png")
parser.add_argument("--output", default="artifacts/benchmark-policy-tiny.json")
args = parser.parse_args()
torch.manual_seed(42)
report = {
    "variant": args.variant,
    "trained_policy": False,
    "gameplay_verified": False,
    "mode": "offline_saved_screenshot",
    "baseline_memory": gpu_memory(),
}
try:
    view, tiles = views(np.asarray(Image.open(args.image).convert("RGB")), device="cuda")
    clip = normalize(torch.stack([view] * 16)).permute(3, 0, 1, 2)[None]
    tiles = normalize(tiles).permute(0, 3, 1, 2)[None]
    encoder = VideoEncoder("models/levjepa-large", variant=args.variant)
    policy = Policy(encoder).cuda().eval()
    report["encoder_parameters"] = sum(p.numel() for p in encoder.parameters())
    previous = torch.zeros(1, 8, 3, device="cuda", dtype=torch.long)
    times = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(35):
            torch.cuda.synchronize()
            start = time.perf_counter()
            hidden, value, _ = policy(clip, tiles, previous)
            policy.actor(hidden)
            torch.cuda.synchronize()
            if i >= 5:
                times.append((time.perf_counter() - start) * 1000)
    report.update(policy_p95_ms=float(np.percentile(times, 95)), inference_memory=gpu_memory())
    policy.train()
    auxiliary = PredictiveAuxiliary(feature_dim=encoder.dim, mode="sparse").cuda()
    params = [p for p in [*policy.parameters(), *auxiliary.parameters()] if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        memories = []
        features = []
        hidden = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(2):
                hidden, value, feature = policy(
                    clip.expand(2, -1, -1, -1, -1),
                    tiles.expand(2, -1, -1, -1, -1),
                    previous.expand(2, -1, -1),
                    hidden,
                )
                memories.append(hidden)
                features.append(feature)
            memory = torch.stack(memories, 1)
            actions = previous.expand(4, -1, -1)
            loss = -policy.actor(memory.flatten(0, 1), actions)[1].mean()
            loss = loss + 0.1 * auxiliary(
                memory,
                torch.stack(features, 1),
                actions.reshape(2, 2, 8, 3),
                torch.ones(2, 2, device="cuda", dtype=torch.bool),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
    torch.cuda.synchronize()
    report.update(
        two_training_updates=True,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        training_memory=gpu_memory(),
        loss=float(loss.detach()),
    )
except Exception as error:
    report["error"] = f"{type(error).__name__}: {error}"
Path(args.output).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

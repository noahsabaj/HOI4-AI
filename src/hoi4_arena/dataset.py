from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .actions import SLOTS, encode_interval


def quadrants(h, w):
    """The four spatially ordered half-resolution boxes, as (top, left, height, width)."""
    return [
        (0, 0, h // 2, w // 2),
        (0, w // 2, h // 2, w - w // 2),
        (h // 2, 0, h - h // 2, w // 2),
        (h // 2, w // 2, h - h // 2, w - w // 2),
    ]


def views(rgb, size=224, device="cpu"):
    """Global view plus four quadrant crops, as uint8 tensors on `device`.

    The resampler is `area`, an exact average over integer bins: output pixel k covers
    source rows [floor(k*H/size), ceil((k+1)*H/size)). That rule is the whole point. A
    3840x2160 frame is a 17x reduction, and the filter choice dominates the result --
    plain bilinear without antialiasing differs from a properly filtered downscale by a
    mean of 72/255. The two defensible filters here differ from each other by a mean of
    11/255, which is far too much to let training and deployment disagree about.

    `area` is chosen over antialiased bilinear because it is the one both sides can
    reproduce exactly: the desktop worker implements the same integer binning in Rust so
    it can downscale before transport, and test_worker_downscale_matches_training_resize
    pins the two implementations against each other. Whatever this function does, the
    worker must do bit-for-bit, or the policy sees different pixels than it trained on.

    Accumulation stays in float32. Half precision would cost about 90 MiB of peak
    allocation, but it rounds a handful of output pixels differently, and any divergence
    here is exactly what the worker must not have. CPU and CUDA float32 agree exactly.
    """
    source = torch.as_tensor(np.ascontiguousarray(rgb), device=device).permute(2, 0, 1)[None]
    h, w = source.shape[-2:]
    # Spatially ordered quadrants retain twice the global view's linear resolution.
    boxes = [source] + [
        source[..., top : top + bh, left : left + bw] for top, left, bh, bw in quadrants(h, w)
    ]
    resized = []
    for box in boxes:
        # One box at a time: the intermediate for a full 4K frame dominates the peak.
        scaled = F.interpolate(box.float(), (size, size), mode="area")
        resized.append(scaled.round().clamp(0, 255).to(torch.uint8))
        del scaled
    stacked = torch.cat(resized).permute(0, 2, 3, 1)
    return stacked[0], stacked[1:]


def normalize(array):
    x = array.float() if torch.is_tensor(array) else torch.as_tensor(np.array(array, copy=True))
    x = x.float() / 255
    mean = x.new_tensor([0.485, 0.456, 0.406])
    std = x.new_tensor([0.229, 0.224, 0.225])
    return (x - mean) / std


def prepare_session(source, destination):
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source / "manifest.json").read_text())
    if not manifest["complete"] or manifest["source"] != "human":
        raise ValueError("Only complete human recordings enter imitation training")
    rows = [json.loads(line) for line in (source / "frames.jsonl").read_text().splitlines()]
    if len(rows) != manifest["frames"] or len(rows) < 32:
        raise ValueError("Missing frames or session too short")
    times = np.array([row["t_ns"] for row in rows], dtype=np.int64)
    if np.any(np.diff(times) <= 0) or max(np.diff(times)) > 1e9:
        raise ValueError("Nonmonotonic timestamps or capture gap exceeding one second")
    events = [e for row in rows for e in row["events"]]
    tail = source / "trailing-events.json"
    if tail.exists():
        events += json.loads(tail.read_text())["events"]
    events.sort(key=lambda e: e["t_ns"])
    decisions = np.arange(
        times[0] + 2_100_000_000, times[-1] - 200_000_000, 200_000_000, dtype=np.int64
    )
    frame_ids = np.searchsorted(times, decisions, side="right") - 1
    destination.mkdir(parents=True, exist_ok=False)
    global_frames = np.lib.format.open_memmap(
        destination / "global.npy", "w+", np.uint8, (len(rows), 224, 224, 3)
    )
    details = np.lib.format.open_memmap(
        destination / "details.npy", "w+", np.uint8, (len(decisions), 4, 224, 224, 3)
    )
    selected = {}
    for decision, frame_id in enumerate(frame_ids):
        selected.setdefault(int(frame_id), []).append(decision)
    w, h = manifest["width"], manifest["height"]
    # Offline work with the game closed, so the GPU is free; this is thousands of frames.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = subprocess.Popen(
        [
            shutil.which("ffmpeg"),
            "-v",
            "error",
            "-i",
            str(source / "screen.mkv"),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
    )
    try:
        for i in range(len(rows)):
            buf = decoder.stdout.read(w * h * 3)
            if len(buf) != w * h * 3:
                raise ValueError("Video truncated relative to timestamps")
            global_view, detail = views(
                np.frombuffer(buf, np.uint8).reshape(h, w, 3), device=device
            )
            global_frames[i] = global_view.cpu().numpy()
            if selected.get(i):
                detail = detail.cpu().numpy()
            for decision in selected.get(i, []):
                details[decision] = detail
        if decoder.stdout.read(1):
            raise ValueError("Video has unindexed extra frames")
        if decoder.wait() != 0:
            raise RuntimeError("FFmpeg decode failed")
    finally:
        if decoder.poll() is None:
            decoder.kill()
            decoder.wait()
    global_frames.flush()
    details.flush()
    event_times = np.array([e["t_ns"] for e in events], dtype=np.int64)
    actions = np.zeros((len(decisions), SLOTS, 3), dtype=np.int64)
    valid = np.ones(len(decisions), dtype=bool)
    excluded = []
    for i, t in enumerate(decisions):
        lo, hi = np.searchsorted(event_times, [t, t + 200_000_000])
        try:
            actions[i] = encode_interval(events[lo:hi], int(t))
        except ValueError as error:
            valid[i] = False
            excluded.append({"decision": i, "reason": str(error)})
    for name, value in {
        "times": times,
        "decisions": decisions,
        "frame_ids": frame_ids,
        "actions": actions,
        "valid": valid,
    }.items():
        np.save(destination / f"{name}.npy", value)
    (destination / "manifest.json").write_text(
        json.dumps({**manifest, "prepared": True, "excluded": excluded}, indent=2)
    )
    return {"decisions": len(decisions), "excluded": len(excluded), "split": manifest["split"]}


class Sessions(Dataset):
    def __init__(self, root, split="train", length=8, burn_in=2):
        self.sessions, self.index = [], []
        self.length, self.burn_in = length, burn_in
        for path in sorted(Path(root).glob("*/manifest.json")):
            meta = json.loads(path.read_text())
            if not meta.get("prepared") or meta["split"] != split:
                continue
            data = {
                name: np.load(path.parent / f"{name}.npy", mmap_mode="r")
                for name in [
                    "global",
                    "details",
                    "times",
                    "decisions",
                    "frame_ids",
                    "actions",
                    "valid",
                ]
            }
            number = len(self.sessions)
            self.sessions.append(data)
            for start in range(0, len(data["actions"]) - length - burn_in + 1, length):
                if data["valid"][start : start + length + burn_in].all():
                    self.index.append((number, start))
        if not self.index:
            raise ValueError(
                f"No complete valid {split} sequences; record and prepare human sessions first"
            )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        session, start = self.index[index]
        data = self.sessions[session]
        n = self.length + self.burn_in
        decision_times = data["decisions"][start : start + n]
        clip_times = decision_times[:, None] - np.arange(15, -1, -1)[None] * (1e9 / 7.5)
        clip_ids = np.searchsorted(data["times"], clip_times, side="right") - 1
        if clip_ids.min() < 0:
            raise ValueError("Clip reaches before session start")
        clips = normalize(data["global"][clip_ids]).permute(0, 4, 1, 2, 3)
        tiles = normalize(data["details"][start : start + n]).permute(0, 1, 4, 2, 3)
        actions = torch.from_numpy(data["actions"][start : start + n].copy())
        previous = torch.zeros_like(actions)
        previous[1:] = actions[:-1]
        if start:
            previous[0] = torch.from_numpy(data["actions"][start - 1].copy())
        return {
            "clips": clips,
            "tiles": tiles,
            "actions": actions,
            "previous": previous,
            "valid": torch.ones(n, dtype=torch.bool),
            "session": session,
            "start": start,
        }

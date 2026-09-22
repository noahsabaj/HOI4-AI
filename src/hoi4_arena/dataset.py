from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .actions import PERIOD, SLOTS, encode_interval

# How many past global views one decision looks at. Eight frames is what fits the
# 200 ms tick: on the 4060 Ti the encoder forward fell from 131.5 ms at sixteen frames
# to 64.9 ms at eight, which is the difference between a policy that fits the interval
# alongside capture and one that does not. A shorter window is less history. It is taken
# because a policy that misses its deadline observes nothing at all. The encoder imposes
# no limit of its own: it positions tokens with RoPE rather than a fixed table, so any
# length is legal and only the context changes.
#
# Lookback spacing is the decision interval, one view per tick, in integer nanoseconds.
# The live actor stores one view per decision, so a faster lookback repeats neighbours
# there while a 10 Hz recording still has a distinct frame. Both paths call
# clip_frame_ids. Eight frames then cover 1.6 s.
CLIP_FRAMES = 8
PERIOD_NS = int(round(PERIOD * 1e9))
# The global frame and the four quadrants are area-averaged down to this square. The
# last view is not averaged: it is this many native pixels, centered on the pointer.
# At 3840x2160 that global view shrinks an ~88 px province to about five pixels, and a
# self-play step stores only the tiles it was given, so a recording that omits this
# crop cannot grow it afterwards.
VIEW_SIZE = 224
VIEW_COUNT = 6
TILES = VIEW_COUNT - 1
# Wall seconds of one in-game hour: the game's GAME_SPEED_SECONDS, speeds 1 through 5.
# Speed 1 was measured at 2.0 (48 s per in-game day) and speed 4 at 0.1. Speed 5 is 0
# because the simulation does not sleep, so that clip has no fixed game length. Eight
# frames at the decision interval are 1.6 s of wall time: 3.2 in-game hours at speed 2
# and 16 at speed 4.
GAME_SPEED_SECONDS = (2.0, 0.5, 0.2, 0.1, 0.0)


def clip_frame_ids(frame_times_ns, decision_ns):
    """Index of the latest frame at each lookback time, oldest first.

    A scalar decision returns shape (CLIP_FRAMES,). Several decisions return
    (N, CLIP_FRAMES). An index of -1 means the sample falls before the first stored
    frame: the live actor repeats its oldest view, and Sessions refuses the clip.
    """
    decision_ns = np.asarray(decision_ns, dtype=np.int64)
    offsets = np.arange(CLIP_FRAMES - 1, -1, -1, dtype=np.int64) * PERIOD_NS
    return np.searchsorted(frame_times_ns, decision_ns[..., None] - offsets, side="right") - 1


def quadrants(h, w):
    """The four spatially ordered half-resolution boxes, as (top, left, height, width)."""
    return [
        (0, 0, h // 2, w // 2),
        (0, w // 2, h // 2, w - w // 2),
        (h // 2, 0, h - h // 2, w // 2),
        (h // 2, w // 2, h - h // 2, w - w // 2),
    ]


def recorded_speed(value):
    """Manifest fields for the speed the operator set.

    There is no default. A recording that omits the speed cannot be assigned one
    afterwards, and the documented 2 is not what the long runs used.
    """
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise ValueError("game speed must be an integer from 1 to 5")
    seconds = GAME_SPEED_SECONDS[value - 1]
    return {"game_speed": value, "seconds_per_hour": None if seconds == 0 else seconds}


def require_one_game_speed(speeds):
    """One training set has one speed. Two speeds make the same clip two amounts of game time."""
    chosen = sorted(set(speeds))
    if len(chosen) != 1:
        raise ValueError(
            f"the same clip is a different amount of game time at game speeds {chosen}"
        )
    return chosen[0]


def parse_cursor(value):
    """Client-pixel pointer the native crop is centered on."""
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    ):
        return int(value[0]), int(value[1])
    raise ValueError("cursor must be two client-pixel integers")


def cursor_crop(rgb, x, y, size):
    """Native `size` square with the pointer pixel at `(size // 2, size // 2)`.

    Samples outside the frame are zero. Sliding the window until it fits would move
    the pointer off that pixel, so the same click would look different at a screen edge.
    An even size cannot be symmetric: 224 leaves 112 pixels to the left and 111 to the right.
    """
    image = np.asarray(rgb)
    out = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
    height, width = image.shape[:2]
    origin_x = int(x) - size // 2
    origin_y = int(y) - size // 2
    src_x0 = max(0, origin_x)
    src_y0 = max(0, origin_y)
    src_x1 = min(width, origin_x + size)
    src_y1 = min(height, origin_y + size)
    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return out
    dst_x0 = src_x0 - origin_x
    dst_y0 = src_y0 - origin_y
    out[
        dst_y0 : dst_y0 + (src_y1 - src_y0),
        dst_x0 : dst_x0 + (src_x1 - src_x0),
    ] = image[src_y0:src_y1, src_x0:src_x1]
    return out


def views(rgb, size=VIEW_SIZE, device="cpu", cursor=None):
    """Global view, four quadrant crops, and the native cursor crop, as uint8 on `device`.

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
    The last tile is the cursor crop, copied rather than resized. `cursor` is required
    because a stand-in would be a different view, and the archive could not be repaired.

    Accumulation stays in float32. Half precision would cost about 90 MiB of peak
    allocation, but it rounds a handful of output pixels differently, and any divergence
    here is exactly what the worker must not have. CPU and CUDA float32 agree exactly.
    """
    x, y = parse_cursor(cursor)
    # prepare_session decodes into a read-only buffer. Torch will not share one, so
    # copy that case only; a capture the worker already copied stays as it is.
    frame = np.ascontiguousarray(rgb)
    if not frame.flags.writeable:
        frame = frame.copy()
    # The window is `size` squared, so the crop does not duplicate the whole frame.
    fovea = torch.as_tensor(cursor_crop(frame, x, y, size), device=device)
    source = torch.as_tensor(frame, device=device).permute(2, 0, 1)[None]
    h, w = source.shape[-2:]
    # Spatially ordered quadrants retain twice the global view's linear resolution.
    boxes = [source[..., top : top + bh, left : left + bw] for top, left, bh, bw in quadrants(h, w)]
    # The four quadrants go through as one batch when they are the same size, which on an
    # even-dimensioned frame they always are. `area` pools each sample over its own bins,
    # so batching cannot change a pixel -- a test pins that on both devices, including
    # the fallback -- and it turns four launches into one: a 4K frame goes from 7.04 ms
    # to 6.07 ms on the GPU, across thousands of frames per prepared session. It is not
    # free. Holding four quadrants at once raises peak allocation from 215.9 MiB to
    # 239.9 MiB, still under the whole-frame intermediate the global view already needs.
    # An odd dimension makes the bottom or right quadrant a pixel larger, and that case
    # falls back to the loop rather than padding, because padding is exactly the kind of
    # almost-right resize this function exists to avoid.
    if len({box.shape for box in boxes}) == 1:
        boxes = [torch.cat(boxes)]
    resized = []
    # The global view stays on its own: its source is the whole frame, so it can never
    # join the batch, and its float32 intermediate is what dominates the peak anyway.
    for box in [source, *boxes]:
        scaled = F.interpolate(box.float(), (size, size), mode="area")
        resized.append(scaled.round().clamp(0, 255).to(torch.uint8))
        del scaled
    stacked = torch.cat(resized).permute(0, 2, 3, 1)
    return stacked[0], torch.cat([stacked[1:], fovea[None]], 0)


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
    # Before any frame is decoded. A session with no recorded speed cannot grow one.
    speed = recorded_speed(manifest.get("game_speed"))
    rows = [json.loads(line) for line in (source / "frames.jsonl").read_text().splitlines()]
    if len(rows) != manifest["frames"] or len(rows) < 32:
        raise ValueError("Missing frames or session too short")
    times = np.array([row["t_ns"] for row in rows], dtype=np.int64)
    if np.any(np.diff(times) <= 0) or max(np.diff(times)) > 1e9:
        raise ValueError("Nonmonotonic timestamps or capture gap exceeding one second")
    # Before the destination exists, so a session that cannot grow the crop leaves nothing.
    cursors = [parse_cursor(row.get("cursor")) for row in rows]
    events = [e for row in rows for e in row["events"]]
    tail = source / "trailing-events.json"
    if tail.exists():
        events += json.loads(tail.read_text())["events"]
    events.sort(key=lambda e: e["t_ns"])
    # The first decision needs a whole clip of recorded video behind it, plus one
    # interval of margin. The lead-in is derived from the clip rather than written down:
    # a hardcoded 2.1 s was correct only while a clip spanned 2.0 s, and raising
    # CLIP_FRAMES past that would have produced a dataset that only failed later, at
    # __getitem__, with "Clip reaches before session start".
    lead_in = (CLIP_FRAMES + 1) * PERIOD_NS
    decisions = np.arange(times[0] + lead_in, times[-1] - PERIOD_NS, PERIOD_NS, dtype=np.int64)
    frame_ids = np.searchsorted(times, decisions, side="right") - 1
    destination.mkdir(parents=True, exist_ok=False)
    global_frames = np.lib.format.open_memmap(
        destination / "global.npy", "w+", np.uint8, (len(rows), VIEW_SIZE, VIEW_SIZE, 3)
    )
    details = np.lib.format.open_memmap(
        destination / "details.npy",
        "w+",
        np.uint8,
        (len(decisions), TILES, VIEW_SIZE, VIEW_SIZE, 3),
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
                np.frombuffer(buf, np.uint8).reshape(h, w, 3),
                device=device,
                cursor=cursors[i],
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
        lo, hi = np.searchsorted(event_times, [t, t + PERIOD_NS])
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
        json.dumps(
            {
                **manifest,
                **speed,
                "prepared": True,
                "excluded": excluded,
                "tiles": TILES,
                "cursor_crop": VIEW_SIZE,
            },
            indent=2,
        )
    )
    return {"decisions": len(decisions), "excluded": len(excluded), "split": manifest["split"]}


class Sessions(Dataset):
    def __init__(self, root, split="train", length=8, burn_in=2):
        self.sessions, self.index = [], []
        self.length, self.burn_in = length, burn_in
        speeds = []
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
            count = int(data["details"].shape[1]) if data["details"].ndim == 5 else None
            if count != TILES:
                raise ValueError(
                    f"detail tiles must include the cursor crop ({TILES} tiles, got {count})"
                )
            speeds.append(recorded_speed(meta.get("game_speed"))["game_speed"])
            number = len(self.sessions)
            self.sessions.append(data)
            for start in range(0, len(data["actions"]) - length - burn_in + 1, length):
                if data["valid"][start : start + length + burn_in].all():
                    self.index.append((number, start))
        if speeds:
            self.game_speed = require_one_game_speed(speeds)
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
        clip_ids = clip_frame_ids(data["times"], decision_times)
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

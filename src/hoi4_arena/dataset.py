from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

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
# Three views of each decision's frame, all area-averaged the same way (see `views`):
#
# - The global view, the whole screen at VIEW_SIZE square, is what the video encoder
#   reads, eight frames at a time. Its cost sets the tick, so it stays small.
# - The four quadrants at DETAIL_SIZE. HOI4 is read from text: 10 px text at 1080p. At
#   224 a quadrant shrank 4.3x across and that text to about 2 px, which nothing can read.
#   At 448 it is about 4.5 px tall and 2.1x narrower, which a convolutional reader can.
#   These go through a small CNN, not the encoder, so the extra pixels are cheap.
# - The fovea, FOVEA_SIZE native pixels centred on the pointer, never resized: whatever
#   the pointer is over is seen at full resolution.
VIEW_SIZE = 224
DETAIL_SIZE = 448
FOVEA_SIZE = 224
QUADRANTS = 4
# Wall seconds of one in-game hour: the game's GAME_SPEED_SECONDS, speeds 1 through 5.
# Speed 1 was measured at 2.0 (48 s per in-game day) and speed 4 at 0.1. Speed 5 is 0
# because the simulation does not sleep, so that clip has no fixed game length. Eight
# frames at the decision interval are 1.6 s of wall time: 3.2 in-game hours at speed 2
# and 16 at speed 4. The policy is told the speed (Policy's speed input), so one model can
# learn from recordings made at different speeds.
GAME_SPEED_SECONDS = (2.0, 0.5, 0.2, 0.1, 0.0)
# Where `hoi4-arena label` writes the inverse dynamics model's labels in a recording.
IDM_LABELS = "labels-idm.npz"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class Views(NamedTuple):
    """One frame as the policy sees it. All uint8, channels last."""

    global_view: torch.Tensor  # (VIEW_SIZE, VIEW_SIZE, 3)
    quadrants: torch.Tensor  # (4, DETAIL_SIZE, DETAIL_SIZE, 3)
    fovea: torch.Tensor  # (FOVEA_SIZE, FOVEA_SIZE, 3)


def clip_frame_ids(frame_times_ns, decision_ns):
    """Index of the latest frame at each lookback time, oldest first.

    A scalar decision returns shape (CLIP_FRAMES,). Several decisions return
    (N, CLIP_FRAMES). An index of -1 means the sample falls before the first stored
    frame: the live actor repeats its oldest view, and training refuses the clip.
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


def parse_cursor(value):
    """Client-pixel pointer the fovea is centered on."""
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


def _area(box, size):
    scaled = F.interpolate(box.float(), (size, size), mode="area")
    return scaled.round().clamp(0, 255).to(torch.uint8)


def views(rgb, size=VIEW_SIZE, detail=DETAIL_SIZE, fovea=FOVEA_SIZE, device="cpu", cursor=None):
    """The global view, four quadrants and the fovea, as uint8 on `device`.

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
    The fovea is copied rather than resized. `cursor` is required because a stand-in would
    be a different view, and a recording could not be repaired afterwards.

    Accumulation stays in float32. Half precision would cost about 90 MiB of peak
    allocation, but it rounds a handful of output pixels differently, and any divergence
    here is exactly what the worker must not have. CPU and CUDA float32 agree exactly.
    """
    x, y = parse_cursor(cursor)
    # A decoded frame arrives in a read-only buffer. Torch will not share one, so copy
    # that case only; a capture the worker already copied stays as it is.
    frame = np.ascontiguousarray(rgb)
    if not frame.flags.writeable:
        frame = frame.copy()
    # The window is `fovea` squared, so the crop does not duplicate the whole frame.
    centre = torch.as_tensor(cursor_crop(frame, x, y, fovea), device=device)
    source = torch.as_tensor(frame, device=device).permute(2, 0, 1)[None]
    h, w = source.shape[-2:]
    boxes = [source[..., top : top + bh, left : left + bw] for top, left, bh, bw in quadrants(h, w)]
    # The four quadrants go through as one batch when they are the same size, which on an
    # even-dimensioned frame they always are. `area` pools each sample over its own bins,
    # so batching cannot change a pixel -- a test pins that on both devices, including
    # the fallback -- and it turns four launches into one. An odd dimension makes the
    # bottom or right quadrant a pixel larger, and that case falls back to the loop rather
    # than padding, because padding is exactly the kind of almost-right resize this
    # function exists to avoid.
    if len({box.shape for box in boxes}) == 1:
        quads = _area(torch.cat(boxes), detail)
    else:
        quads = torch.cat([_area(box, detail) for box in boxes])
    whole = _area(source, size)
    return Views(whole[0].permute(1, 2, 0), quads.permute(0, 2, 3, 1), centre)


def normalize(array):
    """uint8 channels-last pixels to the encoder's normalized float input."""
    x = array.float() if torch.is_tensor(array) else torch.as_tensor(np.array(array, copy=True))
    x = x.float() / 255
    return (x - x.new_tensor(MEAN)) / x.new_tensor(STD)


def batch_to_device(batch, device):
    """A collated training batch on `device`, with pixels normalized there.

    Pixels travel as uint8 and are normalized on the device: a float copy is four times
    the bytes to move, and the quadrants alone are 2.4 MB per decision in uint8.
    """
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    # (B, n, T, H, W, 3) -> (B, n, 3, T, H, W): the encoder takes channels first.
    out["clips"] = normalize(out["clips"]).permute(0, 1, 5, 2, 3, 4)
    out["quadrants"] = normalize(out["quadrants"]).permute(0, 1, 2, 5, 3, 4)
    out["fovea"] = normalize(out["fovea"]).permute(0, 1, 4, 2, 3)
    return out


def session_labels(source, *, sources=("human",), clip_shift=0, detail_shift=0):
    """Everything about a recording except its pixels: times, pointer, actions per decision.

    Checked before any frame is decoded, so a recording that cannot train fails here
    rather than an hour into a run. `sources` are the manifest sources accepted: "human"
    recordings carry the player's own inputs; "ai" recordings carry the scripted camera's
    (see ai_games), which are real inputs but not a player's.

    The shifts, in decision intervals, move each decision's clip and detail views later
    than the decision itself. The policy uses none: it acts on the past. The inverse
    dynamics model looks ahead, at frames that already show what the input did. A decision
    whose shifted frames run past the end of the video is not valid for that reader.
    """
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text())
    own = manifest.get("source") in sources
    # A recording with no inputs of its own can be labelled by the inverse dynamics model
    # (hoi4-arena label), and "idm" in `sources` accepts those labels.
    inferred = not own and "idm" in sources and (source / IDM_LABELS).exists()
    if not manifest.get("complete") or not (own or inferred):
        raise ValueError(f"Only complete recordings from {sorted(sources)} enter training")
    # Before any frame is decoded. A session with no recorded speed cannot grow one.
    speed = recorded_speed(manifest.get("game_speed"))
    rows = [json.loads(line) for line in (source / "frames.jsonl").read_text().splitlines()]
    if len(rows) != manifest["frames"] or len(rows) < 32:
        raise ValueError("Missing frames or session too short")
    times = np.array([row["t_ns"] for row in rows], dtype=np.int64)
    if np.any(np.diff(times) <= 0) or max(np.diff(times)) > 1e9:
        raise ValueError("Nonmonotonic timestamps or capture gap exceeding one second")
    cursors = [parse_cursor(row.get("cursor")) for row in rows]
    key = "events" if manifest["source"] == "human" else "scripted_events"
    events = [e for row in rows for e in row.get(key, [])]
    tail = source / "trailing-events.json"
    if tail.exists() and key == "events":
        events += json.loads(tail.read_text())["events"]
    events.sort(key=lambda e: e["t_ns"])
    # The first decision needs a whole clip of recorded video behind it, plus one
    # interval of margin. The lead-in is derived from the clip rather than written down:
    # a hardcoded 2.1 s was correct only while a clip spanned 2.0 s.
    lead_in = (CLIP_FRAMES + 1) * PERIOD_NS
    decisions = np.arange(times[0] + lead_in, times[-1] - PERIOD_NS, PERIOD_NS, dtype=np.int64)
    frame_ids = np.searchsorted(times, decisions + detail_shift * PERIOD_NS, side="right") - 1
    clip_ids = clip_frame_ids(times, decisions + clip_shift * PERIOD_NS)
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
    label_source = manifest["source"]
    if inferred:
        stored = np.load(source / IDM_LABELS)
        if not np.array_equal(stored["decisions"], decisions):
            raise ValueError("IDM labels were made on a different decision grid; label again")
        actions, valid = stored["actions"].copy(), stored["valid"].copy()
        label_source = "idm"
    # Frames the reader needs that the video does not have.
    readable = decisions + max(clip_shift, detail_shift) * PERIOD_NS <= times[-1]
    valid &= readable
    return {
        "root": source,
        "manifest": manifest,
        "speed": speed["game_speed"],
        "times": times,
        "cursors": cursors,
        "decisions": decisions,
        "frame_ids": frame_ids,
        "clip_ids": clip_ids,
        # The last frame each decision reads, so a window is cut only once it is decoded.
        "last_frame": np.maximum(clip_ids.max(-1), frame_ids),
        "readable": readable,
        "actions": actions,
        "valid": valid,
        "excluded": excluded,
        "label_source": label_source,
    }


def sequence_starts(valid, length, burn_in):
    """Where each whole, valid training window starts: burn-in plus scored steps."""
    return [
        start
        for start in range(0, len(valid) - length - burn_in + 1, length)
        if valid[start : start + length + burn_in].all()
    ]


def cover_starts(labels, length):
    """Windows of `length` that cover every decision whose frames exist, for labelling.

    Unlike training, validity does not matter: a decision with no usable label is exactly
    what labelling is for. The last window is pulled back to end on the last such decision.
    """
    readable = int(labels["readable"].sum())
    if readable < length:
        return []
    starts = list(range(0, readable - length + 1, length))
    if starts[-1] + length < readable:
        starts.append(readable - length)
    return starts


class _Stream:
    """One recording decoded front to back, yielding its windows as they fill."""

    def __init__(self, labels, length, burn_in, device, starts=None):
        self.labels, self.length, self.burn_in, self.device = labels, length, burn_in, device
        manifest = labels["manifest"]
        self.w, self.h = manifest["width"], manifest["height"]
        self.starts = (
            list(starts)
            if starts is not None
            else sequence_starts(labels["valid"], length, burn_in)
        )
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg is required to read recordings")
        self.decoder = subprocess.Popen(
            [ffmpeg, "-v", "error", "-i", str(labels["root"] / "screen.mkv")]
            + ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
            stdout=subprocess.PIPE,
        )
        # Global views are kept only while a pending window's clip can still reach them.
        self.globals = {}
        self.details = {}
        self.index = 0

    def close(self):
        if self.decoder.poll() is None:
            self.decoder.kill()
        self.decoder.wait()

    def _window(self, start):
        labels = self.labels
        n = self.length + self.burn_in
        steps = range(start, start + n)
        clips = torch.stack(
            [torch.stack([self.globals[int(i)] for i in labels["clip_ids"][d]]) for d in steps]
        )
        quads = torch.stack([self.details[d][0] for d in steps])
        fovea = torch.stack([self.details[d][1] for d in steps])
        actions = torch.from_numpy(labels["actions"][start : start + n].copy())
        previous = torch.zeros_like(actions)
        previous[1:] = actions[:-1]
        if start:
            previous[0] = torch.from_numpy(labels["actions"][start - 1].copy())
        return {
            "clips": clips,
            "quadrants": quads,
            "fovea": fovea,
            "actions": actions,
            "previous": previous,
            "valid": torch.from_numpy(labels["valid"][start : start + n].copy()),
            "speed": torch.full((n,), labels["speed"], dtype=torch.long),
            "start": start,
        }

    def advance(self):
        """Decode one frame. Returns the windows it completed, or None when none are left.

        Decoding stops at the last window rather than at the end of the video: the frames
        after it train nothing.
        """
        labels = self.labels
        if not self.starts:
            self.close()
            return None
        size = self.w * self.h * 3
        buffer = self.decoder.stdout.read(size)
        if len(buffer) != size:
            raise ValueError("Video truncated relative to timestamps")
        i = self.index
        self.index += 1
        first = self.starts[0]
        needed = [int(d) for d in np.flatnonzero(labels["frame_ids"] == i) if d >= first]
        if i < int(labels["clip_ids"][first].min()) and not needed:
            return []
        frame = np.frombuffer(buffer, np.uint8).reshape(self.h, self.w, 3)
        seen = views(frame, device=self.device, cursor=labels["cursors"][i])
        self.globals[i] = seen.global_view.cpu()
        if needed:
            quads, fovea = seen.quadrants.cpu(), seen.fovea.cpu()
            for d in needed:
                self.details[d] = (quads, fovea)
        done = []
        n = self.length + self.burn_in
        while self.starts and labels["last_frame"][self.starts[0] + n - 1] <= i:
            done.append(self._window(self.starts.pop(0)))
            if not self.starts:
                break
            # Nothing before the next window's first clip is needed again. Clip indices
            # only grow with the decision, so its first decision's clip is the earliest.
            keep_from = int(labels["clip_ids"][self.starts[0]].min())
            self.globals = {k: v for k, v in self.globals.items() if k >= keep_from}
            self.details = {k: v for k, v in self.details.items() if k >= self.starts[0]}
        return done


class VideoSessions(IterableDataset):
    """Training windows read straight from the recordings' video.

    Nothing is prepared ahead: each recording is decoded front to back and cut into
    windows of `burn_in + length` decisions as it plays. The views are computed on the
    fly, so their sizes can change without redoing any data, and no 43 GB per hour of
    448 px quadrants is ever written to disk. Several recordings play at once and their
    windows pass through a shuffle buffer, so a batch mixes games.
    """

    def __init__(
        self,
        root,
        split="train",
        length=8,
        burn_in=2,
        *,
        sources=("human",),
        streams=4,
        shuffle=32,
        seed=0,
        device=None,
        clip_shift=0,
        detail_shift=0,
    ):
        self.length, self.burn_in = length, burn_in
        self.streams, self.shuffle, self.seed = streams, shuffle, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sessions = []
        for path in sorted(Path(root).glob("*/manifest.json")):
            meta = json.loads(path.read_text())
            labelled = "idm" in sources and (path.parent / IDM_LABELS).exists()
            if meta.get("split") != split or not (meta.get("source") in sources or labelled):
                continue
            if not meta.get("complete"):
                continue
            self.sessions.append(
                session_labels(
                    path.parent,
                    sources=sources,
                    clip_shift=clip_shift,
                    detail_shift=detail_shift,
                )
            )
        self.windows = sum(len(sequence_starts(s["valid"], length, burn_in)) for s in self.sessions)
        if not self.windows:
            raise ValueError(f"No complete valid {split} sequences; record human sessions first")
        self.epoch = 0

    def __len__(self):
        return self.windows

    def __iter__(self):
        rng = random.Random(f"{self.seed}:{self.epoch}")
        self.epoch += 1
        order = list(self.sessions)
        rng.shuffle(order)
        active, buffer = [], []
        try:
            while order or active:
                while order and len(active) < self.streams:
                    active.append(_Stream(order.pop(), self.length, self.burn_in, self.device))
                for stream in list(active):
                    done = stream.advance()
                    if done is None:
                        stream.close()
                        active.remove(stream)
                        continue
                    buffer.extend(done)
                    while len(buffer) > self.shuffle:
                        yield buffer.pop(rng.randrange(len(buffer)))
            rng.shuffle(buffer)
            yield from buffer
        finally:
            for stream in active:
                stream.close()

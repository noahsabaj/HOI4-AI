from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .actions import PERIOD, SLOTS, VOCAB, encode_interval

# The view sizes, the game speeds, the pointer and the speed record are in layout.py, which
# needs no torch, so the recorders and the worker's client can use them without it. They
# are imported here so every `from .dataset import ...` keeps working.
from .layout import (  # noqa: F401
    DETAIL_SIZE,
    FOVEA_SIZE,
    GAME_SPEED_SECONDS,
    QUADRANTS,
    VIEW_SIZE,
    Views,
    hw,
    parse_cursor,
    recorded_speed,
)
from .learning import GAMMA

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
# The longest wait between two frames a decision may read across.
MAX_GAP_NS = 1_000_000_000
# Where `hoi4-arena label` writes the inverse dynamics model's labels in a recording.
IDM_LABELS = "labels-idm.npz"
# Where `hoi4-arena advantage` writes a player's recording's weights (offline.py).
ADVANTAGE_LABELS = "labels-advantage.npz"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


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
    scaled = F.interpolate(box.float(), hw(size), mode="area")
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

    `size` None skips the global view, for training an encoder that reads no clip.
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
    whole = None if size is None else _area(source, size)[0].permute(1, 2, 0)
    return Views(whole, quads.permute(0, 2, 3, 1), centre)


def normalize(array):
    """uint8 channels-last pixels to the encoder's normalized float input."""
    x = array.float() if torch.is_tensor(array) else torch.as_tensor(np.array(array, copy=True))
    x = x.float() / 255
    return (x - x.new_tensor(MEAN)) / x.new_tensor(STD)


def batch_to_device(batch, device, clips=True):
    """A collated training batch on `device`, with pixels normalized there.

    Pixels travel as uint8 and are normalized on the device: a float copy is four times
    the bytes to move, and the quadrants alone are 2.4 MB per decision in uint8.

    Without `clips`, or when the windows were cut without them, the batch has none. An
    encoder that reads no clip (models.reads_clip) is spared 52 MB of uint8 a batch of
    two, 210 MB once normalized.
    """
    out = {}
    for key, value in batch.items():
        if key == "clips" and not clips:
            continue
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    if "clips" in out:
        # (B, n, T, H, W, 3) -> (B, n, 3, T, H, W): the encoder takes channels first.
        out["clips"] = normalize(out["clips"]).permute(0, 1, 5, 2, 3, 4)
    out["quadrants"] = normalize(out["quadrants"]).permute(0, 1, 2, 5, 3, 4)
    out["fovea"] = normalize(out["fovea"]).permute(0, 1, 4, 2, 3)
    return out


PRESS_KINDS = [i for i, e in enumerate(VOCAB) if e and e["kind"] == "button" and e["down"]]

# Where the scripted player puts the pointer only so that it can read the screen
# (scripted.py, ai_games.py): the centre, before each search for a button (forming the
# army, selecting it, clearing its orders, executing, checking the arrow, reinforcing,
# pausing) and before each full zoom out (recentre); the top bar's blank middle, while it
# reads the map (overview); below the middle, before the space bar that starts the game
# (run_at). A lit alert or a tooltip under the pointer would spoil the search.
PARKING = ((0.5, 0.5), (0.65, 0.012), (0.5, 0.75))

# The camera's arrow-key pans. Until #90 (2026-09-24 13:10) the recorder's camera chose
# 30% of its pans at random, whatever the screen showed. A policy that learned them walked
# its camera off the top of the map in a live game and had never seen the way back
# (2026-09-24); since #90 the camera keeps the front in view.
ARROW_KEYS = (0x25, 0x26, 0x27, 0x28)


def camera_keys_dropped(manifest, camera_since):
    """The keys to leave out of a recording's labels for its camera: the arrows, when a
    script's recording (not a player's) was made before `camera_since` (unix seconds), or
    when it has no start time; else none."""
    if camera_since is None or manifest.get("source") == "human":
        return ()
    started = (manifest.get("recorder") or {}).get("started_unix")
    return () if started is not None and started >= camera_since else ARROW_KEYS


def parking_moves(events):
    """Indices of the moves in `events` (in time order) that only park the pointer.

    A move onto a PARKING point, with no button held, where nothing uses the position
    before the next move: no button press or release, and no wheel but a zoom out, which
    ends fully out wherever the pointer is. In the scripted games 10.7% of all moves went
    to the centre's one point, 1 in 18 of them while the army was formed, so the pointer
    learned to put about 60% of its mass there, and the setup's buttons were missed by 500
    to 1000 px (2026-09-24). A click on something at the centre keeps its move, and so
    does a zoom in, which closes in on what the pointer is on.
    """
    found, held = [], set()
    for i, item in enumerate(events):
        event = item["event"]
        if event["kind"] == "button":
            (held.add if event["down"] else held.discard)(event["button"])
            continue
        if event["kind"] != "move" or held:
            continue
        x, y = event["x"], event["y"]
        if not any(abs(x - px) < 1e-6 and abs(y - py) < 1e-6 for px, py in PARKING):
            continue
        used = False
        for later in events[i + 1 :]:
            kind = later["event"]["kind"]
            if kind == "move":
                break
            if kind == "button" or (kind == "wheel" and later["event"].get("delta", 0) > 0):
                used = True
                break
        if not used:
            found.append(i)
    return found


def setup_end(manifest, times, seconds=30.0):
    """When a game's setup ends, on the recording's clock (ns).

    The scripted player sets up while the game is paused (the army, its general, the front
    and the offensive) and then starts it; its `run` order is stamped with the frames
    recorded by then. A recording without one (a policy's) takes the first `seconds`.
    """
    for order in manifest.get("orders") or []:
        if order.get("order") == "run":
            return int(times[min(max(int(order["frame"]), 0), len(times) - 1)])
    return int(times[0] + seconds * 1e9)


def presses_after_move(actions):
    """Per decision, whether a button is pressed in a slot after a move."""
    kinds = actions[..., 0]
    moved = np.cumsum(kinds == 1, axis=-1) > 0
    # A slot's press counts only if a move came in an earlier slot.
    before = np.concatenate([np.zeros_like(moved[..., :1]), moved[..., :-1]], axis=-1)
    return (np.isin(kinds, PRESS_KINDS) & before).any(-1)


def player_outcome(manifest):
    """ "win" or "loss" for the recording's own player, or None when it names no result.

    The player is the one country the arena logged a human for (`players`), else the
    country the recorder started as. A game that ended without a surrender (a draw at
    the cap, `winner` "timeout") counts as not won: "loss".
    """
    players = manifest.get("players") or []
    player = players[0] if len(players) == 1 else manifest.get("started_as")
    winner = manifest.get("winner")
    if player not in ("BLU", "RED") or winner is None:
        return None
    return "win" if winner == player else "loss"


def session_labels(
    source,
    *,
    sources=("human",),
    clip_shift=0,
    detail_shift=0,
    idm_min_logp=None,
    idm_weight=1.0,
    advantage=False,
    look_before_click=False,
    lead_in=None,
    drop_keys=(),
    loser_weight=1.0,
    state=False,
    orders=False,
    press_weight=1.0,
    drop_parking=False,
    setup_weight=1.0,
    setup_seconds=30.0,
):
    """Everything about a recording except its pixels: times, pointer, actions per decision.

    Checked before any frame is decoded, so a recording that cannot train fails here
    rather than an hour into a run. `sources` are the manifest sources accepted: "human"
    recordings carry the player's own inputs; "ai" recordings carry the scripted camera's
    (see ai_games), which are real inputs but not a player's.

    The shifts, in decision intervals, move each decision's clip and detail views later
    than the decision itself. The policy uses none: it acts on the past. The inverse
    dynamics model looks ahead, at frames that already show what the input did. A decision
    whose shifted frames run past the end of the video is not valid for that reader.

    Labels the inverse dynamics model inferred are noisier than recorded ones, so they
    can count for less. D2E found its IDM's labels helped navigation but hurt precise
    manipulation (LIBERO, 96.6% to 92.2%; arXiv 2510.05684), and a click is precise.
    `idm_min_logp` marks invalid every inferred label whose log-likelihood under the IDM
    (summed over the slots; 0 is certain) falls below it, and `weight` gives each
    decision's share of the imitation loss: `idm_weight` for an inferred label, 1 for a
    recorded one. The defaults change nothing.

    `lead_in` is how many decision intervals of video come before the first decision
    (None: a whole clip and one interval more, CLIP_FRAMES + 1). An encoder that reads no
    clip (models.reads_clip) needs none, and the scripted player forms its army in the
    first 1-2.5 s of a game: inside the clip's 1.8 s lead-in, 23 of 31 games never showed
    that click as a label. `drop_keys` are key codes whose events are not the player's to
    learn: the harness presses them (space, which unpauses the game, when a learned
    policy plays), so they are left out of the labels rather than invalidating the
    decision. `loser_weight` scales every decision of a game the player did not win
    (player_outcome). `state` adds each decision's true state from the arena log
    (privileged.decision_states), and `orders` the scripted player's next order and the
    time until it (privileged.decision_orders), which the agent never sees: targets for
    training only.

    `drop_parking` leaves out the moves that only park the pointer so the scripted player
    can read the screen (parking_moves); the events around them keep their times, and
    the count is returned as `parking`. `setup_weight` scales every decision of the setup
    (setup_end: before the run order, else the first `setup_seconds`), the clicks that
    form the army and give it its general, front and offensive.
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
    if np.any(np.diff(times) <= 0):
        raise ValueError("Nonmonotonic timestamps")
    # Frames more than a second apart: over the network the second PC's capture stalls
    # now and then, for 1.1 to 1.5 s (six of its 44 speed-5 games, 2026-09-23). Only the
    # decisions whose frames would span the stall are lost, not the recording.
    gap = np.flatnonzero(np.diff(times) > MAX_GAP_NS)
    cursors = [parse_cursor(row.get("cursor")) for row in rows]
    key = "events" if manifest["source"] == "human" else "scripted_events"
    events = [e for row in rows for e in row.get(key, [])]
    tail = source / "trailing-events.json"
    if tail.exists() and key == "events":
        events += json.loads(tail.read_text())["events"]
    if drop_keys:
        dropped = set(drop_keys)
        events = [
            e
            for e in events
            if not (e["event"].get("kind") == "key" and e["event"].get("vk") in dropped)
        ]
    events.sort(key=lambda e: e["t_ns"])
    parked = parking_moves(events) if drop_parking else []
    if parked:
        skip = set(parked)
        events = [e for i, e in enumerate(events) if i not in skip]
    # The first decision needs a whole clip of recorded video behind it, plus one
    # interval of margin. The lead-in is derived from the clip rather than written down:
    # a hardcoded 2.1 s was correct only while a clip spanned 2.0 s.
    lead_in = (CLIP_FRAMES + 1 if lead_in is None else int(lead_in)) * PERIOD_NS
    if lead_in < 0:
        raise ValueError("the lead-in cannot be negative")
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
    if look_before_click:
        # A policy that looks before it clicks (models.ActionHead `look`) cannot press
        # after a move in the same decision, so such a decision cannot be its label.
        for i in np.flatnonzero(valid & presses_after_move(actions)):
            valid[i] = False
            excluded.append({"decision": int(i), "reason": "press after a move"})
    label_source = manifest["source"]
    weight = np.ones(len(decisions), dtype=np.float32)
    if inferred:
        stored = np.load(source / IDM_LABELS)
        if not np.array_equal(stored["decisions"], decisions):
            raise ValueError("IDM labels were made on a different decision grid; label again")
        actions, valid = stored["actions"].copy(), stored["valid"].copy()
        if idm_min_logp is not None:
            unsure = valid & (stored["logp"] < idm_min_logp)
            for d in np.flatnonzero(unsure):
                excluded.append({"decision": int(d), "reason": "IDM label below confidence"})
            valid &= ~unsure
        weight[:] = idm_weight
        label_source = "idm"
    # Offline RL (offline.py): with `advantage`, a player's recording that has been
    # weighed counts each decision by its advantage; one that has not counts as before.
    if advantage and (source / ADVANTAGE_LABELS).exists():
        stored = np.load(source / ADVANTAGE_LABELS)
        if not np.array_equal(stored["decisions"], decisions):
            raise ValueError(
                "Advantage weights were made on a different decision grid; weigh again"
            )
        weight = weight * stored["weight"]
    if loser_weight != 1.0 and player_outcome(manifest) == "loss":
        weight = weight * np.float32(loser_weight)
    if press_weight != 1.0:
        weight = weight * np.where(acting(actions), np.float32(press_weight), np.float32(1))
    if setup_weight != 1.0:
        setup = decisions < setup_end(manifest, times, setup_seconds)
        weight = weight * np.where(setup, np.float32(setup_weight), np.float32(1))
    # A recorded AI game names its winner. Every decision then has a return to predict:
    # the win (+1) or loss (-1) from Blue's side, the side the observer's view keeps,
    # discounted by the wall time left until the recording ends. It pre-trains the
    # critic (train-critic) before any self-play; other recordings have none (NaN).
    sign = {"BLU": 1.0, "RED": -1.0}.get(manifest.get("winner"))
    left = (times[-1] - decisions) / PERIOD_NS
    outcome = GAMMA**left * sign if sign is not None else np.full(len(decisions), np.nan)
    # Frames the reader needs that the video does not have.
    readable = decisions + max(clip_shift, detail_shift) * PERIOD_NS <= times[-1]
    valid &= readable
    # A decision reads frames from a clip before it to its interval (and any shift) after.
    first = decisions + min(0, clip_shift) * PERIOD_NS - lead_in
    last = decisions + (max(clip_shift, detail_shift) + 1) * PERIOD_NS
    for i in gap:
        spans = (first < times[i + 1]) & (last > times[i])
        for d in np.flatnonzero(spans & valid):
            excluded.append({"decision": int(d), "reason": "capture gap"})
        valid &= ~spans
    extra = {}
    if state:
        from .privileged import decision_states

        extra["state"] = decision_states(source, manifest, frame_ids)
    if orders:
        from .privileged import decision_orders

        extra["order_kind"], extra["order_eta"] = decision_orders(manifest, frame_ids)
    return {
        **extra,
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
        "outcome": outcome.astype(np.float32),
        "actions": actions,
        "valid": valid,
        "weight": weight,
        "excluded": excluded,
        "label_source": label_source,
        "parking": len(parked),
    }


def recording_splits(root):
    """The `splits.json` in a data folder, {recording folder name: split}, or {}."""
    path = Path(root) / "splits.json"
    if not path.exists():
        return {}
    chosen = json.loads(path.read_text())
    unknown = set(chosen.values()) - {"train", "validation", "test"}
    if unknown:
        raise ValueError(f"splits.json names unknown splits: {sorted(unknown)}")
    return chosen


def acting(actions, ahead=3):
    """Per decision, whether it presses a key or a button, or moves onto what the next
    `ahead` decisions press: the orders, as against waiting or moving the camera, which
    are over 97% of a scripted game's decisions."""
    kinds = actions[..., 0]
    press = np.isin(
        kinds,
        [i for i, e in enumerate(VOCAB) if e and e["kind"] in ("key", "button") and e["down"]],
    )
    pressed = press.any(-1)
    moved = (kinds == 1).any(-1)
    soon = np.zeros_like(pressed)
    for step in range(1, ahead + 1):
        soon[:-step] |= pressed[step:]
    return pressed | (moved & soon)


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
    """One recording decoded front to back, yielding its windows as they fill.

    Without `clips` no global view is made or kept, and a frame no decision reads the
    details of is decoded and dropped without computing any view of it.
    """

    def __init__(self, labels, length, burn_in, device, starts=None, clips=True):
        self.labels, self.length, self.burn_in, self.device = labels, length, burn_in, device
        self.clips = clips
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
        # The tower cache's arrays (tower_cache.py), opened here, in the process that
        # reads them: a memory map does not travel to a DataLoader worker.
        self.tower = None
        if labels.get("tower"):
            self.tower = tuple(
                np.load(labels["tower"][key], mmap_mode="r") for key in ("summary", "grid")
            )

    def close(self):
        if self.decoder.poll() is None:
            self.decoder.kill()
        self.decoder.wait()

    def _window(self, start):
        labels = self.labels
        n = self.length + self.burn_in
        steps = range(start, start + n)
        quads = torch.stack([self.details[d][0] for d in steps])
        fovea = torch.stack([self.details[d][1] for d in steps])
        actions = torch.from_numpy(labels["actions"][start : start + n].copy())
        previous = torch.zeros_like(actions)
        previous[1:] = actions[:-1]
        if start:
            previous[0] = torch.from_numpy(labels["actions"][start - 1].copy())
        window = {
            "quadrants": quads,
            "fovea": fovea,
            "actions": actions,
            "previous": previous,
            "valid": torch.from_numpy(labels["valid"][start : start + n].copy()),
            "weight": torch.from_numpy(labels["weight"][start : start + n].copy()),
            "speed": torch.full((n,), labels["speed"], dtype=torch.long),
            "outcome": torch.from_numpy(labels["outcome"][start : start + n].copy()),
            "start": start,
        }
        for key in ("state", "order_kind", "order_eta"):
            if key in labels:
                window[key] = torch.from_numpy(labels[key][start : start + n].copy())
        if self.tower is not None:
            from .tower_cache import from_bits

            frames = labels["frame_ids"][start : start + n]
            window["tower_summary"] = from_bits(self.tower[0][frames])
            window["tower_grid"] = from_bits(self.tower[1][frames])
        if self.clips:
            window["clips"] = torch.stack(
                [torch.stack([self.globals[int(i)] for i in labels["clip_ids"][d]]) for d in steps]
            )
        return window

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
        # No window can complete before the first pending one's clip begins.
        if i < int(labels["clip_ids"][first].min()) and not needed:
            return []
        if needed or self.clips:
            frame = np.frombuffer(buffer, np.uint8).reshape(self.h, self.w, 3)
            size = VIEW_SIZE if self.clips else None
            seen = views(frame, size, device=self.device, cursor=labels["cursors"][i])
            if self.clips:
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

    `idm_min_logp` and `idm_weight` pass to session_labels. Since a window is trained only
    when all its decisions are valid, one inferred label below the threshold drops the
    window that holds it.

    `device` is where the views are computed. On the CPU they are the same bytes as on
    CUDA (see `views`), and that is what lets a DataLoader's worker processes prepare
    windows while the GPU trains (`window_loader`). Each worker then plays its own share
    of the recordings, so every window is read exactly once, with its own share of the
    streams and of the shuffle buffer. `clips` False cuts windows without the global
    clips, for an encoder that does not read them (models.reads_clip).

    A `splits.json` in `root`, {recording folder name: split}, overrides the split each
    recording's manifest drew, so a study can choose its held-out games without touching
    the recordings. `lead_in`, `drop_keys`, `loser_weight`, `state`, `orders`,
    `press_weight`, `drop_parking` and `setup_weight` pass to session_labels; a lead-in
    shorter than a clip needs `clips` off. `tower`, a tower cache (tower_cache.py), adds
    each decision's frozen-tower reading to its window. `camera_since` (unix seconds)
    also drops the arrow keys from recordings made before it (camera_keys_dropped).
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
        idm_min_logp=None,
        idm_weight=1.0,
        clips=True,
        advantage=False,
        look_before_click=False,
        lead_in=None,
        drop_keys=(),
        loser_weight=1.0,
        state=False,
        orders=False,
        tower=None,
        press_weight=1.0,
        drop_parking=False,
        setup_weight=1.0,
        camera_since=None,
    ):
        if clips and lead_in is not None and lead_in < CLIP_FRAMES + 1:
            raise ValueError(
                f"a lead-in under {CLIP_FRAMES + 1} intervals leaves the first clips "
                "without frames; it is for an encoder that reads no clip"
            )
        self.length, self.burn_in, self.clips = length, burn_in, clips
        self.streams, self.shuffle, self.seed = streams, shuffle, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sessions = []
        overrides = recording_splits(root)
        for path in sorted(Path(root).glob("*/manifest.json")):
            meta = json.loads(path.read_text())
            labelled = "idm" in sources and (path.parent / IDM_LABELS).exists()
            chosen = overrides.get(path.parent.name, meta.get("split"))
            if chosen != split or not (meta.get("source") in sources or labelled):
                continue
            if not meta.get("complete"):
                continue
            self.sessions.append(
                session_labels(
                    path.parent,
                    sources=sources,
                    clip_shift=clip_shift,
                    detail_shift=detail_shift,
                    idm_min_logp=idm_min_logp,
                    idm_weight=idm_weight,
                    advantage=advantage,
                    look_before_click=look_before_click,
                    lead_in=lead_in,
                    drop_keys=tuple(drop_keys) + camera_keys_dropped(meta, camera_since),
                    loser_weight=loser_weight,
                    state=state,
                    orders=orders,
                    press_weight=press_weight,
                    drop_parking=drop_parking,
                    setup_weight=setup_weight,
                )
            )
        self.tower_stamp = None
        if tower is not None:
            from .tower_cache import tower_paths

            stamps = set()
            for labels in self.sessions:
                paths = tower_paths(tower, labels["root"])
                if paths is None:
                    raise ValueError(
                        f"{labels['root'].name} is not in the tower cache {tower}; "
                        "run cache-tower on this data first"
                    )
                labels["tower"] = paths
                stamps.add(paths["tower"])
            if len(stamps) > 1:
                raise ValueError("the tower cache mixes the readings of different towers")
            self.tower_stamp = stamps.pop() if stamps else None
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
        streams, shuffle = self.streams, self.shuffle
        worker = get_worker_info()
        if worker is not None and worker.num_workers > 1:
            # Every worker shuffles the same order and takes every n-th recording of it.
            share = worker.num_workers
            order = order[worker.id :: share]
            rng = random.Random(f"{self.seed}:{self.epoch - 1}:{worker.id}")
            streams, shuffle = -(-streams // share), -(-shuffle // share)
        active, buffer = [], []
        try:
            while order or active:
                while order and len(active) < streams:
                    active.append(
                        _Stream(
                            order.pop(), self.length, self.burn_in, self.device, clips=self.clips
                        )
                    )
                for stream in list(active):
                    done = stream.advance()
                    if done is None:
                        stream.close()
                        active.remove(stream)
                        continue
                    buffer.extend(done)
                    while len(buffer) > shuffle:
                        yield buffer.pop(rng.randrange(len(buffer)))
            rng.shuffle(buffer)
            yield from buffer
        finally:
            for stream in active:
                stream.close()


class GameSequences(IterableDataset):
    """Each game's decisions in order, in windows of `length`, `slots` games side by side.

    For a memory carried through whole games, trained by truncated backpropagation
    through time: batch slot b plays one recording from its first decision on, window
    after window; when the recording ends, the slot starts the next one and marks that
    window `fresh`, so the trainer empties that slot's memory. The memory study
    (2026-09-24, 44 AI games, 5 seeds) found carrying the memory between 16-decision
    windows the largest effect it measured: held-out loss 3.109 against 3.169 for windows
    started empty, which is how train-bc trained.

    Every decision is in a window, valid or not: an invalid one is still seen by the
    memory, and train_bc scores it with weight 0. A game's last partial window is left
    out. Items are whole batches, with `slot` (unique across DataLoader workers) and
    `fresh`; `sessions` is a VideoSessions, for its recordings and their labels.
    """

    def __init__(self, sessions, length=16, slots=2, *, seed=0, device="cpu", clips=True):
        self.sessions = sessions.sessions
        self.length, self.slots, self.seed = length, slots, seed
        self.device, self.clips = device, clips
        self.windows = sum(
            max(0, int(labels["readable"].sum())) // length for labels in self.sessions
        )
        self.epoch = 0

    def __len__(self):
        """Batches in an epoch, about: windows over slots."""
        return -(-self.windows // self.slots)

    def _open(self, labels):
        count = int(labels["readable"].sum())
        starts = list(range(0, count - self.length + 1, self.length))
        if not starts:
            return None
        return _Stream(labels, self.length, 0, self.device, starts=starts, clips=self.clips)

    def __iter__(self):
        rng = random.Random(f"{self.seed}:{self.epoch}")
        self.epoch += 1
        order = list(self.sessions)
        rng.shuffle(order)
        worker = get_worker_info()
        base = 0
        if worker is not None and worker.num_workers > 1:
            order = order[worker.id :: worker.num_workers]
            base = worker.id * self.slots
        streams = [None] * self.slots
        fresh = [True] * self.slots
        # Windows a stream finished together (after a capture gap), in order.
        pending = [[] for _ in range(self.slots)]
        try:
            while True:
                windows, ids, starts = [], [], []
                for b in range(self.slots):
                    while not pending[b]:
                        if streams[b] is None:
                            if not order:
                                break
                            streams[b], fresh[b] = self._open(order.pop()), True
                            continue
                        done = streams[b].advance()
                        if done is None:
                            streams[b] = None
                            continue
                        pending[b].extend(done)
                    if not pending[b]:
                        continue
                    window = pending[b].pop(0)
                    windows.append(window)
                    ids.append(base + b)
                    starts.append(fresh[b])
                    fresh[b] = False
                if not windows:
                    return
                batch = torch.utils.data.default_collate(windows)
                batch["slot"] = torch.tensor(ids)
                batch["fresh"] = torch.tensor(starts)
                yield batch
        finally:
            for stream in streams:
                if stream is not None:
                    stream.close()


def window_loader(dataset, batch_size, *, workers=0, device="cpu", drop_last=False):
    """Batches of VideoSessions' windows, prepared by `workers` background processes.

    Measured on this PC (2026-09-23): decoding 1080p x264 4:4:4 takes 5.9 ms a frame and
    the views of one 14.8 ms on the CPU, for about 34 frames a training step, all of it
    on the training thread when `workers` is 0. The dataset must then compute its views
    on the CPU. Batches are pinned for the card, so their copy to it does not wait.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=bool(workers) and torch.device(device).type == "cuda",
        worker_init_fn=_worker_threads if workers else None,
        persistent_workers=bool(workers),
    )


def _worker_threads(_worker_id, threads=2):
    """More than the one thread a DataLoader worker is given: with one, the views of a
    frame took 46 ms (2026-09-24)."""
    torch.set_num_threads(threads)

"""Where the policy wants to point, drawn as a heat map over the screen.

The action head places the pointer by scoring every one of the screen's 32x32 cells
against what it wants, then a position inside the chosen cell (models.ActionHead). That
distribution is a heat map, and this draws it over the recorded frame: hot where the
policy would put the pointer, cold elsewhere, with the demonstrated move marked. The
policy runs through the recording with its memory carried from the start, as it plays.

Two numbers summarise a set of maps: how far the hottest point lies from the
demonstrated one, in pixels, and how often the demonstrated point falls inside the
hottest 1% of the screen.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from .actions import GRID
from .dataset import _Stream, batch_to_device, cover_starts, session_labels
from .models import PRESSES, reads_clip

# Cold to hot, the way a heat camera shows it: dark blue, purple, red, orange, yellow.
RAMP = np.array(
    [[20, 20, 80], [110, 30, 150], [210, 40, 90], [250, 120, 30], [255, 240, 120]], np.float32
)


def colour(heat):
    """RGB for heat values in [0, 1]."""
    position = np.clip(heat, 0, 1) * (len(RAMP) - 1)
    low = np.floor(position).astype(int).clip(0, len(RAMP) - 2)
    frac = (position - low)[..., None]
    return RAMP[low] * (1 - frac) + RAMP[low + 1] * frac


def first_move(actions):
    """The slot of a decision's first move, or None."""
    moves = np.flatnonzero(actions[:, 0] == 1)
    return int(moves[0]) if len(moves) else None


def target_pixel(actions, slot, width, height):
    """The demonstrated move of `slot`, in pixels."""
    x, y = actions[slot, 1], actions[slot, 2]
    return x / (GRID - 1) * (width - 1), y / (GRID - 1) * (height - 1)


@torch.no_grad()
def heat_maps(policy, labels, device, decisions, window=64):
    """(decision, slot, kind probabilities, (GRID, GRID) map) for each of `decisions`.

    The slot is the decision's first demonstrated move, with the slots before it
    teacher-forced, or slot 0 when the decision has no move.
    """
    wanted = set(int(d) for d in decisions)
    last = max(wanted)
    clips = reads_clip(policy.encoder)
    stream = _Stream(labels, window, 0, device, starts=cover_starts(labels, window), clips=clips)
    # On the CPU too: load_policy keeps the frozen blocks in bfloat16 on any device.
    autocast = {"device_type": device, "dtype": torch.bfloat16}
    hidden, done_until, found = None, 0, []
    try:
        while (done := stream.advance()) is not None and done_until <= last:
            for piece in done:
                start = piece.pop("start")
                batch = batch_to_device(default_collate([piece]), device)
                with torch.autocast(**autocast):
                    summary, cells, centre = policy.perceive_window(
                        batch.get("clips"), batch["quadrants"], batch["fovea"]
                    )
                    if hidden is None:
                        hidden = summary.new_zeros(1, policy.memory_dim)
                    for t in range(summary.shape[1]):
                        if start + t < done_until:
                            continue
                        hidden, _ = policy.recall(
                            summary[:, t], cells[:, t], centre[:, t], batch["previous"][:, t],
                            batch["speed"][:, t], hidden, batch["quadrants"].dtype,
                        )  # fmt: skip
                        d = start + t
                        if d in wanted:
                            actions = batch["actions"][:, t]
                            slot = first_move(labels["actions"][d]) or 0
                            kind_p, grid = policy.actor.pointer_map(
                                hidden, cells[:, t], actions, slot
                            )
                            found.append((d, slot, kind_p.cpu().numpy(), grid.cpu().numpy()))
                done_until = max(done_until, start + summary.shape[1])
    finally:
        stream.close()
    return found


def frames_at(video, indices, folder):
    """Decode the frames at `indices` of `video` into `folder`, as {index: path}."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is needed to read the recording's frames")
    folder.mkdir(parents=True, exist_ok=True)
    order = sorted(set(indices))
    select = "+".join(f"eq(n\\,{i})" for i in order)
    pattern = folder / "frame-%04d.png"
    subprocess.run(
        [ffmpeg, "-loglevel", "error", "-y", "-i", str(video), "-vf", f"select={select}",
         "-vsync", "0", str(pattern)],
        check=True,
    )  # fmt: skip
    return {index: folder / f"frame-{k + 1:04d}.png" for k, index in enumerate(order)}


def render(rgb, grid, kind_p, target=None):
    """The frame with the heat map laid over it and the demonstrated move marked."""
    import cv2

    height, width = rgb.shape[:2]
    # Averaged when shrinking, so a narrow peak is not stepped over; interpolated when not.
    shrink = width < grid.shape[1] or height < grid.shape[0]
    method = cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
    heat = cv2.resize(grid.astype(np.float32), (width, height), interpolation=method)
    # Normalised to its peak, with a gamma that keeps the warm margin of a narrow peak
    # visible: most of the mass of a sure policy sits in a few hundred pixels.
    heat = (heat / max(float(heat.max()), 1e-12)) ** 0.5
    alpha = np.clip(heat * 1.2, 0, 0.85)[..., None]
    out = (rgb.astype(np.float32) * (1 - alpha) + colour(heat) * alpha).astype(np.uint8)
    hot_y, hot_x = np.unravel_index(int(np.argmax(heat)), heat.shape)
    cv2.drawMarker(out, (int(hot_x), int(hot_y)), (255, 255, 255), cv2.MARKER_CROSS, 22, 2)
    if target is not None:
        cv2.circle(out, (round(target[0]), round(target[1])), 14, (0, 255, 255), 2)
    move, press = float(kind_p[1]), float(kind_p[PRESSES].sum())
    cell = float(grid.reshape(32, 32, 32, 32).sum((1, 3)).max())
    text = f"move {move:.2f}  press {press:.2f}  hottest cell {cell:.2f}"
    cv2.putText(out, text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    return out


def hot_share(grid, target_xy):
    """Whether the lattice point nearest `target_xy` is in the hottest 1% of the lattice."""
    x, y = (round(v) for v in target_xy)
    return bool(grid[y, x] >= np.quantile(grid, 0.99))


def aimed(actions, decision, ahead=3):
    """Whether a decision's first move is aimed: a button is pressed where it lands.

    The press may come later in the same decision or in the next `ahead` decisions, as
    long as no other move comes first. The scripted player moves onto a button and looks
    for 0.25 to 0.45 s before pressing it; its camera's pointing about does not press.
    """
    slot = first_move(actions[decision])
    if slot is None:
        return False
    rows = [actions[decision][slot + 1 :]]
    rows += [actions[d] for d in range(decision + 1, min(decision + 1 + ahead, len(actions)))]
    for kind in np.concatenate(rows)[:, 0]:
        if kind == 1:
            return False
        if kind in PRESSES:
            return True
    return False


def draw_heatmaps(
    checkpoint, recording, output, *, decisions=None, count=24, model_path=None, targets="moves"
):
    """Heat maps for `decisions` of a recording (default: `count` spread over its moves).

    `targets` "clicks" spreads them over the aimed moves only (`aimed`): where the
    pointer went to press something, the pointing that decides a game, rather than the
    camera's looking about. The recording is cut into decisions as the checkpoint's
    training cut them (its lead-in and dropped keys).
    """
    from PIL import Image

    from .runner import load_policy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    recording, output = Path(recording), Path(output)
    manifest = json.loads((recording / "manifest.json").read_text())
    config = json.loads(Path(checkpoint).with_suffix(".json").read_text())["config"]
    labels = session_labels(
        recording,
        sources=(manifest["source"],),
        lead_in=config.get("lead_in"),
        drop_keys=tuple(config.get("drop_keys") or ()),
    )
    moves = [
        d
        for d in np.flatnonzero(labels["valid"] & labels["readable"])
        if first_move(labels["actions"][d]) is not None
        and (targets == "moves" or aimed(labels["actions"], d))
    ]
    if decisions is None:
        if not moves:
            raise ValueError("the recording has no demonstrated moves to compare with")
        spread = np.linspace(0, len(moves) - 1, min(count, len(moves))).astype(int)
        decisions = [moves[i] for i in spread]
    policy, config, digest = load_policy(checkpoint, model_path, device)
    policy.eval()
    maps = heat_maps(policy, labels, device, decisions)
    frames = frames_at(
        recording / "screen.mkv", [int(labels["frame_ids"][d]) for d, *_ in maps], output / "frames"
    )
    rows, distances, inside = [], [], []
    for d, slot, kind_p, grid in maps:
        rgb = np.asarray(Image.open(frames[int(labels["frame_ids"][d])]).convert("RGB"))
        height, width = rgb.shape[:2]
        demonstrated = labels["actions"][d][slot][0] == 1
        target = target_pixel(labels["actions"][d], slot, width, height) if demonstrated else None
        Image.fromarray(render(rgb, grid, kind_p, target)).save(output / f"decision-{d:05d}.png")
        row = {"decision": int(d), "slot": slot, "move": float(kind_p[1])}
        if target is not None:
            hot_y, hot_x = np.unravel_index(int(np.argmax(grid)), grid.shape)
            hot = (hot_x / (GRID - 1) * (width - 1), hot_y / (GRID - 1) * (height - 1))
            row["miss_px"] = float(np.hypot(hot[0] - target[0], hot[1] - target[1]))
            lattice = labels["actions"][d][slot][1:]
            row["in_hottest_1pct"] = hot_share(grid, (lattice[0], lattice[1]))
            distances.append(row["miss_px"])
            inside.append(row["in_hottest_1pct"])
        rows.append(row)
    shutil.rmtree(output / "frames", ignore_errors=True)
    summary = {
        "checkpoint": digest,
        "recording": recording.name,
        "maps": len(rows),
        "median_miss_px": float(np.median(distances)) if distances else None,
        "in_hottest_1pct": float(np.mean(inside)) if inside else None,
    }
    (output / "heatmaps.json").write_text(json.dumps({"summary": summary, "maps": rows}, indent=2))
    return summary

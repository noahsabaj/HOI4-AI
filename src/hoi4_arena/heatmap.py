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
from .dataset import _Stream, batch_to_device, camera_keys_dropped, cover_starts, session_labels
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
    teacher-forced, or slot 0 when the decision has no move. `decisions` may instead map
    each decision to the slot to draw.
    """
    slots = dict(decisions) if isinstance(decisions, dict) else {}
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
                    tower = None
                    if "tower_grid" in batch:
                        tower = (batch["tower_summary"], batch["tower_grid"])
                    summary, cells, centre = policy.perceive_window(
                        batch.get("clips"), batch["quadrants"], batch["fovea"], tower=tower
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
                            slot = slots.get(d, first_move(labels["actions"][d]) or 0)
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


def trained_labels(config, recording, manifest):
    """A recording's labels cut as a checkpoint's training cut them (`config`: its lead-in,
    dropped keys, the old camera's keys and parking moves), with its frozen tower's cache
    when it has one."""
    keys = tuple(config.get("drop_keys") or ())
    labels = session_labels(
        recording,
        sources=(manifest["source"],),
        lead_in=config.get("lead_in"),
        drop_keys=keys + camera_keys_dropped(manifest, config.get("camera_since")),
        drop_parking=bool(config.get("drop_parking")),
    )
    if config.get("tower_cache"):
        # A policy trained on the frozen tower's cache reads it here too, when the
        # recording is in it: the same numbers, without running the tower.
        from .tower_cache import tower_paths

        found = tower_paths(config["tower_cache"], recording)
        if found is not None:
            labels["tower"] = found
    return labels


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
    labels = trained_labels(config, recording, manifest)
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


# The setup's first clicks, at the places the second PC's interface always shows them
# (play.MILESTONES), and its first front line: the pointing that must work before the
# army forms and takes a plan. No learned policy made the first click live (2026-09-24).
SETUP_TARGETS = ("alert", "plus", "portrait", "front")
FRONT_KEY = 0x5A


def setup_targets(actions, width, height, within=15):
    """{name: (decision, slot, (x, y) pixels)}: the move before the first left press on
    the unassigned-divisions alert, the create-army +, the commander portrait, and the
    first front-line click (a left press within `within` decisions after Z), from a
    recording's labels (N, SLOTS, 3). A target the recording never clicked is absent.
    """
    from .actions import VOCAB
    from .play import MILESTONES

    front_key = VOCAB.index({"kind": "key", "vk": FRONT_KEY, "down": True})
    found, last_move, key_at = {}, None, None
    for d in range(len(actions)):
        for slot in range(actions.shape[1]):
            kind = int(actions[d, slot, 0])
            if kind == 1:
                last_move = (d, slot, target_pixel(actions[d], slot, width, height))
            elif kind == front_key:
                key_at = d
            elif kind in PRESSES and VOCAB[kind]["button"] == 0 and last_move is not None:
                x, y = last_move[2]
                for name in SETUP_TARGETS[:3]:
                    (cx, cy), (dx, dy) = MILESTONES[name]
                    if name not in found and abs(x - cx) <= dx and abs(y - cy) <= dy:
                        found[name] = last_move
                if "front" not in found and key_at is not None and d - key_at <= within:
                    found["front"] = last_move
        if len(found) == len(SETUP_TARGETS):
            break
    return found


def near_mass(grid, target, width, height, radius):
    """The pointer's probability of landing within `radius` px of `target` (pixels)."""
    lattice = np.arange(GRID) / (GRID - 1)
    dx = (lattice * (width - 1) - target[0]) ** 2
    dy = (lattice * (height - 1) - target[1]) ** 2
    return float(grid[(dy[:, None] + dx[None, :]) <= radius * radius].sum())


def setup_pointing(checkpoint, recordings, *, model_path=None, radius=30.0):
    """How well a policy points at the setup's targets (setup_targets) in each recording.

    The policy plays each recording from its start with its memory carried, as it plays
    live, and at the scripted player's move onto each target gives its pointer map. Per
    target: `miss_px`, from the map's hottest point to the target; `near`, the chance a
    sampled move lands within `radius` px; `move`, the chance the slot moves at all. The
    summary gives each target's median miss and mean chances over the recordings, and
    `miss_px` the median over every target in every recording.
    """
    from .runner import load_policy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = json.loads(Path(checkpoint).with_suffix(".json").read_text())["config"]
    policy, _, digest = load_policy(checkpoint, model_path, device)
    policy.eval()
    games, rows = [], []
    for recording in recordings:
        recording = Path(recording)
        manifest = json.loads((recording / "manifest.json").read_text())
        labels = trained_labels(config, recording, manifest)
        width, height = manifest["width"], manifest["height"]
        targets = setup_targets(labels["actions"], width, height)
        wanted = {d: slot for d, slot, _ in targets.values() if labels["valid"][d]}
        maps = (
            {d: (kind_p, grid) for d, _, kind_p, grid in heat_maps(policy, labels, device, wanted)}
            if wanted
            else {}
        )
        game = {"recording": recording.name}
        for name, (d, slot, target) in targets.items():
            if d not in maps:
                continue
            kind_p, grid = maps[d]
            hot_y, hot_x = np.unravel_index(int(np.argmax(grid)), grid.shape)
            hot = (hot_x / (GRID - 1) * (width - 1), hot_y / (GRID - 1) * (height - 1))
            row = {
                "decision": int(d),
                "miss_px": round(float(np.hypot(hot[0] - target[0], hot[1] - target[1])), 1),
                "near": round(near_mass(grid, target, width, height, radius), 4),
                "move": round(float(kind_p[1]), 4),
            }
            game[name] = row
            rows.append((name, row))
        games.append(game)
    summary = {"checkpoint": digest, "games": len(games), "radius_px": radius}
    for name in SETUP_TARGETS:
        mine = [row for n, row in rows if n == name]
        if mine:
            summary[name] = {
                "miss_px": float(np.median([r["miss_px"] for r in mine])),
                "near": float(np.mean([r["near"] for r in mine])),
                "move": float(np.mean([r["move"] for r in mine])),
                "count": len(mine),
            }
    if rows:
        summary["miss_px"] = float(np.median([row["miss_px"] for _, row in rows]))
        summary["near"] = float(np.mean([row["near"] for _, row in rows]))
    return {"summary": summary, "games": games}

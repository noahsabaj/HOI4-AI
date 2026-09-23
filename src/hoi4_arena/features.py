"""Memory trained on long windows over cached perception, and the cells compared on it.

A decision is 200 ms. The policy trained on 16 of them after 2 of burn-in from an empty
memory, so it never learned to use more than 3.6 s of the past, and never saw a memory
older than that, though in a game it carries one for the whole match. Training on long
windows through the whole policy is out of reach: the vision tower runs at every step.
So the perception is frozen after behaviour cloning and what it reads at each decision
is cached (`cache_features`); the memory, the action head and the layers that feed the
memory (models.fuse) are then trained on that cache (`train_memory`), where a window of
hundreds of decisions costs a few milliseconds a step.

Every cell in memory.py is trained the same way, at the same budget: the same decisions
per update and the same number of passes. What each is judged on, on held-out games:

- the imitation loss, with the memory carried from the start of each game, which is
  how the policy runs, and with it cleared every 18 decisions, which is how it trained;
- linear probes of what the memory holds that the screen does not show: the camera's
  zoom (counted from its wheel notches), how long since it last zoomed fully out (it
  does every 20 to 60 s, 100 to 300 decisions), where the pointer was 5, 25 and 100
  decisions ago, and which side goes on to win.

The probes read the memory's output with a ridge regression fitted on training games.
The cell with no memory is the floor: whatever it scores came from the screen alone.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import default_collate

from .actions import SLOTS
from .dataset import _Stream, batch_to_device, cover_starts, session_labels
from .memory import build_memory, detach, reset, state_size
from .models import CELL_DIM, CELLS, SPEEDS, ActionHead, fuse, reads_clip

# How far back the pointer probes look, in decisions.
POINTER_LAGS = (5, 25, 100)
# The scripted camera's deepest zoom, in wheel notches (ai_games.ZOOM_MAX), and how many
# notches out in a row mark its recentre: it zooms out by ZOOM_MAX + 4 at once.
ZOOM_MAX = 26
RECENTRE_RUN = 20


def camera_targets(root, decisions):
    """The scripted camera's zoom and seconds since its last recentre, at each decision.

    Rebuilt from its wheel events: each notch moves the zoom one step, clamped as the game
    clamps it, and a run of RECENTRE_RUN notches out marks a recentre. NaN before the
    first recentre, whose time is unknown.
    """
    rows = [json.loads(line) for line in (Path(root) / "frames.jsonl").read_text().splitlines()]
    events = sorted(
        (e for row in rows for e in row.get("scripted_events", [])), key=lambda e: e["t_ns"]
    )
    wheel = [
        (e["t_ns"], 1 if e["event"]["delta"] > 0 else -1)
        for e in events
        if e["event"].get("kind") == "wheel"
    ]
    times = np.array([t for t, _ in wheel], np.int64)
    zoom_after, level = [], 0
    recentres, run, run_start = [], 0, None
    for t, step in wheel:
        level = min(ZOOM_MAX, max(0, level + step))
        zoom_after.append(level)
        if step < 0:
            run_start = t if run == 0 else run_start
            run += 1
            if run == RECENTRE_RUN:
                recentres.append(run_start)
        else:
            run = 0
    index = np.searchsorted(times, decisions, side="left") - 1
    zoom = np.where(index >= 0, np.array(zoom_after + [0])[index], 0).astype(np.float32)
    recentres = np.array(recentres, np.int64)
    last = np.searchsorted(recentres, decisions, side="right") - 1
    since = np.where(last >= 0, (decisions - recentres[np.maximum(last, 0)]) / 1e9, np.nan).astype(
        np.float32
    )
    return zoom, since


@torch.no_grad()
def cache_features(data, checkpoint, output, *, model_path=None, sources=("ai",), device=None):
    """What a behaviour-cloned policy's frozen perception reads, at every decision.

    For each complete recording from `sources`, in every split, writes to
    output/<recording>/: `summary` (N, encoder dim), `cells` (N, 1024, 256), `centre`
    (N, 256), all float16, and `labels.npz` with the actions, validity, outcome, speed,
    the pointer, the camera's zoom and time since its recentre. N counts the decisions
    whose frames exist. About 0.5 MB a decision, nearly all of it cells.
    """
    from .runner import load_policy

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(output)
    policy, config, digest = load_policy(checkpoint, model_path, device)
    policy.eval()
    autocast = {"device_type": device, "dtype": torch.bfloat16, "enabled": device == "cuda"}
    written = []
    for manifest_path in sorted(Path(data).glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("complete") or manifest.get("source") not in sources:
            continue
        root = manifest_path.parent
        target = output / root.name
        if (target / "labels.npz").exists():
            continue
        labels = session_labels(root, sources=sources)
        count = int(labels["readable"].sum())
        window = 16
        starts = cover_starts(labels, window)
        if not starts:
            continue
        target.mkdir(parents=True, exist_ok=True)
        summary = np.zeros((count, policy.encoder.dim), np.float16)
        centre = np.zeros((count, CELL_DIM), np.float16)
        cells = np.lib.format.open_memmap(
            target / "cells.npy", "w+", np.float16, (count, CELLS * CELLS, CELL_DIM)
        )
        stream = _Stream(labels, window, 0, device, starts=starts, clips=reads_clip(policy.encoder))
        began = time.monotonic()
        try:
            while (done := stream.advance()) is not None:
                for piece in done:
                    start = piece.pop("start")
                    batch = batch_to_device(default_collate([piece]), device)
                    clips = batch.get("clips")
                    with torch.autocast(**autocast):
                        seen = policy.perceive(
                            None if clips is None else clips[0],
                            batch["quadrants"][0],
                            batch["fovea"][0],
                        )
                    s, c, f = (x.float().cpu().numpy().astype(np.float16) for x in seen)
                    summary[start : start + window] = s
                    cells[start : start + window] = c
                    centre[start : start + window] = f
        finally:
            stream.close()
        cells.flush()
        del cells
        np.save(target / "summary.npy", summary)
        np.save(target / "centre.npy", centre)
        decisions = labels["decisions"][:count]
        frame_ids = labels["frame_ids"][:count]
        pointer = np.array([labels["cursors"][i] for i in frame_ids], np.float32)
        pointer /= np.array([manifest["width"], manifest["height"]], np.float32)
        zoom, since = camera_targets(root, decisions)
        np.savez(
            target / "labels.npz",
            actions=labels["actions"][:count],
            valid=labels["valid"][:count],
            outcome=labels["outcome"][:count],
            speed=np.full(count, labels["speed"], np.int64),
            pointer=pointer,
            zoom=zoom,
            since_recentre=since,
            winner=np.float32({"BLU": 1.0, "RED": -1.0}.get(manifest.get("winner"), np.nan)),
        )
        meta = {
            "recording": str(root.resolve()),
            "split": manifest["split"],
            "decisions": count,
            "checkpoint": digest,
            "seconds": round(time.monotonic() - began, 1),
        }
        (target / "meta.json").write_text(json.dumps(meta, indent=2))
        written.append(meta)
    return written


class CachedGame:
    """One recording's cached features, the cells read from disk only when used."""

    def __init__(self, root):
        root = Path(root)
        self.name = root.name
        self.meta = json.loads((root / "meta.json").read_text())
        self.summary = np.load(root / "summary.npy")
        self.centre = np.load(root / "centre.npy")
        self.cells = np.load(root / "cells.npy", mmap_mode="r")
        labels = np.load(root / "labels.npz")
        self.labels = {key: labels[key] for key in labels.files}
        self.length = len(self.summary)

    def piece(self, start, length):
        """Steps [start, start + length), padded past the end with invalid steps."""
        stop = min(self.length, start + length)
        n = stop - start

        def padded(array, dtype=None):
            out = np.zeros((length, *array.shape[1:]), dtype or array.dtype)
            out[:n] = array[start:stop]
            return torch.from_numpy(out)

        actions = self.labels["actions"]
        previous = np.zeros_like(actions[start:stop])
        previous[1:] = actions[start : stop - 1]
        if start:
            previous[0] = actions[start - 1]
        valid = np.zeros(length, bool)
        valid[:n] = self.labels["valid"][start:stop]
        prev = np.zeros((length, SLOTS, 3), np.int64)
        prev[:n] = previous
        return {
            "summary": padded(self.summary),
            "cells": padded(self.cells),
            "centre": padded(self.centre),
            "actions": padded(actions),
            "previous": torch.from_numpy(prev),
            "speed": padded(self.labels["speed"]),
            "valid": torch.from_numpy(valid),
        }


def load_cache(cache, split):
    games = [
        CachedGame(path.parent)
        for path in sorted(Path(cache).glob("*/meta.json"))
        if json.loads(path.read_text())["split"] == split
    ]
    if not games:
        raise ValueError(f"No cached {split} games in {cache}")
    return games


class MemoryHead(nn.Module):
    """Everything of a Policy after its perception: the layers models.fuse uses, the
    memory, the action head and the value. The names are Policy's, so a GRU trained here
    loads into a Policy with the perception it was cached from."""

    def __init__(self, summary_dim, memory="gru", memory_dim=512):
        super().__init__()
        self.previous_action = nn.Linear(SLOTS * 3, 64)
        self.speed = nn.Embedding(SPEEDS, 32)
        self.read = nn.Linear(memory_dim, CELL_DIM)
        self.fusion = nn.Linear(summary_dim + 3 * CELL_DIM + 64 + 32, memory_dim)
        self.memory = build_memory(memory, memory_dim)
        self.actor = ActionHead(memory_dim)
        self.value = nn.Linear(memory_dim, 1)
        self.memory_dim = memory_dim

    def initial(self, batch, device):
        return torch.zeros(batch, self.memory_dim, device=device), self.memory.initial(
            batch, device
        )

    def unroll(self, batch, out, state, fresh, grad_from=0):
        """Step through a batch of pieces. `fresh` marks samples starting a game here.

        Steps before `grad_from` only warm the memory. Returns the outputs of the rest,
        (B, T', memory_dim), and the carried (out, state).
        """
        out = out * (~fresh)[:, None].to(out.dtype)
        state = reset(state, fresh)
        outs = []
        for t in range(batch["summary"].shape[1]):
            with torch.set_grad_enabled(torch.is_grad_enabled() and t >= grad_from):
                merged = fuse(
                    self,
                    batch["summary"][:, t],
                    batch["cells"][:, t],
                    batch["centre"][:, t],
                    batch["previous"][:, t],
                    batch["speed"][:, t],
                    out,
                )
                out, state = self.memory(merged, state)
            if t < grad_from:
                out, state = out.detach(), detach(state)
            else:
                outs.append(out)
        return torch.stack(outs, 1), (out, state)


def _to(batch, device):
    """A batch on `device`; the features in bfloat16 on the GPU, where autocast runs."""
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    for key in ("summary", "cells", "centre"):
        batch[key] = batch[key].to(torch.bfloat16 if device == "cuda" else torch.float32)
    return batch


def _nll(head, outs, batch, first=0):
    """Per-step negative log-likelihood of the demonstrated actions, and the valid mask."""
    actions = batch["actions"][:, first:]
    cells = batch["cells"][:, first:]
    valid = batch["valid"][:, first:]
    flat = valid.flatten()
    if not flat.any():
        return outs.sum() * 0, valid
    _, logp, _ = head.actor(
        outs.flatten(0, 1)[flat], cells.flatten(0, 1)[flat], actions.flatten(0, 1)[flat]
    )
    return -logp, valid


def _carried_batches(games, window, streams, rng):
    """Consecutive windows of whole games, `streams` games side by side.

    Each stream plays its games front to back, so the memory entering a window has seen
    everything before it in that game; `fresh` marks a stream's first window of a game.
    """
    order = list(games)
    rng.shuffle(order)
    queues = [order[i::streams] for i in range(streams)]
    positions = [[g, 0] for g in (q.pop(0) if q else None for q in queues)]
    while any(game is not None for game, _ in positions):
        pieces, fresh = [], []
        for i, (game, start) in enumerate(positions):
            if game is None:
                pieces.append(None)
                fresh.append(True)
                continue
            pieces.append(game.piece(start, window))
            fresh.append(start == 0)
            start += window
            if start >= game.length:
                game, start = (queues[i].pop(0) if queues[i] else None), 0
            positions[i] = [game, start]
        template = next(p for p in pieces if p is not None)
        pieces = [p or {k: torch.zeros_like(v) for k, v in template.items()} for p in pieces]
        yield default_collate(pieces), torch.tensor(fresh)


def _reset_batches(games, window, burn_in, batch_size, rng):
    """Windows cut anywhere, each from an empty memory after `burn_in` steps: the old way."""
    span = window + burn_in
    cuts = [(g, s) for g in games for s in range(0, g.length - span + 1, window)]
    rng.shuffle(cuts)
    for i in range(0, len(cuts), batch_size):
        group = cuts[i : i + batch_size]
        yield (
            default_collate([g.piece(s, span) for g, s in group]),
            torch.ones(len(group), dtype=torch.bool),
        )


@torch.no_grad()
def evaluate(head, games, device, clear_every=None):
    """Imitation loss over whole held-out games, and each decision's memory output.

    The memory is carried from each game's start, or cleared every `clear_every`
    decisions (after which the step's output is the first after an empty memory, as in
    training from windows). Returns the mean loss, the loss on decisions with an input,
    and the outputs per game.
    """
    head.eval()
    losses, active, outputs = [], [], []
    autocast = {"device_type": device, "dtype": torch.bfloat16, "enabled": device == "cuda"}
    for game in games:
        out, state = head.initial(1, device)
        kept = []
        chunk = 128
        for start in range(0, game.length, chunk):
            batch = _to(default_collate([game.piece(start, chunk)]), device)
            with torch.autocast(**autocast):
                if clear_every:
                    parts = []
                    for t in range(chunk):
                        if (start + t) % clear_every == 0:
                            out, state = head.initial(1, device)
                        piece = {k: v[:, t : t + 1] for k, v in batch.items()}
                        o, (out, state) = head.unroll(
                            piece, out, state, torch.zeros(1, dtype=torch.bool, device=device)
                        )
                        parts.append(o)
                    outs = torch.cat(parts, 1)
                else:
                    outs, (out, state) = head.unroll(
                        batch, out, state, torch.tensor([start == 0], device=device)
                    )
                nll, valid = _nll(head, outs, batch)
            n = min(chunk, game.length - start)
            kept.append(outs[0, :n].float().cpu())
            losses.extend(nll.float().cpu().tolist())
            acting = (batch["actions"][:, :, :, 0] != 0).any(-1)[valid]
            active.extend(nll[acting].float().cpu().tolist())
        outputs.append(torch.cat(kept).numpy())
    head.train()
    return float(np.mean(losses)), float(np.mean(active)) if active else None, outputs


def _probe_targets(game):
    labels = game.labels
    n = game.length
    targets = {
        "zoom": labels["zoom"][:n],
        "since_recentre": np.log1p(labels["since_recentre"][:n]),
        "winner": np.full(n, labels["winner"], np.float32),
    }
    for lag in POINTER_LAGS:
        past = np.full((n, 2), np.nan, np.float32)
        if lag < n:
            past[lag:] = labels["pointer"][: n - lag]
        targets[f"pointer_{lag}"] = past
    return targets


def probe(train_outputs, train_games, test_outputs, test_games, ridge=1.0):
    """Linear read-outs of the memory, fitted on training games, scored on held-out ones.

    R^2 for each target, and for the winner, how often the read-out's sign is right.
    """
    report = {}
    names = _probe_targets(train_games[0]).keys()
    for name in names:

        def stack(outputs, games):
            xs, ys = [], []
            for out, game in zip(outputs, games, strict=True):
                y = _probe_targets(game)[name]
                y = y.reshape(len(y), -1)
                keep = np.isfinite(y).all(1)
                xs.append(out[keep])
                ys.append(y[keep])
            return np.concatenate(xs), np.concatenate(ys)

        x, y = stack(train_outputs, train_games)
        tx, ty = stack(test_outputs, test_games)
        if not len(x) or not len(tx):
            continue
        mean_x, mean_y = x.mean(0), y.mean(0)
        xc = x - mean_x
        weights = np.linalg.solve(
            xc.T @ xc + ridge * len(x) * np.eye(x.shape[1]), xc.T @ (y - mean_y)
        )
        guess = (tx - mean_x) @ weights + mean_y
        if name == "winner":
            report[name] = float((np.sign(guess) == np.sign(ty)).mean())
        else:
            residual = ((ty - guess) ** 2).sum()
            total = ((ty - ty.mean(0)) ** 2).sum()
            report[name] = float(1 - residual / total)
    return report


def train_memory(
    cache,
    output,
    *,
    memory="gru",
    window=256,
    carry=True,
    burn_in=0,
    decisions=1024,
    epochs=4,
    lr=1e-4,
    seed=0,
    probe_games=None,
    device=None,
):
    """Train a MemoryHead on cached features and report how it does on held-out games.

    `decisions` per update is fixed, so a longer window means fewer games side by side,
    never more data. With `carry`, windows run through whole games and the memory enters
    each one where the last left off; without it, each window starts from an empty memory
    after `burn_in` steps, as train_bc does.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(output)
    if (output / "report.json").exists():
        raise FileExistsError("Results are immutable")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train, validation = load_cache(cache, "train"), load_cache(cache, "validation")
    head = MemoryHead(train[0].summary.shape[1], memory).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    streams = max(1, decisions // window)
    autocast = {"device_type": device, "dtype": torch.bfloat16, "enabled": device == "cuda"}
    output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    steps = 0
    with (output / "metrics.jsonl").open("a") as log:
        for epoch in range(epochs):
            if carry:
                batches = _carried_batches(train, window, streams, rng)
            else:
                batches = _reset_batches(train, window, burn_in, streams, rng)
            out, state = head.initial(streams, device)
            for batch, fresh in batches:
                batch, fresh = _to(batch, device), fresh.to(device)
                if not carry:
                    out, state = head.initial(len(fresh), device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(**autocast):
                    outs, (out, state) = head.unroll(
                        batch, out, state, fresh, grad_from=0 if carry else burn_in
                    )
                    nll, _ = _nll(head, outs, batch, 0 if carry else burn_in)
                    loss = nll.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite memory objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                out, state = out.detach(), detach(state)
                steps += 1
                log.write(json.dumps({"epoch": epoch, "step": steps, "nll": loss.item()}) + "\n")
            log.flush()
    trained = time.monotonic() - began
    nll, nll_acting, outputs = evaluate(head, validation, device)
    nll_cleared, _, _ = evaluate(head, validation, device, clear_every=18)
    probed = train[: probe_games or len(train)]
    _, _, train_outputs = evaluate(head, probed, device)
    report = {
        "memory": memory,
        "window": window,
        "carry": carry,
        "burn_in": burn_in,
        "decisions_per_update": decisions,
        "epochs": epochs,
        "seed": seed,
        "updates": steps,
        "parameters": sum(p.numel() for p in head.memory.parameters()),
        "state_floats": state_size(head.memory.initial(1, "cpu")) or 0,
        "train_seconds": round(trained, 1),
        "validation_nll": nll,
        "validation_nll_acting": nll_acting,
        "validation_nll_cleared_every_18": nll_cleared,
        "probes": probe(train_outputs, probed, outputs, validation),
        "train_games": len(train),
        "validation_games": len(validation),
    }
    torch.save({"head": head.state_dict(), "report": report}, output / "head.pt")
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report

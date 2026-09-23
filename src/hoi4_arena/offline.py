"""Offline reinforcement learning on recordings: imitation weighted by each input's advantage.

Every game this project can play is real time, so the first improvement over imitation
has to come from recordings already made. AlphaStar Unplugged (Mathieu et al., 2023,
arXiv 2308.03526) beat its own imitation agent 90% of the time from replays alone, and
RECAP (Physical Intelligence, 2025, arXiv 2511.14759) trains a value function on
outcomes and then the policy on each action's advantage. This is the simplest form of
that, advantage-weighted regression (Peng et al., 2019, arXiv 1910.00177):

1. A critic that predicts who wins (train-critic on the AI games) values every decision
   of a recording, with the memory carried from the game's start as when it plays.
2. A decision's advantage is how far the value moved, from the player's side, over the
   next `n_step` decisions; for the last of a game, up to its outcome.
3. `train-bc --advantage` counts each decision in proportion to exp(advantage / beta),
   capped at `max_weight` and scaled to average one over the recording.

Only recordings whose inputs decided the game can be weighted: a player's own games and,
later, the agent's. In the AI games the game's AI plays both sides and the recorded
inputs are the scripted camera's and the popup clicks; weighting camera moves by who won
would teach nothing, so those recordings are refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from .dataset import ADVANTAGE_LABELS, _Stream, batch_to_device, cover_starts, session_labels
from .learning import value_estimate
from .models import reads_clip


def player_side(manifest):
    """+1 if the recording's player was Blue, -1 if Red: the critic values Blue's side.

    Refused for an AI game, and for a recording that does not name exactly one player
    (record reads it from the arena log, `players`).
    """
    if manifest.get("source") == "ai":
        raise ValueError("an AI game's inputs did not decide it; there is nothing to weight")
    players = manifest.get("players") or []
    if len(players) != 1 or players[0] not in ("BLU", "RED"):
        raise ValueError(f"the recording must name one player, Blue or Red; it names {players}")
    return 1.0 if players[0] == "BLU" else -1.0


def advantage_weights(values, outcome, side, valid, *, n_step=25, beta=0.05, max_weight=20.0):
    """Each decision's advantage and its weight, from the critic's values over one game.

    `values` are the critic's returns from Blue's side, one per decision; `outcome` the
    recording's discounted result from Blue's side (NaN where unknown). With the discount
    at 0.9999 a decision, the n steps between two values change a return by 0.25% at
    most, so the advantage is the plain difference. A decision too close to the end to
    look n ahead looks to the outcome, or to the last value when the game has none.
    Weights are exp(advantage / beta), capped at `max_weight`, then scaled to average one
    over the valid decisions, so the loss keeps its size and only its emphasis moves.
    """
    v = side * np.asarray(values, np.float64)
    count = len(v)
    end = side * float(outcome[-1]) if np.isfinite(outcome[-1]) else v[-1]
    future = np.full(count, end)
    if count > n_step:
        future[: count - n_step] = v[n_step:]
    advantage = future - v
    weight = np.exp(np.minimum(advantage / beta, np.log(max_weight)))
    ok = np.asarray(valid, bool) & np.isfinite(weight)
    weight = np.where(ok, weight, 0.0)
    if ok.any():
        weight = weight / weight[ok].mean()
    return advantage.astype(np.float32), weight.astype(np.float32)


@torch.no_grad()
def value_recording(policy, labels, device, window=64):
    """The critic's value of every readable decision, the memory carried through the game."""
    count = int(labels["readable"].sum())
    values = np.full(len(labels["decisions"]), np.nan, np.float32)
    clips = reads_clip(policy.encoder)
    stream = _Stream(labels, window, 0, device, starts=cover_starts(labels, window), clips=clips)
    autocast = {"device_type": device, "dtype": torch.bfloat16, "enabled": device == "cuda"}
    hidden, done_until = None, 0
    try:
        while (done := stream.advance()) is not None:
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
                        # The last window is pulled back to end on the last decision, so
                        # its first steps were already taken; the memory must not see
                        # them twice.
                        if start + t < done_until:
                            continue
                        hidden, value = policy.recall(
                            summary[:, t],
                            cells[:, t],
                            centre[:, t],
                            batch["previous"][:, t],
                            batch["speed"][:, t],
                            hidden,
                            batch["quadrants"].dtype,
                        )
                        values[start + t] = float(value_estimate(value)[0])
                done_until = max(done_until, start + summary.shape[1])
    finally:
        stream.close()
    return values[:count] if count else values


def advantage_labels(
    checkpoint,
    recording,
    *,
    model_path=None,
    n_step=25,
    beta=0.05,
    max_weight=20.0,
    device=None,
):
    """Write `labels-advantage.npz` beside a player's recording: values, advantages, weights."""
    from .runner import load_policy

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    recording = Path(recording)
    manifest = json.loads((recording / "manifest.json").read_text())
    side = player_side(manifest)
    labels = session_labels(recording, sources=(manifest["source"],))
    policy, config, digest = load_policy(checkpoint, model_path, device)
    policy.eval()
    values = value_recording(policy, labels, device)
    count = len(values)
    advantage, weight = advantage_weights(
        values,
        labels["outcome"][:count],
        side,
        labels["valid"][:count],
        n_step=n_step,
        beta=beta,
        max_weight=max_weight,
    )
    full = np.zeros(len(labels["decisions"]), np.float32)
    full[:count] = weight
    np.savez(
        recording / ADVANTAGE_LABELS,
        decisions=labels["decisions"],
        value=np.pad(values, (0, len(full) - count), constant_values=np.nan),
        advantage=np.pad(advantage, (0, len(full) - count)),
        weight=full,
        checkpoint=digest,
        n_step=n_step,
        beta=beta,
        max_weight=max_weight,
    )
    return {
        "decisions": count,
        "player": manifest["players"][0],
        "winner": manifest.get("winner"),
        "checkpoint": digest,
        "advantage_mean": float(np.nanmean(advantage)),
        "weight_max": float(weight.max()) if count else 0.0,
        "weight_above_2": float((weight > 2).mean()) if count else 0.0,
    }

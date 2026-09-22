from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gae(rewards, values, bootstrap, terminated, valid, elapsed, gamma=0.9999, lam=0.95):
    """Discount actual wall time in units of 200ms. Never learn from invalid episodes."""
    if not valid.all():
        raise ValueError("Invalid episodes must be quarantined, not assigned zero reward")
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(bootstrap)
    next_value = bootstrap
    for t in reversed(range(rewards.shape[0])):
        continuation = (~terminated[t]).float()
        discount = gamma ** (elapsed[t] / 0.2)
        delta = rewards[t] + discount * next_value * continuation - values[t]
        carry = delta + discount * lam * continuation * carry
        advantages[t] = carry
        next_value = values[t]
    return advantages, advantages + values


def normalize_advantages(advantages):
    """One scale for the whole episode.

    An 8-step window of a flat reward has a tiny standard deviation. Dividing by that
    window turns the noise into a unit-scale advantage, and the ratio then trains on it.
    A constant episode has no relative advantage; leave it at zero rather than dividing
    by a floor. A single transition is already the whole episode: centering it would
    erase the only reward, including a one-step win. The divisor is floored at 1e-2, so
    a spread too small to be a terminal return stays proportionally small and flicker
    cannot grow to 1, and the scale is continuous across the floor.
    """
    advantages = advantages.float()
    if advantages.numel() < 2:
        return advantages.clone()
    scale = advantages.std(unbiased=False)
    if not torch.isfinite(scale) or float(scale) < 1e-6:
        return torch.zeros_like(advantages)
    centered = advantages - advantages.mean()
    return centered / max(float(scale), 1e-2)


def approximate_kl(logp, old_logp):
    """Schulman's non-negative approximation of the mean KL from the behavior policy."""
    log_ratio = logp - old_logp
    return (log_ratio.exp() - 1 - log_ratio).mean()


def ppo_loss(logp, old_logp, values, returns, advantages, entropy, clip=0.2):
    """Score one window. Advantages are already scaled over the episode; do not rescale."""
    ratio = (logp - old_logp).exp()
    policy = -torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages).mean()
    value = 0.5 * (values - returns).square().mean()
    return policy + 0.5 * value - 0.001 * entropy.mean()


def save_checkpoint(path, policy, config, *, auxiliary=None, optimizer=None, provenance=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Checkpoints are immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"policy": policy.state_dict(), "config": config, "provenance": provenance or {}}
    if auxiliary is not None:
        payload["auxiliary"] = auxiliary.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)
    digest = file_hash(path)
    path.with_suffix(".json").write_text(
        json.dumps({"sha256": digest, "config": config, "provenance": provenance}, indent=2)
    )
    return digest


class League:
    def __init__(self, path, seed=42):
        self.path = Path(path)
        self.rng = np.random.default_rng(seed)
        self.entries = json.loads(self.path.read_text()) if self.path.exists() else []

    def add(self, checkpoint):
        checkpoint = Path(checkpoint).resolve()
        digest = file_hash(checkpoint)
        if not any(e["sha256"] == digest for e in self.entries):
            self.entries.append({"path": str(checkpoint), "sha256": digest})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps(self.entries, indent=2))
            temp.replace(self.path)

    def sample(self):
        if not self.entries:
            raise ValueError("League needs a trained imitation checkpoint")
        entry = (
            self.entries[-1]
            if self.rng.random() < 0.5
            else self.entries[int(self.rng.integers(len(self.entries)))]
        )
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("League checkpoint changed after registration")
        return entry


def paired_evaluation(rows, seed=42):
    """Bootstrap whole side-swapped pairs; draws score half, invalid pairs excluded."""
    groups = {}
    for row in rows:
        groups.setdefault(row["pair_id"], []).append(row)
    scores = []
    win_scores = []
    counts = {"win": 0, "draw": 0, "loss": 0}
    excluded = 0
    for pair in groups.values():
        if (
            len(pair) != 2
            or {r["side"] for r in pair} != {"left", "right"}
            or not all(r["valid"] for r in pair)
        ):
            excluded += 1
            continue
        if len({r["scenario"] for r in pair}) != 1:
            raise ValueError("A pair must share its scenario")
        scores.append(np.mean([{"win": 1.0, "draw": 0.5, "loss": 0.0}[r["outcome"]] for r in pair]))
        win_scores.append(np.mean([r["outcome"] == "win" for r in pair]))
        for row in pair:
            counts[row["outcome"]] += 1
    if not scores:
        return {"pairs": 0, "excluded_pairs": excluded, "score": None, "ci95": None}
    x = np.array(scores)
    samples = np.random.default_rng(seed).choice(x, (10000, len(x)), replace=True).mean(1)
    radius = float(np.sqrt(np.log(40) / (2 * len(x))))

    def bounded(mean):
        return [max(0.0, float(mean) - radius), min(1.0, float(mean) + radius)]

    return {
        "pairs": len(x),
        "excluded_pairs": excluded,
        "score": float(x.mean()),
        "ci95": bounded(x.mean()),
        "ci_method": "Hoeffding bound over independent side-swapped pairs",
        "bootstrap_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "win_rate": float(np.mean(win_scores)),
        "win_rate_ci95": bounded(np.mean(win_scores)),
        "outcomes": counts,
        "acceptance_sample_complete": len(x) >= 50,
    }

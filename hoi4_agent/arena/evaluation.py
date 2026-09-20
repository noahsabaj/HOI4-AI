"""Paired, side-balanced evaluation; current-self-play win rate is not a gate."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import ArenaError


@dataclass(frozen=True)
class EvaluationGame:
    candidate: str  # initialization or candidate
    training_seed: int
    pair_id: str  # groups the two sides for both initialization and candidate
    side: str
    scenario: str
    opponent: str
    outcome: str
    source: str = "hoi4_vision"

    @property
    def score(self) -> float:
        return {"win": 1.0, "draw": 0.5, "loss": 0.0}[self.outcome]

    def __post_init__(self) -> None:
        if (self.candidate not in ("initialization", "candidate") or self.side not in ("BLU", "RED") or
            self.outcome not in ("win", "loss", "draw") or not all((self.pair_id, self.scenario, self.opponent))):
            raise ArenaError("invalid evaluation game")


def improvement_report(games: list[EvaluationGame], bootstrap_seed: int = 0) -> dict[str, Any]:
    if not games or any(game.source != "hoi4_vision" for game in games):
        raise ArenaError("strength reports require real HOI4 results")
    blocks: dict[tuple[int, str], list[EvaluationGame]] = defaultdict(list)
    for game in games:
        blocks[game.training_seed, game.pair_id].append(game)
    expected = {(candidate, side) for candidate in ("initialization", "candidate") for side in ("BLU", "RED")}
    differences: dict[int, list[float]] = defaultdict(list)
    for (seed, _), block in blocks.items():
        if len(block) != 4 or {(row.candidate, row.side) for row in block} != expected:
            raise ArenaError("every comparison block requires both policies on both sides")
        if len({(row.scenario, row.opponent) for row in block}) != 1:
            raise ArenaError("paired games must use the same scenario and fixed opponent")
        differences[seed].append(sum(row.score * (1 if row.candidate == "candidate" else -1) for row in block) / 2)
    rng = np.random.default_rng(bootstrap_seed)
    per_seed = []
    for seed, values in sorted(differences.items()):
        array = np.asarray(values)
        samples = rng.choice(array, (10_000, len(array)), replace=True).mean(axis=1)
        interval = np.quantile(samples, [0.025, 0.975]).tolist()
        counts = {}
        for candidate in ("initialization", "candidate"):
            counts[candidate] = dict(Counter(row.outcome for row in games
                                            if row.training_seed == seed and row.candidate == candidate))
        per_seed.append({"seed": seed, "games_per_policy": len(array) * 2,
                         "score_improvement": float(array.mean()), "paired_bootstrap_95_ci": interval,
                         "outcomes": counts,
                         "statistical_gate": bool(len(array) >= 50 and array.mean() >= 0.15 and interval[0] > 0)})
    return {"schema_version": 1, "metric": "win=1, draw=0.5, loss=0; paired side-block bootstrap",
            "per_seed": per_seed,
            "three_seed_statistical_gate": len(per_seed) >= 3 and all(row["statistical_gate"] for row in per_seed),
            "acceptance": "requires separate held-out scenario/opponent and integration evidence"}

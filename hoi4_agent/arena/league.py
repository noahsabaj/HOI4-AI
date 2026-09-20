"""Immutable opponents, episode-level sampling, and resumable league randomness."""
from __future__ import annotations

import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contracts import ArenaError
from .fingerprint import file_hash


@dataclass
class Opponent:
    id: str
    kind: str
    checkpoint: str | None = None
    sha256: str | None = None
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def player_score(self) -> float:
        return (self.wins + 0.5 * self.draws + 1) / (self.wins + self.losses + self.draws + 2)


class League:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.opponents: dict[str, Opponent] = {}
        self.champion: str | None = None
        self.evaluation_panel: tuple[str, ...] = ()

    def add_scripted(self, opponent_id: str) -> None:
        if opponent_id in self.opponents:
            raise ArenaError("opponent ID already exists")
        self.opponents[opponent_id] = Opponent(opponent_id, "scripted")

    def freeze(self, checkpoint: Path, directory: Path, *, champion: bool = False) -> str:
        digest = file_hash(checkpoint)
        identifier = "policy-" + digest
        destination = directory.resolve() / (digest + ".pt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if file_hash(destination) != digest:
                raise ArenaError("frozen checkpoint was modified")
        else:
            temporary = destination.with_suffix(".tmp")
            shutil.copyfile(checkpoint, temporary)
            if file_hash(temporary) != digest:
                raise ArenaError("checkpoint changed while freezing")
            temporary.replace(destination)
        self.opponents.setdefault(identifier, Opponent(identifier, "historical", str(destination), digest))
        if champion:
            self.champion = identifier
        return identifier

    def set_evaluation_panel(self, opponents: tuple[str, ...]) -> None:
        if self.evaluation_panel or not opponents or len(set(opponents)) != len(opponents):
            raise ArenaError("evaluation panel must be nonempty, unique, and set only once")
        if not set(opponents) <= self.opponents.keys():
            raise ArenaError("unknown evaluation opponent")
        self.evaluation_panel = opponents

    def sample_episode(self) -> Opponent:
        """Call once per episode; returned snapshot does not change mid-match."""
        historical = [opponent for opponent in self.opponents.values() if opponent.kind == "historical"]
        scripted = [opponent for opponent in self.opponents.values() if opponent.kind == "scripted"]
        if not historical or not scripted or self.champion not in self.opponents:
            raise ArenaError("league requires an initial frozen champion, history, and scripted opponents")
        category = self.rng.random()
        if category < 0.5:
            selected = self.rng.choices(historical, weights=[max(0.05, (1 - item.player_score) ** 2)
                                                          for item in historical])[0]
        elif category < 0.75:
            assert self.champion is not None
            selected = self.opponents[self.champion]
        else:
            selected = self.rng.choice(scripted)
        if selected.checkpoint and file_hash(Path(selected.checkpoint)) != selected.sha256:
            raise ArenaError("frozen opponent content changed")
        return Opponent(**asdict(selected))

    def record(self, opponent_id: str, result: str) -> None:
        opponent = self.opponents[opponent_id]
        if result == "win":
            opponent.wins += 1
        elif result == "loss":
            opponent.losses += 1
        elif result == "draw":
            opponent.draws += 1
        else:
            raise ArenaError("unknown match result")

    def state_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "opponents": [asdict(item) for item in self.opponents.values()],
                "champion": self.champion, "evaluation_panel": list(self.evaluation_panel),
                "rng": self.rng.getstate()}

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> League:
        if data["schema_version"] != 1:
            raise ArenaError("unsupported league schema")
        league = cls()
        league.opponents = {row["id"]: Opponent(**row) for row in data["opponents"]}
        league.champion = data["champion"]
        league.evaluation_panel = tuple(data["evaluation_panel"])
        state = data["rng"]
        league.rng.setstate((state[0], tuple(state[1]), state[2]))
        return league

"""Observed transitions with receipts and immutable episode provenance."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .actions import choice_index
from .contracts import ArenaError, BuildFingerprint, Country, Intent, Order, OrderReceipt, PlayerObservation, Verb
from .diagnostics import write_json


REAL_SOURCE = "hoi4_vision"
SOURCES = (REAL_SOURCE, "simulator", "fixture")


@dataclass(frozen=True)
class Provenance:
    # hoi4_vision (real game through screen and input), simulator, or fixture.
    # Only hoi4_vision counts as strength evidence; fixtures never train.
    source: str
    scenario_id: str
    scenario_split: str
    scenario_seed: int
    fingerprint: BuildFingerprint
    engine_state_fingerprint: str
    policy_id: str
    opponent_id: str
    demonstration_source: str | None = None

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ArenaError("unknown trajectory source")
        if self.scenario_split not in ("train", "validation", "held_out"):
            raise ArenaError("scenario split must be assigned before collecting data")
        if self.demonstration_source not in (None, "human", "scripted"):
            raise ArenaError("unknown demonstration source")
        if not all((self.scenario_id, self.engine_state_fingerprint, self.policy_id, self.opponent_id)):
            raise ArenaError("trajectory provenance is incomplete")


@dataclass(frozen=True)
class Transition:
    observation: PlayerObservation
    order: Order
    receipt: OrderReceipt
    next_observation: PlayerObservation
    log_probability: float
    value: float
    observation_to_submission_ms: float
    intent: Intent | None = None  # the guidance the policy was conditioned on, replayed in updates

    @property
    def elapsed_hours(self) -> int:
        return self.next_observation.game_hour - self.observation.game_hour

    def __post_init__(self) -> None:
        import math
        choice_index(self.observation, self.order)
        if self.receipt.order_id != self.order.id or self.receipt.episode_id != self.observation.episode_id:
            raise ArenaError("transition receipt does not match its submitted order")
        if (self.observation.terminal or self.elapsed_hours < 0 or
            self.observation.episode_id != self.next_observation.episode_id or
            self.observation.country != self.next_observation.country):
            raise ArenaError("invalid transition episode, country, or elapsed time")
        if not all(math.isfinite(x) for x in (self.log_probability, self.value, self.observation_to_submission_ms)):
            raise ArenaError("non-finite trajectory metadata")
        if self.log_probability > 1e-5 or self.observation_to_submission_ms < 0:
            raise ArenaError("invalid log probability or latency")


@dataclass(frozen=True)
class Trajectory:
    provenance: Provenance
    transitions: tuple[Transition, ...]

    def __post_init__(self) -> None:
        if not self.transitions:
            raise ArenaError("empty trajectory")
        seen = set()
        for index, transition in enumerate(self.transitions):
            if transition.order.id in seen:
                raise ArenaError("trajectory contains duplicate submitted commands")
            seen.add(transition.order.id)
            if index and self.transitions[index - 1].next_observation != transition.observation:
                raise ArenaError("trajectory is not a continuous player-observation sequence")

    @property
    def complete(self) -> bool:
        return self.transitions[-1].next_observation.terminal

    def require_live_training_data(self) -> None:
        if self.provenance.source not in (REAL_SOURCE, "simulator") or self.provenance.scenario_split != "train":
            raise ArenaError("training requires real HOI4 or simulator trajectories from training scenarios")
        if not self.complete:
            raise ArenaError("this learner requires complete episodes")

    def save(self, path: Path) -> None:
        # Continuity is enforced above, so store each observation once and let
        # transition i span observations[i] -> observations[i + 1].
        observations = [self.transitions[0].observation, *(t.next_observation for t in self.transitions)]
        rows = [{"order": asdict(t.order), "receipt": asdict(t.receipt), "log_probability": t.log_probability,
                 "value": t.value, "observation_to_submission_ms": t.observation_to_submission_ms,
                 "intent": None if t.intent is None else asdict(t.intent)}
                for t in self.transitions]
        write_json(path, {"schema_version": 2, "provenance": asdict(self.provenance),
                          "observations": [o.to_dict() for o in observations], "transitions": rows})

    @classmethod
    def load(cls, path: Path) -> Trajectory:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.pop("schema_version", None) != 2:
            raise ArenaError("unsupported trajectory schema")
        metadata = data["provenance"]
        metadata["fingerprint"] = BuildFingerprint(**metadata["fingerprint"])
        if len(data["observations"]) != len(data["transitions"]) + 1:
            raise ArenaError("trajectory observations do not bracket its transitions")
        transitions = []
        for index, row in enumerate(data["transitions"]):
            row["observation"] = data["observations"][index]
            row["next_observation"] = data["observations"][index + 1]
            order = row["order"]
            order.update(country=Country(order["country"]), verb=Verb(order["verb"]),
                         unit_ids=tuple(order["unit_ids"]))
            intent = row.get("intent")
            if intent is not None:
                intent["stances"] = tuple((sector, stance) for sector, stance in intent["stances"])
                row["intent"] = Intent(**intent)
            row.update(observation=PlayerObservation.from_dict(row["observation"]), order=Order(**order),
                       receipt=OrderReceipt(**row["receipt"]),
                       next_observation=PlayerObservation.from_dict(row["next_observation"]))
            transitions.append(Transition(**row))
        return cls(Provenance(**metadata), tuple(transitions))


def validate_demonstrations(trajectories: list[Trajectory], scenario_splits: dict[str, str]) -> dict[str, Any]:
    """Split by whole match and preassigned scenario; reject leakage, never shuffle frames."""
    matches: dict[str, str] = {}
    counts: dict[str, int] = {}
    for trajectory in trajectories:
        provenance = trajectory.provenance
        if provenance.source != REAL_SOURCE or provenance.demonstration_source is None or not trajectory.complete:
            raise ArenaError("demonstrations must be complete real-engine human/scripted matches")
        expected = scenario_splits.get(provenance.scenario_id)
        if expected != provenance.scenario_split:
            raise ArenaError("demonstration scenario was not assigned to this split")
        match = trajectory.transitions[0].observation.episode_id
        if match in matches and matches[match] != expected:
            raise ArenaError("complete match appears in multiple dataset splits")
        matches[match] = expected
        if any(not transition.receipt.accepted for transition in trajectory.transitions):
            raise ArenaError("demonstrations require accepted engine commands")
        counts[expected] = counts.get(expected, 0) + len(trajectory.transitions)
    return {"matches": matches, "transitions_by_split": counts}

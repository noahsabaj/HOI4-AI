"""Reinforcement learning in the SIMULATOR: episode collection, league training, evaluation.

Everything produced here has provenance source "simulator". It shows whether the learning
machinery works and gives a policy to start real-game fine-tuning from; it is never evidence
of strength in real HOI4 (``evaluation.improvement_report`` refuses these results by design).

The learner trains without Tier-2 guidance (``intent=None``) unless an ``intent_provider`` is
given. The learner is the speed/pause authority of its session: the opponent's speed and
pause orders are rejected, as a LAN host's would override a guest's.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from .contracts import ArenaError, ArenaSpec, Country, Intent, PlayerObservation, Policy
from .league import League, Opponent
from .learner import PPOConfig, PPOLearner
from .policy import RecurrentPolicy
from .scripted import IntentExecutor, ScriptedPolicy
from .sim.maps import SIZES, VARIANTS, scenario_ids, scenario_split
from .sim.session import SIM_FINGERPRINT, NoiseConfig, SimConfig, SimSession, SpeedModel
from .trajectory import Provenance, Trajectory, Transition

PANEL = tuple(f"{style}-v1" for style in ScriptedPolicy.STYLES)  # ScriptedPolicy.id values
IntentProvider = Callable[[PlayerObservation], Intent | None]


@dataclass(frozen=True)
class SimTask:
    """Which simulator scenarios are played and how decisions are paced in them."""
    size: str = "tiny"
    variants: tuple[str, ...] = VARIANTS
    divisions: int | None = None
    armor: int = 0
    horizon_days: int = 45
    interval_hours: float = 12.0  # the policy decides at most this often in game time
    latency_ms: float = 1000.0  # placeholder per-decision wall latency, see SpeedModel
    max_decisions: int = 400
    noise: bool = True
    speeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    scenario_pool: int = 64

    def __post_init__(self) -> None:
        if self.size not in SIZES or not set(self.variants) <= set(VARIANTS) or not self.variants:
            raise ArenaError("unknown simulator size or variant")
        if not self.speeds or any(speed not in range(1, 6) for speed in self.speeds):
            raise ArenaError("speeds must be within 1-5")

    def scenarios(self, split: str, count: int | None = None) -> tuple[str, ...]:
        return scenario_ids(split, self.size, count or self.scenario_pool, self.variants, self.divisions, self.armor)

    def spec(self, scenario_id: str, seed: int) -> ArenaSpec:
        return ArenaSpec(scenario_id, SIM_FINGERPRINT, seed, self.horizon_days * 24)

    def sim_config(self, speed: int, authority: Country | None) -> SimConfig:
        return SimConfig(SpeedModel(decision_latency_ms=self.latency_ms, decision_interval_hours=self.interval_hours),
                         NoiseConfig() if self.noise else None, speed, False, self.max_decisions, authority)


@dataclass
class Episode:
    outcome: str  # win / loss / draw, from the learner's side
    side: str
    scenario_id: str
    speed: int
    opponent_id: str
    decisions: int
    game_hours: int
    rejected: int
    truncated: bool
    reason: str
    trajectory: Trajectory | None = None


def scripted_opponent(opponent_id: str, seed: int) -> Policy:
    if opponent_id == "intent-executor":
        return IntentExecutor("attack")
    style = opponent_id.removesuffix("-v1")
    return ScriptedPolicy(style, seed)


def play_episode(policy: RecurrentPolicy, opponent: Policy, task: SimTask, scenario_id: str, seed: int,
                 side: Country, speed: int, *, policy_id: str, opponent_id: str, record: bool = True,
                 sample: bool = True, intent_provider: IntentProvider | None = None,
                 log: Callable[[str], None] | None = None) -> Episode:
    """One simulated match. With ``record`` the learner's transitions become a Trajectory."""
    split = scenario_split(scenario_id)
    session = SimSession(task.sim_config(speed, side))
    session.reset(task.spec(scenario_id, seed), (Country.BLUE, Country.RED))
    memory, opponent_memory = None, None
    transitions: list[Transition] = []
    rejected = decisions = 0
    observation = session.observe(side)
    while not observation.terminal:
        started = time.perf_counter()
        intent = intent_provider(observation) if intent_provider else None
        with torch.no_grad():
            output = policy(observation, memory, intent)
            index = output.distribution.sample() if sample else output.distribution.probs.argmax()
            log_probability = min(0.0, float(output.distribution.log_prob(index)))
        memory = output.memory
        order = output.actions[int(index)].order(observation, f"{policy_id}-{decisions}")
        latency_ms = (time.perf_counter() - started) * 1000
        receipt = session.submit(order)
        rejected += not receipt.accepted
        enemy_view = session.observe(side.opponent)
        enemy_order, opponent_memory = opponent.act(enemy_view, opponent_memory)
        enemy_receipt = session.submit(enemy_order)
        session.step()
        decisions += 1
        following = session.observe(side)
        if record:
            transitions.append(Transition(observation, order, receipt, following, log_probability,
                                          float(output.value), latency_ms, intent))
        if log:
            target = f"->{order.target_province_id}" if order.target_province_id else ""
            units = " ".join(f"{'E' if u.country != side else 'u'}{u.id}@{u.province_id}x{u.count}"
                             f"[{u.organization:.2f}/{u.strength:.2f}]" for u in following.units)
            log(f"h{observation.game_hour:4d} s{observation.game_speed}{'P' if observation.paused else ' '} "
                f"{order.verb.value}{order.unit_ids or ''}{target}{order.speed or ''} "
                f"{'ok' if receipt.accepted else 'REJECTED ' + receipt.reason} | opp {enemy_order.verb.value}"
                f"{'' if enemy_receipt.accepted else ' REJECTED'} | {units}")
        observation = following
    assert session.engine is not None
    outcome = "draw" if observation.winner is None else "win" if observation.winner == side else "loss"
    trajectory = None
    if record:
        trajectory = Trajectory(Provenance("simulator", scenario_id, split, seed, SIM_FINGERPRINT,
                                           f"sim:{scenario_id}:{seed}:{speed}", policy_id, opponent_id),
                                tuple(transitions))
    return Episode(outcome, side.value, scenario_id, speed, opponent_id, decisions, observation.game_hour, rejected,
                   session.truncated, session.engine.reason, trajectory)


def wilson(successes: float, games: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (95% by default)."""
    if games <= 0:
        return 0.0, 1.0
    p = successes / games
    centre = (p + z * z / (2 * games)) / (1 + z * z / games)
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / (1 + z * z / games)
    return max(0.0, centre - half), min(1.0, centre + half)


def evaluate(policy: RecurrentPolicy, task: SimTask, *, split: str = "validation", opponents: tuple[str, ...] = PANEL,
             speeds: tuple[int, ...] = (1, 5), scenarios: int = 6, seed: int = 0, sample: bool = True,
             policy_id: str = "candidate") -> dict[str, Any]:
    """Side-swapped games against fixed scripted opponents; W/L/D with Wilson intervals.

    Rows are per opponent and per INITIAL game speed (the policy may change speed itself), plus
    an "all" row per opponent. ``score`` counts a draw as half. SIMULATOR results only.
    """
    was_training = policy.training
    policy.eval()
    rows: list[dict[str, Any]] = []
    total: Counter[str] = Counter()
    lengths: list[int] = []
    for opponent_id in opponents:
        per_opponent: Counter[str] = Counter()
        for speed in speeds:
            counts: Counter[str] = Counter()
            for index, scenario_id in enumerate(task.scenarios(split, scenarios)):
                for side in Country:  # the pair shares scenario, seed and opponent seed
                    episode = play_episode(policy, scripted_opponent(opponent_id, seed + index), task, scenario_id,
                                           seed + index, side, speed, policy_id=policy_id, opponent_id=opponent_id,
                                           record=False, sample=sample)
                    counts[episode.outcome] += 1
                    lengths.append(episode.decisions)
            rows.append(_row(opponent_id, str(speed), counts))
            per_opponent.update(counts)
        rows.append(_row(opponent_id, "all", per_opponent))
        total.update(per_opponent)
    policy.train(was_training)
    return {"source": "simulator", "split": split, "sampled_actions": sample, "rows": rows,
            "overall": _row("panel", "all", total), "mean_decisions": sum(lengths) / max(1, len(lengths))}


def _row(opponent: str, speed: str, counts: Counter[str]) -> dict[str, Any]:
    games = sum(counts.values())
    wins, draws = counts["win"], counts["draw"]
    return {"opponent": opponent, "speed": speed, "games": games, "wins": wins, "losses": counts["loss"],
            "draws": draws, "win_rate": wins / max(1, games), "win_rate_wilson95": list(wilson(wins, games)),
            "score": (wins + 0.5 * draws) / max(1, games),
            "score_wilson95": list(wilson(wins + 0.5 * draws, games))}


def format_table(report: dict[str, Any]) -> str:
    lines = [f"{'opponent':12s} {'speed':>5s} {'n':>4s} {'W':>4s} {'L':>4s} {'D':>4s}  win-rate [95% Wilson]   score"]
    for row in (*report["rows"], report["overall"]):
        low, high = row["win_rate_wilson95"]
        lines.append(f"{row['opponent']:12s} {row['speed']:>5s} {row['games']:4d} {row['wins']:4d} {row['losses']:4d} "
                     f"{row['draws']:4d}  {row['win_rate']:.2f} [{low:.2f}, {high:.2f}]      {row['score']:.2f}")
    return "\n".join(lines)


def benchmark_device(task: SimTask, width: int, device: str, steps: int = 40) -> float:
    """Seconds per decision (forward + its share of a backward pass) on one device."""
    policy = RecurrentPolicy(width).to(device)
    session = SimSession(task.sim_config(3, None))
    session.reset(task.spec(task.scenarios("train", 1)[0], 0), (Country.BLUE,))
    observation = session.observe(Country.BLUE)
    for phase in range(2):  # first pass warms up kernels
        started = time.perf_counter()
        memory, terms = None, []
        for _ in range(steps):
            output = policy(observation, memory)
            memory = output.memory
            terms.append(output.distribution.entropy() + output.value)
        torch.stack(terms).mean().backward()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    return elapsed / steps


def choose_device(task: SimTask, width: int, requested: str = "auto") -> tuple[str, dict[str, float]]:
    if requested != "auto":
        return requested, {}
    timings = {"cpu": benchmark_device(task, width, "cpu")}
    if torch.cuda.is_available():
        timings["cuda"] = benchmark_device(task, width, "cuda")
    return min(timings, key=lambda name: timings[name]), timings


@dataclass(frozen=True)
class TrainConfig:
    output: str = "artifacts/sim/run"
    seed: int = 0
    width: int = 64
    updates: int = 100
    minutes: float = 0.0  # wall-clock budget for the loop; 0 means no budget
    episodes_per_update: int = 16
    league_fraction: float = 1.0  # share of episodes drawn by League.sample_episode (50/25/25);
    # the rest face a uniformly drawn scripted panel member
    shaping: float = 0.5
    shaping_updates: int = 20  # potential shaping is switched off after this many updates
    freeze_every: int = 10
    eval_every: int = 0
    eval_scenarios: int = 4
    eval_speeds: tuple[int, ...] = (1, 5)
    device: str = "auto"
    threads: int = 1
    ppo: PPOConfig = PPOConfig()
    task: SimTask = SimTask()


class Trainer:
    """Alternates simulator collection and PPO updates; checkpoints are atomic and resumable."""

    def __init__(self, config: TrainConfig, log: Callable[[str], None] = print) -> None:
        self.config, self.log = config, log
        self.output = Path(config.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.checkpoint = self.output / "learner.pt"
        self.metrics_path = self.output / "metrics.jsonl"
        torch.set_num_threads(max(1, config.threads))
        self.device, self.device_timings = choose_device(config.task, config.width, config.device)
        self.frozen: dict[str, RecurrentPolicy] = {}
        self.episodes, self.wall_seconds = 0, 0.0
        if self.checkpoint.exists():
            self.learner, extra = PPOLearner.resume(self.checkpoint, SIM_FINGERPRINT, self.device)
            self.league = League.from_state_dict(extra["league"])
            self.episodes = int(extra["provenance"].get("episodes", 0))
            self.wall_seconds = float(extra["provenance"].get("wall_seconds", 0.0))
            self.log(f"resumed {self.checkpoint} at update {self.learner.updates}")
        else:
            torch.manual_seed(config.seed)
            self.learner = PPOLearner(RecurrentPolicy(config.width).to(self.device), config.ppo)
            self.league = League(config.seed)
            for opponent_id in PANEL:
                self.league.add_scripted(opponent_id)
            self.league.set_evaluation_panel(PANEL)  # permanent: League refuses to change it later
            self.save()
            self.league.freeze(self.checkpoint, self.output / "league", champion=True)
            self.save()

    def save(self) -> None:
        self.learner.save(self.checkpoint, SIM_FINGERPRINT, self.league.state_dict(),
                          {"source": "simulator", "episodes": self.episodes, "wall_seconds": self.wall_seconds,
                           "task": asdict(self.config.task), "seed": self.config.seed,
                           "strength_evidence": False})

    def opponent(self, entry: Opponent, seed: int) -> Policy:
        if entry.kind == "scripted":
            return scripted_opponent(entry.id, seed)
        if entry.id not in self.frozen:
            assert entry.checkpoint is not None
            self.frozen[entry.id] = load_policy(Path(entry.checkpoint), self.device)
        return self.frozen[entry.id]

    def collect(self, rng: random.Random) -> list[Episode]:
        config, task = self.config, self.config.task
        scenarios = task.scenarios("train")
        policy_id = f"sim-learner-u{self.learner.updates}"
        episodes = []
        for _ in range(config.episodes_per_update):
            if rng.random() < config.league_fraction:
                entry = self.league.sample_episode()
            else:
                entry = self.league.opponents[rng.choice(PANEL)]
            seed = rng.randrange(2**31)
            episode = play_episode(self.learner.policy, self.opponent(entry, seed), task, rng.choice(scenarios), seed,
                                   rng.choice(tuple(Country)), rng.choice(task.speeds),
                                   policy_id=policy_id, opponent_id=entry.id)
            assert episode.trajectory is not None
            self.league.record(entry.id, episode.outcome)
            episodes.append(episode)
        return episodes

    def run(self) -> dict[str, Any]:
        config = self.config
        started = time.perf_counter()
        session_episodes = 0
        if config.eval_every and self.learner.updates == 0:
            self.evaluate()  # the untrained baseline, on the same games as every later evaluation
        while self.learner.updates < config.updates:
            if config.minutes and (time.perf_counter() - started) / 60 >= config.minutes:
                break
            tick = time.perf_counter()
            shaping = config.shaping if self.learner.updates < config.shaping_updates else 0.0
            self.learner.config = replace(self.learner.config, shaping=shaping)
            rng = random.Random(f"{config.seed}:{self.learner.updates}")
            self.learner.policy.eval()
            episodes = self.collect(rng)
            collected = time.perf_counter()
            self.learner.policy.train()
            stats = self.learner.update([e.trajectory for e in episodes if e.trajectory is not None])
            finished = time.perf_counter()
            self.episodes += len(episodes)
            session_episodes += len(episodes)
            self.wall_seconds += finished - tick
            outcomes = Counter(e.outcome for e in episodes)
            decisions = sum(e.decisions for e in episodes)
            row: dict[str, Any] = {
                "kind": "update", "source": "simulator", "update": self.learner.updates, "episodes": self.episodes,
                "wins": outcomes["win"], "losses": outcomes["loss"], "draws": outcomes["draw"],
                "mean_decisions": decisions / len(episodes),
                "mean_game_hours": sum(e.game_hours for e in episodes) / len(episodes),
                "invalid_order_rate": sum(e.rejected for e in episodes) / max(1, decisions),
                "truncated": sum(e.truncated for e in episodes), "entropy": stats["entropy"], "loss": stats["loss"],
                "shaping": shaping, "collect_seconds": collected - tick, "update_seconds": finished - collected,
                "games_per_hour": 3600 * len(episodes) / (finished - tick),
                "cumulative_games_per_hour": 3600 * self.episodes / max(1e-9, self.wall_seconds),
                "by_opponent_kind": {kind: dict(Counter(e.outcome for e in episodes
                                                        if self.league.opponents[e.opponent_id].kind == kind))
                                     for kind in ("scripted", "historical")}}
            self._write(row)
            self.log(f"u{row['update']:4d} W/L/D {row['wins']}/{row['losses']}/{row['draws']} "
                     f"len {row['mean_decisions']:.0f} H {row['entropy']:.2f} loss {row['loss']:.3f} "
                     f"invalid {row['invalid_order_rate']:.2f} {row['games_per_hour']:.0f} games/h")
            self.save()
            if config.freeze_every and self.learner.updates % config.freeze_every == 0:
                self.league.freeze(self.checkpoint, self.output / "league", champion=True)
                self.save()
            if config.eval_every and self.learner.updates % config.eval_every == 0:
                self.evaluate()
        return {"updates": self.learner.updates, "episodes": self.episodes, "wall_seconds": self.wall_seconds,
                "games_per_hour": 3600 * self.episodes / max(1e-9, self.wall_seconds), "device": self.device,
                "device_seconds_per_decision": self.device_timings, "session_episodes": session_episodes}

    def evaluate(self) -> dict[str, Any]:
        report = evaluate(self.learner.policy, self.config.task, scenarios=self.config.eval_scenarios,
                          speeds=self.config.eval_speeds, seed=self.config.seed + 10_000,
                          policy_id=f"sim-learner-u{self.learner.updates}")
        self._write({"kind": "evaluation", "update": self.learner.updates, **report})
        self.log(f"validation at update {self.learner.updates}\n{format_table(report)}")
        return report

    def _write(self, row: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")


def load_policy(checkpoint: Path, device: str = "cpu") -> RecurrentPolicy:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    policy = RecurrentPolicy(payload["width"]).to(device)
    policy.load_state_dict(payload["policy"])
    return policy.eval()


# ----- CLI ---------------------------------------------------------------------------------
def _task(args: argparse.Namespace) -> SimTask:
    return SimTask(args.size, tuple(args.variants.split(",")) if args.variants else VARIANTS, args.divisions,
                   args.armor, args.horizon_days, args.interval_hours, args.latency_ms, args.max_decisions,
                   not args.no_noise)


def _add_task_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", choices=sorted(SIZES), default="tiny")
    parser.add_argument("--variants", default="", help="comma-separated; default all")
    parser.add_argument("--divisions", type=int)
    parser.add_argument("--armor", type=int, default=0)
    parser.add_argument("--horizon-days", type=int, default=45)
    parser.add_argument("--interval-hours", type=float, default=12.0)
    parser.add_argument("--latency-ms", type=float, default=1000.0)
    parser.add_argument("--max-decisions", type=int, default=400)
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--seed", type=int, default=0)


def _speeds(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split(","))


def command_train(args: argparse.Namespace) -> int:
    ppo = PPOConfig(learning_rate=args.learning_rate, entropy_coefficient=args.entropy, epochs=args.epochs)
    config = TrainConfig(str(args.output), args.seed, args.width, args.updates, args.minutes, args.episodes,
                         args.league_fraction, args.shaping, args.shaping_updates, args.freeze_every,
                         args.eval_every, args.eval_scenarios, _speeds(args.eval_speeds), args.device, 1, ppo,
                         _task(args))
    trainer = Trainer(config)
    print(f"SIMULATOR training on {trainer.device} {trainer.device_timings}; not real-HOI4 evidence")
    summary = trainer.run()
    final = trainer.evaluate()
    print(json.dumps({**summary, "final_validation_score": final["overall"]["score"]}, indent=2))
    return 0


def command_eval(args: argparse.Namespace) -> int:
    task = _task(args)
    torch.manual_seed(args.seed)  # before construction, so an untrained policy is reproducible
    policy = load_policy(args.checkpoint, args.device) if args.checkpoint else RecurrentPolicy(args.width)
    report = evaluate(policy, task, split=args.split, speeds=_speeds(args.speeds), scenarios=args.scenarios,
                      seed=args.seed, sample=not args.greedy)
    print("SIMULATOR evaluation; not real-HOI4 evidence")
    print(format_table(report))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def command_play(args: argparse.Namespace) -> int:
    task = _task(args)
    torch.manual_seed(args.seed)
    policy = load_policy(args.checkpoint) if args.checkpoint else RecurrentPolicy(args.width)
    scenario_id = args.scenario or task.scenarios("validation", 1)[0]
    episode = play_episode(policy, scripted_opponent(args.opponent, args.seed), task, scenario_id, args.seed,
                           Country(args.side), args.speed, policy_id="sim-play", opponent_id=args.opponent,
                           record=False, sample=not args.greedy, log=print)
    print(json.dumps({k: v for k, v in asdict(episode).items() if k != "trajectory"} | {"source": "simulator"}))
    return 0


def add_commands(commands: Any) -> dict[str, Callable[[argparse.Namespace], int]]:
    """Register sim-train, sim-eval and sim-play on an argparse subparsers object."""
    train = commands.add_parser("sim-train", help="PPO league training in the SIMULATOR (not real HOI4)")
    _add_task_arguments(train)
    train.add_argument("--output", type=Path, default=Path("artifacts/sim/run"))
    train.add_argument("--width", type=int, default=64)
    train.add_argument("--updates", type=int, default=100)
    train.add_argument("--minutes", type=float, default=0.0)
    train.add_argument("--episodes", type=int, default=16, help="episodes per PPO update")
    train.add_argument("--league-fraction", type=float, default=1.0)
    train.add_argument("--shaping", type=float, default=0.5)
    train.add_argument("--shaping-updates", type=int, default=20)
    train.add_argument("--freeze-every", type=int, default=10)
    train.add_argument("--eval-every", type=int, default=0)
    train.add_argument("--eval-scenarios", type=int, default=4)
    train.add_argument("--eval-speeds", default="1,5")
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--entropy", type=float, default=0.01)
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--device", default="auto")
    evaluation = commands.add_parser("sim-eval", help="W/L/D against the scripted panel in the SIMULATOR")
    _add_task_arguments(evaluation)
    evaluation.add_argument("--checkpoint", type=Path, help="omit to score an untrained policy")
    evaluation.add_argument("--width", type=int, default=64)
    evaluation.add_argument("--split", default="validation", choices=("train", "validation", "held_out"))
    evaluation.add_argument("--speeds", default="1,5")
    evaluation.add_argument("--scenarios", type=int, default=6)
    evaluation.add_argument("--greedy", action="store_true")
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--output", type=Path)
    play = commands.add_parser("sim-play", help="one verbose SIMULATOR episode")
    _add_task_arguments(play)
    play.add_argument("--checkpoint", type=Path)
    play.add_argument("--width", type=int, default=64)
    play.add_argument("--opponent", default="advance-v1", choices=(*PANEL, "intent-executor"))
    play.add_argument("--scenario", default="")
    play.add_argument("--side", default="BLU", choices=("BLU", "RED"))
    play.add_argument("--speed", type=int, default=3, choices=range(1, 6))
    play.add_argument("--greedy", action="store_true")
    return {"sim-train": command_train, "sim-eval": command_eval, "sim-play": command_play}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HOI4 arena SIMULATOR training (never real-game evidence)")
    handlers = add_commands(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    try:
        return handlers[args.command](args)
    except ArenaError as exc:
        print(f"arena: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

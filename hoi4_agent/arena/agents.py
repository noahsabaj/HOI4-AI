"""Agent families behind one interface, and the generic episode loop that runs them.

Families (docs/architecture-v0.1.md): random and scripted floors, Cloud (Jev + DeepSeek driving
the scripted intent executor), Structured (``PolicyAgent``, the learned recurrent policy) and
Hierarchical (the policy conditioned on the Commander's latest Intent). Every agent sees only a
``PlayerObservation`` (+ an optional screenshot) and returns exactly one ``Order`` per call, so
the same agent runs against the simulator, a fake session or the real game.

Nothing here blocks on the cloud: ``CloudLink.step`` only submits to and reads from the tiers'
mailboxes. No action caps; pausing is allowed unless the ``allow_pause`` track switch forbids it;
all five game speeds are reachable. Importing this module does not import torch.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .actions import choices
from .contracts import (ArenaError, ArenaSession, ArenaSpec, BuildFingerprint, Country, Intent, Order,
                        PlayerObservation, Province, UnitView, Verb)
from .scripted import IntentExecutor, ScriptedPolicy
from .tiers import Briefer, Commander, Tiers, live_tiers


@runtime_checkable
class Agent(Protocol):
    id: str

    def reset(self) -> None: ...
    def act(self, observation: PlayerObservation, frame: Any = None) -> Order: ...
    def close(self) -> None: ...


def control_order(observation: PlayerObservation, verb: Verb, speed: int | None = None) -> Order:
    return Order(uuid.uuid4().hex, observation.episode_id, observation.sequence, observation.country, verb,
                 (), None, speed)


# --------------------------------------------------------------------------- floors

class RandomAgent:
    """Uniform over the structural action mask. Pause is excluded (a random pauser only stalls
    the match); set-speed is included only with ``controls=True``."""

    def __init__(self, seed: int = 0, controls: bool = False) -> None:
        self.id, self.seed, self.controls = "random-v1", seed, controls
        self.rng = random.Random(seed)

    def reset(self) -> None:
        self.rng = random.Random(self.seed)

    def act(self, observation: PlayerObservation, frame: Any = None) -> Order:
        skip = (Verb.PAUSE,) if self.controls else (Verb.PAUSE, Verb.SET_SPEED)
        options = [c for c in choices(observation) if c.verb not in skip]
        return self.rng.choice(options).order(observation, uuid.uuid4().hex)

    def close(self) -> None:
        pass


class ScriptedAgent:
    def __init__(self, style: str = "advance", seed: int = 0) -> None:
        self.style, self.seed = style, seed
        self.policy = ScriptedPolicy(style, seed)
        self.id, self.memory = f"scripted-{self.policy.id}", None

    def reset(self) -> None:
        self.policy, self.memory = ScriptedPolicy(self.style, self.seed), None

    def act(self, observation: PlayerObservation, frame: Any = None) -> Order:
        order, self.memory = self.policy.act(observation, self.memory)
        return order

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- cloud link

@dataclass
class LinkMetrics:
    pauses: int = 0
    pause_timeouts: int = 0
    pauses_not_applied: int = 0
    speed_changes: int = 0
    escalations: int = 0
    commander_submits: int = 0
    briefer_submits: int = 0


class CloudLink:
    """Feeds the tiers, reads their latest output, and owns pause/speed control orders.

    ``step`` returns a control order (pause toggle, set-speed, or no-op while a deliberate pause
    waits for guidance) or None when the agent's body should act. Pause and think: when the
    Commander requests it and pausing is allowed, toggle pause, escalate (Briefer with thinking
    when there is one, else a fresh Commander call), then toggle back as soon as guidance newer
    than the pause arrives, or after ``max_pause_s`` of wall clock, whichever is first.
    """

    def __init__(self, commander: Commander, briefer: Briefer | None = None, *, intent_every_hours: int = 6,
                 brief_every_hours: int = 72, allow_pause: bool = True, allow_speed: bool = True,
                 max_pause_s: float = 30.0, pause_grace_s: float = 2.0, min_pause_gap_hours: int = 24,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.commander, self.briefer = commander, briefer
        self.intent_every_hours, self.brief_every_hours = intent_every_hours, brief_every_hours
        self.allow_pause, self.allow_speed = allow_pause, allow_speed
        self.max_pause_s, self.pause_grace_s, self.min_pause_gap_hours = max_pause_s, pause_grace_s, min_pause_gap_hours
        self.clock = clock
        self.metrics = LinkMetrics()
        self.reset()

    def reset(self) -> None:
        guidance, brief = self.commander.guidance(), self.briefer.latest() if self.briefer else None
        # Output that predates this episode is never "fresh" and never triggers anything.
        self._seen_guidance = guidance.sequence if guidance else 0
        self._brief_floor = self._forwarded_brief = brief.sequence if brief else 0
        self._episode = ""
        self._intent_hour: int | None = None
        self._brief_hour: int | None = None
        self._pause_started: float | None = None
        self._pause_guidance = self._pause_brief = 0
        self._last_pause_hour: int | None = None
        self._escalated = self._speed_handled = self._pause_handled = self._seen_guidance

    def intent(self) -> Intent | None:
        guidance = self.commander.guidance()
        return guidance.intent if guidance is not None and guidance.sequence > self._seen_guidance else None

    @property
    def pausing(self) -> bool:
        return self._pause_started is not None

    def step(self, observation: PlayerObservation, frame: Any = None, popup_text: str = "") -> Order | None:
        if observation.terminal:
            return None
        if observation.episode_id != self._episode:
            self.reset()
            self._episode = observation.episode_id
        hour, briefer = observation.game_hour, self.briefer
        brief = briefer.latest() if briefer is not None else None
        if brief is not None and brief.sequence <= self._brief_floor:
            brief = None  # a brief from an earlier episode
        guidance = self.commander.guidance()
        if guidance is not None and guidance.sequence <= self._seen_guidance:
            guidance = None
        if briefer is not None and (popup_text or self._brief_hour is None or
                                    hour - self._brief_hour >= self.brief_every_hours):
            briefer.submit(observation, frame, popup_text)
            self._brief_hour, self.metrics.briefer_submits = hour, self.metrics.briefer_submits + 1
        if brief is not None and brief.sequence > self._forwarded_brief:
            self._forwarded_brief = brief.sequence  # generator -> judge: new plans go straight to Jev
            self._ask_commander(observation, brief)
        elif self._intent_hour is None or hour - self._intent_hour >= self.intent_every_hours:
            self._ask_commander(observation, brief)
        if guidance is None:
            return self._wait_or_none(observation, None)
        if guidance.needs_escalation and guidance.sequence > self._escalated and briefer is not None:
            self._escalate(observation, guidance.sequence, frame, popup_text)
        control = self._wait_or_none(observation, guidance)
        if control is not None or self.pausing:
            return control
        gap_ok = self._last_pause_hour is None or hour - self._last_pause_hour >= self.min_pause_gap_hours
        wanted = guidance.pause_requested and guidance.sequence > self._pause_handled
        self._pause_handled = max(self._pause_handled, guidance.sequence)  # act on a request fresh or never
        if self.allow_pause and wanted and not observation.paused and gap_ok:
            self._last_pause_hour = hour
            self._pause_started, self._pause_guidance = self.clock(), guidance.sequence
            self._pause_brief = self._forwarded_brief
            if briefer is not None:
                if guidance.sequence > self._escalated:
                    self._escalate(observation, guidance.sequence, frame, popup_text)
            else:
                self._ask_commander(observation, None)
            self.metrics.pauses += 1
            return control_order(observation, Verb.PAUSE)
        if self.allow_speed and guidance.speed is not None and guidance.sequence > self._speed_handled:
            self._speed_handled = guidance.sequence
            if guidance.speed != observation.game_speed:
                self.metrics.speed_changes += 1
                return control_order(observation, Verb.SET_SPEED, guidance.speed)
        return None

    def _ask_commander(self, observation: PlayerObservation, brief: Any) -> None:
        self.commander.submit(observation, brief)
        self._intent_hour, self.metrics.commander_submits = observation.game_hour, self.metrics.commander_submits + 1

    def _escalate(self, observation: PlayerObservation, sequence: int, frame: Any, popup_text: str) -> None:
        assert self.briefer is not None
        self.briefer.submit(observation, frame, popup_text, escalate=True)
        self._escalated, self._brief_hour = sequence, observation.game_hour
        self.metrics.escalations, self.metrics.briefer_submits = (self.metrics.escalations + 1,
                                                                  self.metrics.briefer_submits + 1)

    def _wait_or_none(self, observation: PlayerObservation, guidance: Any) -> Order | None:
        """While we hold a deliberate pause: wait (no-op), or toggle back when it is time."""
        if self._pause_started is None:
            return None
        waited = self.clock() - self._pause_started
        fresh = guidance is not None and guidance.sequence > self._pause_guidance and (
            self.briefer is None or guidance.brief_sequence > self._pause_brief)
        if fresh or waited >= self.max_pause_s:
            self._pause_started = None
            self.metrics.pause_timeouts += int(not fresh)
            if guidance is not None:  # the answer we paused for must not start another pause
                self._pause_handled = max(self._pause_handled, guidance.sequence)
            return control_order(observation, Verb.PAUSE) if observation.paused else None
        if observation.paused:
            return control_order(observation, Verb.NOOP)
        if waited > self.pause_grace_s:  # the toggle never showed up on screen; stop waiting for it
            self._pause_started = None
            self.metrics.pauses_not_applied += 1
        return None

    def report(self, game_hour: int | None = None) -> dict[str, Any]:
        return {"link": asdict(self.metrics), **Tiers(self.commander, self.briefer).metrics(game_hour)}


# --------------------------------------------------------------------------- families

class CloudAgent:
    """Commander [+ Briefer] -> scripted ``IntentExecutor``. No training; first end-to-end number."""

    def __init__(self, commander: Commander, briefer: Briefer | None = None, *, default_stance: str = "hold",
                 owns_tiers: bool = True, **link_options: Any) -> None:
        self.id = "cloud-jev+deepseek-v1" if briefer is not None else "cloud-jev-v1"
        self.link = CloudLink(commander, briefer, **link_options)
        self.executor, self.cursor, self.owns_tiers = IntentExecutor(default_stance), 0, owns_tiers
        self.last_game_hour: int | None = None

    def reset(self) -> None:
        self.cursor = 0
        self.link.reset()

    def act(self, observation: PlayerObservation, frame: Any = None) -> Order:
        self.last_game_hour = observation.game_hour
        control = self.link.step(observation, frame)
        if control is not None:
            return control
        order, self.cursor = self.executor.act(observation, self.cursor, self.link.intent())
        return order

    def metrics(self) -> dict[str, Any]:
        return self.link.report(self.last_game_hour)

    def close(self) -> None:
        if self.owns_tiers:
            Tiers(self.link.commander, self.link.briefer).close()


def load_policy(path: Path, device: str = "cpu", expected: BuildFingerprint | None = None) -> tuple[Any, dict]:
    """Policy weights from a ``PPOLearner.save`` checkpoint. The optimizer and RNG state are ignored;
    the fingerprint is verified only when ``expected`` is given (inference, not resumed training)."""
    import torch

    from .policy import RecurrentPolicy
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ArenaError("unsupported checkpoint version")
    if expected is not None:
        from .fingerprint import verify_fingerprint
        verify_fingerprint(expected, BuildFingerprint(**payload["fingerprint"]))
    policy = RecurrentPolicy(payload["width"]).to(device)
    policy.load_state_dict(payload["policy"])
    policy.eval()
    return policy, {"updates": payload.get("updates"), "provenance": payload.get("provenance"),
                    "fingerprint": payload.get("fingerprint")}


class PolicyAgent:
    """Structured family: the learned recurrent policy alone. Keeps GRU memory across decisions
    and clears it on reset or when the episode ID changes."""

    def __init__(self, policy: Any, *, agent_id: str = "policy", greedy: bool = False, allow_pause: bool = True,
                 seed: int | None = None) -> None:
        import torch
        self.policy, self.id, self.greedy, self.allow_pause, self.seed = policy, agent_id, greedy, allow_pause, seed
        self.memory: Any = None
        self._episode = ""
        self._generator = torch.Generator(device="cpu")
        self.reset()

    @classmethod
    def load(cls, path: Path, device: str = "cpu", **options: Any) -> PolicyAgent:
        policy, _ = load_policy(path, device)
        return cls(policy, agent_id=options.pop("agent_id", f"policy:{Path(path).stem}"), **options)

    def reset(self) -> None:
        self.memory, self._episode = None, ""
        if self.seed is not None:
            self._generator.manual_seed(self.seed)
        else:
            self._generator.seed()

    def decide(self, observation: PlayerObservation, intent: Intent | None = None) -> Order:
        import torch
        if observation.episode_id != self._episode:
            self.memory, self._episode = None, observation.episode_id
        with torch.no_grad():
            output = self.policy(observation, self.memory, intent)
            self.memory = output.memory
            logits = output.distribution.logits.detach().cpu().clone()
            if not self.allow_pause:  # "pause forbidden" track: mask, never resample
                for index, action in enumerate(output.actions):
                    if action.verb is Verb.PAUSE:
                        logits[index] = float("-inf")
            if self.greedy:
                index = int(logits.argmax().item())
            else:
                index = int(torch.multinomial(logits.softmax(-1), 1, generator=self._generator).item())
        return output.actions[index].order(observation, uuid.uuid4().hex)

    def act(self, observation: PlayerObservation, frame: Any = None) -> Order:
        return self.decide(observation, None)

    def close(self) -> None:
        pass


class HierarchicalAgent:
    """Structured + Jev intent + DeepSeek brief. The policy runs every tick (so its memory stays
    continuous) even when the link's control order is the one that is sent."""

    def __init__(self, body: PolicyAgent, commander: Commander, briefer: Briefer | None = None, *,
                 owns_tiers: bool = True, **link_options: Any) -> None:
        link_options.setdefault("allow_pause", body.allow_pause)
        self.body, self.owns_tiers = body, owns_tiers
        self.link = CloudLink(commander, briefer, **link_options)
        self.id = f"hier:{body.id.removeprefix('policy:')}"
        self.last_game_hour: int | None = None

    def reset(self) -> None:
        self.body.reset()
        self.link.reset()

    def act(self, observation: PlayerObservation, frame: Any = None) -> Order:
        self.last_game_hour = observation.game_hour
        control = self.link.step(observation, frame)
        order = self.body.decide(observation, self.link.intent())
        return order if control is None else control

    def metrics(self) -> dict[str, Any]:
        return self.link.report(self.last_game_hour)

    def close(self) -> None:
        if self.owns_tiers:
            Tiers(self.link.commander, self.link.briefer).close()


FAMILIES = {
    "random": "uniform over the action mask; sanity floor",
    "scripted:<hold|advance|flank|random>": "fixed-style ScriptedPolicy; panel opponent",
    "cloud": "Jev Commander + DeepSeek Briefer -> scripted intent executor; no training",
    "cloud-nobrief": "Jev Commander alone -> scripted intent executor (ablation)",
    "policy:<checkpoint>": "learned recurrent policy alone (PPOLearner.save checkpoint)",
    "hier:<checkpoint>": "learned policy conditioned on the Commander's Intent, plus Briefer",
    "hier-nobrief:<checkpoint>": "learned policy + Commander, no Briefer (ablation)",
}


def build_agent(spec: str, *, seed: int = 0, tiers: Tiers | None = None, device: str = "cpu",
                greedy: bool = False, allow_pause: bool = True, **link_options: Any) -> Agent:
    """``tiers`` injects ready (possibly fake) tiers; without it the cloud families build live clients."""
    kind, _, argument = spec.partition(":")
    if kind == "random" and not argument:
        return RandomAgent(seed)
    if kind == "scripted":
        return ScriptedAgent(argument or "advance", seed)
    if kind in ("cloud", "cloud-nobrief") and not argument:
        t = tiers if tiers is not None else live_tiers(briefer=kind == "cloud")
        return CloudAgent(t.commander, t.briefer if kind == "cloud" else None, allow_pause=allow_pause,
                          owns_tiers=tiers is None, **link_options)
    if kind in ("policy", "hier", "hier-nobrief") and argument:
        body = PolicyAgent.load(Path(argument), device, greedy=greedy, allow_pause=allow_pause, seed=seed)
        if kind == "policy":
            return body
        t = tiers if tiers is not None else live_tiers(briefer=kind == "hier")
        return HierarchicalAgent(body, t.commander, t.briefer if kind == "hier" else None,
                                 owns_tiers=tiers is None, **link_options)
    raise ArenaError(f"unknown agent spec {spec!r}; known: {', '.join(FAMILIES)}")


# --------------------------------------------------------------------------- episode loop

@dataclass
class AgentEpisodeMetrics:
    agent_id: str
    decisions: int = 0
    accepted: int = 0
    rejected: int = 0
    verbs: dict[str, int] = field(default_factory=dict)
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    act_seconds_total: float = 0.0
    act_seconds_max: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    episode_id: str
    winner: Country | None
    outcomes: dict[str, str]  # country value -> win / loss / draw
    decisions: int
    final_game_hour: int
    truncated: bool  # stopped by max_decisions or max_seconds, not by a terminal observation
    agents: dict[str, AgentEpisodeMetrics]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "winner": self.winner.value if self.winner else None}


def play_episode(session: ArenaSession, agents_by_country: Mapping[Country, Agent], spec: ArenaSpec, *,
                 max_decisions: int = 100_000, max_seconds: float | None = None, pace_s: float = 0.0,
                 frames: Callable[[Country], Any] | None = None, advance: Callable[[], Any] | None = None,
                 on_step: Callable[[PlayerObservation, Order, Any], None] | None = None) -> EpisodeResult:
    """Run one episode over any ``ArenaSession``: observe -> act -> submit, round-robin by country.

    How game time advances belongs to the session: the real game runs on its own clock, while a
    stepped session (the simulator's ``SimSession.step``) is advanced once per round, after every
    country has acted on the same tick. ``advance`` overrides that hook; by default a callable
    ``session.step`` is used when present. The loop ends at the first terminal observation. A truncated
    episode is reported as a draw and flagged; it is the caller's job not to count it as evidence.
    The caller owns the session and the agents (nothing is closed here). ``pace_s`` sleeps between
    rounds for real-time sessions; the contract allows at most two decisions per second there.
    """
    if not agents_by_country:
        raise ArenaError("an episode needs at least one agent")
    episode_id = session.reset(spec, tuple(agents_by_country))
    metrics = {country: AgentEpisodeMetrics(agent.id) for country, agent in agents_by_country.items()}
    verbs: dict[Country, Counter[str]] = {country: Counter() for country in agents_by_country}
    reasons: dict[Country, Counter[str]] = {country: Counter() for country in agents_by_country}
    for agent in agents_by_country.values():
        agent.reset()
    if advance is None:
        hook = getattr(session, "step", None)
        advance = hook if callable(hook) else None
    started, decisions, final = time.monotonic(), 0, None
    while final is None and decisions < max_decisions and (
            max_seconds is None or time.monotonic() - started < max_seconds):
        for country, agent in agents_by_country.items():
            observation = session.observe(country)
            if observation.episode_id != episode_id or observation.country != country:
                raise ArenaError("session returned an observation for another episode or player")
            if observation.terminal:
                final = observation
                break
            tick = time.perf_counter()
            order = agent.act(observation, frames(country) if frames is not None else None)
            spent = time.perf_counter() - tick
            if order.country != country or order.episode_id != episode_id:
                raise ArenaError(f"agent {agent.id} produced an order for another player or episode")
            receipt = session.submit(order)
            m = metrics[country]
            m.decisions, m.act_seconds_total = m.decisions + 1, m.act_seconds_total + spent
            m.act_seconds_max = max(m.act_seconds_max, spent)
            m.accepted, m.rejected = m.accepted + int(receipt.accepted), m.rejected + int(not receipt.accepted)
            verbs[country][order.verb.value] += 1
            if not receipt.accepted:
                reasons[country][receipt.reason] += 1
            decisions += 1
            if on_step is not None:
                on_step(observation, order, receipt)
        if final is None and advance is not None:
            advance()
        if pace_s > 0 and final is None:
            time.sleep(pace_s)
    last_hour = final.game_hour if final is not None else max(
        session.observe(country).game_hour for country in agents_by_country)
    winner = final.winner if final is not None else None
    for country, agent in agents_by_country.items():
        metrics[country].verbs, metrics[country].rejection_reasons = dict(verbs[country]), dict(reasons[country])
        report = getattr(agent, "metrics", None)
        if callable(report):
            metrics[country].extra = report()
    outcomes = {country.value: "draw" if winner is None else "win" if winner is country else "loss"
                for country in agents_by_country}
    return EpisodeResult(episode_id, winner, outcomes, decisions, last_hour, final is None,
                         {country.value: m for country, m in metrics.items()})


# --------------------------------------------------------------------------- CLI

def canned_observation() -> PlayerObservation:
    """SYNTHETIC three-sector situation for smoke tests: never evidence about HOI4.

    North: 2 worn divisions face a stack of 6. Center: 5 against 5. South: 5 fresh against 1 weak.
    """
    provinces, units = [], []
    for column, sector in enumerate(("north", "center", "south")):
        home, middle, far = 1 + column * 3, 2 + column * 3, 3 + column * 3
        side = tuple(m for m in (middle - 3, middle + 3) if 1 <= m <= 9)
        y = 0.2 + 0.3 * column
        provinces += [Province(home, 0.2, y, "plains", (middle,), Country.BLUE, 1.0, sector=sector),
                      Province(middle, 0.5, y, "forest" if column == 0 else "plains", (home, far, *side), None,
                               2.0, sector=sector),
                      Province(far, 0.8, y, "plains", (middle,), Country.RED, 1.0, sector=sector)]
    own = {"north": (2, 0.35, 0.7), "center": (5, 0.7, 0.9), "south": (5, 0.95, 1.0)}
    enemy = {"north": (6, 0.9, 0.95), "center": (5, 0.7, 0.9), "south": (1, 0.2, 0.4)}
    for column, sector in enumerate(("north", "center", "south")):
        count, org, strength = own[sector]
        units += [UnitView(100 + column * 10 + i, Country.BLUE, 1 + column * 3, org, strength, 1.0, False, "infantry")
                  for i in range(count)]
        count, org, strength = enemy[sector]
        units.append(UnitView(200 + column, Country.RED, 3 + column * 3, org, strength, None, None, "infantry",
                              count=count, confidence=0.9))
    return PlayerObservation("synthetic-smoke", Country.BLUE, 0, 240, time.monotonic_ns(), tuple(provinces),
                             tuple(units), game_speed=3)


def _agent_info(args: argparse.Namespace) -> int:
    from .envfile import available
    info: dict[str, Any] = {"families": FAMILIES, "keys_present": available("TYPESAFE_API_KEY", "DEEPSEEK_API_KEY"),
                            "interface": "Agent: id, reset(), act(observation, frame=None) -> Order, close()"}
    if args.spec:
        offline = args.spec.split(":")[0] in ("random", "scripted", "policy")
        if offline:
            agent = build_agent(args.spec)
            order = agent.act(canned_observation())
            info["agent"] = {"id": agent.id, "order_on_synthetic_observation": order.verb.value}
            agent.close()
        else:
            info["agent"] = "cloud families are exercised by tiers-smoke, not here"
    print(json.dumps(info, indent=2))
    return 0


def _tiers_smoke(args: argparse.Namespace) -> int:
    """One live Commander + Briefer round trip on the SYNTHETIC observation. Prints no secrets."""
    from ..errors import AgentError
    try:
        tiers = live_tiers(briefer=not args.no_briefer)
    except AgentError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    observation, brief, screenshot = canned_observation(), None, None
    if args.image:
        from PIL import Image
        screenshot = Image.open(args.image).convert("RGB")
    try:
        if tiers.briefer is not None:
            tiers.briefer.submit(observation, screenshot, args.popup, escalate=args.think)
            tiers.briefer.wait_idle(args.wait)
            brief = tiers.briefer.latest()
        tiers.commander.submit(observation, brief)
        tiers.commander.wait_idle(args.wait)
        guidance = tiers.commander.guidance()
        report = {
            "ok": guidance is not None and (tiers.briefer is None or brief is not None),
            "observation": "SYNTHETIC canned observation; not HOI4 evidence",
            "brief": None if brief is None else asdict(brief),
            "intent": None if guidance is None else asdict(guidance.intent),
            "guidance": None if guidance is None else {k: v for k, v in asdict(guidance).items() if k != "intent"},
            "metrics": tiers.metrics(observation.game_hour)}
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    finally:
        tiers.close()


def add_commands(commands: Any) -> dict[str, Callable[[argparse.Namespace], int]]:
    info = commands.add_parser("agent-info", help="list agent families, specs and which cloud keys are present")
    info.add_argument("--spec", help="also build this offline agent and act once on a synthetic observation")
    smoke = commands.add_parser("tiers-smoke", help="one live Jev + DeepSeek round trip on a synthetic observation")
    smoke.add_argument("--no-briefer", action="store_true", help="Commander only")
    smoke.add_argument("--think", action="store_true", help="run the Briefer in thinking (escalation) mode")
    smoke.add_argument("--image", type=Path, help="optional screenshot to attach to the brief request")
    smoke.add_argument("--popup", default="", help="optional popup text to attach")
    smoke.add_argument("--wait", type=float, default=90.0, help="seconds to wait for each tier")
    return {"agent-info": _agent_info, "tiers-smoke": _tiers_smoke}

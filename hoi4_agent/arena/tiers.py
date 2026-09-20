"""The two cloud tiers: Commander (Tier 2, Jev) and Briefer (Tier 3, DeepSeek V4.1 Flash).

Each tier owns one daemon thread and a one-slot mailbox. ``submit`` replaces whatever is
waiting (newest request wins) and returns immediately; ``latest`` only reads a field under
a lock. THE TICK LOOP THEREFORE NEVER BLOCKS ON THE CLOUD: a timeout, rate limit or outage
is counted in ``metrics`` and the previous intent/brief stays in force, stamped with the
game hour it was produced at so consumers can see its age.

Numbers and positions come only from the local ``PlayerObservation``. The Briefer supplies
text and candidate plans; its output is validated strictly and never written into the
numeric part of the state. Transports are injected callables, so tests run offline.
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..errors import AgentError, SchemaError
from .contracts import STANCES, ArenaError, Intent, PlayerObservation, UnitView

# ask(state, questions, timeout) -> object with .answers/.choice()/.noul(), token counts, latency_s
AskJev = Callable[[Any, dict[str, dict[str, Any]], float], Any]
# chat(system=, user=, images=, thinking=, timeout=) -> object with .text, token counts, latency_s
ChatDeepSeek = Callable[..., Any]

NO_PLAN = "none_of_these"
SPEED_OPTIONS = ("slow_down", "keep_speed", "speed_up")
MAX_PLANS = 4
MAX_TEXT = 600
_SAFE_KEY = re.compile(r"[^a-z0-9_]+")


# --------------------------------------------------------------------------- summary

def _words(ratio: float | None) -> str:
    """Own/enemy division ratio as words, because Jev is not a calculator."""
    if ratio is None:
        return "no enemy seen"
    for bound, word in ((0.5, "heavily outnumbered"), (0.8, "outnumbered"), (1.25, "roughly even"),
                        (2.0, "we outnumber them")):
        if ratio < bound:
            return word
    return "we heavily outnumber them"


def _condition(value: float | None) -> str:
    if value is None:
        return "unknown"
    return "exhausted" if value < 0.25 else "worn" if value < 0.5 else "fair" if value < 0.75 else "fresh"


def _average(units: list[UnitView], name: str) -> float | None:
    """Division-weighted mean of a bar; a stack counter already shows its own average."""
    known = [(getattr(u, name), u.count) for u in units if getattr(u, name) is not None]
    total = sum(count for _, count in known)
    return None if not total else round(sum(value * count for value, count in known) / total, 2)


def summarize(observation: PlayerObservation, horizon_hours: int = 90 * 24) -> dict[str, Any]:
    """Compact JSON state for Jev, built only from the player view.

    Comparisons are pre-computed (booleans and bucket words) next to the raw figures, and so is
    the clock: the match ends at ``horizon_hours`` and is then decided on victory points, so a
    level race is a draw. Without that, holding everywhere looks safe and scores nothing, which
    is exactly what the first live matches did.
    "front_pressure" counts visible enemy divisions standing in, or adjacent to, a province
    of the sector that we control or occupy. Fog of war holds by construction: enemies that
    perception did not see are not here.
    """
    mine, provinces = observation.country, {p.id: p for p in observation.provinces}
    sectors: dict[str, dict[str, Any]] = {}
    for name in sorted({p.sector for p in observation.provinces if p.sector}):
        members = [p for p in observation.provinces if p.sector == name]
        ids = {p.id for p in members}
        own = [u for u in observation.units if u.country == mine and u.province_id in ids]
        enemy = [u for u in observation.units if u.country != mine and u.province_id in ids]
        ours = {p.id for p in members if p.controller == mine} | {u.province_id for u in own}
        contact = ours | {n for pid in ours for n in provinces[pid].neighbors}
        pressing = sum(u.count for u in observation.units if u.country != mine and u.province_id in contact)
        own_n, enemy_n = sum(u.count for u in own), sum(u.count for u in enemy)
        own_org, enemy_org = _average(own, "organization"), _average(enemy, "organization")
        vp = [p for p in members if p.victory_points > 0]
        sectors[name] = {
            "own_divisions": own_n, "own_avg_org": own_org, "own_avg_strength": _average(own, "strength"),
            "own_condition": _condition(own_org), "own_in_combat": sum(1 for u in own if u.in_combat),
            "enemy_stacks": len(enemy), "enemy_divisions": enemy_n, "enemy_avg_org": enemy_org,
            "enemy_condition": _condition(enemy_org),
            "force_balance": "we have no divisions here" if not own_n else
                             _words(own_n / enemy_n if enemy_n else None),
            "outnumbered": bool(enemy_n > own_n),
            "enemy_weaker_than_us": bool(own_org is not None and enemy_org is not None and enemy_org < own_org),
            "victory_points_we_hold": sum(1 for p in vp if p.controller == mine),
            "victory_points_enemy_holds": sum(1 for p in vp if p.controller == mine.opponent),
            "victory_points_unknown_or_neutral": sum(1 for p in vp if p.controller not in (mine, mine.opponent)),
            "front_pressure": "none" if not pressing else "light" if pressing <= max(1, own_n // 2) else
                              "matched" if pressing <= own_n else "heavy",
        }
    ours_vp = sum(p.victory_points for p in observation.provinces if p.controller == mine)
    theirs_vp = sum(p.victory_points for p in observation.provinces if p.controller == mine.opponent)
    left = max(0, horizon_hours - observation.game_hour) // 24
    total = max(1, horizon_hours // 24)
    return {
        "we_are": mine.value, "game_hour": observation.game_hour, "game_day": observation.game_hour // 24,
        "game_speed": observation.game_speed, "paused": observation.paused,
        "victory_point_race": "level" if ours_vp == theirs_vp else "we lead" if ours_vp > theirs_vp else "we trail",
        "own_victory_points": ours_vp, "enemy_victory_points": theirs_vp,
        "days_left": left,
        "time_left": "over" if not left else "almost over" if left <= total // 10 else
                     "running out" if left <= total // 3 else "plenty",
        "result_if_nothing_changes": "draw" if ours_vp == theirs_vp else
                                     "we win" if ours_vp > theirs_vp else "we lose",
        "own_divisions_total": sum(u.count for u in observation.units if u.country == mine),
        "enemy_divisions_visible": sum(u.count for u in observation.units if u.country != mine),
        "any_own_division_in_combat": any(bool(u.in_combat) for u in observation.units if u.country == mine),
        "sectors": sectors,
    }


# --------------------------------------------------------------------------- data

@dataclass(frozen=True)
class Plan:
    name: str
    stances: tuple[tuple[str, str], ...]  # (sector, stance), sorted; only known sectors/stances
    rationale: str = ""


@dataclass(frozen=True)
class Brief:
    """Tier-3 output. Text and plans only: no coordinates, no quantities for the state."""
    text: str
    candidate_plans: tuple[Plan, ...]
    read_text: str
    produced_game_hour: int
    thinking: bool = False
    sequence: int = 0

    def age_hours(self, game_hour: int) -> int:
        return max(0, game_hour - self.produced_game_hour)


@dataclass(frozen=True)
class Guidance:
    """Everything one Commander round trip decided. ``intent`` is what Tier 1 consumes."""
    intent: Intent
    needs_escalation: bool
    pause_probability: float
    pause_requested: bool
    speed: int | None  # preferred game speed 1-5, None to leave it alone
    plan: str | None  # name of the Briefer plan that overrode the per-sector answers
    sector_probabilities: dict[str, dict[str, float]]  # distillation targets for the intent head
    sequence: int
    brief_sequence: int  # the brief this judgment saw, 0 for none


@dataclass
class TierMetrics:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    superseded: int = 0  # requests replaced in the mailbox before they were sent
    last_latency_s: float | None = None
    total_latency_s: float = 0.0
    max_latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    last_error: str = ""
    last_success_game_hour: int | None = None

    def as_dict(self, game_hour: int | None = None) -> dict[str, Any]:
        done = self.completed
        data: dict[str, Any] = {**self.__dict__, "mean_latency_s": self.total_latency_s / done if done else None}
        if game_hour is not None:
            data["staleness_hours"] = (None if self.last_success_game_hour is None
                                       else max(0, game_hour - self.last_success_game_hour))
        return data


# --------------------------------------------------------------------------- worker

class _LatestWorker:
    """One daemon thread, one-slot mailbox, newest request wins. Subclasses implement ``_run``."""

    def __init__(self, name: str) -> None:
        self.metrics = TierMetrics()
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._pending: Any = None
        self._busy = False
        self._closed = False
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._started = False

    def _submit(self, request: Any) -> None:
        with self._wake:
            if self._closed:
                return
            if self._pending is not None:
                self.metrics.superseded += 1
            self._pending = request
            self.metrics.submitted += 1
            if not self._started:
                self._started = True
                self._thread.start()
            self._wake.notify_all()

    def _loop(self) -> None:
        while True:
            with self._wake:
                while self._pending is None and not self._closed:
                    self._wake.wait()
                if self._closed:
                    return
                request, self._pending, self._busy = self._pending, None, True
            try:
                self._run(request)
            except (AgentError, ArenaError) as exc:
                with self._lock:
                    self.metrics.failed += 1
                    self.metrics.last_error = f"{type(exc).__name__}: {exc}"[:300]
            except Exception as exc:  # a transport bug must not kill the tier; it is counted loudly
                with self._lock:
                    self.metrics.failed += 1
                    self.metrics.last_error = f"UNEXPECTED {type(exc).__name__}: {exc}"[:300]
            finally:
                with self._wake:
                    self._busy = False
                    self._wake.notify_all()

    def _run(self, request: Any) -> None:
        raise NotImplementedError

    def _record(self, reply: Any, game_hour: int, tokens: tuple[str, str]) -> None:
        """Caller holds the lock."""
        latency = float(getattr(reply, "latency_s", 0.0) or 0.0)
        m = self.metrics
        m.completed, m.last_latency_s, m.last_success_game_hour = m.completed + 1, latency, game_hour
        m.total_latency_s, m.max_latency_s = m.total_latency_s + latency, max(m.max_latency_s, latency)
        m.input_tokens += int(getattr(reply, tokens[0], 0) or 0)
        m.output_tokens += int(getattr(reply, tokens[1], 0) or 0)

    @property
    def idle(self) -> bool:
        with self._lock:
            return self._pending is None and not self._busy

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """For tests, smoke runs and deliberate pauses; the tick loop never calls this."""
        deadline = time.monotonic() + timeout
        with self._wake:
            while self._pending is not None or self._busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._wake.wait(remaining)
        return True

    def close(self) -> None:
        with self._wake:
            self._closed, self._pending = True, None
            self._wake.notify_all()
        if self._started and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)


# --------------------------------------------------------------------------- Tier 2

def _key(text: str) -> str:
    return _SAFE_KEY.sub("_", text.lower()).strip("_") or "x"


def build_questions(observation: PlayerObservation, brief: Brief | None) -> tuple[dict[str, dict[str, Any]],
                                                                                 dict[str, str], dict[str, Plan]]:
    """The fan-out: one request, every judgment. Returns (questions, sector by key, plan by option)."""
    side = observation.country.value
    questions: dict[str, dict[str, Any]] = {}
    sector_keys: dict[str, str] = {}
    for sector in sorted({p.sector for p in observation.provinces if p.sector}):
        key = f"stance_{len(sector_keys)}_{_key(sector)}"
        sector_keys[key] = sector
        questions[key] = {"type": "choice", "instructions": (
            f"You command the land forces of {side} in Hearts of Iron IV. The match ends on the day in "
            f"state.days_left and is decided on victory points, so state.result_if_nothing_changes is what we "
            f"get by doing nothing. Look at sectors.{sector} and at the whole-match state. Which stance should "
            f"our divisions in the {sector} sector take now?"), "criteria": {
            "attack": "Advance on enemy-held victory points. Right when we are not outnumbered and our divisions "
                      "are fresh or fair, or the enemy here is absent, weaker or exhausted. Also the only stance "
                      "that can change a drawn or losing race before the match ends: holding a level race scores "
                      "nothing, and attacking somewhere is better than a certain draw as the days run out.",
            "hold": "Stay and defend the current line. Right when forces are roughly even AND we are already "
                    "ahead on victory points, or our divisions are worn and need to recover. Wrong as a way to "
                    "protect a level race, because the match is then decided as a draw.",
            "retreat": "Fall back toward our own victory points. Right when we are heavily outnumbered or our "
                       "divisions are exhausted under heavy front pressure."}}
    plans: dict[str, Plan] = {}
    if brief is not None and brief.candidate_plans and sector_keys:
        for plan in brief.candidate_plans[:MAX_PLANS]:
            plans[f"plan_{len(plans)}_{_key(plan.name)}"[:60]] = plan
        criteria: dict[str, str | None] = {
            option: (f"{plan.name}: " + ", ".join(f"{s} {stance}" for s, stance in plan.stances) +
                     (f". {plan.rationale}" if plan.rationale else ""))[:400] for option, plan in plans.items()}
        criteria[NO_PLAN] = "None of the proposed plans fits the current state."
        questions["plan"] = {"type": "choice", "instructions": (
            f"An advisor proposed these whole-front plans for {side}. Judge them against the current sectors "
            "in the state. Which plan fits the state best?"), "criteria": criteria}
    questions["pause"] = {"type": "noul", "instructions": (
        "Is the situation in the state both dangerous and unclear enough that the game should be paused to "
        "think longer before ordering anything?"), "criteria": {
        "true": "A sector is heavily outnumbered or under heavy front pressure and the right response is not obvious.",
        "false": "The situation is calm, or the right stance in each sector is obvious."}}
    questions["speed"] = {"type": "choice", "instructions": (
        f"The game runs at speed {observation.game_speed} of 5. Should the game speed change?"), "criteria": {
        "slow_down": "Our divisions are in combat or under front pressure, so decisions need to keep up.",
        "keep_speed": "The current speed suits what is happening.",
        "speed_up": "Nothing is happening: no combat, no front pressure, units are only marching or waiting."}}
    return questions, sector_keys, plans


def commander_state(observation: PlayerObservation, brief: Brief | None) -> dict[str, Any]:
    state = summarize(observation)
    if brief is not None:
        # Text only. Plans go into the Choice options; nothing from Tier 3 overwrites a figure above.
        state["advisor_note"] = {"text": brief.text, "screen_text": brief.read_text,
                                 "age_in_game_hours": brief.age_hours(observation.game_hour)}
    return state


def interpret(result: Any, observation: PlayerObservation, sector_keys: dict[str, str], plans: dict[str, Plan], *,
              escalation_threshold: float, pause_threshold: float, speed_threshold: float, sequence: int,
              brief_sequence: int) -> Guidance:
    """Typed Jev answers -> Guidance. Confidence is the minimum over sectors.

    Generator and judge: when Jev picks one of the Briefer's plans with more confidence than
    the weakest per-sector answer, that plan's stances replace the per-sector answers for the
    sectors it names, and its confidence replaces theirs.
    """
    stances: dict[str, str] = {}
    confidences: dict[str, float] = {}
    probabilities: dict[str, dict[str, float]] = {}
    for key, sector in sector_keys.items():
        answer = result.choice(key)
        if answer.choice not in STANCES:
            raise SchemaError(f"sector {sector!r} answered {answer.choice!r}")
        stances[sector], confidences[sector] = answer.choice, answer.confidence
        probabilities[sector] = {stance: float(answer.probabilities.get(stance, 0.0)) for stance in STANCES}
    if not stances:
        raise SchemaError("observation defines no sectors; nothing to command")
    chosen: str | None = None
    if plans:
        verdict = result.choice("plan")
        plan = plans.get(verdict.choice)
        if plan is not None and verdict.confidence > min(confidences.values()):
            chosen = plan.name
            for sector, stance in plan.stances:
                if sector in stances:
                    stances[sector], confidences[sector] = stance, verdict.confidence
    confidence = min(confidences.values())
    pause = float(result.noul("pause").probability)
    speed_answer, speed = result.choice("speed"), None
    if speed_answer.confidence >= speed_threshold and speed_answer.choice != "keep_speed":
        wanted = observation.game_speed + (1 if speed_answer.choice == "speed_up" else -1)
        speed = wanted if 1 <= wanted <= 5 else None
    return Guidance(Intent.of(stances, observation.game_hour, source="jev", confidence=confidence),
                    confidence < escalation_threshold, pause, pause >= pause_threshold, speed, chosen,
                    probabilities, sequence, brief_sequence)


class Commander(_LatestWorker):
    """Tier 2. ``submit`` never blocks; ``latest`` returns the newest Intent that arrived."""

    def __init__(self, ask: AskJev, *, timeout_s: float = 4.0, escalation_threshold: float = 0.5,
                 pause_threshold: float = 0.7, speed_threshold: float = 0.6) -> None:
        super().__init__("arena-commander")
        self._ask, self.timeout_s = ask, timeout_s
        self.escalation_threshold, self.pause_threshold = escalation_threshold, pause_threshold
        self.speed_threshold = speed_threshold
        self._guidance: Guidance | None = None
        self._sequence = 0

    def submit(self, observation: PlayerObservation, brief: Brief | None = None) -> None:
        self._submit((observation, brief))

    def _run(self, request: tuple[PlayerObservation, Brief | None]) -> None:
        observation, brief = request
        questions, sector_keys, plans = build_questions(observation, brief)
        if not sector_keys:
            raise SchemaError("observation defines no sectors; nothing to command")
        result = self._ask(commander_state(observation, brief), questions, self.timeout_s)
        with self._lock:
            sequence = self._sequence + 1
        guidance = interpret(result, observation, sector_keys, plans,
                             escalation_threshold=self.escalation_threshold, pause_threshold=self.pause_threshold,
                             speed_threshold=self.speed_threshold, sequence=sequence,
                             brief_sequence=brief.sequence if brief is not None else 0)
        with self._lock:
            self._sequence, self._guidance = sequence, guidance
            self._record(result, observation.game_hour, ("input_tokens", "output_tokens"))

    def guidance(self) -> Guidance | None:
        with self._lock:
            return self._guidance

    def latest(self) -> Intent | None:
        with self._lock:
            return None if self._guidance is None else self._guidance.intent


# --------------------------------------------------------------------------- Tier 3

BRIEFER_SYSTEM = (
    "You advise an automated Hearts of Iron IV player in a small two-country land-combat arena. You receive "
    "a JSON summary measured by local perception, sometimes a screenshot and popup text. Reply with one JSON "
    'object: {"brief": string, "candidate_plans": [{"name": string, "stances": {sector: stance}, '
    '"rationale": string}], "read_text": string}. "brief" is at most three sentences on what matters now. '
    "Give two to four distinct candidate plans that each assign a stance to every sector. \"read_text\" is a "
    "transcription of any popup, alert or unfamiliar screen text you can see, else an empty string. Never "
    "output coordinates, pixel positions, bar fill or unit counts of your own: the summary's figures are "
    "authoritative and you cannot measure the image.")


def _clean(value: Any, limit: int = MAX_TEXT) -> str:
    return " ".join(value.split())[:limit] if isinstance(value, str) else ""


def validate_brief(raw: str | dict[str, Any], sectors: tuple[str, ...], game_hour: int, *,
                   thinking: bool = False, sequence: int = 0) -> Brief:
    """Strict: the shape must be right; unknown sectors and stances are dropped, empty plans too."""
    if isinstance(raw, str):
        text = raw
        try:
            raw = json.loads(text)
        except ValueError:
            raise SchemaError(f"briefer reply is not JSON: {text[:120]!r}") from None
    if not isinstance(raw, dict) or not isinstance(raw.get("brief"), str) or not raw["brief"].strip():
        raise SchemaError("briefer reply needs a non-empty string 'brief'")
    listed = raw.get("candidate_plans", [])
    if not isinstance(listed, list):
        raise SchemaError("'candidate_plans' must be a list")
    plans: list[Plan] = []
    for entry in listed:
        if not isinstance(entry, dict) or not isinstance(entry.get("stances"), dict):
            continue
        name = _clean(entry.get("name"), 60)
        stances = tuple(sorted((sector, stance) for sector, stance in entry["stances"].items()
                               if sector in sectors and isinstance(stance, str) and stance in STANCES))
        if name and stances and name not in {p.name for p in plans} and stances not in {p.stances for p in plans}:
            plans.append(Plan(name, stances, _clean(entry.get("rationale"), 240)))
    return Brief(_clean(raw["brief"]), tuple(plans[:MAX_PLANS]), _clean(raw.get("read_text")), game_hour,
                 thinking, sequence)


@dataclass(frozen=True)
class _BriefRequest:
    observation: PlayerObservation
    screenshot: Any = None  # PIL.Image
    popup_text: str = ""
    escalate: bool = False


class Briefer(_LatestWorker):
    """Tier 3. Slow and asynchronous; thinking mode is used on escalation only."""

    def __init__(self, chat: ChatDeepSeek, *, timeout_s: float = 20.0, thinking_timeout_s: float = 60.0) -> None:
        super().__init__("arena-briefer")
        self._chat, self.timeout_s, self.thinking_timeout_s = chat, timeout_s, thinking_timeout_s
        self._brief: Brief | None = None
        self._sequence = 0
        self.escalations = 0

    def submit(self, observation: PlayerObservation, screenshot: Any = None, popup_text: str = "",
               escalate: bool = False) -> None:
        self._submit(_BriefRequest(observation, screenshot, popup_text, escalate))

    def _run(self, request: _BriefRequest) -> None:
        observation = request.observation
        sectors = tuple(sorted({p.sector for p in observation.provinces if p.sector}))
        user = json.dumps({"summary": summarize(observation), "sectors": list(sectors), "stances": list(STANCES),
                           "popup_text": request.popup_text[:2000],
                           "screenshot_attached": request.screenshot is not None}, separators=(",", ":"))
        reply = self._chat(system=BRIEFER_SYSTEM, user=user,
                           images=[] if request.screenshot is None else [request.screenshot],
                           thinking=request.escalate,
                           timeout=self.thinking_timeout_s if request.escalate else self.timeout_s)
        with self._lock:
            sequence = self._sequence + 1
        brief = validate_brief(reply.text, sectors, observation.game_hour, thinking=request.escalate,
                               sequence=sequence)
        with self._lock:
            self._sequence, self._brief = sequence, brief
            self.escalations += int(request.escalate)
            self._record(reply, observation.game_hour, ("prompt_tokens", "completion_tokens"))

    def latest(self) -> Brief | None:
        with self._lock:
            return self._brief


# --------------------------------------------------------------------------- live wiring

@dataclass
class Tiers:
    commander: Commander
    briefer: Briefer | None = None

    def metrics(self, game_hour: int | None = None) -> dict[str, Any]:
        data = {"commander": self.commander.metrics.as_dict(game_hour)}
        if self.briefer is not None:
            data["briefer"] = {**self.briefer.metrics.as_dict(game_hour), "escalations": self.briefer.escalations}
        return data

    def close(self) -> None:
        self.commander.close()
        if self.briefer is not None:
            self.briefer.close()


def live_tiers(*, briefer: bool = True, commander_timeout_s: float = 4.0) -> Tiers:
    """Real clients, keys from the environment or ``.env``. Raises ConfigError when a key is missing."""
    from ..brain.deepseek import DEEPSEEK_KEY_NAME, DeepSeekClient
    from ..brain.jev import JEV_KEY_NAME, JevClient
    from .envfile import require
    jev = JevClient(api_key=require(JEV_KEY_NAME))
    deepseek = DeepSeekClient(api_key=require(DEEPSEEK_KEY_NAME)) if briefer else None
    return Tiers(Commander(jev.ask, timeout_s=commander_timeout_s),
                 Briefer(deepseek.chat) if deepseek is not None else None)


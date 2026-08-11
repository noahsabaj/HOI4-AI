"""Typed data structures (dataclasses, not pydantic) + validators.

Everything is ``frozen`` so values can't be mutated behind the controller's back.
``WorldState`` holds perceived *facts only*; a field of ``None`` means "uncertain",
never an assumed value. ``validate_intent`` enforces that each tool gets exactly the
args it needs, with real enum members — so a malformed intent never reaches I/O.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace

from .enums import (
    BuildingType,
    GermanState,
    PanelId,
    PreconditionKind,
    Tech,
    ToolName,
    Verdict,
)
from .errors import IntentValidationError

# Date shapes that appear in UI-read text. Numeric Y.M.D ("1936.1.1") is the
# format the original design assumed; the HOI4 top bar actually renders
# "12:00, 1 Jan, 1936" (day, month NAME, year), so both are accepted.
_YMD_UI_RE = re.compile(r"(\d{4})\D{1,3}(\d{1,2})\D{1,3}(\d{1,2})")
_DMY_UI_RE = re.compile(r"(\d{1,2})\s*,?\s*([A-Za-z]{3,9})\.?,?\s*(\d{4})")
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
# As the top bar renders them ("1 Jan, 1936"). The deterministic glyph tier has
# to classify these letters, so perception derives its required alphabet here
# rather than hardcoding a second copy of the month list.
MONTH_ABBREVS = tuple(name.title() for name in _MONTHS)


# --- in-game date -----------------------------------------------------------
@dataclass(frozen=True, slots=True, order=True)
class GameDate:
    """Ordered by (year, month, day) — drives date-based cadence.

    Month/day are range-checked at construction so a misread (e.g. a VLM
    hallucinating month 13) can never become an orderable-but-wrong date.
    """

    year: int
    month: int
    day: int

    def __post_init__(self) -> None:
        if not (1 <= self.month <= 12):
            raise ValueError(f"month {self.month} out of range 1..12")
        if not (1 <= self.day <= 31):
            raise ValueError(f"day {self.day} out of range 1..31")

    @classmethod
    def from_str(cls, s: str) -> "GameDate":
        sep = "." if "." in s else "-"
        parts = s.strip().split(sep)
        if len(parts) != 3:
            raise ValueError(f"bad date {s!r} (want YYYY.MM.DD or YYYY-MM-DD)")
        y, m, d = (int(p) for p in parts)
        return cls(y, m, d)

    @classmethod
    def from_ui_text(cls, text: str) -> "GameDate | None":
        """Parse a date out of noisy UI-read text, or None — never guess.

        Accepts numeric Y.M.D ("1936. 1. 14") and the HOI4 top-bar rendering
        ("12:00, 1 Jan, 1936" — the clock is ignored). Out-of-range values
        read as None so a misread can't become an orderable-but-wrong date.

        The month-NAME form is tried FIRST because it is unambiguous. Y.M.D is
        not: if a reader drops the clock's colon, "12:00, 1 Jan, 1936" arrives
        as "1200,1Jan,1936", where the numeric pattern happily matches
        1200-01-19 (year "1200", "Jan" eaten as a separator) — an orderable
        wrong date. Anchoring on the month name reads that same string as
        1936-01-01, correctly.
        """
        m = _DMY_UI_RE.search(text)
        if m:
            month = _MONTHS.get(m.group(2)[:3].lower())
            if month is not None:
                try:
                    return cls(int(m.group(3)), month, int(m.group(1)))
                except ValueError:
                    return None
        m = _YMD_UI_RE.search(text)
        if m:
            try:
                return cls(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                return None
        return None

    def to_str(self) -> str:
        return f"{self.year:04d}.{self.month:02d}.{self.day:02d}"

    def plus_days(self, days: int) -> "GameDate":
        """Approximate calendar advance (30-day months) — good enough for cadence."""
        total = (self.year * 360) + (self.month - 1) * 30 + (self.day - 1) + days
        year, rem = divmod(total, 360)
        month, day = divmod(rem, 30)
        return GameDate(year, month + 1, day + 1)


# --- preconditions ----------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Precondition:
    kind: PreconditionKind = PreconditionKind.ALWAYS
    date: GameDate | None = None

    @property
    def handler_enforced(self) -> bool:
        """True if the tool handler re-checks this precondition itself (slot facts
        are only readable with their panel open, so the self-contained tool — which
        opens that panel — is the place that can actually decide)."""
        return self.kind in (PreconditionKind.FREE_CIV_SLOT, PreconditionKind.IDLE_RESEARCH_SLOT)

    def satisfied(self, world: "WorldState") -> bool | None:
        """True/False if decidable from the world, or None if the needed fact is uncertain."""
        k = self.kind
        if k is PreconditionKind.ALWAYS:
            return True
        if k is PreconditionKind.FREE_CIV_SLOT:
            return None if world.free_civ_slots is None else world.free_civ_slots >= 1
        if k is PreconditionKind.IDLE_RESEARCH_SLOT:
            return None if world.idle_research_slots is None else world.idle_research_slots >= 1
        if k in (PreconditionKind.DATE_BEFORE, PreconditionKind.DATE_AFTER):
            if world.date is None or self.date is None:
                return None
            return world.date < self.date if k is PreconditionKind.DATE_BEFORE else world.date >= self.date
        return None


# --- model intent (the typed action) ----------------------------------------
@dataclass(frozen=True, slots=True)
class Intent:
    tool: ToolName
    building: BuildingType | None = None
    state: GermanState | None = None
    tech: Tech | None = None
    speed: int | None = None
    paused: bool | None = None

    def to_dict(self) -> dict:
        d: dict[str, object] = {"tool": self.tool.value}
        for k in ("building", "state", "tech"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v.value
        if self.speed is not None:
            d["speed"] = self.speed
        if self.paused is not None:
            d["paused"] = self.paused
        return d


# tool -> required arg attribute(s)
_REQUIRED_ARGS: dict[ToolName, tuple[str, ...]] = {
    ToolName.BUILD_IN_STATE: ("state",),
    ToolName.ASSIGN_RESEARCH: ("tech",),
    ToolName.SET_SPEED: ("speed",),
    ToolName.ENSURE_PAUSED: ("paused",),
}
_ENUM_TYPES = {"building": BuildingType, "state": GermanState, "tech": Tech}


def validate_intent(intent: Intent) -> None:
    """Raise IntentValidationError unless the intent is well-formed for its tool."""
    if not isinstance(intent.tool, ToolName):
        raise IntentValidationError(str(intent.tool), "unknown tool")
    tool = intent.tool
    required = _REQUIRED_ARGS.get(tool, ())
    for name in required:
        if getattr(intent, name) is None:
            raise IntentValidationError(tool.value, f"missing required arg {name!r}")
    # enum-typed args must be real members
    for name, typ in _ENUM_TYPES.items():
        v = getattr(intent, name)
        if v is not None and not isinstance(v, typ):
            raise IntentValidationError(tool.value, f"{name} must be {typ.__name__}, got {v!r}")
    if intent.speed is not None and not (1 <= intent.speed <= 5):
        raise IntentValidationError(tool.value, f"speed {intent.speed} out of range 1..5")
    if tool is ToolName.ENSURE_PAUSED and not isinstance(intent.paused, bool):
        raise IntentValidationError(tool.value, "paused must be a bool")


# --- perceived world state --------------------------------------------------
@dataclass(frozen=True, slots=True)
class WorldState:
    """Facts read from the screen. ``None`` == uncertain (never assumed)."""

    date: GameDate | None = None
    paused: bool | None = None
    speed: int | None = None
    open_panel: PanelId = PanelId.NONE
    free_civ_slots: int | None = None
    idle_research_slots: int | None = None
    construction_queue_len: int | None = None
    event_popup: bool = False
    pause_menu: bool = False
    confidence: dict[str, float] = field(default_factory=dict)
    captured_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "date": self.date.to_str() if self.date else None,
            "paused": self.paused,
            "speed": self.speed,
            "open_panel": self.open_panel.value,
            "free_civ_slots": self.free_civ_slots,
            "idle_research_slots": self.idle_research_slots,
            "construction_queue_len": self.construction_queue_len,
            "event_popup": self.event_popup,
            "pause_menu": self.pause_menu,
            "confidence": dict(self.confidence),
            "captured_at": self.captured_at,
        }


# --- tool result ------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ToolResult:
    tool: ToolName
    verdict: Verdict
    pre: WorldState | None = None
    post: WorldState | None = None
    assertion: str = ""
    error: Exception | None = None
    retries: int = 0
    latency_s: float = 0.0
    # False only when an UNCERTAIN result was produced AFTER a non-idempotent
    # mutation (e.g. clicked, then couldn't read the effect): re-executing the
    # handler could repeat the mutation, so retry must not re-run it.
    retry_safe: bool = True

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.OK


# --- playbook goal + state --------------------------------------------------
@dataclass(frozen=True, slots=True)
class Goal:
    """One ordered, idempotent step in a playbook.

    Carries the same arg fields as ``Intent`` so deriving the intent is trivial.
    ``needs_judgment`` means a VLM must resolve an arg (e.g. "best free state")
    before execution; otherwise the goal names everything outright.
    """

    id: str
    tool: ToolName
    building: BuildingType | None = None
    state: GermanState | None = None
    tech: Tech | None = None
    speed: int | None = None
    paused: bool | None = None
    precondition: Precondition = field(default_factory=Precondition)
    repeatable: bool = False
    needs_judgment: bool = False

    def to_intent(self) -> Intent:
        return Intent(
            tool=self.tool,
            building=self.building,
            state=self.state,
            tech=self.tech,
            speed=self.speed,
            paused=self.paused,
        )


@dataclass(frozen=True, slots=True)
class PlaybookState:
    """Persisted progress; survives restarts.

    ``pending_etas`` are predicted completion dates (tech value -> date string)
    used purely as WAKE-UP hints for time advance — correctness always comes
    from perception, never from a prediction.
    """

    completed_goal_ids: tuple[str, ...] = ()
    last_seen_date: GameDate | None = None
    cycle_count: int = 0
    pending_etas: tuple[tuple[str, str], ...] = ()

    def with_completed(self, goal_id: str) -> "PlaybookState":
        if goal_id in self.completed_goal_ids:
            return self
        return replace(self, completed_goal_ids=self.completed_goal_ids + (goal_id,))

    def with_date(self, d: GameDate | None) -> "PlaybookState":
        return self if d is None else replace(self, last_seen_date=d)

    def advance_cycle(self) -> "PlaybookState":
        return replace(self, cycle_count=self.cycle_count + 1)

    def with_eta(self, key: str, date: GameDate) -> "PlaybookState":
        kept = tuple((k, v) for k, v in self.pending_etas if k != key)
        return replace(self, pending_etas=kept + ((key, date.to_str()),))

    def drop_etas_through(self, date: GameDate | None) -> "PlaybookState":
        """Retire ETAs that have arrived — as wake hints their job is done."""
        if date is None:
            return self
        kept = tuple((k, v) for k, v in self.pending_etas if GameDate.from_str(v) > date)
        return self if kept == self.pending_etas else replace(self, pending_etas=kept)

    def to_dict(self) -> dict:
        return {
            "completed_goal_ids": list(self.completed_goal_ids),
            "last_seen_date": self.last_seen_date.to_str() if self.last_seen_date else None,
            "cycle_count": self.cycle_count,
            "pending_etas": [list(e) for e in self.pending_etas],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlaybookState":
        ds = d.get("last_seen_date")
        return cls(
            completed_goal_ids=tuple(d.get("completed_goal_ids", ())),
            last_seen_date=GameDate.from_str(ds) if ds else None,
            cycle_count=int(d.get("cycle_count", 0)),
            pending_etas=tuple((str(k), str(v)) for k, v in d.get("pending_etas", ())),
        )


# --- trace record (plain DTO of primitives, round-trippable) -----------------
# Every record kind the controller emits. Kept as a constant (and asserted in
# tests against the loop) so the set can't drift out of sync with the code the
# way a hand-maintained comment did.
TRACE_KINDS = ("action", "advance", "error", "consult", "skipped")


@dataclass(frozen=True, slots=True)
class TraceRecord:
    cycle: int
    ts: float
    verdict: str
    kind: str = "action"  # one of TRACE_KINDS
    date: str | None = None
    plan_step: str | None = None
    pre_screenshot: str | None = None
    prompt: str | None = None
    raw_model_output: str | None = None
    parsed_intent: dict | None = None
    actions: tuple[dict, ...] = ()
    post_screenshot: str | None = None
    verification_question: str | None = None
    retries: int = 0
    latency_s: float = 0.0
    mode: str = "robust"
    vlm_used: bool = False       # the model made a JUDGMENT (prompt/raw describe it)
    # Model round trips during this record's work, judgment and perception reads
    # alike. vlm_used alone under-reports: the brain is the last tier of the
    # perception chain, so a time advance polling the date can be many calls on a
    # record whose vlm_used is False.
    vlm_calls: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["actions"] = list(self.actions)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TraceRecord":
        d = dict(d)
        d["actions"] = tuple(d.get("actions", ()))
        return cls(**d)

"""Typed data structures (dataclasses, not pydantic) + validators.

Everything is ``frozen`` so values can't be mutated behind the controller's back.
``WorldState`` holds perceived *facts only*; a field of ``None`` means "uncertain",
never an assumed value. ``validate_intent`` enforces that each tool gets exactly the
args it needs, with real enum members — so a malformed intent never reaches I/O.
"""

from __future__ import annotations

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


# --- in-game date -----------------------------------------------------------
@dataclass(frozen=True, slots=True, order=True)
class GameDate:
    """Ordered by (year, month, day) — drives date-based cadence."""

    year: int
    month: int
    day: int

    @classmethod
    def from_str(cls, s: str) -> "GameDate":
        sep = "." if "." in s else "-"
        parts = s.strip().split(sep)
        if len(parts) != 3:
            raise ValueError(f"bad date {s!r} (want YYYY.MM.DD or YYYY-MM-DD)")
        y, m, d = (int(p) for p in parts)
        return cls(y, m, d)

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
        d = {"tool": self.tool.value}
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
    ToolName.SELECT_BUILDING: ("building",),
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
    """Persisted progress; survives restarts."""

    completed_goal_ids: tuple[str, ...] = ()
    last_seen_date: GameDate | None = None
    cycle_count: int = 0

    def with_completed(self, goal_id: str) -> "PlaybookState":
        if goal_id in self.completed_goal_ids:
            return self
        return replace(self, completed_goal_ids=self.completed_goal_ids + (goal_id,))

    def with_date(self, d: GameDate | None) -> "PlaybookState":
        return self if d is None else replace(self, last_seen_date=d)

    def advance_cycle(self) -> "PlaybookState":
        return replace(self, cycle_count=self.cycle_count + 1)

    def to_dict(self) -> dict:
        return {
            "completed_goal_ids": list(self.completed_goal_ids),
            "last_seen_date": self.last_seen_date.to_str() if self.last_seen_date else None,
            "cycle_count": self.cycle_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlaybookState":
        ds = d.get("last_seen_date")
        return cls(
            completed_goal_ids=tuple(d.get("completed_goal_ids", ())),
            last_seen_date=GameDate.from_str(ds) if ds else None,
            cycle_count=int(d.get("cycle_count", 0)),
        )


# --- trace record (plain DTO of primitives, round-trippable) -----------------
@dataclass(frozen=True, slots=True)
class TraceRecord:
    cycle: int
    ts: float
    verdict: str
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
    vlm_used: bool = False
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

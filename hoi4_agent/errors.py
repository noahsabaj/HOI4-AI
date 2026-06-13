"""Typed error hierarchy.

The controller catches ``AgentError`` (never bare ``Exception``), so an
unexpected non-AgentError surfaces loudly instead of being swallowed — the exact
opposite of v3's ``except Exception: return None``. Every raise site attaches the
observed state where it helps debugging.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base for all expected, handled agent errors."""


# --- config / environment ---------------------------------------------------
class ConfigError(AgentError):
    """Missing/invalid configuration."""


class ResolutionError(AgentError):
    """Live client resolution does not match the configured/locked resolution."""

    def __init__(self, expected: tuple[int, int], actual: tuple[int, int]) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"expected client resolution {expected}, got {actual}")


# --- perception -------------------------------------------------------------
class PerceptionError(AgentError):
    """Base for perception failures."""


class TemplateMissingError(PerceptionError):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"no template/ROI named {name!r}")


class NccUncertainError(PerceptionError):
    """A template match scored below threshold — treat as uncertain, never success."""

    def __init__(self, roi: str, score: float, threshold: float) -> None:
        self.roi = roi
        self.score = score
        self.threshold = threshold
        super().__init__(f"NCC for {roi!r} = {score:.3f} < threshold {threshold:.3f}")


# --- brain / LLM ------------------------------------------------------------
class BrainError(AgentError):
    """Base for LLM/model failures."""


class OllamaUnavailableError(BrainError):
    """Could not reach the model endpoint (connection refused / DNS / etc.)."""


class OllamaTimeoutError(BrainError):
    """The model call exceeded its timeout."""


class ParseError(BrainError):
    def __init__(self, raw: str) -> None:
        self.raw = raw
        super().__init__(f"could not parse JSON from model output: {raw[:200]!r}")


class EnumError(BrainError):
    def __init__(self, field: str, value: object, allowed: list[str]) -> None:
        self.field = field
        self.value = value
        self.allowed = allowed
        super().__init__(f"{field}={value!r} not in {allowed}")


class SchemaError(BrainError):
    """Model output was valid JSON but the wrong shape for the requested schema."""


# --- intent validation ------------------------------------------------------
class IntentValidationError(AgentError):
    def __init__(self, tool: str, reason: str) -> None:
        self.tool = tool
        self.reason = reason
        super().__init__(f"invalid intent for {tool!r}: {reason}")


# --- tools ------------------------------------------------------------------
class ToolError(AgentError):
    """Base for tool execution failures; carries the observed post-state."""

    def __init__(self, message: str, observed: object | None = None) -> None:
        self.observed = observed
        super().__init__(message)


class PreconditionError(ToolError):
    pass


class PostconditionError(ToolError):
    pass


class PauseToggleError(ToolError):
    def __init__(self, observed: object, desired: bool) -> None:
        super().__init__(f"pause still {observed!r}, wanted {desired}", observed=observed)
        self.desired = desired


class SpeedSetError(ToolError):
    pass


class PanelOpenError(ToolError):
    def __init__(self, expected: str, observed: object) -> None:
        super().__init__(f"expected panel {expected!r}, observed {observed!r}", observed=observed)
        self.expected = expected


class BuildingSelectError(ToolError):
    pass


class BuildInStateError(ToolError):
    pass


class AssignResearchError(ToolError):
    pass


class ResetFailedError(ToolError):
    """The known-good reset macro did not return the UI to a home/no-panel state."""


# --- controller -------------------------------------------------------------
class ControllerError(AgentError):
    pass


class StuckError(ControllerError):
    def __init__(self, reason: str, trace_ref: str | None = None) -> None:
        self.trace_ref = trace_ref
        super().__init__(reason)


class HaltAndFlag(ControllerError):
    """Unrecoverable: stop the loop and surface the full trace to the operator."""

    def __init__(self, reason: str, trace_ref: str | None = None) -> None:
        self.trace_ref = trace_ref
        super().__init__(reason)

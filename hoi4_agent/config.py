"""Config loading via stdlib ``tomllib`` (no pyyaml). Fails loudly on bad config."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .enums import AgentMode
from .errors import ConfigError


@dataclass(frozen=True, slots=True)
class LLMConfig:
    backend: str          # "ollama" | "openai_compat"
    endpoint: str
    model: str
    timeout_s: float


@dataclass(frozen=True, slots=True)
class DisplayConfig:
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class TimingConfig:
    action_dwell_ms: int
    settle_ms: int
    verify_read_retries: int
    max_retries: int
    ncc_threshold: float
    run_speed: int
    cycle_days: int
    max_advance_days: int


@dataclass(frozen=True, slots=True)
class PathsConfig:
    calibration: str
    templates: str
    playbook: str
    trace_dir: str
    corpus: str


@dataclass(frozen=True, slots=True)
class Config:
    mode: AgentMode
    llm: LLMConfig
    display: DisplayConfig
    timing: TimingConfig
    paths: PathsConfig
    # [recovery] vlm_consult: one bounded VLM hint when deterministic recovery
    # fails, before halt-and-flag. Default on; explicit in agent.toml.
    recovery_vlm_consult: bool = True


def _require(table: dict, key: str, section: str):
    if key not in table:
        raise ConfigError(f"missing config key [{section}] {key!r}")
    return table[key]


def _read_raw(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        with open(p, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"invalid TOML in {p}: {e}") from e


def _resolve_llm(raw: dict, profile_override: str | None = None) -> LLMConfig:
    """Resolve [llm] into an LLMConfig.

    Two layouts: named profiles ([llm] profile = "x" + [llm.profiles.x] tables,
    the shipped form — one config, many models, switch by name) or the legacy
    flat [llm] keys. An override (CLI --llm-profile) beats the file's choice.
    """
    llm = raw.get("llm", {})
    profiles = llm.get("profiles", {})
    name = profile_override or llm.get("profile")
    if name is not None:
        if name not in profiles:
            raise ConfigError(f"unknown llm profile {name!r} (available: {sorted(profiles)})")
        table = profiles[name]
        section = f"llm.profiles.{name}"
    elif profiles:
        raise ConfigError("[llm.profiles.*] defined but no [llm] profile selected")
    else:
        table = llm
        section = "llm"
    return LLMConfig(
        backend=_require(table, "backend", section),
        endpoint=_require(table, "endpoint", section),
        model=_require(table, "model", section),
        timeout_s=float(_require(table, "timeout_s", section)),
    )


def list_llm_profiles(path: str | Path = "config/agent.toml") -> list[str]:
    """Profile names defined in the config file (empty for legacy flat [llm])."""
    return sorted(_read_raw(path).get("llm", {}).get("profiles", {}))


def load_config(path: str | Path = "config/agent.toml", llm_profile: str | None = None) -> Config:
    raw = _read_raw(path)

    try:
        mode = AgentMode(raw.get("mode", "robust"))
    except ValueError as e:
        raise ConfigError(f"invalid mode: {raw.get('mode')!r}") from e
    if mode is AgentMode.PURIST:
        # Accepting-but-ignoring a mode would be config that lies; fail loudly.
        raise ConfigError("mode 'purist' is a designed seam, not implemented — use 'robust'")

    disp = raw.get("display", {})
    tim = raw.get("timing", {})
    paths = raw.get("paths", {})

    return Config(
        mode=mode,
        llm=_resolve_llm(raw, llm_profile),
        display=DisplayConfig(
            width=int(_require(disp, "width", "display")),
            height=int(_require(disp, "height", "display")),
        ),
        timing=TimingConfig(
            action_dwell_ms=int(_require(tim, "action_dwell_ms", "timing")),
            settle_ms=int(_require(tim, "settle_ms", "timing")),
            verify_read_retries=int(_require(tim, "verify_read_retries", "timing")),
            max_retries=int(_require(tim, "max_retries", "timing")),
            ncc_threshold=float(_require(tim, "ncc_threshold", "timing")),
            run_speed=int(_require(tim, "run_speed", "timing")),
            cycle_days=int(_require(tim, "cycle_days", "timing")),
            max_advance_days=int(_require(tim, "max_advance_days", "timing")),
        ),
        recovery_vlm_consult=bool(raw.get("recovery", {}).get("vlm_consult", True)),
        paths=PathsConfig(
            calibration=_require(paths, "calibration", "paths"),
            templates=_require(paths, "templates", "paths"),
            playbook=_require(paths, "playbook", "paths"),
            trace_dir=_require(paths, "trace_dir", "paths"),
            corpus=_require(paths, "corpus", "paths"),
        ),
    )

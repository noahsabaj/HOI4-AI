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
    max_retries: int
    ncc_threshold: float
    run_speed: int
    cycle_days: int


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


def _require(table: dict, key: str, section: str):
    if key not in table:
        raise ConfigError(f"missing config key [{section}] {key!r}")
    return table[key]


def load_config(path: str | Path = "config/agent.toml") -> Config:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        with open(p, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"invalid TOML in {p}: {e}") from e

    try:
        mode = AgentMode(raw.get("mode", "robust"))
    except ValueError as e:
        raise ConfigError(f"invalid mode: {raw.get('mode')!r}") from e

    llm = raw.get("llm", {})
    disp = raw.get("display", {})
    tim = raw.get("timing", {})
    paths = raw.get("paths", {})

    return Config(
        mode=mode,
        llm=LLMConfig(
            backend=_require(llm, "backend", "llm"),
            endpoint=_require(llm, "endpoint", "llm"),
            model=_require(llm, "model", "llm"),
            timeout_s=float(_require(llm, "timeout_s", "llm")),
        ),
        display=DisplayConfig(
            width=int(_require(disp, "width", "display")),
            height=int(_require(disp, "height", "display")),
        ),
        timing=TimingConfig(
            action_dwell_ms=int(_require(tim, "action_dwell_ms", "timing")),
            max_retries=int(_require(tim, "max_retries", "timing")),
            ncc_threshold=float(_require(tim, "ncc_threshold", "timing")),
            run_speed=int(_require(tim, "run_speed", "timing")),
            cycle_days=int(_require(tim, "cycle_days", "timing")),
        ),
        paths=PathsConfig(
            calibration=_require(paths, "calibration", "paths"),
            templates=_require(paths, "templates", "paths"),
            playbook=_require(paths, "playbook", "paths"),
            trace_dir=_require(paths, "trace_dir", "paths"),
            corpus=_require(paths, "corpus", "paths"),
        ),
    )

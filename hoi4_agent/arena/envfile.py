"""Tiny ``.env`` reader for the cloud-tier API keys. No python-dotenv dependency.

Precedence: a non-empty process environment variable wins, otherwise the repo-root ``.env``.
Values are secrets: nothing here logs, prints or embeds them in an exception message.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"


def parse_env(text: str) -> dict[str, str]:
    """``KEY=value`` lines; ``#`` comments, blank lines, ``export`` prefixes and quotes are tolerated."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip().removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if name:
            values[name] = value
    return values


def read_env_file(path: Path | None = None) -> dict[str, str]:
    path = DEFAULT_ENV_PATH if path is None else path
    try:
        return parse_env(path.read_text(encoding="utf-8-sig"))
    except OSError:
        return {}


def lookup(name: str, path: Path | None = None) -> str | None:
    """The secret called ``name``, or None. The environment variable wins over the file."""
    value = os.environ.get(name) or read_env_file(path).get(name)
    return value or None


def require(name: str, path: Path | None = None) -> str:
    value = lookup(name, path)
    if value is None:
        raise ConfigError(f"{name} is not set in the environment or in {path or DEFAULT_ENV_PATH}")
    return value


def available(*names: str, path: Path | None = None) -> dict[str, bool]:
    """Presence only, safe to print."""
    return {name: lookup(name, path) is not None for name in names}

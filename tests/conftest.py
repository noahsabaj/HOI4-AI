"""Shared fixtures. pyproject sets pythonpath=["."] so `import hoi4_agent` works."""

from __future__ import annotations

from pathlib import Path

import pytest

from hoi4_agent.calibration import default_calibration
from hoi4_agent.config import load_config
from hoi4_agent.context import AgentContext
from hoi4_agent.geometry import WindowGeometry
from hoi4_agent.io.backends import RecordingInput
from hoi4_agent.perception.templates import TemplateStore

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config(REPO_ROOT / "config" / "agent.toml")


@pytest.fixture
def scripted_ctx(cfg):
    """Factory: build an AgentContext whose perceive() pops scripted WorldStates."""

    def _make(states):
        it = iter(states)
        inp = RecordingInput()
        ctx = AgentContext(
            config=cfg,
            geometry=WindowGeometry(1, 0, 0, cfg.display.width, cfg.display.height),
            input=inp,
            capture=None,
            calibration=default_calibration(cfg.display.width, cfg.display.height),
            templates=TemplateStore(),
            brain=None,
            mode=cfg.mode,
            perceive=lambda read_numbers=True: next(it),
        )
        return ctx, inp

    return _make

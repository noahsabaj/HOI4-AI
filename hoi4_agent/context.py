"""AgentContext: the wired bundle of components passed to tools and the controller.

It exposes ``perceive(read_numbers=True)`` as a plain callable so tests can inject
scripted WorldStates without monkeypatching, while the real build wires it to the
deterministic perception pipeline + the brain reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .brain.decide import Brain
from .calibration import Calibration
from .config import Config
from .enums import AgentMode
from .geometry import WindowGeometry
from .io.backends import CaptureBackend, InputBackend
from .perception.perceive import perceive as _perceive
from .perception.templates import TemplateStore
from .schemas import WorldState


@dataclass
class AgentContext:
    config: Config
    geometry: WindowGeometry
    input: InputBackend
    capture: CaptureBackend
    calibration: Calibration
    templates: TemplateStore
    brain: Brain
    mode: AgentMode
    perceive: Callable[..., WorldState]

    @classmethod
    def build(
        cls,
        *,
        config: Config,
        geometry: WindowGeometry,
        input: InputBackend,
        capture: CaptureBackend,
        calibration: Calibration,
        templates: TemplateStore,
        brain: Brain,
    ) -> "AgentContext":
        ctx = cls(
            config=config,
            geometry=geometry,
            input=input,
            capture=capture,
            calibration=calibration,
            templates=templates,
            brain=brain,
            mode=config.mode,
            perceive=lambda read_numbers=True: WorldState(),  # replaced below
        )

        def _do(read_numbers: bool = True) -> WorldState:
            return _perceive(
                capture=ctx.capture,
                geo=ctx.geometry,
                calib=ctx.calibration,
                templates=ctx.templates,
                threshold=ctx.config.timing.ncc_threshold,
                reader=ctx.brain,
                read_numbers=read_numbers,
            )

        ctx.perceive = _do
        return ctx

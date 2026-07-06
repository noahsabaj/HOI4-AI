"""AgentContext: the wired bundle of components passed to tools and the controller.

It exposes ``perceive(read_numbers=True, fields=None)`` as a plain callable so
tests can inject scripted WorldStates without monkeypatching, while the real
build wires it to the deterministic perception pipeline with a glyph-first
numeric reader (VLM fallback). ``sleep`` is injectable so tests run at full
speed while live handlers get real settle delays.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .brain.decide import Brain
from .calibration import Calibration
from .config import Config
from .enums import AgentMode
from .geometry import WindowGeometry
from .io.backends import CaptureBackend, InputBackend
from .perception.digits import FallbackReader, GlyphReader
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
    brain: Brain | None  # None is legal until a goal needs judgment
    mode: AgentMode
    perceive: Callable[..., WorldState]
    sleep: Callable[[float], None] = field(default=time.sleep)

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
        brain: Brain | None,
        sleep: Callable[[float], None] = time.sleep,
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
            perceive=lambda read_numbers=True, fields=None: WorldState(),  # replaced below
            sleep=sleep,
        )

        glyphs = GlyphReader(templates, threshold=config.timing.ncc_threshold)
        reader = FallbackReader(glyphs if glyphs.available() else None, brain)

        def _do(read_numbers: bool = True, fields=None) -> WorldState:
            return _perceive(
                capture=ctx.capture,
                geo=ctx.geometry,
                calib=ctx.calibration,
                templates=ctx.templates,
                threshold=ctx.config.timing.ncc_threshold,
                reader=reader,
                read_numbers=read_numbers,
                fields=fields,
            )

        ctx.perceive = _do
        return ctx

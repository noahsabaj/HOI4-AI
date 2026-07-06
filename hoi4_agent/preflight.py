"""Startup preflight: validate every wired asset BEFORE touching the game.

Catches at launch what would otherwise surface as perpetual UNCERTAIN verdicts
or repeated mid-run failures: missing perception templates, calibration that
doesn't cover the playbook, a calibration recorded at a different resolution,
and playbook goals that can never validate as intents.

``errors`` block a live run; ``warnings`` mean degraded-but-functional (e.g. no
glyph templates yet -> numeric reads fall back to the VLM).
"""

from __future__ import annotations

from .calibration import (
    GLYPH_TEMPLATE_PREFIX,
    REQUIRED_TEMPLATES,
    ROI_NAMES,
    SPEED_TEMPLATES,
    UI_POINT_NAMES,
    Calibration,
)
from .config import Config
from .enums import GermanState, Tech, ToolName
from .errors import IntentValidationError
from .perception.templates import TemplateStore
from .schemas import Goal, validate_intent
from .tools.macros import MAP_MODE_KEYS


def preflight(
    config: Config,
    calibration: Calibration,
    templates: TemplateStore,
    goals: list[Goal],
) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Empty errors == safe to start the live loop."""
    errors: list[str] = []
    warnings: list[str] = []

    # --- templates the T0 tier matches against -------------------------------
    for name in REQUIRED_TEMPLATES:
        if not templates.has(name):
            errors.append(f"missing required template {name!r} (run `calibrate` to capture it)")
    if not any(templates.has(n) for n in SPEED_TEMPLATES):
        warnings.append("no speed_1..speed_5 templates: game speed will read as unreadable")
    if not any(n.startswith(GLYPH_TEMPLATE_PREFIX) for n in templates.names()):
        from .perception.ocr import OcrReader

        if OcrReader().available():
            warnings.append("no glyph_* templates: numeric reads fall back to OCR, then the VLM")
        else:
            warnings.append(
                "no glyph_* templates: numeric reads fall back to the VLM (slow, less "
                'reliable) — `pip install -e ".[ocr]"` adds a faster OCR middle tier'
            )

    if config.grounding is not None:
        warnings.append(
            f"grounding fallback enabled ({config.grounding.model}) — validate locate-task "
            "accuracy in M0 before trusting grounded clicks"
        )

    # --- calibration ----------------------------------------------------------
    for name in ROI_NAMES:
        if name not in calibration.rois:
            errors.append(f"calibration missing ROI {name!r}")
    if (calibration.width, calibration.height) != (config.display.width, config.display.height):
        errors.append(
            f"calibration recorded at {calibration.width}x{calibration.height} but config "
            f"expects {config.display.width}x{config.display.height} — recalibrate"
        )
    if calibration.home_map_mode not in MAP_MODE_KEYS:
        errors.append(f"calibrated home map mode {calibration.home_map_mode!r} has no known hotkey")
    _UI_DEGRADATION = {
        "event_option": "event popups can only be escaped, not clicked away",
        "minimap_anchor": "no camera re-anchoring — state clicks assume zero pan drift",
    }
    for name in UI_POINT_NAMES:
        if name not in calibration.ui_points:
            warnings.append(f"no calibrated ui point {name!r}: {_UI_DEGRADATION.get(name, 'degraded')}")

    # --- playbook coverage + validity -----------------------------------------
    states = sorted({g.state for g in goals if g.state is not None}, key=lambda s: s.value)
    techs = sorted({g.tech for g in goals if g.tech is not None}, key=lambda t: t.value)
    buildings = sorted({g.building for g in goals if g.building is not None}, key=lambda b: b.value)
    for missing in calibration.missing_points_for(states, techs, buildings):
        errors.append(f"no calibrated click-point for playbook-referenced {missing}")

    for g in goals:
        if g.needs_judgment:
            if g.tool is ToolName.BUILD_IN_STATE and g.state is None:
                state_opts = [s for s in GermanState if s.value in calibration.state_points]
                if not state_opts:
                    errors.append(f"goal {g.id!r} needs state judgment but no state has a calibrated point")
            elif g.tool is ToolName.ASSIGN_RESEARCH and g.tech is None:
                tech_opts = [t for t in Tech if t.value in calibration.tech_points]
                if not tech_opts:
                    errors.append(f"goal {g.id!r} needs tech judgment but no tech has a calibrated point")
            else:
                warnings.append(f"goal {g.id!r} sets needs_judgment but has nothing to resolve")
            continue
        try:
            validate_intent(g.to_intent())
        except IntentValidationError as e:
            errors.append(f"goal {g.id!r} does not form a valid intent: {e}")

    return errors, warnings

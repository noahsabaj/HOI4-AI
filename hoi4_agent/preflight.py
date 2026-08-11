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
    REQUIRED_TEMPLATES,
    ROI_NAMES,
    SPEED_TEMPLATES,
    UI_POINT_NAMES,
    Calibration,
)
from .config import Config
from .enums import TECH_TABS, GermanState, Tech, ToolName
from .errors import IntentValidationError
from .perception.digits import DATE_STRIP_CHARS, GlyphReader, missing_date_glyphs
from .perception.perceive import VLM_ONLY_FIELDS
from .perception.templates import TemplateStore
from .schemas import Goal, validate_intent
from .tools.macros import MAP_MODE_KEYS


def preflight(
    config: Config,
    calibration: Calibration,
    templates: TemplateStore,
    goals: list[Goal],
    measured_fields: frozenset[str] = frozenset(),
) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Empty errors == safe to start the live loop.

    ``measured_fields`` names the numeric fields the M0 corpus actually covers.
    Passed in rather than read here so preflight stays free of I/O; the CLI
    derives it from the corpus.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # --- templates the T0 tier matches against -------------------------------
    for name in REQUIRED_TEMPLATES:
        if not templates.has(name):
            errors.append(f"missing required template {name!r} (run `calibrate` to capture it)")
    if not any(templates.has(n) for n in SPEED_TEMPLATES):
        warnings.append("no speed_1..speed_5 templates: game speed will read as unreadable")
    # Ask the reader itself rather than pattern-matching file names: a
    # "glyph_*.png" whose name doesn't decode to a character is not a usable
    # glyph tier, and preflight must not report one where the reader sees none.
    if not GlyphReader(templates).available():
        from .perception.ocr import OcrReader

        if OcrReader().available():
            warnings.append("no glyph_* templates: numeric reads fall back to OCR, then the VLM")
        else:
            warnings.append(
                "no glyph_* templates: numeric reads fall back to the VLM (slow, less "
                'reliable) — `pip install -e ".[ocr]"` adds a faster OCR middle tier'
            )
    elif missing_chars := missing_date_glyphs(templates):
        # A PARTIAL glyph set reads numbers fine but silently cannot read the
        # date: the strip carries a colon, commas and a month NAME, and the
        # reader is all-or-nothing. Time advance polls the date every couple of
        # seconds, so a quiet fallback here is a VLM call per poll.
        shown = ", ".join(repr(c) for c in missing_chars[:8])
        more = f" (+{len(missing_chars) - 8} more)" if len(missing_chars) > 8 else ""
        warnings.append(
            f"glyph set cannot read the in-game date — {len(missing_chars)} of "
            f"{len(DATE_STRIP_CHARS)} date characters have no template: {shown}{more}. "
            "The reader is all-or-nothing, so every date poll falls back to OCR/the VLM; "
            "run `calibrate --only glyphs` at more in-game dates to close the gap"
        )

    # The two postconditions the load-bearing tools assert on are COUNTS of what
    # a crop shows, not printed numbers, so no glyph or OCR tier can produce them
    # and both reads are the model's. That is a real qualification on the
    # "verified deterministically, not by re-asking the model" claim. It stops
    # being a risk once M0 has measured the model on those fields, so the warning
    # names exactly the ones the corpus does not yet cover.
    unmeasured = sorted(set(VLM_ONLY_FIELDS) - set(measured_fields))
    if unmeasured:
        warnings.append(
            "no deterministic tier for " + ", ".join(unmeasured)
            + " — these are counted from the crop, not printed by the UI, so the "
            "postconditions build_in_state/assign_research assert on are model "
            "reads, and M0 has not measured the model on them; add labeled "
            "read_number crops to the corpus and run `eval`"
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

    # --- research tabs: tech points are tab-relative, so every tab holding a
    # calibrated tech needs its own calibrated click-point, or the executor
    # can't select it and the tech click lands on whatever tab is showing.
    # Checks ALL calibrated techs (not just fixed goals): the judgment refill
    # goal can pick any tech that has a point.
    tech_by_value = {t.value: t for t in Tech}
    needed_tabs = set()
    for tv in calibration.tech_points:
        t = tech_by_value.get(tv)
        if t is None:
            continue
        tab = TECH_TABS.get(t)
        if tab is None:  # pragma: no cover - TECH_TABS covers all Tech (asserted in tests)
            errors.append(f"tech {tv!r} is calibrated but has no known research tab")
        else:
            needed_tabs.add(tab)
    for tab in sorted(needed_tabs, key=lambda x: x.value):
        if tab.value not in calibration.research_tabs:
            errors.append(
                f"research tab {tab.value!r} has calibrated techs but no tab click-point "
                "— run `calibrate --only tabs`"
            )

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

"""Closed vocabularies. String-valued so they serialize straight to TOML/JSON.

These enums ARE the contract: tool args are constrained to these members and
validated before execution, which makes out-of-range actions unrepresentable.
The M1 subsets are intentionally small; extending them is a data change
(enum member + a calibrated click-point), not a code change.
"""

from __future__ import annotations

from enum import StrEnum


class BuildingType(StrEnum):
    CIVILIAN_FACTORY = "civilian_factory"
    MILITARY_FACTORY = "military_factory"


class GermanState(StrEnum):
    # High-slot German states prioritized for M1 (calibrated map click-points).
    RUHR = "ruhr"
    SAXONY = "saxony"
    RHINELAND = "rhineland"
    WESTPHALIA = "westphalia"
    BRANDENBURG = "brandenburg"
    SILESIA = "silesia"
    BAVARIA = "bavaria"
    HANNOVER = "hannover"


class Tech(StrEnum):
    CONSTRUCTION_1 = "construction_1"
    CONSTRUCTION_2 = "construction_2"
    CONSTRUCTION_3 = "construction_3"
    INDUSTRY_1 = "industry_1"
    INDUSTRY_2 = "industry_2"
    ELECTRONICS_1 = "electronics_1"
    RADAR_1 = "radar_1"
    LAND_DOCTRINE_1 = "land_doctrine_1"


class ToolName(StrEnum):
    OBSERVE = "observe"
    ENSURE_PAUSED = "ensure_paused"
    SET_SPEED = "set_speed"
    OPEN_CONSTRUCTION = "open_construction"
    BUILD_IN_STATE = "build_in_state"
    OPEN_RESEARCH = "open_research"
    ASSIGN_RESEARCH = "assign_research"
    CLOSE_PANELS = "close_panels"


class PanelId(StrEnum):
    """Mutually-exclusive main panels. Blocking overlays (event popup, pause
    menu) are independent WorldState booleans, not panel classifications."""

    NONE = "none"            # home / map, no panel open
    CONSTRUCTION = "construction"
    RESEARCH = "research"


class Verdict(StrEnum):
    OK = "ok"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class AgentMode(StrEnum):
    ROBUST = "robust"       # deterministic verification (default)
    # "purist" (VLM verifies too) is a designed seam, not built: load_config
    # REJECTS it rather than silently running robust behavior under that name.
    PURIST = "purist"


class MapMode(StrEnum):
    DEFAULT = "default"     # F1 / default map mode = the calibrated "home view"


class ResearchTab(StrEnum):
    """Research-panel tabs. A tech's calibrated click-point is a screen position
    that only lands correctly when the tech's tab is the one showing, and HOI4
    remembers the last-opened tab across panel opens — so the executor selects
    the tab before clicking the tech. Members carry the tabs M1 techs live on;
    extending research to a new tab is an enum member + a calibrated tab-point."""

    INDUSTRY = "industry"           # construction, industry, synthetic oil
    ENGINEERING = "engineering"     # electronic engineering (radio/radar), forts, atomic
    LAND_DOCTRINE = "land_doctrine"


class PreconditionKind(StrEnum):
    ALWAYS = "always"
    FREE_CIV_SLOT = "free_civ_slot"            # free_civ_slots >= 1
    IDLE_RESEARCH_SLOT = "idle_research_slot"  # idle_research_slots >= 1
    DATE_BEFORE = "date_before"                # date < param
    DATE_AFTER = "date_after"                  # date >= param


# Which research tab each tech lives on. Every Tech MUST appear here (asserted in
# tests) so a tech can never be assigned without first selecting its tab.
TECH_TABS: dict[Tech, ResearchTab] = {
    Tech.CONSTRUCTION_1: ResearchTab.INDUSTRY,
    Tech.CONSTRUCTION_2: ResearchTab.INDUSTRY,
    Tech.CONSTRUCTION_3: ResearchTab.INDUSTRY,
    Tech.INDUSTRY_1: ResearchTab.INDUSTRY,
    Tech.INDUSTRY_2: ResearchTab.INDUSTRY,
    Tech.ELECTRONICS_1: ResearchTab.ENGINEERING,
    Tech.RADAR_1: ResearchTab.ENGINEERING,
    Tech.LAND_DOCTRINE_1: ResearchTab.LAND_DOCTRINE,
}

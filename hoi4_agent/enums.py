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
    SELECT_BUILDING = "select_building"
    BUILD_IN_STATE = "build_in_state"
    OPEN_RESEARCH = "open_research"
    ASSIGN_RESEARCH = "assign_research"
    CLOSE_PANELS = "close_panels"


class PanelId(StrEnum):
    NONE = "none"            # home / map, no panel open
    CONSTRUCTION = "construction"
    RESEARCH = "research"
    EVENT_POPUP = "event_popup"


class Verdict(StrEnum):
    OK = "ok"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class AgentMode(StrEnum):
    ROBUST = "robust"       # deterministic verification (default)
    PURIST = "purist"       # VLM verifies too (designed seam, not built in M1)


class DecisionType(StrEnum):
    WHICH_STATE = "which_state"
    WHICH_TECH = "which_tech"
    READ_NUMBER = "read_number"
    YES_NO = "yes_no"


class MapMode(StrEnum):
    DEFAULT = "default"     # F1 / default map mode = the calibrated "home view"


class PreconditionKind(StrEnum):
    ALWAYS = "always"
    FREE_CIV_SLOT = "free_civ_slot"            # free_civ_slots >= 1
    IDLE_RESEARCH_SLOT = "idle_research_slot"  # idle_research_slots >= 1
    DATE_BEFORE = "date_before"                # date < param
    DATE_AFTER = "date_after"                  # date >= param

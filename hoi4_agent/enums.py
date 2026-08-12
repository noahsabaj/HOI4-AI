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
    """Where a tech lives, and what has to be showing before its point is clickable.

    A tech's calibrated click-point is a screen position that only lands correctly
    when its tab is the one showing, and HOI4 remembers the last selection across
    panel opens — so the executor selects the tab before clicking the tech.

    Verified against the installed game's interface data, which shows these are
    NOT all the same kind of thing:

    - INDUSTRY and ENGINEERING are folder tabs of ONE view. In
      ``countrytechtreeview.gui`` a ``folder_tabs`` row holds nine buttons
      evenly spaced 89 units apart from x=22: infantry, support, armour,
      artillery, naval, mtg-naval-support, air, ``electronics_folder_tab``
      (x=645 — the one HOI4 also labels ``highlight_engineering_folder``, hence
      our name), ``industry_folder_tab`` (x=734).
    - LAND_DOCTRINE is a SEPARATE VIEW, not a folder of that row.
      ``countrydoctrinetreeview.gui`` defines its own ``land_doctrine_folder_tab``
      alongside naval/air/special-forces. Its calibrated "tab point" is therefore
      a view switch, not a sibling of the two above.

    That distinction matters when extending: adding a tech in a new tech-tree
    folder is an enum member plus a point on the known tab row, while adding one
    in another separate view means calibrating a different navigation control.
    """

    INDUSTRY = "industry"           # tech-tree folder tab: construction, industry
    ENGINEERING = "engineering"     # tech-tree folder tab: electronics (radio/radar)
    LAND_DOCTRINE = "land_doctrine"  # separate doctrine VIEW, not a folder tab


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

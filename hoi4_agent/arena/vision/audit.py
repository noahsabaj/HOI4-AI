"""Audit a perceived ``PlayerObservation`` against a text save taken at the same moment.

The save is parsed by ``hoi4_agent.clausewitz``; it is the game's own state, so it may be used
to GRADE perception but never to feed the agent. The division schema of HOI4 saves is
UNVERIFIED here (no arena save existed when this was written), so extraction is a defensive
search: any block reached through a ``division``/``divisions`` key, at any depth, that names a
province (``location``/``province``/``position``) counts as a division, organisation and
strength are taken as ratios when a maximum is present or the value is already within 0..1,
and everything that cannot be found is listed under ``missing`` instead of being guessed.

Checks: own stacks per province (presence, division count, mean organisation and strength);
every enemy stack perception reported must exist in the save, and should be "plausibly
visible", approximated as within one province of an own division. That last test is a
heuristic: real HOI4 visibility also depends on intel, radar and air recon.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ... import clausewitz
from ..contracts import Country, PlayerObservation

_DIVISION_KEYS = ("division", "divisions", "land_division")  # a block under one of these is a division
_PROVINCE_KEYS = ("location", "province", "position")
_ORG_KEYS = ("organisation", "organization", "org")
_STRENGTH_KEYS = ("strength", "str")
_MAX_KEYS = {"organisation": "max_organisation", "organization": "max_organization", "org": "max_org",
             "strength": "max_strength", "str": "max_str"}


@dataclass(frozen=True)
class SaveDivision:
    country: str
    province_id: int
    organization: float | None
    strength: float | None


def _ratio(block: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = block.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        maximum = block.get(_MAX_KEYS[key])
        if isinstance(maximum, (int, float)) and not isinstance(maximum, bool) and maximum > 0:
            return min(1.0, max(0.0, float(value) / float(maximum)))
        if 0.0 <= float(value) <= 1.0:
            return float(value)
        return None  # an absolute value without its maximum cannot be compared with a bar
    return None


def _walk(node: Any, inside: bool, country: str, found: list[SaveDivision]) -> None:
    if isinstance(node, list):
        for item in node:
            _walk(item, inside, country, found)
        return
    if not isinstance(node, dict):
        return
    province = next((node[k] for k in _PROVINCE_KEYS if isinstance(node.get(k), int)
                     and not isinstance(node.get(k), bool)), None)
    if inside and province is not None:
        found.append(SaveDivision(country, int(province), _ratio(node, _ORG_KEYS),
                                  _ratio(node, _STRENGTH_KEYS)))
        return
    for key, value in node.items():
        _walk(value, inside or key in _DIVISION_KEYS, country, found)


_TAGS = (Country.BLUE.value, Country.RED.value)


def extract_divisions(data: Any, countries: tuple[str, ...] = _TAGS) -> tuple[list[SaveDivision], list[str]]:
    """(divisions, missing). Looks under ``countries.<TAG>`` first, then a top-level ``<TAG>``."""
    found: list[SaveDivision] = []
    missing: list[str] = []
    root = data if isinstance(data, dict) else {}
    for tag in countries:
        holder = root.get("countries") if isinstance(root.get("countries"), dict) else root
        block = holder.get(tag) if isinstance(holder, dict) else None
        if not isinstance(block, dict):
            block = root.get(tag)
        if not isinstance(block, dict):
            missing.append(f"country block {tag}")
            continue
        before = len(found)
        _walk(block, False, tag, found)
        if len(found) == before:
            missing.append(f"divisions of {tag}")
    return found, missing


def _mean(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


def _accuracy(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def audit_observation(observation: PlayerObservation, divisions: list[SaveDivision],
                      missing: list[str] | tuple[str, ...] = (), province_map: dict[int, int] | None = None,
                      bar_tolerance: float = 0.1) -> dict[str, Any]:
    """Per-field accuracy. ``province_map`` translates save province ids to layout ids."""
    translate = province_map or {}
    own_tag, enemy_tag = observation.country.value, observation.country.opponent.value
    truth: dict[str, dict[int, list[SaveDivision]]] = {own_tag: {}, enemy_tag: {}}
    for division in divisions:
        if division.country in truth:
            truth[division.country].setdefault(translate.get(division.province_id, division.province_id),
                                               []).append(division)
    seen: dict[int, list[Any]] = {}
    for unit in observation.units:
        if unit.country is observation.country:
            seen.setdefault(unit.province_id, []).append(unit)
    own_truth = truth[own_tag]
    count_hits = org_n = str_n = org_hits = str_hits = 0
    org_error = str_error = 0.0
    for province, real in own_truth.items():
        units = seen.get(province, [])
        if sum(u.count for u in units) == len(real):
            count_hits += 1
        for field_name in ("organization", "strength"):
            expected = _mean([getattr(d, field_name) for d in real])
            weights = [(getattr(u, field_name), u.count) for u in units if getattr(u, field_name) is not None]
            if expected is None or not weights:
                continue
            perceived = sum(v * w for v, w in weights) / sum(w for _, w in weights)
            error = abs(perceived - expected)
            if field_name == "organization":
                org_n, org_error, org_hits = org_n + 1, org_error + error, org_hits + (error <= bar_tolerance)
            else:
                str_n, str_error, str_hits = str_n + 1, str_error + error, str_hits + (error <= bar_tolerance)
    neighbors = {p.id: set(p.neighbors) for p in observation.provinces}
    near_own = set(own_truth) | {n for province in own_truth for n in neighbors.get(province, set())}
    enemy_units = [u for u in observation.units if u.country is not observation.country]
    exists = [u for u in enemy_units if u.province_id in truth[enemy_tag]]
    return {
        "own_province_recall": _accuracy(sum(p in seen for p in own_truth), len(own_truth)),
        "own_province_precision": _accuracy(sum(p in own_truth for p in seen), len(seen)),
        "own_count_accuracy": _accuracy(count_hits, len(own_truth)),
        "own_organization_accuracy": _accuracy(org_hits, org_n),
        "own_organization_mae": round(org_error / org_n, 4) if org_n else None,
        "own_strength_accuracy": _accuracy(str_hits, str_n),
        "own_strength_mae": round(str_error / str_n, 4) if str_n else None,
        "enemy_reported_exists": _accuracy(len(exists), len(enemy_units)),
        "enemy_count_accuracy": _accuracy(
            sum(u.count == len(truth[enemy_tag][u.province_id]) for u in exists), len(exists)),
        "enemy_plausibly_visible": _accuracy(sum(u.province_id in near_own for u in enemy_units),
                                             len(enemy_units)),
        "bar_tolerance": bar_tolerance,
        "save_divisions": {tag: sum(len(v) for v in provinces.values()) for tag, provinces in truth.items()},
        "missing": list(missing),
    }


def audit_file(observation: PlayerObservation, save_path: str | Path,
               province_map: dict[int, int] | None = None) -> dict[str, Any]:
    divisions, missing = extract_divisions(clausewitz.parse_file(save_path))
    return audit_observation(observation, divisions, missing, province_map)


def counter_truth_report(readings: list[Any], truth: dict[str, Any], reach: tuple[float, float] = (16.0, 8.0)
                         ) -> dict[str, Any]:
    """Score counter detections against a hand-labelled truth file (see tests/data/arena/*.truth.json).

    Per relation: recall over ``full`` and ``partial`` labels, precision over detections (one on a
    ``sliver`` label is neither hit nor false positive), and count accuracy over matched labels
    whose count is known, where an unread count (None) scores as wrong. ``readings`` need
    ``bbox``, ``relation`` and ``count``. Labels are a human reading, not game state.
    """
    labels = truth["counters"]
    used: set[int] = set()
    pairs: list[tuple[Any, dict[str, Any]]] = []
    extra: list[Any] = []
    for reading in readings:
        cx, cy = (reading.bbox[0] + reading.bbox[2]) / 2.0, (reading.bbox[1] + reading.bbox[3]) / 2.0
        near = [(abs(label["center"][0] - cx) + abs(label["center"][1] - cy), index)
                for index, label in enumerate(labels)
                if index not in used and label["relation"] == reading.relation
                and abs(label["center"][0] - cx) <= reach[0] and abs(label["center"][1] - cy) <= reach[1]]
        if near:
            used.add(min(near)[1])
            pairs.append((reading, labels[min(near)[1]]))
        else:
            extra.append(reading)
    report: dict[str, Any] = {}
    for relation in sorted({label["relation"] for label in labels} | {r.relation for r in readings}):
        def scored(label: dict[str, Any]) -> bool:
            return bool(label["relation"] == relation and label["visibility"] != "sliver")

        wanted = [label for label in labels if scored(label)]
        hits = [(r, label) for r, label in pairs if scored(label)]
        counted = [(r, label) for r, label in hits if label["count"] is not None]
        false = [r for r in extra if r.relation == relation]
        report[relation] = {
            "labelled": len(wanted), "found": len(hits), "recall": _accuracy(len(hits), len(wanted)),
            "recall_full": _accuracy(sum(label["visibility"] == "full" for _, label in hits),
                                     sum(label["visibility"] == "full" for label in wanted)),
            "false_positives": len(false), "precision": _accuracy(len(hits), len(hits) + len(false)),
            "count_labelled": len(counted),
            "count_accuracy": _accuracy(sum(r.count == label["count"] for r, label in counted), len(counted)),
        }
    inside = [r for r in readings for x0, y0, x1, y1 in truth.get("naval_boxes", [])
              if x0 <= (r.bbox[0] + r.bbox[2]) / 2 <= x1 and y0 <= (r.bbox[1] + r.bbox[3]) / 2 <= y1]
    report["naval_misread_as_land"] = len(inside)
    return report

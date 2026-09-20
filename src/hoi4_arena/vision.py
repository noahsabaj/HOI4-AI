"""Pixel-only visual confirmation for setup, outcomes, and fault detection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

# A screen state must repeat this many times before it is believed.
OUTCOME_FRAMES = 3

# Two separate budgets guard a screen that has stopped matching "healthy".
#
# UNKNOWN_FRAMES bounds a screen matching no template at all: nothing is in flight, so
# there is nothing to wait for. TERMINAL_GRACE_FRAMES bounds the whole unhealthy stretch
# once some terminal template is matching. It must exceed OUTCOME_FRAMES, because the
# debounce cannot even start until the panel first renders, and the transition frames in
# between are spent before it does. A single budget of OUTCOME_FRAMES is wrong in both
# directions: it expires before a real victory converges, and it was the absence of any
# bound at all that previously let a flickering candidate suppress the liveness gates.
UNKNOWN_FRAMES = OUTCOME_FRAMES
TERMINAL_GRACE_FRAMES = 4 * OUTCOME_FRAMES


def _open_spec(screenshot, rules):
    path = Path(rules)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(screenshot).convert("RGB")
    spec = (
        json.loads(path.read_text())
        if path.exists()
        else {"resolution": list(image.size), "rules": {}}
    )
    if spec["resolution"] != list(image.size):
        raise ValueError("Resolution mismatch")
    return path, image, spec


def _checked_rect(image, rect, label):
    x, y, w, h = rect
    if min(x, y) < 0 or min(w, h) < 1 or x + w > image.width or y + h > image.height:
        raise ValueError(f"{label} rectangle outside screenshot")
    return [int(v) for v in rect]


def add_template(screenshot, rules, name, rect):
    import re

    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("Use a simple rule name")
    path, image, spec = _open_spec(screenshot, rules)
    x, y, w, h = _checked_rect(image, rect, "Template")
    if name in spec["rules"]:
        raise ValueError("Use a new rule name or explicitly remove the old calibration")
    image.crop((x, y, x + w, y + h)).save(path.parent / f"{name}.png")
    spec["rules"][name] = {"rect": [x, y, w, h], "template": f"{name}.png", "max_mae": 5}
    path.write_text(json.dumps(spec, indent=2))
    return {"rule": name, "calibration": str(path)}


def set_clock_rect(screenshot, rules, rect):
    """Calibrate the changing-clock ROI. Collection refuses to start without it."""
    path, image, spec = _open_spec(screenshot, rules)
    checked = _checked_rect(image, rect, "Clock")
    existing = spec.get("clock_rect")
    if existing is not None and list(existing) != checked:
        raise ValueError("Explicitly remove the old clock calibration before recalibrating")
    spec["clock_rect"] = checked
    path.write_text(json.dumps(spec, indent=2))
    return {"clock_rect": checked, "calibration": str(path)}


class ScreenRules:
    def __init__(self, path):
        path = Path(path)
        spec = json.loads(path.read_text())
        self.width, self.height = spec["resolution"]
        self.clock_rect = spec.get("clock_rect")
        if self.clock_rect is not None:
            # A hand-edited rules.json can carry floats, which would slice the frame with a
            # TypeError deep inside the match loop. Reject them here instead.
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in self.clock_rect):
                raise ValueError("clock_rect must be four integers")
            x, y, w, h = self.clock_rect
            if min(x, y) < 0 or min(w, h) < 1 or x + w > self.width or y + h > self.height:
                raise ValueError("Calibrated clock_rect lies outside the calibrated resolution")
        self.rules = spec["rules"]
        self.templates = {}
        for name, rule in self.rules.items():
            self.templates[name] = np.asarray(
                Image.open(path.parent / rule["template"]).convert("RGB"), dtype=np.float32
            )
        self.last = None
        self.count = 0

    def require_match_rules(self):
        required = {"ready", "healthy", "running_speed_two", "win", "loss", "disconnect", "desync"}
        if not required.issubset(self.rules) or self.clock_rect is None:
            raise ValueError(
                "Calibrate all match outcomes, running speed two, and clock_rect before collection"
            )

    def matches(self, name, rgb):
        if rgb.shape[:2] != (self.height, self.width):
            raise ValueError("Screen rules require their calibrated resolution")
        x, y, w, h = self.rules[name]["rect"]
        roi = rgb[y : y + h, x : x + w].astype(np.float32)
        target = self.templates[name]
        if target.shape != roi.shape:
            raise ValueError("Template and screen region disagree")
        return float(np.abs(roi - target).mean()) <= self.rules[name].get("max_mae", 5)

    def outcome(self, rgb):
        matched = [
            name
            for name in ("win", "loss", "disconnect", "desync")
            if name in self.rules and self.matches(name, rgb)
        ]
        if len(matched) > 1:
            return "invalid"
        state = matched[0] if matched else None
        self.count = self.count + 1 if state == self.last else 1
        self.last = state
        return state if self.count >= OUTCOME_FRAMES else None


def run_setup(desktop, rules: ScreenRules, recipe):
    """Guard each setup action by its screenshot; never run this during an episode."""
    import time

    desktop.arm(setup=True)
    try:
        for step in recipe:
            expiry = time.monotonic() + step.get("timeout", 30)
            while not rules.matches(step["expect"], desktop.capture().rgb):
                if time.monotonic() >= expiry:
                    raise TimeoutError(f"Setup screen not found: {step['expect']}")
                time.sleep(0.1)
            # Re-arm only at an observed setup boundary; match code never auto-rearms.
            desktop.arm(setup=True)
            for event in step.get("events", []):
                desktop.apply([event])
                time.sleep(step.get("event_delay", 0.25))
        if not rules.matches("ready", desktop.capture().rgb):
            raise RuntimeError("Setup did not reach its calibrated ready screen")
    finally:
        desktop.release()

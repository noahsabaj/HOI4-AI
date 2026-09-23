"""Pixel-only visual confirmation for setup, outcomes, and fault detection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

# A screen state must repeat this many times before it is believed.
OUTCOME_FRAMES = 3

# How far a crop may drift from its template and still count as the same screen. Measured
# on a live 3840x2160 match on 2026-09-21: two captures seconds apart differ by a mean of
# 6.9 over the HUD icon row while the game runs, and 1.6 with it paused. The running drift
# is the day/night terminator sweeping the map under a translucent HUD, so nothing on
# screen holds still to within 5. A rule calibrated at the old fixed 5 rejects the very
# screen it was cut from one frame later.
DEFAULT_MAX_MAE = 5

# How far two captures of the clock may differ and still be the same reading. The match
# loop compared them for exact equality, which no two captures of this game ever satisfy:
# measured on 2026-09-21, the clock crop of a *paused* game differs from itself by a mean
# of 6.4 between captures, and by 44.8 while the clock is advancing. Exact equality made
# every frame look like a fresh tick, so `game_clock_stalled` could never fire and a
# paused game would have been stepped for the full match. Sits between the two.
CLOCK_STILL_MAE = 15.0


def clock_advanced(current, previous):
    """Whether the clock reads differently than it did, against capture noise."""
    if previous is None or current.shape != previous.shape:
        return True
    return float(np.abs(current.astype(np.float32) - previous.astype(np.float32)).mean()) > (
        CLOCK_STILL_MAE
    )


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


def add_template(screenshot, rules, name, rect, max_mae=DEFAULT_MAX_MAE):
    import re

    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("Use a simple rule name")
    if not 1 <= max_mae <= 96:
        raise ValueError("max_mae outside 1..96 either never matches or matches anything")
    path, image, spec = _open_spec(screenshot, rules)
    x, y, w, h = _checked_rect(image, rect, "Template")
    if name in spec["rules"]:
        raise ValueError("Use a new rule name or explicitly remove the old calibration")
    image.crop((x, y, x + w, y + h)).save(path.parent / f"{name}.png")
    spec["rules"][name] = {
        "rect": [x, y, w, h],
        "template": f"{name}.png",
        "max_mae": int(max_mae),
    }
    path.write_text(json.dumps(spec, indent=2))
    return {"rule": name, "calibration": str(path), "max_mae": int(max_mae)}


def set_clock_rect(screenshot, rules, rect):
    """Calibrate the changing-clock ROI. Collection refuses to start without it."""
    return _set_rect(screenshot, rules, rect, "clock_rect", "Clock", "clock calibration")


def set_minimap_rect(screenshot, rules, rect):
    """Calibrate the map area the territory reward reads.

    HOI4 has no minimap, so this is the part of the main view where the whole arena sits
    when the camera is zoomed all the way out. The screenshot should show exactly that:
    the width of the land in it is stored as `minimap_span`, and the reward is read only
    from frames whose land spans the same width, so a zoomed-in view that shows part of
    the arena never counts as a change of territory.
    """
    result = _set_rect(screenshot, rules, rect, "minimap_rect", "Minimap", "minimap calibration")
    path, image, spec = _open_spec(screenshot, rules)
    x, y, w, h = result["minimap_rect"]
    span = land_span(np.asarray(image.convert("RGB"))[y : y + h, x : x + w])
    if span:
        spec["minimap_span"] = span
        path.write_text(json.dumps(spec, indent=2))
    result["minimap_span"] = span
    return result


def _set_rect(screenshot, rules, rect, key, label, replacement):
    path, image, spec = _open_spec(screenshot, rules)
    checked = _checked_rect(image, rect, label)
    existing = spec.get(key)
    if existing is not None and list(existing) != checked:
        raise ValueError(f"Explicitly remove the old {replacement} before recalibrating")
    spec[key] = checked
    path.write_text(json.dumps(spec, indent=2))
    return {key: checked, "calibration": str(path)}


# The colours the map generator writes for the two countries.
BLUE = (40, 100, 220)
RED = (220, 60, 60)


def country_pixels(crop):
    """Masks of the pixels that read as Blue's and as Red's land.

    The map draws country colour faintly over terrain, so the land is nowhere near the
    written colours: measured at 1080p, Blue's land is about (120, 134, 145) and Red's
    (168, 145, 131). What survives is the tint. Land is brighter than the sea (about
    (33, 43, 61)), bluish land is Blue's and reddish land is Red's, and grey interface
    chrome, which is neither, is ignored. Occupied land takes its occupier's colour.

    Lit cloud reads as bluish land too, so only the largest connected patch counts:
    the arena is one piece, because the two countries share a border, and clouds drift
    over the sea as separate patches.
    """
    import cv2

    pixels = np.asarray(crop, dtype=np.int32)
    if pixels.size == 0 or pixels.ndim != 3:
        return None, None
    r, b = pixels[..., 0], pixels[..., 2]
    land = pixels.sum(-1) > 250
    blue, red = land & (b - r > 10), land & (r - b > 15)
    # Four-neighbour connectivity, as scipy's label used by default; OpenCV's labelling
    # measured about twice as fast. Label 0 is the background in both.
    count, labels = cv2.connectedComponents((blue | red).astype(np.uint8), connectivity=4)
    count -= 1
    if count > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        arena = labels == int(sizes.argmax())
        blue, red = blue & arena, red & arena
    return blue, red


def find_template(rgb, template, threshold):
    """Where `template` sits anywhere in `rgb`, as the centre in screen fractions, or None.

    The best normalised squared difference must be below `threshold`: 0 is an exact copy.
    A fixed-rect rule reads one known place with ScreenRules; this searches the whole
    frame, for things such as a popup's Ok button that can open anywhere.
    """
    import cv2

    score, _, (x, y), _ = cv2.minMaxLoc(cv2.matchTemplate(rgb, template, cv2.TM_SQDIFF_NORMED))
    if score >= threshold:
        return None
    h, w = template.shape[:2]
    return (x + w / 2) / rgb.shape[1], (y + h / 2) / rgb.shape[0]


def land_span(crop):
    """The width in pixels of the country-coloured land in a crop, or None if none."""
    blue, red = country_pixels(crop)
    if blue is None:
        return None
    columns = np.flatnonzero((blue | red).any(axis=0))
    return int(columns[-1] - columns[0] + 1) if columns.size else None


def occupation_balance(crop, colour=BLUE, span=None, span_tolerance=0.15):
    """`colour`'s share of the pixels that read as one of the two countries.

    A match passes the acting country's colour: Red's reward is Red's share, so the two
    sides move in opposite directions when the front moves. None means the crop is not a
    reading: it contained neither country, or, when `span` is given, its land was not
    that wide, so the camera was not showing the whole arena at the calibrated zoom. The
    caller must not turn None into a swing.
    """
    chosen = tuple(int(v) for v in colour)
    if chosen not in (BLUE, RED):
        raise ValueError("occupation colour must be the blue or red country colour")
    is_blue, is_red = country_pixels(crop)
    if is_blue is None:
        return None
    total = int(is_blue.sum() + is_red.sum())
    if total == 0:
        return None
    if span:
        width = land_span(crop)
        if width is None or abs(width / span - 1) > span_tolerance:
            return None
    owned = is_blue if chosen == BLUE else is_red
    return float(owned.sum() / total)


class ScreenRules:
    def __init__(self, path):
        path = Path(path)
        spec = json.loads(path.read_text())
        self.width, self.height = spec["resolution"]
        self.clock_rect = self._stored_rect(spec.get("clock_rect"), "clock_rect")
        self.minimap_rect = self._stored_rect(spec.get("minimap_rect"), "minimap_rect")
        self.minimap_span = spec.get("minimap_span")
        self.rules = spec["rules"]
        self.templates = {}
        for name, rule in self.rules.items():
            self.templates[name] = np.asarray(
                Image.open(path.parent / rule["template"]).convert("RGB"), dtype=np.float32
            )
        self.last = None
        self.count = 0

    def _stored_rect(self, rect, label):
        # A hand-edited rules.json can carry floats, which would slice the frame with a
        # TypeError deep inside the match loop. Reject them here instead.
        if rect is None:
            return None
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in rect):
            raise ValueError(f"{label} must be four integers")
        x, y, w, h = rect
        if min(x, y) < 0 or min(w, h) < 1 or x + w > self.width or y + h > self.height:
            raise ValueError(f"Calibrated {label} lies outside the calibrated resolution")
        return [int(v) for v in rect]

    def require_match_rules(self):
        required = {
            "ready",
            "healthy",
            "paused",
            "speed",
            "win",
            "loss",
            "disconnect",
            "desync",
        }
        if not required.issubset(self.rules) or self.clock_rect is None:
            raise ValueError(
                "Calibrate all match outcomes, the pause glyph, the speed indicator, "
                "and clock_rect before collection"
            )

    def capture_regions(self):
        """The rects the match loop reads, in a stable order.

        Every rule, then the clock, then the minimap. The worker crops exactly these
        and sends nothing else, so the order here is the wire order. Sorted by name so
        it does not depend on JSON key ordering.
        """
        names = sorted(self.rules)
        rects = [self.rules[name]["rect"] for name in names]
        if self.clock_rect is not None:
            rects.append(self.clock_rect)
        if self.minimap_rect is not None:
            rects.append(self.minimap_rect)
        return names, rects

    def matches_crop(self, name, crop):
        target = self.templates[name]
        if target.shape != crop.shape:
            raise ValueError("Template and screen region disagree")
        return float(np.abs(crop.astype(np.float32) - target).mean()) <= self.rules[name].get(
            "max_mae", DEFAULT_MAX_MAE
        )

    def matches(self, name, rgb):
        if rgb.shape[:2] != (self.height, self.width):
            raise ValueError("Screen rules require their calibrated resolution")
        x, y, w, h = self.rules[name]["rect"]
        return self.matches_crop(name, rgb[y : y + h, x : x + w])

    def observe(self, frame):
        """Bind a capture to these rules, whether it carries a full frame or just crops."""
        return Observation(self, frame)

    def outcome(self, rgb):
        return self.outcome_where(lambda name: self.matches(name, rgb))

    def outcome_where(self, matches):
        matched = [
            name
            for name in ("win", "loss", "disconnect", "desync")
            if name in self.rules and matches(name)
        ]
        if len(matched) > 1:
            return "invalid"
        state = matched[0] if matched else None
        self.count = self.count + 1 if state == self.last else 1
        self.last = state
        return state if self.count >= OUTCOME_FRAMES else None


class Observation:
    """One capture, matched against its rules however the pixels arrived.

    A worker that downscaled on the capture side sends only the calibrated crops, so
    there is no full frame to slice. Everything the match loop asks of a screen goes
    through here, and the two sources must answer identically.
    """

    def __init__(self, rules: ScreenRules, frame):
        self.rules, self.frame = rules, frame
        self.crops = None
        if frame.crops is not None:
            names, rects = rules.capture_regions()
            if len(frame.crops) != len(rects):
                raise ValueError("Worker returned a different region set than was calibrated")
            self.crops = dict(zip(names, frame.crops, strict=False))
            extra = list(frame.crops[len(names) :])
            self.clock = extra.pop(0) if rules.clock_rect is not None else None
            self.minimap = extra.pop(0) if rules.minimap_rect is not None else None
        elif frame.rgb is None:
            raise ValueError("Capture carries neither a full frame nor calibrated crops")
        else:
            self.clock = None
            self.minimap = None

    def matches(self, name):
        if self.crops is not None:
            if name not in self.crops:
                raise KeyError(name)
            return self.rules.matches_crop(name, self.crops[name])
        return self.rules.matches(name, self.frame.rgb)

    def outcome(self):
        return self.rules.outcome_where(self.matches)

    def clock_pixels(self):
        if self.clock is not None:
            return self.clock
        x, y, w, h = self.rules.clock_rect
        return self.frame.rgb[y : y + h, x : x + w]

    def minimap_pixels(self):
        if self.minimap is not None:
            return self.minimap
        x, y, w, h = self.rules.minimap_rect
        return self.frame.rgb[y : y + h, x : x + w]


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

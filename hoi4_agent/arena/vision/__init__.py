"""Vision runtime for the arena: local perception of the map and the order executor.

Screenshots in, mouse and keyboard out; no memory reading. ``add_commands`` exposes the CLI
pieces for the integrator to wire into ``arena/cli.py``. Heavy imports stay inside handlers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable


def _vision_read(args: argparse.Namespace) -> int:
    from PIL import Image

    from .counters import CounterDigits, find_counters
    readings = find_counters(Image.open(args.image), digits=CounterDigits.load(args.digits))
    print(json.dumps({"image": str(args.image), "counters": [r.to_dict() for r in readings]}, indent=2))
    return 0


def _vision_observe(args: argparse.Namespace) -> int:
    """One LIVE observation. Needs the running game; never exercised by the tests."""
    from ...io import windows as win
    from ..contracts import Country
    from ..layout import ArenaLayout
    from .calibration import load_vision_calibration
    from .observe import ArenaObserver
    calibration = load_vision_calibration(args.calibration)
    locator, capture, _ = win.build_io()
    geo = locator.find(calibration.window_title, (calibration.width, calibration.height))
    if geo is None:
        print("vision-observe: HOI4 window not found")
        return 2
    observer = ArenaObserver(ArenaLayout.load(args.layout), calibration, Country(args.country))
    observer.begin_episode("vision-observe")
    observation = observer.observe(capture.grab(geo))
    print(json.dumps({"observation": observation.to_dict(), "confidence": observer.detail.confidence,
                      "unassigned_counters": observer.detail.unassigned,
                      "hour_estimated": observer.detail.hour_estimated}, indent=2))
    return 0


def _calibrate_arena(args: argparse.Namespace) -> int:
    from ..layout import ArenaLayout
    from .calibration import run_wizard
    return run_wizard(ArenaLayout.load(args.layout), args.calibration)


def _vision_audit(args: argparse.Namespace) -> int:
    from ..contracts import PlayerObservation
    from .audit import audit_file
    data = json.loads(args.observation.read_text(encoding="utf-8"))
    observation = PlayerObservation.from_dict(data.get("observation", data))
    mapping = None
    if args.province_map:
        raw = json.loads(args.province_map.read_text(encoding="utf-8"))
        mapping = {int(k): int(v) for k, v in raw.items()}
    print(json.dumps(audit_file(observation, args.save, mapping), indent=2))
    return 0


def add_commands(commands: Any) -> dict[str, Callable[[argparse.Namespace], int]]:
    """Register the vision subcommands on an argparse subparsers object; returns their handlers."""
    from .calibration import DEFAULT_PATH
    read = commands.add_parser("vision-read", help="run the counter reader on an image file and print JSON")
    read.add_argument("--image", type=Path, required=True)
    read.add_argument("--digits", type=Path, default=None, help="directory of calibrated counter_glyph_N.png")
    observe = commands.add_parser("vision-observe", help="one live observation (needs the running game)")
    calibrate = commands.add_parser("calibrate-arena",
                                    help="interactive arena vision calibration (needs the game)")
    for parser in (observe, calibrate):
        parser.add_argument("--layout", type=Path, required=True)
        parser.add_argument("--calibration", type=Path, default=DEFAULT_PATH)
    observe.add_argument("--country", choices=("BLU", "RED"), default="BLU")
    audit = commands.add_parser("vision-audit", help="compare a perceived observation with a text save")
    audit.add_argument("--observation", type=Path, required=True, help="JSON from vision-observe")
    audit.add_argument("--save", type=Path, required=True)
    audit.add_argument("--province-map", type=Path, default=None, help="JSON {save province id: layout id}")
    return {"vision-read": _vision_read, "vision-observe": _vision_observe,
            "calibrate-arena": _calibrate_arena, "vision-audit": _vision_audit}

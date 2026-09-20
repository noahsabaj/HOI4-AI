"""Run a screen-guarded, disposable single-player startup regression.

Templates must come from reviewed prior setup screenshots. This is not the
two-PC reset recipe and must never be invoked during an active match.
"""

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from hoi4_arena.desktop import Desktop
from hoi4_arena.vision import ScreenRules, add_template

parser = argparse.ArgumentParser()
parser.add_argument("output")
args = parser.parse_args()
root = Path(args.output)
root.mkdir(parents=True, exist_ok=False)
steps = [
    ("main", "single/before.png", [1810, 690, 200, 35], 0.5, 0.327),
    ("single", "single/after.png", [1810, 880, 200, 30], 0.5, 0.417),
    ("country", "new/after.png", [1970, 1565, 150, 30], 0.532, 0.733),
    ("map", "select/after.png", [3550, 2090, 170, 30], 0.946, 0.978),
]
rules_path = root / "calibration/rules.json"
for name, image, rect, _, _ in steps:
    add_template("artifacts/worker-menu-v4-" + image, rules_path, name, rect)
rules = ScreenRules(rules_path)
receipts = []
with Desktop() as worker:
    for name, _, _, x, y in steps:
        expiry = time.monotonic() + 15
        while True:
            frame = worker.capture()
            if rules.matches(name, frame.rgb):
                break
            if time.monotonic() >= expiry:
                Image.fromarray(frame.rgb).save(root / f"unexpected-{name}.png")
                raise RuntimeError(f"Expected screen not confirmed: {name}")
            time.sleep(0.2)
        Image.fromarray(frame.rgb).save(root / f"before-{name}.png")
        worker.arm(setup=True)
        for event in [
            {"kind": "move", "x": x, "y": y},
            {"kind": "button", "button": 0, "down": True},
            {"kind": "button", "button": 0, "down": False},
        ]:
            receipts.append(worker.apply([event]))
            time.sleep(0.25)
        worker.release()
        (root / "receipts.json").write_text(json.dumps(receipts, indent=2))
    time.sleep(5)
    Image.fromarray(worker.capture().rgb).save(root / "after-start.png")

"""Audited integration test of one observed menu control; never a combat script."""

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from hoi4_arena.desktop import Desktop
from hoi4_arena.remote import RemoteDesktop

parser = argparse.ArgumentParser()
parser.add_argument("output")
parser.add_argument("--x", type=float, required=True)
parser.add_argument("--y", type=float, required=True)
parser.add_argument("--peer", help="Pairing file for the second-PC worker")
args = parser.parse_args()
root = Path(args.output)
root.mkdir(parents=True, exist_ok=False)
with RemoteDesktop(args.peer) if args.peer else Desktop() as worker:
    frame = worker.capture()
    Image.fromarray(frame.rgb).save(root / "before.png")
    worker.arm(setup=True)
    receipts = [worker.apply([{"kind": "move", "x": args.x, "y": args.y}])]
    time.sleep(0.1)
    receipts.append(worker.apply([{"kind": "button", "button": 0, "down": True}]))
    time.sleep(0.25)
    receipts.append(worker.apply([{"kind": "button", "button": 0, "down": False}]))
    worker.release()
    time.sleep(1.5)
    Image.fromarray(worker.capture().rgb).save(root / "after.png")
    (root / "receipts.json").write_text(json.dumps(receipts, indent=2))

"""Second-PC integration test; start in the observed Single Player submenu."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from hoi4_arena.remote import RemoteDesktop

parser = argparse.ArgumentParser()
parser.add_argument("peer")
parser.add_argument("output")
args = parser.parse_args()
root = Path(args.output)
root.mkdir(parents=True, exist_ok=False)
with RemoteDesktop(args.peer) as worker:
    report = {k: v for k, v in worker.attached.items() if k != "payload"}
    worker.arm(setup=True)
    down = worker.apply([{"kind": "key", "vk": 0x1B, "down": True}])
    time.sleep(0.25)
    up = worker.apply([{"kind": "key", "vk": 0x1B, "down": False}])
    worker.release()
    time.sleep(1.5)
    frame = worker.capture()
    Image.fromarray(frame.rgb).save(root / "after-escape.png")
    worker.arm()
    worker.apply([{"kind": "key", "vk": 0x10, "down": True}])
    held = worker.request("status")["held_keys"]
    time.sleep(1.0)
    status = worker.request("status")
    assert held == [0x10] and status["held_keys"] == [] and not status["armed"]
    captures, pings = [], []
    for _ in range(20):
        start = time.perf_counter()
        worker.request("status")
        pings.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        frame = worker.capture()
        captures.append(
            {
                "roundtrip_ms": (time.perf_counter() - start) * 1000,
                "capture_ms": (frame.meta["t_ns"] - frame.meta["capture_start_ns"]) / 1e6,
                "seq": frame.meta["seq"],
            }
        )
    report.update(
        keyboard_receipts=[down, up],
        keyboard_effect_requires_visual_review=True,
        watchdog_released=True,
        resolution=[frame.meta["width"], frame.meta["height"]],
        ping_p95_ms=float(np.percentile(pings, 95)),
        capture_roundtrip_p95_ms=float(np.percentile([r["roundtrip_ms"] for r in captures], 95)),
        captures=captures,
        game_state="menu",
        gameplay_verified=False,
    )
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "captures"}, indent=2))

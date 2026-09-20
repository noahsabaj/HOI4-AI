"""Explicit live worker integration test. Run with a disposable HOI4 session in focus."""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from hoi4_arena.desktop import Desktop

out = Path("artifacts/worker-smoke")
out.mkdir(parents=True, exist_ok=True)
with Desktop() as desktop:
    before = desktop.capture()
    Image.fromarray(before.rgb).save(out / "before.png")
    desktop.arm(setup=True)
    applied = desktop.apply([{"kind": "key", "vk": 0x1B, "down": True}])
    time.sleep(0.1)
    desktop.apply([{"kind": "key", "vk": 0x1B, "down": False}])
    time.sleep(0.2)
    after = desktop.capture()
    Image.fromarray(after.rgb).save(out / "after.png")
    desktop.arm()
    desktop.apply([{"kind": "key", "vk": 0x10, "down": True}])
    held = desktop.request("status")
    time.sleep(1.0)
    released = desktop.request("status")
    assert held["held_keys"] == [0x10]
    assert released["held_keys"] == [] and not released["armed"]
    result = {
        "watchdog_released": True,
        "capture_shape": list(before.rgb.shape),
        "input_to_capture_ns": after.meta["t_ns"] - applied["t_ns"],
        "pixel_change_mean": float(np.abs(after.rgb.astype(float) - before.rgb).mean()),
        "keyboard_effect_requires_visual_review": True,
        "gameplay_verified": False,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

"""Self-calibration, step 1: hover a grid and read the debug tooltip's province id (DeepSeek OCR).

The game must be running with -debug, paused, at the fixed arena camera. Writes samples JSON:
[{"x":..,"y":..,"province":N|null,"state":N|null}]. Needs the mouse and the HOI4 window in front.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import live  # noqa: E402
from hoi4_agent.brain.deepseek import DeepSeekClient  # noqa: E402
from hoi4_agent.io import windows as w  # noqa: E402

out = Path(sys.argv[1])
step = int(sys.argv[2]) if len(sys.argv) > 2 else 150
geo = live.window()
live.focused(geo)
client = DeepSeekClient()
capture = w.PrintWindowCapture()
samples = json.loads(out.read_text()) if out.exists() else []
done = {(s["x"], s["y"]) for s in samples}
points = [(x, y) for y in range(130, 940, step) for x in range(70, 1850, step)
          if not (y > 900 and 820 < x < 1100) and not (x > 1800 and y > 740) and (x, y) not in done]
for index, (x, y) in enumerate(points):
    if not w.Win32Input().focus(geo):
        raise SystemExit("lost the foreground; stopping")
    live.move(geo, x + 3, y + 3)
    live.move(geo, x, y)
    time.sleep(1.3)
    frame = capture.grab(geo)
    # The tooltip opens beside the cursor and flips near edges: look at a generous window around it.
    box = (max(0, x - 560), max(0, y - 330), min(frame.width, x + 560), min(frame.height, y + 330))
    crop = frame.crop(box)
    try:
        reply = client.chat(system="You read Hearts of Iron IV debug tooltips. Output JSON only.",
                            user=('The black tooltip box contains text like "(State: 973 Province: 6720)". Reply '
                                  '{"province": N, "state": N}. If there is no such tooltip reply {"province": null, "state": null}.'),
                            images=[crop], thinking=False, max_tokens=60, timeout=30)
        data = json.loads(reply.text)
    except Exception as exc:  # keep sampling; a miss is just a missing sample
        data = {"province": None, "state": None, "error": type(exc).__name__}
    samples.append({"x": x, "y": y, **data})
    out.write_text(json.dumps(samples))
    print(index + 1, len(points), x, y, data, flush=True)
live.move(geo, 1900, 700)

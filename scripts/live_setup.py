"""Relaunch the arena, pause, put the camera at the calibrated view and verify it against known starts.

The camera sequence is the one the calibration was made with (wheel +6 at 1150,560; -2 at 960,560;
+1 at 930,620 from the autostart camera). Verification: the four starting counters must map to the
scenario's start provinces through the saved province centres.
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import live  # noqa: E402
from hoi4_agent.arena.launch import hoi4_running  # noqa: E402
from hoi4_agent.arena.vision.counters import find_counters  # noqa: E402
from hoi4_agent.io import windows as w  # noqa: E402

camera = json.loads(Path(sys.argv[1]).read_text())
centres = {int(k): v for k, v in camera["province_centres"].items()}
SKIP_LAUNCH = '--no-launch' in sys.argv
if SKIP_LAUNCH:
    sys.argv.remove('--no-launch')
if hoi4_running() and not SKIP_LAUNCH:
    subprocess.run(["taskkill", "/IM", "hoi4.exe", "/F"], capture_output=True)
    time.sleep(4)
if not SKIP_LAUNCH:
    live.main(["launch"])
log = live.PROFILE / "logs/game.log"
deadline = time.time() + 120
while time.time() < deadline and not (log.exists() and "ARENA_RESET" in log.read_text(errors="replace")):
    time.sleep(2)
time.sleep(6)
geo = live.window()
live.focused(geo)
w.Win32Input().key("space")
time.sleep(0.5)
for ticks, x, y in ((6, 1150, 560), (-2, 960, 560), (1, 930, 620)):
    live.scroll(geo, ticks, x, y)
    time.sleep(1.5)
live.button(geo, 1660, 650, w.MOUSEEVENTF_RIGHTDOWN, w.MOUSEEVENTF_RIGHTUP)  # dismiss the war toast
live.move(geo, 1900, 700)
time.sleep(3)
frame = w.PrintWindowCapture().grab(geo)
frame.save(ROOT / "artifacts/previews/setup_check.png")
# The wheel sequence is not perfectly repeatable (load time changes the anchor), so correct the saved
# centres with a screen-space similarity fitted on the four starting counters, whose provinces are known.
readings = find_counters(frame)
detected = {rel: [((r.bbox[0] + r.bbox[2]) / 2, (r.bbox[1] + r.bbox[3]) / 2) for r in readings if r.relation == rel]
            for rel in ("own", "enemy")}
starts = {"own": (6703, 9690), "enemy": (9660, 3731)}
if any(len(detected[rel]) != 2 for rel in starts):
    print("expected two own and two enemy counters, saw", {k: len(v) for k, v in detected.items()})
    sys.exit(3)
best = None
for own_order in itertools.permutations(detected["own"]):
    for enemy_order in itertools.permutations(detected["enemy"]):
        seen = np.array([*own_order, *enemy_order], float)
        want = np.array([[centres[p][0], centres[p][1] + 38] for p in (*starts["own"], *starts["enemy"])], float)
        design = np.zeros((8, 3))
        design[0::2, 0], design[0::2, 1] = want[:, 0], 1
        design[1::2, 0], design[1::2, 2] = want[:, 1], 1
        solution, *_ = np.linalg.lstsq(design, seen.reshape(-1), rcond=None)
        rms = float(np.sqrt(np.mean((design @ solution - seen.reshape(-1)) ** 2)))
        if best is None or rms < best[0]:
            best = (rms, solution)
rms, (scale, tx, ty) = best
print(f"session camera: scale {scale:.3f} shift ({tx:.0f}, {ty:.0f}) rms {rms:.1f} px")
session = dict(camera, province_centres={str(p): [scale * x + tx, scale * y + ty] for p, (x, y) in centres.items()},
               counter_dy=38 * scale, session_fit_rms=rms)
Path(sys.argv[2] if len(sys.argv) > 2 else "artifacts/previews/camera_session.json").write_text(json.dumps(session))
ok = rms < 12 and 0.5 < scale < 2.0
print("camera verified" if ok else "CAMERA MISMATCH")
sys.exit(0 if ok else 3)

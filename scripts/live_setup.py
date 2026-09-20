"""Relaunch the arena, pause, zoom the camera onto it, and fit this session's screen geometry.

The mouse wheel is not repeatable (where a zoom lands depends on where the camera started), so
instead of a fixed sequence this zooms until the arena is the right SIZE on screen, judged by the
four starting counters, whose provinces are known. The same four counters then give a similarity
(scale and shift) that maps the saved province centres into this session's view.
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
out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "artifacts/previews/camera_session.json")
centres = {int(k): v for k, v in camera["province_centres"].items()}
STARTS = {"own": (6703, 9690), "enemy": (9660, 3731)}
WANT = (0.80, 1.30)  # acceptable on-screen scale relative to the calibration view
SKIP_LAUNCH = "--no-launch" in sys.argv
if SKIP_LAUNCH:
    sys.argv.remove("--no-launch")
if hoi4_running() and not SKIP_LAUNCH:
    subprocess.run(["taskkill", "/IM", "hoi4.exe", "/F"], capture_output=True)
    time.sleep(4)
if not SKIP_LAUNCH:
    live.main(["launch"])
    log = live.PROFILE / "logs/game.log"
    deadline = time.time() + 150
    while time.time() < deadline and not (log.exists() and "ARENA_RESET" in log.read_text(errors="replace")):
        time.sleep(2)
    time.sleep(6)
geo = live.window()
live.focused(geo)
capture = w.PrintWindowCapture()


def fit(frame) -> tuple[float, float, float, float] | None:
    """Similarity (scale, dx, dy) from saved centres to this frame, with its rms, from the four starts."""
    seen: dict[str, list[tuple[float, float]]] = {}
    for r in find_counters(frame):
        if r.relation in STARTS:
            seen.setdefault(r.relation, []).append(((r.bbox[0] + r.bbox[2]) / 2, (r.bbox[1] + r.bbox[3]) / 2))
    if any(len(seen.get(rel, [])) != 2 for rel in STARTS):
        return None
    best = None
    for own_order in itertools.permutations(seen["own"]):
        for enemy_order in itertools.permutations(seen["enemy"]):
            observed = np.array([*own_order, *enemy_order], float).reshape(-1)
            want = np.array([[centres[p][0], centres[p][1] + 38] for p in (*STARTS["own"], *STARTS["enemy"])], float)
            design = np.zeros((8, 3))
            design[0::2, 0], design[0::2, 1] = want[:, 0], 1
            design[1::2, 0], design[1::2, 2] = want[:, 1], 1
            solution, *_ = np.linalg.lstsq(design, observed, rcond=None)
            rms = float(np.sqrt(np.mean((design @ solution - observed) ** 2)))
            if best is None or rms < best[3]:
                best = (float(solution[0]), float(solution[1]), float(solution[2]), rms)
    return best


def counters(frame) -> list[tuple[float, float]]:
    return [((r.bbox[0] + r.bbox[2]) / 2, (r.bbox[1] + r.bbox[3]) / 2)
            for r in find_counters(frame) if r.relation in ("own", "enemy")]


def coarse_arena(frame) -> tuple[float, float] | None:
    """Roughly where the two arena countries are, asked of the cloud vision model.

    Counters merge into a stack icon at wide zoom, so the classical reader sees nothing and
    cannot lead the camera in. The model only has to get within a province or so; the counter
    hunt takes over from there. Pointing is its weakest skill, so the answer is never trusted
    as a click target.
    """
    try:
        from hoi4_agent.brain.deepseek import DeepSeekClient

        reply = DeepSeekClient().chat(
            system="You read Hearts of Iron IV map screenshots. Output JSON only.",
            user=('Most of this world map is one grey country. Exactly two small countries are coloured, one '
                  'light blue and one red, side by side in central Europe. Give the centre of that coloured '
                  'pair as json {"x": X, "y": Y}, integers 0-1000 normalised to this image. If you cannot see '
                  'them, reply {"x": null, "y": null}.'),
            images=[frame], thinking=False, max_tokens=60, timeout=40)
        data = json.loads(reply.text)
        if isinstance(data.get("x"), int) and isinstance(data.get("y"), int):
            return data["x"] / 1000 * frame.width, data["y"] / 1000 * frame.height
    except Exception as exc:  # the hunt still has its own fallbacks
        print("coarse locate unavailable:", type(exc).__name__, exc)
    return None


w.Win32Input().key("space")  # pause: zooming while the clock runs races the AI
time.sleep(0.5)
# Where the camera starts is not fixed, and zooming at a guessed point walks off across Europe
# (one run ended over the Netherlands), so hunt for the units themselves: pull back until some
# counter is on screen, then zoom in on the counters and let them lead the camera in.
for pull in range(6):
    seen = counters(capture.grab(geo))
    if seen:
        break
    live.scroll(geo, -3, 960, 540)
    time.sleep(1.5)
for attempt in range(12):
    live.move(geo, 1900, 700)
    time.sleep(0.8)
    frame = capture.grab(geo)
    result = fit(frame)
    if result is None:  # the four counters are not all visible yet: zoom in on what is
        seen = counters(frame)
        if seen:
            live.scroll(geo, 1, sum(x for x, _ in seen) / len(seen), sum(y for _, y in seen) / len(seen))
        elif (where := coarse_arena(frame)) is not None:
            live.scroll(geo, 2, *where)
        else:
            live.scroll(geo, -2, 960, 540)
        time.sleep(1.5)
        continue
    scale, dx, dy, _ = result
    anchor = ((scale * (centres[6703][0] + centres[3731][0]) / 2 + dx),
              (scale * (centres[6703][1] + centres[3731][1]) / 2 + dy))
    if scale < WANT[0]:
        live.scroll(geo, 1, *anchor)
    elif scale > WANT[1]:
        live.scroll(geo, -1, *anchor)
    else:
        break
    time.sleep(1.5)
frame = capture.grab(geo)
frame.save(ROOT / "artifacts/previews/setup_check.png")
result = fit(frame)
if result is None:
    print("cannot see the four starting counters")
    sys.exit(3)
scale, dx, dy, rms = result
print(f"session camera: scale {scale:.3f} shift ({dx:.0f}, {dy:.0f}) rms {rms:.1f} px")
session = dict(camera, province_centres={str(p): [scale * x + dx, scale * y + dy] for p, (x, y) in centres.items()},
               counter_dy=38 * scale, session_fit_rms=rms, session_scale=scale)
out_path.write_text(json.dumps(session))
ok = rms < 12 and WANT[0] <= scale <= WANT[1]
print("camera verified" if ok else "CAMERA MISMATCH")
sys.exit(0 if ok else 3)

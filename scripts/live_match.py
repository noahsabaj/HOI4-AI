"""Minimal LIVE match loop on the vanilla-region arena: one agent as BLU against the built-in AI.

Bring-up quality, stack-level control: each own counter is one controllable stack (id = its province);
orders are a left-click on the counter and a right-click on the calibrated target centre, unverified.
VP holders, game day and the outcome come from the mod's ARENA_* log lines (a log shortcut, not vision).
Usage: python scripts/live_match.py <agent spec> <camera.json> <out.json> [max_minutes] [speed_ups]
Assumes the game is running, paused, at the calibrated camera. Takes over the mouse.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import live  # noqa: E402
from hoi4_agent.arena.agents import build_agent  # noqa: E402
from hoi4_agent.arena.contracts import Country, PlayerObservation, UnitView, Verb  # noqa: E402
from hoi4_agent.arena.layout import ArenaLayout  # noqa: E402
from hoi4_agent.arena.vision.counters import find_counters  # noqa: E402
from hoi4_agent.io import windows as w  # noqa: E402

spec, camera_path, out_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
max_minutes = float(sys.argv[4]) if len(sys.argv) > 4 else 12
speed_ups = int(sys.argv[5]) if len(sys.argv) > 5 else 3
PROFILE = ROOT / "artifacts/arena_mod/default"
layout = ArenaLayout.load(PROFILE / "arena_layout.json")
camera = json.loads(camera_path.read_text())
centres = {int(k): v for k, v in camera["province_centres"].items()}
COUNTER_DY = -camera.get("counter_dy", 38)  # a counter hangs this far below its province's unit position at the calibrated zoom
log_path = PROFILE / "logs/game.log"
geo = live.window()
capture = w.PrintWindowCapture()
agent = build_agent(spec, seed=0, allow_pause=False)
agent.reset()


def log_state() -> dict:
    text = log_path.read_text(errors="replace")
    state: dict = {"day": 0, "holders": {}, "outcome": None, "tick": None}
    for line in text.splitlines():
        if "ARENA_TICK" in line:
            state["tick"] = line[line.index("ARENA_TICK"):]
            state["day"] = int(re.search(r"day=(\d+)", line).group(1))
        elif "ARENA_VP" in line:
            state["holders"] = {int(k): int(v) for k, v in re.findall(r"(\d+)=(\d)", line[line.index("ARENA_VP"):])
                                if int(k) in centres}
        elif "ARENA_OUTCOME" in line:
            state["outcome"] = line[line.index("ARENA_OUTCOME"):]
    return state


def province_of(reading) -> int:
    cx, cy = (reading.bbox[0] + reading.bbox[2]) / 2, (reading.bbox[1] + reading.bbox[3]) / 2 + COUNTER_DY
    return min(centres, key=lambda p: (centres[p][0] - cx) ** 2 + (centres[p][1] - cy) ** 2)


def observe(sequence: int, state: dict):
    frame = capture.grab(geo)
    readings = [r for r in find_counters(frame) if r.relation in ("own", "enemy")]
    holder = {1: Country.BLUE, 2: Country.RED}
    provinces = tuple(p.view(holder.get(state["holders"].get(p.id, -1),
                                        Country(p.initial_controller) if p.initial_controller else None))
                      for p in layout.provinces)
    units, where = {}, {}
    for r in readings:
        pid = province_of(r)
        own = r.relation == "own"
        uid = pid if own else 100000 + pid
        if uid in units:
            continue
        units[uid] = UnitView(uid, Country.BLUE if own else Country.RED, pid, r.organization, r.strength, None,
                              count=max(1, r.count or 1), confidence=float(r.confidence))
        where[uid] = r
    obs = PlayerObservation("live-episode", Country.BLUE, sequence, state["day"] * 24, time.monotonic_ns(),
                            provinces, tuple(units.values()))
    return obs, where, frame


records, start = [], time.time()
live.focused(geo)
for _ in range(speed_ups):
    w.Win32Input().key("add")
    time.sleep(0.15)
w.Win32Input().key("space")  # unpause
sequence, outcome = 0, None
while time.time() - start < max_minutes * 60:
    state = log_state()
    if state["outcome"]:
        outcome = state["outcome"]
        break
    if not w.Win32Input().focus(geo):
        print("lost the foreground; stopping")
        break
    obs, where, frame = observe(sequence, state)
    if not obs.units and not list(out_path.parent.glob(out_path.stem + '_blank*.png')):
        frame.save(out_path.parent / f'{out_path.stem}_blank{sequence}.png')
    began = time.perf_counter()
    order = agent.act(obs, None)
    decided_ms = (time.perf_counter() - began) * 1000
    done = "noop"
    if order.verb in (Verb.MOVE, Verb.SUPPORT_ATTACK) and order.unit_ids and order.unit_ids[0] in where:
        r = where[order.unit_ids[0]]
        live.button(geo, (r.bbox[0] + r.bbox[2]) / 2 - 8, (r.bbox[1] + r.bbox[3]) / 2,
                    w.MOUSEEVENTF_LEFTDOWN, w.MOUSEEVENTF_LEFTUP)
        time.sleep(0.12)
        tx, ty = centres[order.target_province_id]
        live.button(geo, tx, ty, w.MOUSEEVENTF_RIGHTDOWN, w.MOUSEEVENTF_RIGHTUP)
        time.sleep(0.08)  # no Escape here: with nothing selected it would open the game menu
        done = "sent"
    records.append({"t": round(time.time() - start, 1), "day": state["day"], "verb": order.verb.value,
                    "units": list(order.unit_ids), "target": order.target_province_id, "executed": done,
                    "own": sum(u.country == Country.BLUE for u in obs.units),
                    "enemy_visible": sum(u.country == Country.RED for u in obs.units),
                    "decide_ms": round(decided_ms, 1), "tick": state["tick"]})
    if sequence % 10 == 0:
        print(records[-1], flush=True)
    sequence += 1
    live.move(geo, 1900, 700)
    time.sleep(0.7)
w.Win32Input().key("space")  # pause again
agent.close()
final = log_state()
out_path.write_text(json.dumps({"agent": spec, "source": "hoi4_vision", "bring_up": True, "outcome": outcome,
                                "final_tick": final["tick"], "final_holders": final["holders"],
                                "decisions": len(records), "minutes": round((time.time() - start) / 60, 2),
                                "records": records}, indent=1))
print("outcome:", outcome, "| final:", final["tick"], "| decisions:", len(records))

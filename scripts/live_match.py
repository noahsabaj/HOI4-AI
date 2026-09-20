"""Minimal LIVE match loop on the vanilla-region arena: one agent as BLU against the built-in AI.

Bring-up quality, stack-level control: each own counter is one controllable stack (id = its province).
An order is a left-click on the counter then a right-click on the calibrated target centre, and counts
as confirmed only when the movement arrow appears along that line. Standing orders are remembered, so
the agent is not handed a blank slate every tick: re-issuing an order restarts the battle it is fighting.
Game day, VP holders and the outcome come from the mod's ARENA_* log lines, a log shortcut, not vision.

Usage: python scripts/live_match.py <agent spec> <camera.json> <out.json> [max_minutes] [speed]
Assumes the game is running at the calibrated camera. Takes over the mouse.
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
from hoi4_agent.arena.vision.executor import line_change  # noqa: E402
from hoi4_agent.arena.vision.topbar import read_top_bar  # noqa: E402
from hoi4_agent.io import windows as w  # noqa: E402

spec, camera_path, out_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
max_minutes = float(sys.argv[4]) if len(sys.argv) > 4 else 12
want_speed = int(sys.argv[5]) if len(sys.argv) > 5 else 3
PROFILE = ROOT / "artifacts/arena_mod/default"
layout = ArenaLayout.load(PROFILE / "arena_layout.json")
camera = json.loads(camera_path.read_text())
centres = {int(k): v for k, v in camera["province_centres"].items()}
COUNTER_DY = camera.get("counter_dy", 38)
DESELECT_AT = (200.0, 200.0)  # empty map well away from the arena counters
log_path = PROFILE / "logs/game.log"
geo = live.window()
capture = w.PrintWindowCapture()
agent = build_agent(spec, seed=0, allow_pause=False)
agent.reset()
standing: dict[int, int] = {}  # province of a stack -> province it was last ordered to enter
diagnostics = {"orders_sent": 0, "orders_confirmed": 0, "frames_without_own": 0, "reissue_skipped": 0}


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
    cx = (reading.bbox[0] + reading.bbox[2]) / 2
    cy = (reading.bbox[1] + reading.bbox[3]) / 2 - COUNTER_DY
    return min(centres, key=lambda p: (centres[p][0] - cx) ** 2 + (centres[p][1] - cy) ** 2)


def observe(sequence: int, state: dict):
    frame = capture.grab(geo)
    holder = {1: Country.BLUE, 2: Country.RED}
    provinces = tuple(p.view(holder.get(state["holders"].get(p.id, -1),
                                        Country(p.initial_controller) if p.initial_controller else None))
                      for p in layout.provinces)
    units, where = {}, {}
    for r in find_counters(frame):
        if r.relation not in ("own", "enemy"):
            continue
        pid = province_of(r)
        own = r.relation == "own"
        uid = pid if own else 100000 + pid
        if uid in units:
            continue
        units[uid] = UnitView(uid, Country.BLUE if own else Country.RED, pid, r.organization, r.strength, None,
                              count=max(1, r.count or 1), confidence=float(r.confidence),
                              order_target_province_id=standing.get(pid) if own else None)
        where[uid] = r
    for pid in [p for p in standing if p not in units]:  # the stack left its province, so its order is spent
        standing.pop(pid, None)
    obs = PlayerObservation("live-episode", Country.BLUE, sequence, state["day"] * 24, time.monotonic_ns(),
                            provinces, tuple(units.values()), game_speed=want_speed)
    return obs, where, frame


def issue(order, reading, before) -> bool:
    """Click the counter, right-click the target, and say whether an arrow appeared along the line."""
    sx = (reading.bbox[0] + reading.bbox[2]) / 2 - 8
    sy = (reading.bbox[1] + reading.bbox[3]) / 2
    tx, ty = centres[order.target_province_id]
    live.button(geo, sx, sy, w.MOUSEEVENTF_LEFTDOWN, w.MOUSEEVENTF_LEFTUP)
    time.sleep(0.12)
    live.button(geo, tx, ty, w.MOUSEEVENTF_RIGHTDOWN, w.MOUSEEVENTF_RIGHTUP)
    time.sleep(0.25)
    moved = line_change(before, capture.grab(geo), (sx, sy), (tx, ty), 34.0, 34.0) >= 0.2
    # Deselect: a selected counter is drawn with a cream border and the reader stops seeing it,
    # so leaving a unit selected blinds the next observation. Escape would open the game menu.
    live.button(geo, *DESELECT_AT, w.MOUSEEVENTF_LEFTDOWN, w.MOUSEEVENTF_LEFTUP)
    time.sleep(0.12)
    live.move(geo, 1900, 700)
    time.sleep(0.2)
    return moved


def set_running(frame, running: bool) -> bool | None:
    """Press space only when the screen disagrees with what we want; returns the state that was read."""
    state = read_top_bar(frame).paused
    if state is not None and state == running:
        w.Win32Input().key("space")
        time.sleep(0.3)
    return state


records, start = [], time.time()
live.focused(geo)
if not [r for r in find_counters(capture.grab(geo)) if r.relation in ("own", "enemy")]:
    raise SystemExit("no unit counters on screen: the camera is not on the arena; run live_setup.py first")
bar = read_top_bar(capture.grab(geo))
if bar.paused is False:  # pause while the speed is set so the keys are not racing the clock
    w.Win32Input().key("space")
    time.sleep(0.3)
for key, count in (("add", max(0, want_speed - (bar.speed or 1))), ("subtract", max(0, (bar.speed or 1) - want_speed))):
    for _ in range(count):
        w.Win32Input().key(key)
        time.sleep(0.15)
set_running(capture.grab(geo), True)
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
    if not any(u.country == Country.BLUE for u in obs.units):
        diagnostics["frames_without_own"] += 1
        if diagnostics["frames_without_own"] <= 2:
            frame.save(out_path.parent / f"{out_path.stem}_noown{sequence}.png")
    began = time.perf_counter()
    order = agent.act(obs, None)
    decided_ms = (time.perf_counter() - began) * 1000
    done = "noop"
    if order.verb in (Verb.MOVE, Verb.SUPPORT_ATTACK) and order.unit_ids and order.unit_ids[0] in where:
        unit = order.unit_ids[0]
        if standing.get(unit) == order.target_province_id:
            done, diagnostics["reissue_skipped"] = "already", diagnostics["reissue_skipped"] + 1
        else:
            confirmed = issue(order, where[unit], frame)
            standing[unit] = order.target_province_id
            diagnostics["orders_sent"] += 1
            diagnostics["orders_confirmed"] += confirmed
            done = "confirmed" if confirmed else "unconfirmed"
    if sequence % 25 == 0:  # a popup can pause the game; keep it running
        set_running(frame, True)
    top = read_top_bar(frame)
    records.append({"t": round(time.time() - start, 1), "day": state["day"], "verb": order.verb.value,
                    "units": list(order.unit_ids), "target": order.target_province_id, "executed": done,
                    "own": sum(u.country == Country.BLUE for u in obs.units),
                    "enemy_visible": sum(u.country == Country.RED for u in obs.units),
                    "paused": top.paused, "speed": top.speed,
                    "decide_ms": round(decided_ms, 1), "tick": state["tick"]})
    if sequence % 20 == 0:
        print(records[-1], flush=True)
    sequence += 1
    live.move(geo, 1900, 700)
    time.sleep(0.5)
set_running(capture.grab(geo), False)
agent.close()
final = log_state()
out_path.write_text(json.dumps({"agent": spec, "source": "hoi4_vision", "bring_up": True, "outcome": outcome,
                                "final_tick": final["tick"], "final_holders": final["holders"],
                                "decisions": len(records), "minutes": round((time.time() - start) / 60, 2),
                                **diagnostics, "records": records}, indent=1))
print("outcome:", outcome, "|", final["tick"], "| decisions:", len(records), "|", diagnostics)

"""What the live view knows about the games: which are being played now and where, how
each stands, the games already played, and whether any runs are going at all."""

from __future__ import annotations

import glob
import json
import time
from datetime import datetime
from pathlib import Path

from .media import read_json

# A game is live while its manifest is incomplete and its video grew in the last seconds.
FRESH = 10
# Run folders untouched for this long are not looked into for live games.
DAY = 24 * 3600
SIDES = ("BLU", "RED")


def run_folders(runs, now=None, max_age=DAY):
    """The folders the `runs` globs match, touched within `max_age` seconds."""
    now = time.time() if now is None else now
    found = []
    for pattern in runs:
        for path in glob.glob(pattern):
            run = Path(path)
            try:
                if run.is_dir() and now - run.stat().st_mtime <= max_age:
                    found.append(run)
            except OSError:
                continue
    return found


def live_games(runs, now=None):
    """Every game being recorded under the run folders, as (folder, manifest): an
    incomplete manifest, and a screen.mkv written in the last FRESH seconds. One per
    station at most, the newest (a recorder plays one game at a time on each PC)."""
    now = time.time() if now is None else now
    best = {}
    for run in run_folders(runs, now):
        try:
            games = list(run.iterdir())
        except OSError:
            continue
        for game in games:
            try:
                written = (game / "screen.mkv").stat().st_mtime
            except OSError:
                continue
            if now - written > FRESH:
                continue
            manifest = read_json(game / "manifest.json") or {}
            if manifest.get("complete") or not manifest:
                continue
            station = station_of(game.name, read_json(game / "live-state.json"))
            if station not in best or written > best[station][0]:
                best[station] = (written, game, manifest)
    return {station: (game, manifest) for station, (_, game, manifest) in best.items()}


def live_game(runs, now=None):
    """The newest game being recorded, as (folder, manifest), or None between games."""
    games = live_games(runs, now)
    if not games:
        return None
    return max(games.values(), key=lambda found: (found[0] / "screen.mkv").stat().st_mtime)


def station_of(name, state=None):
    """Where a game is played: its live state says, else its folder's name does
    (scripted-peer-20260925-173110: the second PC; -here-: this PC)."""
    if state and state.get("station"):
        return state["station"]
    parts = name.split("-")
    return parts[1] if len(parts) > 2 else "peer"


def started_at(name, manifest=None):
    """When a game began, in unix seconds: its recorder's start, else the time in its
    folder's name (local time)."""
    started = ((manifest or {}).get("recorder") or {}).get("started_unix")
    if started:
        return started
    try:
        stamp = "-".join(name.split("-")[-2:])
        return datetime.strptime(stamp, "%Y%m%d-%H%M%S").timestamp()
    except ValueError:
        return None


def side_report(state):
    """Each side's latest daily report (weekly on arenas before v3), trimmed for the page:
    surrender progress 0..1, states owned of states, divisions, and the game's estimate of
    its army's strength against the enemy's."""
    reports = {}
    days, weeks = state.get("days") or {}, state.get("weeks") or {}
    for side in SIDES:
        report = days.get(side) or weeks.get(side)
        if not report:
            continue
        reports[side] = {
            key: report.get(key)
            for key in ("surrender", "states", "owned", "divisions", "strength", "casualties")
            if report.get(key) is not None
        }
    date = next((r.get("date") for r in (*days.values(), *weeks.values()) if r.get("date")), None)
    return reports, date


def game_card(game, manifest, run_entry=None, now=None):
    """Everything the page shows about a game in progress: where and what, since when, the
    plan, how each side stands in the game, and the scripted player's orders so far (or a
    learned policy's setup steps)."""
    now = time.time() if now is None else now
    state = read_json(game / "live-state.json") or {}
    entry = run_entry or {}
    plan = state.get("plan") or entry.get("plan") or {}
    started = state.get("started_unix") or started_at(game.name, manifest)
    hz = state.get("hz") or 5
    sides, date = side_report(state)
    orders = [
        {
            "frame": o.get("frame") or 0,
            "seconds": round((o.get("frame") or 0) / hz),
            "order": o.get("order"),
            **_detail(o),
        }
        for o in state.get("orders") or []
    ]
    return {
        "game": game.name,
        "run": game.parent.name,
        "station": station_of(game.name, state),
        "arena": state.get("arena") or entry.get("arena") or manifest.get("arena"),
        "side": state.get("started_as") or entry.get("started_as") or manifest.get("started_as"),
        "plan": plan,
        "started_unix": started,
        "elapsed": round(now - started) if started else None,
        "date": date,
        "sides": sides,
        "declarer": state.get("declarer"),
        "orders": orders[-30:],
        "kicks": state.get("kicks", 0),
        "milestones": state.get("milestones"),
        "winner": state.get("winner"),
    }


def _detail(order):
    """The few details of an order worth a line on the page."""
    keep = {}
    for key in ("attack", "law", "share", "tries"):
        if key in order and isinstance(order[key], (str, int, float)):
            keep[key] = order[key]
    return keep


def result(game):
    """A finished game's result for the side the recorder played: win, loss or timeout."""
    if game.get("winner") == game.get("started_as"):
        return "win"
    return "timeout" if game.get("winner") in ("timeout", None) else "loss"


class History:
    """The games already played, from every run's results files, newest first: each
    file is read again only when it changes."""

    def __init__(self, runs):
        self.runs = list(runs)
        self.files = {}  # path: (mtime, entries)

    def games(self, max_age=30 * DAY):
        seen = set()
        for run in run_folders(self.runs, max_age=max_age):
            for path in run.glob("results-*.json"):
                seen.add(path)
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                if self.files.get(path, (None,))[0] == mtime:
                    continue
                entries = read_json(path)
                if isinstance(entries, list):
                    self.files[path] = (
                        mtime,
                        [self.entry(run, e) for e in entries if e.get("game")],
                    )
        for path in list(self.files):
            if path not in seen:
                del self.files[path]
        games = [g for _, entries in self.files.values() for g in entries if g]
        return sorted(games, key=lambda g: g["started_unix"] or 0, reverse=True)

    @staticmethod
    def entry(run, game):
        if not game.get("winner"):
            return None  # A game that failed to start, or was cut short.
        started = started_at(game["game"])
        seconds = game.get("seconds")
        plan = game.get("plan") or {}
        return {
            "run": run.name,
            "game": game["game"],
            "station": game.get("station") or station_of(game["game"]),
            "arena": game.get("arena"),
            "side": game.get("started_as"),
            "plan": plan.get("variant"),
            "result": result(game),
            "seconds": seconds,
            "started_unix": started,
            "ended_unix": started + seconds if started and seconds else None,
            "milestones": game.get("milestones"),
            "path": str(run / game["game"]),
        }


def recorders():
    """The recorders running on this PC (record-ai or play-policy), as {kind, output}:
    whether any games are going at all, beside whether one is on screen now."""
    import psutil

    found = []
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            args = proc.info["cmdline"] or []
        except (psutil.Error, OSError):
            continue
        if not args or "python" not in (proc.info["name"] or "").lower():
            continue
        for kind in ("record-ai", "play-policy"):
            if kind in args:
                index = args.index(kind)
                output = args[index + 1] if index + 1 < len(args) else ""
                if kind == "play-policy" and index + 2 < len(args):
                    output = args[index + 2]
                found.append({"kind": kind, "output": Path(output).name})
    # The Python launcher of a venv starts the real interpreter as its child: one each.
    unique = []
    for item in found:
        if item not in unique:
            unique.append(item)
    return unique


def stats(games, now=None, hours=48):
    """The record by map and plan over the last `hours`: the best plan, the tuned plans
    and the learned player, each game counted once."""
    now = time.time() if now is None else now
    table = {}
    for game in games:
        if not game["started_unix"] or now - game["started_unix"] > hours * 3600:
            continue
        plan = game["plan"] or "explore"
        row = table.setdefault(game["arena"] or "?", {})
        cell = row.setdefault(plan, {"win": 0, "loss": 0, "timeout": 0})
        cell[game["result"]] += 1
    return {"hours": hours, "by_arena": table}


def training(runs=("artifacts/learned/*",), now=None, max_age=12 * 3600):
    """Training runs going now: each folder whose metrics.jsonl grew in the last
    `max_age` seconds, with its latest step, epoch, loss (averaged over the last 50
    steps) and pace."""
    now = time.time() if now is None else now
    found = []
    for run in run_folders(runs, now, max_age):
        path = run / "metrics.jsonl"
        try:
            if now - path.stat().st_mtime > max_age:
                continue
            with path.open("rb") as file:
                file.seek(max(0, path.stat().st_size - 40_000))
                lines = file.read().decode("utf-8", "replace").splitlines()[1:]
        except OSError:
            continue
        steps = [json.loads(line) for line in lines if '"step"' in line]
        held = [json.loads(line) for line in lines if "validation_nll" in line]
        if not steps:
            continue
        last = steps[-1]
        recent = steps[-50:]
        found.append({
            "run": run.name,
            "epoch": last.get("epoch"),
            "step": last.get("step"),
            "loss": round(sum(s.get("loss", 0) for s in recent) / len(recent), 3),
            "updated_unix": path.stat().st_mtime,
            "validation": held[-1] if held else None,
        })  # fmt: skip
    return found

"""Start HOI4 on an isolated arena profile, then restore the launcher settings byte for byte.

Same mechanism the binding probes verified: the game reads ``gameDataPath`` from the installed
``launcher-settings.json`` at startup. The file is changed only until the game has opened its
logs in the profile, a lock file names the backup, and the game is left running.
"""
from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

from .contracts import ArenaError


def hoi4_running() -> bool:
    listing = subprocess.check_output(["tasklist", "/FI", "IMAGENAME eq hoi4.exe", "/FO", "CSV", "/NH"],
                                      text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    return any(row and row[0].casefold() == "hoi4.exe" for row in csv.reader(listing.splitlines()))


def launch_profile(game: Path, profile: Path, mod: str, extra_args: tuple[str, ...] = ("-debug",),
                   startup_seconds: float = 90.0) -> dict[str, object]:
    """``mod`` is the descriptor path relative to the profile, e.g. ``mod/hoi4_arena_three_lanes.mod``."""
    profile = profile.resolve()
    if not (profile / mod).is_file() or not (profile / "dlc_load.json").is_file():
        raise ArenaError(f"{profile} is not a generated arena profile")
    if hoi4_running():
        raise ArenaError("close existing HOI4 instances before launching an arena profile")
    settings = game / "launcher-settings.json"
    lock = game / "hoi4-arena-launch.lock"
    if lock.exists():
        raise ArenaError(f"a previous launch did not finish cleanly; inspect {lock}")
    original = settings.read_bytes()
    parsed = json.loads(original)
    parsed["gameDataPath"] = profile.as_posix()
    modified = json.dumps(parsed, ensure_ascii=False, indent=2).encode("utf-8")
    backup = profile / f"launcher-settings.original.{int(time.time())}.json"
    backup.write_bytes(original)
    lock.write_text(str(backup), encoding="utf-8")
    system_log = profile / "logs" / "system.log"
    before = system_log.stat().st_mtime_ns if system_log.exists() else 0
    process = None
    try:
        settings.write_bytes(modified)
        process = subprocess.Popen([str(game / "hoi4.exe"), *extra_args, f"-mod={mod}"], cwd=game)
        deadline = time.monotonic() + startup_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ArenaError(f"HOI4 exited during startup with code {process.returncode}")
            if system_log.exists() and system_log.stat().st_mtime_ns != before:
                return {"pid": process.pid, "profile": str(profile), "status": "profile_in_use"}
            time.sleep(0.2)
        raise ArenaError("HOI4 did not open its logs in the profile; it may be using the default user directory")
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
        raise
    finally:
        if settings.read_bytes() == modified:
            settings.write_bytes(original)
            lock.unlink()
        # otherwise something else edited the file: keep the lock so the backup is not lost

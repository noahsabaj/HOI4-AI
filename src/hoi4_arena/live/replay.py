"""Watching a game again.

Every game's recording is kept as training data: screen.mkv, 1080p at 5 frames a second
in H.264 4:4:4, which an iPhone cannot play. So nothing more is stored for replays: when
someone asks for one, ffmpeg makes a 720p 4:2:0 copy with NVENC in a few seconds, and the
copies made last are kept up to a budget (the oldest go first).
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")
# The copies kept for replays, in bytes.
BUDGET = 3 * 1024**3


def replay_command(ffmpeg, video, out, encoder="h264_nvenc"):
    """ffmpeg making a replay: 720p, 4:2:0, a keyframe every 2 s (at the recordings' 5
    frames a second) so seeking is quick, the index at the front so playback starts
    before the whole file has come, and its progress on stdout."""
    quality = ["-preset", "p5", "-cq", "30"] if encoder == "h264_nvenc" else ["-crf", "28"]
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-threads", "4", "-i", str(video),
        "-vf", "scale=-2:720,format=yuv420p", "-c:v", encoder, *quality, "-g", "10",
        "-an", "-movflags", "+faststart", "-progress", "pipe:1", "-nostats",
        "-f", "mp4", str(out),
    ]  # fmt: skip


class Replays:
    """Replays made on request in `folder`, one at a time, the newest kept up to `budget`
    bytes. `request(game_folder)` answers {state: ready|working|error, url, progress}."""

    def __init__(self, folder, ffmpeg, budget=BUDGET, encoder="h264_nvenc"):
        self.folder, self.ffmpeg, self.budget = Path(folder), ffmpeg, budget
        self.encoder = encoder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.jobs = {}  # game name: {"progress": 0..1, "error": str | None}

    def path(self, name):
        return self.folder / f"{name}.mp4"

    def request(self, game):
        game = Path(game)
        name = game.name
        if not SAFE.match(name):
            return {"state": "error", "error": "bad name"}
        done = self.path(name)
        with self.lock:
            job = self.jobs.get(name)
            if job is None and done.exists():
                done.touch()  # Used again: kept longer.
                return {"state": "ready", "url": f"replays/{name}.mp4"}
            if job is None:
                if not (game / "screen.mkv").exists():
                    return {"state": "error", "error": "no recording"}
                if any(j.get("error") is None for j in self.jobs.values()):
                    return {"state": "busy"}
                job = self.jobs[name] = {"progress": 0.0, "error": None}
                threading.Thread(target=self.make, args=(game, job), daemon=True).start()
            if job["error"]:
                self.jobs.pop(name, None)
                return {"state": "error", "error": job["error"]}
            return {"state": "working", "progress": round(job["progress"], 2)}

    def make(self, game, job):
        import json

        name = game.name
        try:
            frames = json.loads((game / "manifest.json").read_text()).get("frames") or 0
        except (OSError, ValueError):
            frames = 0
        partial = self.folder / f"{name}.part.mp4"
        command = replay_command(self.ffmpeg, game / "screen.mkv", partial, self.encoder)
        started = time.monotonic()
        try:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)  # fmt: skip
            for line in proc.stdout:
                if line.startswith("frame=") and frames:
                    job["progress"] = min(0.99, int(line[6:].strip() or 0) / frames)
            error = proc.stderr.read()
            if proc.wait() != 0:
                raise RuntimeError(
                    error.strip().splitlines()[-1] if error.strip() else "ffmpeg failed"
                )
            partial.replace(self.path(name))
            log.info("replay of %s made in %.1f s", name, time.monotonic() - started)
            self.trim()
            with self.lock:
                self.jobs.pop(name, None)
        except Exception as error:  # noqa: BLE001 - reported to the page.
            partial.unlink(missing_ok=True)
            job["error"] = str(error)[:200]
            log.warning("replay of %s failed: %s", name, error)

    def trim(self):
        """Keep the replays used last, up to the budget."""
        kept = sorted(self.folder.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        total = 0
        for path in kept:
            if path.name.endswith(".part.mp4"):
                continue
            total += path.stat().st_size
            if total > self.budget:
                path.unlink(missing_ok=True)

"""Watching the games live from a phone: the recording being written, followed and
re-encoded for the web.

The second PC streams every game here, where record-ai writes it to screen.mkv as it
comes, and ffmpeg can read that file while it grows (-follow). So the live view changes
nothing in either PC's recording: `hoi4-arena live` finds the game being recorded,
re-encodes it as HLS (H.264, which Safari on an iPhone or iPad plays natively) beside a
snapshot of the screen each second (latest.jpg), and serves both on 127.0.0.1 with a page
that shows the game and its run's record. `tailscale serve --bg --https=8443
http://127.0.0.1:8765` publishes the page to the tailnet alone. It plays at the
recording's 5 frames a second, a few seconds behind the game.
"""

from __future__ import annotations

import glob
import http.server
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

PORT = 8765
# A game is live while its manifest is incomplete and its video grew in the last seconds.
FRESH = 10
# Seconds of video before the live edge that a view joining mid-game starts from.
LEAD = 4
# Seconds the video may stop growing before its game counts as over.
STALL = 15
# Seconds a segment lasts, and how many the playlist keeps.
SEGMENT, SEGMENTS = 2, 8
# The recordings' frames a second.
FPS = 5
# What the page may ask for, and as what.
SERVED = {
    ".html": "text/html; charset=utf-8",
    ".m3u8": "application/vnd.apple.mpegurl",
    ".ts": "video/mp2t",
    ".jpg": "image/jpeg",
    ".json": "application/json",
}
NAME = re.compile(r"^(index\.html|live\.m3u8|seg\d+\.ts|latest\.jpg|status\.json)$")


def live_game(runs, now=None):
    """The game being recorded under the run folders the `runs` globs match, as its
    folder and manifest: an incomplete manifest, and a screen.mkv written in the last
    FRESH seconds. The newest if several; None between games. Run folders untouched for
    a day are not looked into."""
    now = time.time() if now is None else now
    best = None
    for run in (Path(p) for pattern in runs for p in glob.glob(pattern)):
        try:
            if not run.is_dir() or now - run.stat().st_mtime > 24 * 3600:
                continue
            games = list(run.iterdir())
        except OSError:
            continue
        for game in games:
            try:
                written = (game / "screen.mkv").stat().st_mtime
            except OSError:
                continue
            if now - written > FRESH or (best and written <= best[0]):
                continue
            try:
                manifest = json.loads((game / "manifest.json").read_text())
            except (OSError, ValueError):
                manifest = {}
            if not manifest.get("complete"):
                best = (written, game, manifest)
    return None if best is None else best[1:]


def hls_command(ffmpeg, video, out, start=0.0, number=0, stall=STALL):
    """ffmpeg following `video` as it grows, from `start` seconds in: an HLS playlist
    whose segments are numbered from `number`, after a discontinuity (the game before
    had its own clock), and latest.jpg once a second."""
    graph = (
        f"[0:v]select='gte(t\\,{start:.2f})',setpts=PTS-STARTPTS,split=2[v][s];"
        "[v]format=yuv420p[hls];[s]fps=1[jpg]"
    )
    flags = "append_list+delete_segments+discont_start+omit_endlist+independent_segments"
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-follow", "1", "-rw_timeout", str(stall * 1_000_000), "-i", str(video),
        "-filter_complex", graph,
        "-map", "[hls]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
        "-maxrate", "3M", "-bufsize", "6M", "-fps_mode", "cfr", "-r", str(FPS),
        "-g", str(SEGMENT * FPS), "-keyint_min", str(SEGMENT * FPS), "-sc_threshold", "0",
        "-f", "hls", "-hls_time", str(SEGMENT), "-hls_list_size", str(SEGMENTS),
        "-hls_flags", flags, "-start_number", str(number),
        "-hls_segment_filename", str(Path(out) / "seg%06d.ts"), str(Path(out) / "live.m3u8"),
        "-map", "[jpg]", "-q:v", "4", "-update", "1", str(Path(out) / "latest.jpg"),
    ]  # fmt: skip


def next_segment(out):
    """The number after the playlist's last segment, or 0 without one."""
    try:
        numbers = [
            int(n) for n in re.findall(r"seg(\d+)\.ts", (Path(out) / "live.m3u8").read_text())
        ]
    except OSError:
        return 0
    return max(numbers) + 1 if numbers else 0


def status(found, runs, now=None):
    """What the page shows: the game being recorded (its run, how long in, and the arena
    and side its recorder wrote to the run's live.json), and the run's finished games
    from its results files, newest first."""
    now = time.time() if now is None else now
    info = {"live": found is not None, "updated": round(now)}
    run = None
    if found:
        game, manifest = found
        run = game.parent
        started = (manifest.get("recorder") or {}).get("started_unix")
        info.update(game=game.name, run=run.name)
        info["elapsed"] = round(now - started) if started else None
        try:
            playing = json.loads((run / "live.json").read_text())
        except (OSError, ValueError):
            playing = {}
        if playing.get("game") == game.name:
            plan = playing.get("plan") or {}
            info.update(arena=playing.get("arena"), side=playing.get("started_as"))
            info["plan"] = plan.get("variant")
    else:
        recent = [Path(p) for pattern in runs for p in glob.glob(pattern)]
        recent = [r for r in recent if r.is_dir() and now - r.stat().st_mtime < 24 * 3600]
        recent = [r for r in recent if any(r.glob("results-*.json"))]
        if recent:
            run = max(recent, key=lambda r: r.stat().st_mtime)
            info["run"] = run.name
    games = []
    for path in sorted(run.glob("results-*.json")) if run else []:
        try:
            games.extend(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    info["record"] = [
        {
            "arena": g.get("arena"),
            "side": g.get("started_as"),
            "result": result(g),
            "seconds": g.get("seconds"),
            "plan": (g.get("plan") or {}).get("variant"),
        }
        for g in reversed(games)
        if g.get("winner")
    ][:12]
    return info


def result(game):
    """A finished game's result for the side the recorder played: win, loss or timeout."""
    if game["winner"] == game.get("started_as"):
        return "win"
    return "timeout" if game["winner"] == "timeout" else "loss"


def read_shared(path):
    """A file's bytes, read without stopping ffmpeg from replacing or deleting it: on
    Windows a file open for reading blocks both unless opened with every sharing flag,
    and ffmpeg renames each playlist over the last and deletes old segments."""
    if sys.platform != "win32":
        return Path(path).read_bytes()
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]  # fmt: skip
    generic_read, share_all, open_existing = 0x80000000, 0x7, 3
    handle = kernel.CreateFileW(str(path), generic_read, share_all, None, open_existing, 0, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        raise FileNotFoundError(path)
    with os.fdopen(msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY), "rb") as file:
        return file.read()


class Handler(http.server.BaseHTTPRequestHandler):
    """The page's files from the live folder (self.server.out), by name alone, never
    cached; nothing else."""

    def do_HEAD(self):
        self.do_GET(head=True)

    def do_GET(self, head=False):
        name = self.path.split("?", 1)[0].lstrip("/") or "index.html"
        if not NAME.match(name):
            self.send_error(404)
            return
        try:
            body = read_shared(self.server.out / name)
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", SERVED[Path(name).suffix])
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def log_message(self, *args):
        pass


def serve(out, port=PORT):
    """The live folder served on 127.0.0.1 in a thread; the server, whose port is the
    one asked for, or the one given for port 0."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.out = Path(out)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    os.replace(temporary, path)


def watch(runs=("artifacts/*",), out=None, port=PORT, poll=2.0, ffmpeg=None, rounds=None):
    """Follow each game as it is recorded, until stopped (or for `rounds` polls): the
    page, the playlist, latest.jpg and status.json in `out` (by default the system's temp
    folder's hoi4-live), served on `port`."""
    out = Path(out) if out else Path(tempfile.gettempdir()) / "hoi4-live"
    out.mkdir(parents=True, exist_ok=True)
    for old in [*out.glob("seg*.ts"), out / "live.m3u8"]:
        old.unlink(missing_ok=True)
    (out / "index.html").write_text(PAGE, encoding="utf-8")
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("the live view needs ffmpeg")
    server = serve(out, port)
    log.info("live view on http://127.0.0.1:%d/ from %s", server.server_port, out)
    proc, current, retry_at = None, None, 0.0
    try:
        while rounds is None or rounds > 0:
            rounds = None if rounds is None else rounds - 1
            found = live_game(runs)
            game = found[0] if found else None
            if proc is not None and (proc.poll() is not None or game != current):
                if proc.poll() is None:
                    proc.terminate()  # A new game began while the last one's file lay still.
                    proc.wait(timeout=10)
                elif game == current:
                    retry_at = time.monotonic() + 10  # It failed mid-game: not at once.
                proc = None
            if game is not None and proc is None and time.monotonic() >= retry_at:
                started = (found[1].get("recorder") or {}).get("started_unix") or time.time()
                start = max(0.0, time.time() - started - LEAD)
                command = hls_command(ffmpeg, game / "screen.mkv", out, start, next_segment(out))
                proc = subprocess.Popen(command, stdin=subprocess.DEVNULL)
                current = game
                log.info("following %s from %.0f s", game, start)
            write_json(out / "status.json", status(found, runs))
            time.sleep(poll)
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
        server.shutdown()


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HOI4 live</title>
<style>
:root { color-scheme: dark; --bg: #101214; --panel: #1b1e22; --fg: #e8e8e8;
  --muted: #9aa0a6; --win: #66bb6a; --loss: #ef5350; --draw: #ffca28; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
video { display: block; width: 100%; max-height: 82vh; background: #000; }
main { padding: 10px 16px 24px; }
#now { font-size: 17px; margin: 4px 0 10px; }
#now .muted, .muted { color: var(--muted); }
#record { display: flex; flex-wrap: wrap; gap: 6px; }
#record span { background: var(--panel); border-radius: 6px; padding: 3px 8px; font-size: 13px; }
.win { color: var(--win); } .loss { color: var(--loss); } .timeout { color: var(--draw); }
</style></head>
<body>
<video id="video" poster="latest.jpg" autoplay muted playsinline controls></video>
<main>
<div id="now">Waiting for a game&hellip;</div>
<div id="record"></div>
<p class="muted">5 frames a second, a few seconds behind the game.</p>
</main>
<script>
const video = document.getElementById("video");
let hls = null, playing = null;
function load() {
  const src = "live.m3u8?" + Date.now();
  if (video.canPlayType("application/vnd.apple.mpegurl")) {
    video.src = src; video.play().catch(() => {});
  } else if (window.Hls && Hls.isSupported()) {
    if (hls) hls.destroy();
    hls = new Hls({ liveSyncDurationCount: 2 }); hls.loadSource(src); hls.attachMedia(video);
  }
}
video.addEventListener("error", () => setTimeout(load, 3000));
const side = { BLU: "Blue", RED: "Red" };
function arena(name) { return (name || "").replace(/^arena-/, "").replace(/-v\\d+$/, ""); }
function clock(s) { return s == null ? "" : Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); }
async function tick() {
  let s;
  try { s = await (await fetch("status.json?" + Date.now())).json(); } catch (e) { return; }
  const now = document.getElementById("now");
  if (s.live) {
    now.innerHTML = (s.arena ? arena(s.arena) + " as " + (side[s.side] || s.side) : s.game) +
      ' <span class="muted">' + clock(s.elapsed) + " in &middot; " + (s.plan || "") + " plan &middot; " + s.run + "</span>";
    if (s.game !== playing) { playing = s.game; setTimeout(load, 5000); }
  } else {
    now.innerHTML = 'Between games <span class="muted">' + (s.run || "") + "</span>";
  }
  document.getElementById("record").innerHTML = (s.record || []).map(g =>
    '<span class="' + g.result + '">' + arena(g.arena) + " " + (side[g.side] || "") + " " +
    (g.result === "win" ? "won" : g.result === "loss" ? "lost" : "timed out") + " " + clock(g.seconds) + "</span>"
  ).join("");
}
if (!video.canPlayType("application/vnd.apple.mpegurl")) {
  const script = document.createElement("script");
  script.src = "https://cdn.jsdelivr.net/npm/hls.js@1";
  script.onload = load; document.head.appendChild(script);
} else { load(); }
setInterval(tick, 5000); tick();
</script>
</body></html>
"""

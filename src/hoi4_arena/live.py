"""Watching the games live from a phone.

`hoi4-arena live --peer PAIRING` asks the second PC's worker for a view of its game
window (the worker's `view`): captured at 30 frames a second on that PC's GPU, apart
from the recording, and sent as H.264 over a read-only connection. Here it is cut into
HLS without encoding it again (Safari on an iPhone or iPad plays HLS natively), with a
snapshot of the screen each second (latest.jpg), and served on 127.0.0.1 with a page
that shows the game and its run's record. It shows the menus and loading between games
too. `tailscale serve --bg --https=8443 http://127.0.0.1:8765` publishes the page to the
tailnet alone.

Without the pairing, or from a worker too old for views, it follows the recording being
written instead: record-ai writes each game's screen.mkv as it comes, and ffmpeg reads
the file while it grows (-follow), at the recording's 5 frames a second. Neither way
changes a recording.

The page is also an app: added to an iPhone's or iPad's home screen (Share, Add to Home
Screen) it opens full screen with its own icon (install_app).
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
    ".webmanifest": "application/manifest+json",
    ".png": "image/png",
}
NAME = re.compile(
    r"^(index\.html|live\.m3u8|seg\d+\.ts|latest\.jpg|status\.json|manifest\.webmanifest"
    r"|icon-\d+\.png|apple-touch-icon\.png)$"
)


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
    had its own clock), and latest.jpg once a second. Every output is overwritten (-y):
    without it the second game's ffmpeg refused the first one's latest.jpg and the stream
    stopped. Segments stay on disk a while after leaving the playlist, for a player that
    lags."""
    graph = (
        f"[0:v]select='gte(t\\,{start:.2f})',setpts=PTS-STARTPTS,split=2[v][s];"
        "[v]format=yuv420p[hls];[s]fps=1[jpg]"
    )
    flags = "append_list+delete_segments+discont_start+omit_endlist+independent_segments"
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-follow", "1", "-rw_timeout", str(stall * 1_000_000), "-i", str(video),
        "-filter_complex", graph,
        "-map", "[hls]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
        "-maxrate", "3M", "-bufsize", "6M", "-fps_mode", "cfr", "-r", str(FPS),
        "-g", str(SEGMENT * FPS), "-keyint_min", str(SEGMENT * FPS), "-sc_threshold", "0",
        "-f", "hls", "-hls_time", str(SEGMENT), "-hls_list_size", str(SEGMENTS),
        "-hls_delete_threshold", "5", "-hls_flags", flags, "-start_number", str(number),
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
    cached, and status.json from memory (self.server.status); nothing else."""

    def do_HEAD(self):
        self.do_GET(head=True)

    def do_GET(self, head=False):
        name = self.path.split("?", 1)[0].lstrip("/") or "index.html"
        if not NAME.match(name):
            self.send_error(404)
            return
        if name == "status.json":
            body = json.dumps(self.server.status).encode()
        else:
            body = None
            for _ in range(3):
                try:
                    body = read_shared(self.server.out / name)
                    break
                except PermissionError:
                    time.sleep(0.02)  # Mid-rename by ffmpeg, or a scanner's moment.
                except OSError:
                    break
            if body is None:
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
    server.status = {"live": False}
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class Follower:
    """The recording being written, followed at its 5 frames a second (hls_command): one
    ffmpeg per game, from near the live edge, restarted 10 s after a failure."""

    def __init__(self, ffmpeg, out):
        self.ffmpeg, self.out = ffmpeg, Path(out)
        self.proc, self.current, self.retry_at = None, None, 0.0

    def step(self, found):
        game = found[0] if found else None
        proc = self.proc
        if proc is not None and (proc.poll() is not None or game != self.current):
            if proc.poll() is None:
                proc.terminate()  # A new game began while the last one's file lay still.
                proc.wait(timeout=10)
            elif game == self.current:
                self.retry_at = time.monotonic() + 10  # It failed mid-game: not at once.
                log.warning("ffmpeg stopped (exit %s); again in 10 s", proc.returncode)
            self.proc = None
        if game is not None and self.proc is None and time.monotonic() >= self.retry_at:
            started = (found[1].get("recorder") or {}).get("started_unix") or time.time()
            start = max(0.0, time.time() - started - LEAD)
            command = hls_command(
                self.ffmpeg, game / "screen.mkv", self.out, start, next_segment(self.out)
            )
            self.proc = subprocess.Popen(command, stdin=subprocess.DEVNULL)
            self.current = game
            log.info("following %s from %.0f s", game, start)

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None


def view_command(ffmpeg, out, number=0):
    """ffmpeg cutting the second PC's live view (MPEG-TS on its stdin, keyframes every
    2 s) into the playlist as it comes, without encoding it again, after a discontinuity,
    with latest.jpg once a second."""
    out = Path(out)
    flags = "append_list+delete_segments+discont_start+omit_endlist+independent_segments"
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "mpegts", "-i", "pipe:0",
        "-map", "0:v", "-c:v", "copy",
        "-f", "hls", "-hls_time", str(SEGMENT), "-hls_list_size", str(SEGMENTS),
        "-hls_delete_threshold", "5", "-hls_flags", flags, "-start_number", str(number),
        "-hls_segment_filename", str(out / "seg%06d.ts"), str(out / "live.m3u8"),
        "-map", "0:v", "-vf", "fps=1", "-q:v", "4", "-update", "1", str(out / "latest.jpg"),
    ]  # fmt: skip


def update_pending(peer, share=None):
    """Whether a new worker waits on the second PC's share (Deploy-Peer stages it as
    .new beside the running one). Its bridge swaps it in only while no connection is
    open there, so a view that never closed would keep every update out."""
    try:
        share = share or f"//{json.loads(Path(peer).read_text())['host']}/HOI4Worker"
        return (Path(share) / "hoi4-desktop-worker.exe.new").exists()
    except (OSError, ValueError, KeyError):
        return False


class PeerView:
    """The second PC's own live view (its worker's `view`): the game window captured at
    `hz` frames a second on that PC's GPU, sent as MPEG-TS over a read-only connection,
    and cut into the playlist here without encoding it again (view_command). It shows
    the menus and the loading between games too, where following the recording is 5
    frames a second and stops. `refused` once a worker without views says no."""

    def __init__(self, peer, ffmpeg, out, hz=30):
        self.peer, self.ffmpeg, self.out, self.hz = peer, ffmpeg, Path(out), hz
        self.desk = self.proc = None
        self.ended = threading.Event()
        self.refused = False

    def start(self):
        from .remote import RemoteDesktop

        self.stop()
        self.ended.clear()
        command = view_command(self.ffmpeg, self.out, next_segment(self.out))
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            self.desk = RemoteDesktop(self.peer, attach=True, observer=True)
            self.desk.streams["view"] = self.deliver
            self.desk.request("view", action="start", key="view", hz=self.hz)
        except Exception as error:
            self.refused = "refused_for_observer" in str(error) or "unknown" in str(error)
            self.stop()
            raise

    def deliver(self, message):
        """A message of the view's stream, on the connection's reader thread."""
        if "end" in message:
            self.ended.set()
            return
        payload, proc = message.get("payload"), self.proc
        if payload and proc is not None and proc.stdin is not None:
            try:
                proc.stdin.write(payload)
            except (OSError, ValueError):
                self.ended.set()

    def running(self):
        return self.proc is not None and self.proc.poll() is None and not self.ended.is_set()

    def stop(self):
        desk, proc = self.desk, self.proc
        self.desk = self.proc = None
        if desk is not None:
            try:
                desk.request("view", action="stop", timeout=5)
            except Exception:  # noqa: BLE001 - closing the connection stops it anyway.
                pass
            try:
                desk.close()
            except Exception:  # noqa: BLE001
                pass
        if proc is not None:
            try:
                proc.stdin.close()
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001 - it is going anyway.
                proc.kill()


def watch(
    runs=("artifacts/*",), out=None, port=PORT, poll=2.0, ffmpeg=None, rounds=None,
    peer=None, hz=30,
):  # fmt: skip
    """Show each game live, until stopped (or for `rounds` polls): the page, the playlist
    and latest.jpg in `out` (by default the system's temp folder's hoi4-live), and the
    status, served on `port`. Returns the last status.

    With `peer` (the second PC's pairing file) the video is that PC's own view of its
    game window at `hz` frames a second (PeerView), which steps aside while a new worker
    waits there; without, or from a worker too old for views, the recording being
    written, at 5 frames a second (Follower).

    Nothing that goes wrong in a round stops it: on 2026-09-24 a status file that Windows
    held for a moment ended the whole view, and the page went blank. The status is kept
    in memory since."""
    out = Path(out) if out else Path(tempfile.gettempdir()) / "hoi4-live"
    out.mkdir(parents=True, exist_ok=True)
    for old in [*out.glob("seg*.ts"), out / "live.m3u8"]:
        old.unlink(missing_ok=True)
    install_app(out)
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("the live view needs ffmpeg")
    server = serve(out, port)
    log.info("live view on http://127.0.0.1:%d/ from %s", server.server_port, out)
    follower = Follower(ffmpeg, out)
    view = PeerView(peer, ffmpeg, out, hz) if peer else None
    retry_at = 0.0
    try:
        while rounds is None or rounds > 0:
            rounds = None if rounds is None else rounds - 1
            try:
                found = live_game(runs)
                fps = FPS
                if view is not None and not view.refused:
                    if view.running() and update_pending(peer):
                        view.stop()
                        log.info("a new worker waits on the second PC: the view steps aside")
                    elif not view.running() and time.monotonic() >= retry_at:
                        if update_pending(peer):
                            retry_at = time.monotonic() + 5
                        else:
                            try:
                                view.start()
                                log.info("viewing the second PC at %d frames a second", hz)
                            except Exception as error:  # noqa: BLE001 - tried again.
                                retry_at = time.monotonic() + 5
                                log.info("no view yet: %s", error)
                    fps = hz if view.running() else 0
                else:
                    follower.step(found)
                server.status = {**status(found, runs), "fps": fps}
            except Exception:  # noqa: BLE001 - the view goes on; the next round tries again.
                log.exception("live view round failed")
            time.sleep(poll)
    finally:
        follower.stop()
        if view is not None:
            view.stop()
        server.shutdown()
    return server.status


def draw_icon(size):
    """The home-screen icon: the arena's two countries, Blue and Red, halves of a disc
    round a play mark, on the page's dark ground; drawn large and shrunk for smooth
    edges. iOS rounds the corners itself; the disc keeps inside a maskable icon's safe
    zone (the middle 80%)."""
    from PIL import Image, ImageDraw

    big = size * 4
    image = Image.new("RGB", (big, big), (16, 18, 20))
    draw = ImageDraw.Draw(image)
    middle, radius = big / 2, big * 0.36
    disc = (middle - radius, middle - radius, middle + radius, middle + radius)
    draw.pieslice(disc, 90, 270, fill=(59, 111, 214))
    draw.pieslice(disc, 270, 90, fill=(214, 69, 69))
    mark = radius * 0.55
    draw.polygon(
        [
            (middle - mark * 0.45, middle - mark * 0.6),
            (middle - mark * 0.45, middle + mark * 0.6),
            (middle + mark * 0.62, middle),
        ],
        fill=(245, 245, 245),
    )
    return image.resize((size, size), Image.LANCZOS)


MANIFEST = {
    "name": "HOI4 Live",
    "short_name": "HOI4 Live",
    "start_url": "/",
    "scope": "/",
    "display": "standalone",
    "background_color": "#101214",
    "theme_color": "#101214",
    "icons": [
        {"src": "icon-192.png", "sizes": "192x192", "type": "image/png"},
        {"src": "icon-512.png", "sizes": "512x512", "type": "image/png"},
        {"src": "icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
    ],
}


def install_app(out):
    """The page, and what makes it an app on a home screen: its manifest (standalone,
    without the browser's bars) and icons, 180 px for iOS and 192 and 512 px for the
    rest."""
    out = Path(out)
    (out / "index.html").write_text(PAGE, encoding="utf-8")
    (out / "manifest.webmanifest").write_text(json.dumps(MANIFEST), encoding="utf-8")
    for size, name in ((180, "apple-touch-icon.png"), (192, "icon-192.png"), (512, "icon-512.png")):
        draw_icon(size).save(out / name)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="HOI4 Live">
<meta name="theme-color" content="#101214">
<link rel="manifest" href="manifest.webmanifest">
<link rel="apple-touch-icon" href="apple-touch-icon.png">
<title>HOI4 Live</title>
<style>
:root { color-scheme: dark; --bg: #101214; --panel: #1b1e22; --fg: #e8e8e8;
  --muted: #9aa0a6; --win: #66bb6a; --loss: #ef5350; --draw: #ffca28; --live: #e53935; }
html, body { background: var(--bg); overscroll-behavior: none; }
body { margin: 0; color: var(--fg); -webkit-text-size-adjust: 100%;
  font: 15px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  padding: env(safe-area-inset-top) env(safe-area-inset-right)
    env(safe-area-inset-bottom) env(safe-area-inset-left); }
header { display: flex; align-items: center; justify-content: space-between;
  padding: 10px 16px; -webkit-user-select: none; user-select: none; }
header h1 { margin: 0; font-size: 18px; font-weight: 650; }
.badge { font-size: 12px; font-weight: 700; letter-spacing: 0.6px; padding: 3px 10px;
  border-radius: 999px; background: var(--panel); color: var(--muted); }
.badge.on { background: var(--live); color: #fff; }
.badge.on::before { content: ""; display: inline-block; width: 7px; height: 7px;
  margin-right: 6px; border-radius: 50%; background: #fff; vertical-align: 1px;
  animation: pulse 1.6s ease-in-out infinite; }
@keyframes pulse { 50% { opacity: 0.25; } }
video { display: block; width: 100%; max-height: 78vh; background: #000; }
main { padding: 12px 16px 8px; }
#now { font-size: 17px; margin: 0 0 10px; }
.muted { color: var(--muted); }
#record { display: flex; flex-wrap: wrap; gap: 6px; }
#record span { background: var(--panel); border-radius: 6px; padding: 3px 8px; font-size: 13px; }
.win { color: var(--win); } .loss { color: var(--loss); } .timeout { color: var(--draw); }
footer { padding: 8px 16px 16px; font-size: 12px; }
</style></head>
<body>
<header><h1>HOI4 Live</h1><span id="badge" class="badge">OFF AIR</span></header>
<video id="video" poster="latest.jpg" autoplay muted playsinline controls></video>
<main>
<div id="now">Waiting for a game&hellip;</div>
<div id="record"></div>
</main>
<footer class="muted">A few seconds behind the game.</footer>
<script>
const video = document.getElementById("video");
const badge = document.getElementById("badge");
let hls = null, playing = null, lastTime = -1, still = 0;
// Safari, and every browser on an iPhone or iPad, plays HLS itself; elsewhere hls.js does
// (Chromium answers "maybe" for HLS without playing it).
const ua = navigator.userAgent;
const apple = /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) &&
  (navigator.maxTouchPoints > 1 || !/Chrome|Chromium|Firefox|Edg/.test(ua)));
function load() {
  const src = "live.m3u8?" + Date.now();
  still = 0;
  if (!apple && window.Hls && Hls.isSupported()) {
    if (hls) hls.destroy();
    hls = new Hls({ liveSyncDurationCount: 2 }); hls.loadSource(src); hls.attachMedia(video);
  } else {
    video.src = src;
  }
  video.play().catch(() => {});
}
video.addEventListener("error", () => setTimeout(load, 3000));
// Back in the app after a while away: straight to the live edge.
document.addEventListener("visibilitychange", () => { if (!document.hidden) load(); });
const side = { BLU: "Blue", RED: "Red" };
function arena(name) { return (name || "").replace(/^arena-/, "").replace(/-v\\d+$/, ""); }
function clock(s) { return s == null ? "" : Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); }
async function tick() {
  let s;
  try { s = await (await fetch("status.json?" + Date.now())).json(); } catch (e) { return; }
  const now = document.getElementById("now");
  badge.className = s.live ? "badge on" : "badge";
  badge.textContent = s.live ? "LIVE" : "BETWEEN GAMES";
  if (s.live) {
    now.innerHTML = (s.arena ? arena(s.arena) + " as " + (side[s.side] || s.side) : s.game) +
      ' <span class="muted">' + clock(s.elapsed) + " in &middot; " +
      (s.plan ? s.plan + " plan &middot; " : "") + s.run + "</span>";
    // A new game: from its start, once a few segments of it are out. The first status
    // only notes the game (reloading then cut every view off after 5 s).
    if (playing !== null && s.game !== playing) setTimeout(load, 6000);
    playing = s.game;
    // The same frame for 20 s while it should play: start again at the live edge.
    if (!video.paused && video.currentTime === lastTime) { if (++still >= 4) load(); }
    else still = 0;
    lastTime = video.currentTime;
  } else {
    now.innerHTML = 'Between games <span class="muted">' + (s.run || "") + "</span>";
  }
  document.querySelector("footer").textContent = s.fps ?
    s.fps + " frames a second, a few seconds behind the game." :
    "The second PC's game window is closed: back when it opens.";
  document.getElementById("record").innerHTML = (s.record || []).map(g =>
    '<span class="' + g.result + '">' + arena(g.arena) + " " + (side[g.side] || "") + " " +
    (g.result === "win" ? "won" : g.result === "loss" ? "lost" : "timed out") + " " + clock(g.seconds) + "</span>"
  ).join("");
}
if (apple) { load(); } else {
  const script = document.createElement("script");
  script.src = "https://cdn.jsdelivr.net/npm/hls.js@1";
  script.onload = load; script.onerror = load; document.head.appendChild(script);
}
setInterval(tick, 5000); tick();
</script>
</body></html>
"""

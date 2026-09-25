"""The live view's web server, on 127.0.0.1 (tailscale serve publishes it to the tailnet).

GET  /                          the page (page/, built from web/live with SvelteKit), its
                                manifest and icons
GET  /_app/...                  the page's scripts and styles (immutable: named by hash)
GET  /s/<station>/<file>        a PC's stream: live.m3u8, its segments, latest.jpg
GET  /api/status                the stations, their games, and whether runs are going
GET  /api/games                 the games already played, newest first
GET  /api/chat?after=ID         the feed since a message
GET  /api/stats                 the record by map and plan, and training in progress
GET  /api/replay?run=R&game=G   a replay: made on request, then its URL
GET  /replays/<game>.mp4        a replay's file, in byte ranges as iPhones ask for it
POST /api/chat {who, text}      a watcher's message
POST /api/flag {..., note}      a moment marked for a closer look

Nothing else is served, and names are checked before any file is opened.
"""

from __future__ import annotations

import http.server
import json
import re
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .media import read_shared

PAGE = Path(__file__).with_name("page")
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".m3u8": "application/vnd.apple.mpegurl",
    ".ts": "video/mp2t",
    ".jpg": "image/jpeg",
    ".json": "application/json",
    ".webmanifest": "application/manifest+json",
    ".png": "image/png",
    ".mp4": "video/mp4",
}
STREAM_FILE = re.compile(r"^(live\.m3u8|seg\d+\.ts|latest\.jpg)$")
STATION = re.compile(r"^[a-z]{1,12}$")
SAFE = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")
APP_FILES = {"manifest.webmanifest", "apple-touch-icon.png", "icon-192.png", "icon-512.png"}
CHUNK = 1 << 16


class Handler(http.server.BaseHTTPRequestHandler):
    """Routes above; the app (self.server.app) answers the API."""

    def do_HEAD(self):
        self.do_GET(head=True)

    def do_GET(self, head=False):
        url = urlsplit(self.path)
        path, query = url.path, {k: v[-1] for k, v in parse_qs(url.query).items()}
        app = self.server.app
        if path.startswith("/api/"):
            answer = app.api(path[5:], query)
            if answer is None:
                self.send_error(404)
            else:
                self.send(json.dumps(answer, default=str).encode(), ".json", head)
            return
        parts = [p for p in path.split("/") if p]
        if not parts or parts == ["index.html"]:
            self.send_file(PAGE / "index.html", head)
        elif (
            parts[0] == "_app"
            and all(SAFE.match(p) and p not in (".", "..") for p in parts)
            and Path(parts[-1]).suffix in (".js", ".css", ".json")
        ):
            # SvelteKit names what never changes by its hash: cached for good.
            immutable = len(parts) > 2 and parts[1] == "immutable"
            self.send_file(PAGE.joinpath(*parts), head, cache=immutable)
        elif len(parts) == 1 and parts[0] in APP_FILES:
            self.send_file(app.out / parts[0], head)
        elif (
            len(parts) == 3
            and parts[0] == "s"
            and STATION.match(parts[1])
            and STREAM_FILE.match(parts[2])
        ):
            self.send_file(app.out / parts[1] / parts[2], head, shared=True)
        elif (
            len(parts) == 2
            and parts[0] == "replays"
            and SAFE.match(parts[1])
            and parts[1].endswith(".mp4")
        ):
            self.send_range(app.replays.folder / parts[1], head)
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > 4096:
            self.send_error(413)
            return
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self.send_error(400)
            return
        answer = self.server.app.post(urlsplit(self.path).path.removeprefix("/api/"), body)
        if answer is None:
            self.send_error(400)
        else:
            self.send(json.dumps(answer).encode(), ".json")

    def send(self, body, suffix, head=False, status=200, headers=(), cache=False):
        self.send_response(status)
        self.send_header("Content-Type", TYPES[suffix])
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Cache-Control", "public, max-age=31536000, immutable" if cache else "no-cache"
        )
        for key, value in headers:
            self.send_header(key, value)
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def send_file(self, path, head=False, shared=False, cache=False):
        body = None
        for _ in range(3):
            try:
                body = read_shared(path) if shared else Path(path).read_bytes()
                break
            except PermissionError:
                time.sleep(0.02)  # Mid-rename by ffmpeg, or a scanner's moment.
            except OSError:
                break
        if body is None:
            self.send_error(404)
            return
        self.send(body, Path(path).suffix, head, cache=cache)

    def send_range(self, path, head=False):
        """A file in the byte range asked for (Safari asks for ranges of every video)."""
        try:
            size = path.stat().st_size
        except OSError:
            self.send_error(404)
            return
        start, end = 0, size - 1
        asked = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
        if asked and (asked.group(1) or asked.group(2)):
            if asked.group(1):
                start = int(asked.group(1))
                end = int(asked.group(2)) if asked.group(2) else size - 1
            else:
                start = max(0, size - int(asked.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_error(416)
                return
        self.send_response(206 if asked else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if asked:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head:
            return
        try:
            with path.open("rb") as file:
                file.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = file.read(min(CHUNK, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (ConnectionError, OSError):
            pass  # The player moved on to another range.

    def log_message(self, *args):
        pass


def serve(app, port):
    """The server in a thread, answering for `app`; its port is the one asked for, or the
    one given for port 0."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.app = app
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server

"""The live view's loop: each round it finds the games being played, keeps one stream per
PC going while there is something to show, works out what the page says, and narrates
the games into the feed. The web server (server.py) asks it for everything."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path

from . import state as games
from .feed import Feed, Flags, Narrator
from .media import Follower, LocalView, PeerView, update_pending
from .replay import SAFE, Replays
from .server import serve

log = logging.getLogger(__name__)

PORT = 8765
LABELS = {"peer": "Second PC", "here": "This PC"}
# Seconds between looks for running recorders (a scan of every process).
RECORDER_EVERY = 10
# Seconds a watcher waits between messages.
CHAT_GAP = 1.0


class LiveApp:
    def __init__(self, runs, out, ffmpeg, *, peer=None, hz=30, labels=None, feed="artifacts/live"):
        self.runs, self.out, self.ffmpeg = list(runs), Path(out), ffmpeg
        self.peer, self.hz = peer, hz
        self.labels = {**LABELS, **(labels or {})}
        self.views = {"here": LocalView(ffmpeg, self.out / "here", hz)}
        if peer:
            self.views["peer"] = PeerView(peer, ffmpeg, self.out / "peer", hz)
        self.followers = {"peer": Follower(ffmpeg, self.out / "peer")}
        for station in ("peer", "here"):
            (self.out / station).mkdir(parents=True, exist_ok=True)
        self.history = games.History(self.runs)
        self.feed = Feed(Path(feed) / "chat.jsonl")
        self.flags = Flags(Path(feed) / "flags.jsonl")
        self.narrator = Narrator(self.feed, self.labels)
        self.replays = Replays(self.out / "replays", ffmpeg)
        self.retry_at = 0.0
        self.recorders, self.recorders_at = [], 0.0
        self.last_post = 0.0
        self.played = []
        self.status = {"updated": time.time(), "stations": [], "running": []}

    # The loop.

    def round(self):
        now = time.time()
        live = games.live_games(self.runs, now)
        self.keep_views(live)
        self.played = self.history.games()
        if time.monotonic() >= self.recorders_at:
            self.recorders = games.recorders()
            self.recorders_at = time.monotonic() + RECORDER_EVERY
        cards = {}
        for station, (game, manifest) in live.items():
            entry = games.read_json(game.parent / "live.json") or {}
            if entry.get("game") != game.name:
                entry = {}
            cards[station] = games.game_card(game, manifest, entry, now)
        finished = {g["game"]: g for g in self.played}
        self.narrator.step(cards, finished.get)
        self.feed.absorb()
        self.status = self.describe(cards, now)

    def keep_views(self, live):
        """A stream for each PC while there is something to show: the second PC's view
        whenever its game window is open (menus and loading too), this PC's only while a
        recorder plays a game here, and the recording followed where no view is had."""
        view = self.views.get("peer")
        if view is not None and not view.refused:
            if view.running() and update_pending(self.peer):
                view.stop()
                log.info("a new worker waits on the second PC: the view steps aside")
            elif not view.running() and time.monotonic() >= self.retry_at:
                if update_pending(self.peer):
                    self.retry_at = time.monotonic() + 5
                else:
                    try:
                        view.start()
                        log.info("viewing the second PC at %d frames a second", self.hz)
                    except Exception as error:  # noqa: BLE001 - tried again.
                        self.retry_at = time.monotonic() + 5
                        log.info("no view of the second PC: %s", error)
        else:
            self.followers["peer"].step(live.get("peer"))
        here = self.views["here"]
        if "here" in live and not here.running():
            try:
                here.start()
                log.info("viewing this PC's game")
            except OSError as error:
                log.warning("no view of this PC: %s", error)
        elif "here" not in live and here.since is not None:
            here.stop()

    def streaming(self, station):
        view = self.views.get(station)
        if view is not None and view.running():
            return True
        follower = self.followers.get(station)
        return follower is not None and follower.running()

    def describe(self, cards, now):
        stations = []
        for station in ("peer", "here"):
            last = next((g for g in self.played if g["station"] == station), None)
            card = cards.get(station)
            streaming = self.streaming(station)
            if station == "here" and not streaming and card is None and last is None:
                continue  # This PC never played: no tab for it.
            stations.append({
                "id": station,
                "label": self.labels.get(station, station),
                "streaming": streaming,
                "fps": self.hz if streaming and station in self.views else 5 if streaming else 0,
                "game": card,
                "last": _public(last),
            })  # fmt: skip
        ended = [g["ended_unix"] for g in self.played if g.get("ended_unix")]
        idle = not self.recorders and not cards
        return {
            "updated": round(now),
            "stations": stations,
            "running": self.recorders,
            "last_end": max(ended) if ended else None,
            "idle_since": (max(ended) if ended else None) if idle else None,
            "record": [_public(g) for g in self.played[:12]],
        }

    # The API.

    def api(self, name, query):
        if name == "status":
            return self.status
        if name == "games":
            limit = min(500, int(query.get("limit") or 100))
            run = query.get("run")
            chosen = [g for g in self.played if not run or g["run"] == run]
            return [_public(g) for g in chosen[:limit]]
        if name == "chat":
            try:
                after = int(query.get("after") or 0)
            except ValueError:
                after = 0
            return self.feed.since(after)
        if name == "stats":
            return {**games.stats(self.played), "training": games.training()}
        if name == "replay":
            game = query.get("game") or ""
            if not SAFE.match(game):
                return {"state": "error", "error": "bad name"}
            found = next((g for g in self.played if g["game"] == game), None)
            if found is None:
                return {"state": "error", "error": "no such game"}
            answer = self.replays.request(found["path"])
            if answer.get("state") == "ready":
                answer["orders"] = replay_orders(Path(found["path"]))
            return answer
        return None

    def post(self, name, body):
        if name == "chat":
            if time.monotonic() - self.last_post < CHAT_GAP:
                return {"ok": False, "error": "slow down"}
            self.last_post = time.monotonic()
            message = self.feed.add(body.get("who") or "anon", body.get("text") or "")
            return {"ok": message is not None, "message": message}
        if name == "flag":
            flag = {k: body.get(k) for k in ("station", "game", "seconds", "note", "replay")}
            if flag["game"] and not SAFE.match(str(flag["game"])):
                return None
            self.flags.add(flag)
            where = (
                f"{flag['game'] or '?'} at {int(flag.get('seconds') or 0) // 60}:"
                f"{int(flag.get('seconds') or 0) % 60:02d}"
            )
            note = f": {flag['note']}" if flag.get("note") else ""
            self.feed.add(body.get("who") or "anon", f"flagged {where}{note}", kind="flag",
                          game=flag["game"])  # fmt: skip
            return {"ok": True}
        return None

    def stop(self):
        for view in self.views.values():
            view.stop()
        for follower in self.followers.values():
            follower.stop()


def replay_orders(game):
    """A played game's orders, with the second of the recording each came at, for the
    replay to jump to (the recordings run at 5 frames a second)."""
    manifest = games.read_json(game / "manifest.json") or {}
    hz = manifest.get("nominal_fps") or 5
    return [
        {"seconds": round((o.get("frame") or 0) / hz, 1), "order": o.get("order")}
        for o in manifest.get("orders") or []
        if o.get("order") not in ("clear", "pause")
    ][:80]


def _public(game):
    """A history entry without its folder's path."""
    if game is None:
        return None
    return {k: v for k, v in game.items() if k != "path"}


def watch(
    runs=("artifacts/*", "artifacts/learned/*"), out=None, port=PORT, poll=2.0, ffmpeg=None,
    rounds=None, peer=None, hz=30, labels=None, feed="artifacts/live",
):  # fmt: skip
    """Show the games live, until stopped (or for `rounds` rounds): each PC's stream in
    `out` (by default the system's temp folder's hoi4-live), served on `port` with the
    page. Returns the last status.

    With `peer` (the second PC's pairing file) its stream is that PC's own view of its
    game window at `hz` frames a second, which steps aside while a new worker waits there;
    without, or from a worker too old for views, its recording being written, at 5 frames
    a second. This PC's game window is shown while a recorder plays a game here.

    Nothing that goes wrong in a round stops it: on 2026-09-24 a status file that Windows
    held for a moment ended the whole view, and the page went blank."""
    out = Path(out) if out else Path(tempfile.gettempdir()) / "hoi4-live"
    out.mkdir(parents=True, exist_ok=True)
    install_app(out)
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("the live view needs ffmpeg")
    app = LiveApp(runs, out, ffmpeg, peer=peer, hz=hz, labels=labels, feed=feed)
    server = serve(app, port)
    log.info("live view on http://127.0.0.1:%d/ from %s", server.server_port, out)
    try:
        while rounds is None or rounds > 0:
            rounds = None if rounds is None else rounds - 1
            try:
                app.round()
            except Exception:  # noqa: BLE001 - the view goes on; the next round tries again.
                log.exception("live view round failed")
            time.sleep(poll)
    finally:
        app.stop()
        server.shutdown()
    return app.status


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
    """What makes the page an app on a home screen: its manifest (standalone, without the
    browser's bars) and icons, 180 px for iOS and 192 and 512 px for the rest."""
    out = Path(out)
    (out / "manifest.webmanifest").write_text(json.dumps(MANIFEST), encoding="utf-8")
    for size, name in ((180, "apple-touch-icon.png"), (192, "icon-192.png"), (512, "icon-512.png")):
        draw_icon(size).save(out / name)

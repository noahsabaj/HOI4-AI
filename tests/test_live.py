import io
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import pytest

from hoi4_arena import live


def recording(run, name, complete=False, age=0.0, started_ago=300.0):
    now = time.time()
    game = run / name
    game.mkdir(parents=True)
    manifest = {"complete": complete, "recorder": {"started_unix": now - started_ago}}
    (game / "manifest.json").write_text(json.dumps(manifest))
    video = game / "screen.mkv"
    video.write_bytes(b"\x1a\x45\xdf\xa3")
    os.utime(video, (now - age, now - age))
    return game


def test_the_live_game_is_the_newest_recording_still_being_written(tmp_path):
    runs = [str(tmp_path / "*")]
    recording(tmp_path / "run-a", "finished", complete=True)
    recording(tmp_path / "run-a", "stalled", age=60)
    assert live.live_game(runs) is None
    recording(tmp_path / "run-b", "older", age=3)
    newest = recording(tmp_path / "run-b", "newest", age=1)
    game, manifest = live.live_game(runs)
    assert game == newest and manifest["complete"] is False


def test_the_stream_follows_the_file_from_near_its_end_and_numbers_on(tmp_path):
    command = live.hls_command("ffmpeg", tmp_path / "screen.mkv", tmp_path, start=12.5, number=40)
    joined = " ".join(command)
    assert "-follow 1" in joined and "gte(t\\,12.50)" in joined
    assert "-y" in command  # The last game's latest.jpg is overwritten, not refused.
    assert command[command.index("-start_number") + 1] == "40"
    assert "discont_start" in command[command.index("-hls_flags") + 1]
    assert str(tmp_path / "live.m3u8") in command and command[-1].endswith("latest.jpg")
    assert live.next_segment(tmp_path) == 0
    (tmp_path / "live.m3u8").write_text(
        "#EXTM3U\nseg000041.ts\n#EXT-X-DISCONTINUITY\nseg000042.ts\n"
    )
    assert live.next_segment(tmp_path) == 43


def playing(run, name, state, **manifest):
    """A game being recorded, with the live state its recorder publishes each second."""
    game = recording(run, name, **manifest)
    (game / "live-state.json").write_text(json.dumps(state))
    return game


DAY = "13:00, 1 February, 1937"
STATE = {
    "station": "peer", "arena": "arena-bay-v6", "started_as": "BLU", "hz": 5,
    "plan": {"variant": "tuned", "wait": 300, "redraw": 20},
    "days": {
        "BLU": {"date": DAY, "surrender": 0.1, "states": 8, "owned": 7, "divisions": 8},
        "RED": {"date": DAY, "surrender": 0.55, "states": 8, "owned": 5, "divisions": 3},
    },
    "orders": [{"frame": 30, "order": "army"}, {"frame": 127, "order": "offensive", "attack": "broad"}],
    "kicks": 1,
}  # fmt: skip


def test_each_pc_s_game_is_found_and_described(tmp_path):
    peer = playing(
        tmp_path / "scripted-f14", "scripted-peer-20260925-173110", STATE, started_ago=95
    )
    here = recording(tmp_path / "scripted-f15", "scripted-here-20260925-173500")
    found = live.live_games([str(tmp_path / "*")])
    assert set(found) == {"peer", "here"} and found["peer"][0] == peer and found["here"][0] == here
    card = live.game_card(peer, found["peer"][1])
    assert card["station"] == "peer" and card["arena"] == "arena-bay-v6" and card["side"] == "BLU"
    assert card["plan"]["variant"] == "tuned" and 94 <= card["elapsed"] <= 97
    assert card["date"] == DAY and card["sides"]["RED"]["surrender"] == 0.55
    assert [(o["seconds"], o["order"]) for o in card["orders"]] == [(6, "army"), (25, "offensive")]
    assert card["orders"][1]["attack"] == "broad" and card["kicks"] == 1
    # Without a live state (a recorder from before it), the run's live.json still names it.
    entry = {"game": here.name, "arena": "arena-12x8-v4"}
    assert live.game_card(here, found["here"][1], entry)["arena"] == "arena-12x8-v4"


def test_the_learned_player_s_live_games_are_followed_too(tmp_path):
    """play-policy records one level down (artifacts/learned/<test>/<game>)."""
    run = tmp_path / "learned" / "live-bc4c-e1"
    game = recording(run, "policy-peer-20260924-192149", started_ago=40)
    assert live.live_game([str(tmp_path / "*")]) is None, "a level above the games"
    assert live.live_game([str(tmp_path / "*"), str(tmp_path / "learned" / "*")])[0] == game


def results(run, games):
    run.mkdir(parents=True, exist_ok=True)
    (run / "results-peer-1.json").write_text(json.dumps(games))


def test_the_games_played_are_listed_newest_first_with_their_times(tmp_path):
    results(tmp_path / "scripted-f13", [
        {"game": "scripted-peer-20260924-181935", "arena": "arena-plains-v6", "started_as": "RED",
         "winner": "timeout", "seconds": 603, "plan": {"variant": "best"}},
        {"game": "scripted-peer-20260924-183018", "arena": "arena-plains-v6", "started_as": "BLU",
         "winner": "BLU", "seconds": 456, "plan": {"variant": "tuned"}},
        {"game": "scripted-peer-20260924-183915", "started_as": "RED", "error": "start failed"},
    ])  # fmt: skip
    played = live.History([str(tmp_path / "*")]).games()
    assert [(g["plan"], g["result"]) for g in played] == [("tuned", "win"), ("best", "timeout")]
    assert played[0]["ended_unix"] - played[0]["started_unix"] == 456
    assert played[0]["path"] == str(tmp_path / "scripted-f13" / "scripted-peer-20260924-183018")


def test_the_page_says_when_runs_have_stopped_not_between_games(tmp_path, monkeypatch):
    from hoi4_arena.live import state

    game = {"game": "scripted-peer-20260924-232952", "arena": "arena-bay-v6", "started_as": "RED",
            "winner": "RED", "seconds": 463}  # fmt: skip
    results(tmp_path / "scripted-f13", [game])
    app = live.LiveApp([str(tmp_path / "*")], tmp_path / "out", "ffmpeg", feed=tmp_path / "feed")
    monkeypatch.setattr(state, "recorders", lambda: [])
    app.round()
    ended = app.played[0]["ended_unix"]
    assert app.status["idle_since"] == ended and app.status["running"] == []
    assert not any(s["streaming"] or s["game"] for s in app.status["stations"])
    # A recorder still running: between games, not stopped.
    monkeypatch.setattr(
        state, "recorders", lambda: [{"kind": "record-ai", "output": "scripted-f14"}]
    )
    app.recorders_at = 0
    app.round()
    assert app.status["idle_since"] is None and app.status["last_end"] == ended


def test_the_feed_narrates_a_game_and_keeps_watchers_messages(tmp_path):
    feed = live.Feed(tmp_path / "chat.jsonl")
    narrator = live.Narrator(feed, {"peer": "Lent PC"})
    card = {"game": "g1", "station": "peer", "arena": "arena-bay-v6", "side": "BLU",
            "plan": {"variant": "best"}, "orders": [{"frame": 30, "seconds": 6, "order": "army"}],
            "sides": {"RED": {"surrender": 0.3, "owned": 8, "states": 8}}}  # fmt: skip
    narrator.step({"peer": card}, lambda name: None)
    orders = [*card["orders"], {"frame": 60, "seconds": 12, "order": "general"}]
    card = {**card, "orders": orders, "sides": {"RED": {"surrender": 0.6, "owned": 7, "states": 7}}}
    narrator.step({"peer": card}, lambda name: None)
    texts = [m["text"] for m in feed.since()]
    assert texts[0] == "Lent PC: bay as Blue, best plan"
    assert texts.count("0:06 formed the army") == 1 and "0:12 gave the army its general" in texts
    assert "Red is 25% of the way to surrender" in texts
    assert "Red is 50% of the way to surrender" in texts
    assert "Red lost a state (7 of 8 held)" in texts
    # Its end, once its results are written.
    done = {"result": "win", "side": "BLU", "arena": "arena-bay-v6", "seconds": 463}
    narrator.step({}, lambda name: done)
    assert feed.since()[-1]["text"] == "Lent PC: Blue won on bay in 7:43"
    # Watchers and Claude; the file keeps them all, and a new feed reads them back.
    feed.add("watcher", "  go   blue  ")
    live.say(tmp_path / "chat.jsonl", "the guard just redrew")
    feed.absorb()
    assert [m["text"] for m in feed.since()[-2:]] == ["go blue", "the guard just redrew"]
    assert feed.since()[-1]["kind"] == "claude"
    after = feed.since()[-2]["id"]
    again = live.Feed(tmp_path / "chat.jsonl").since(after)
    assert [m["text"] for m in again] == ["the guard just redrew"]


def test_the_server_gives_the_page_s_files_and_nothing_else(tmp_path):
    app = live.LiveApp(
        [str(tmp_path / "runs" / "*")], tmp_path / "out", "ffmpeg", feed=tmp_path / "feed"
    )
    (tmp_path / "out" / "peer" / "live.m3u8").write_text("#EXTM3U")
    (tmp_path / "out" / "notes.txt").write_text("x")
    app.replays.path("scripted-peer-1").write_bytes(bytes(range(256)) * 4)
    server = live.serve(app, 0)
    base = f"http://127.0.0.1:{server.server_port}"

    def post(path, body):
        request = urllib.request.Request(base + path, json.dumps(body).encode(), method="POST")
        with urllib.request.urlopen(request) as answer:
            return json.loads(answer.read())

    try:
        with urllib.request.urlopen(base + "/") as page:
            assert b"HOI4 Live" in page.read()
        from hoi4_arena.live.server import PAGE

        built = next((PAGE / "_app" / "immutable" / "entry").glob("*.js"))
        with urllib.request.urlopen(base + "/_app/immutable/entry/" + built.name) as script:
            assert script.headers["Content-Type"].startswith("text/javascript")
            assert "immutable" in script.headers["Cache-Control"]
        with urllib.request.urlopen(base + "/s/peer/live.m3u8?123") as playlist:
            assert playlist.headers["Content-Type"] == "application/vnd.apple.mpegurl"
            assert playlist.headers["Cache-Control"] == "no-cache"
        head = urllib.request.Request(base + "/s/peer/live.m3u8", method="HEAD")
        with urllib.request.urlopen(head) as probed:
            assert probed.headers["Content-Length"] == "7" and probed.read() == b""
        app.status = {"stations": [], "note": "from memory"}
        with urllib.request.urlopen(base + "/api/status") as shown:
            assert json.loads(shown.read()) == app.status
        # A replay in byte ranges, as Safari asks for it.
        ranged = urllib.request.Request(
            base + "/replays/scripted-peer-1.mp4", headers={"Range": "bytes=10-19"}
        )
        with urllib.request.urlopen(ranged) as part:
            assert part.status == 206 and part.read() == bytes(range(10, 20))
            assert part.headers["Content-Range"] == "bytes 10-19/1024"
        assert post("/api/chat", {"who": "watcher", "text": "hello"})["ok"]
        flag = {"who": "watcher", "game": "scripted-peer-1", "seconds": 75, "note": "stuck"}
        assert post("/api/flag", flag)["ok"]
        with urllib.request.urlopen(base + "/api/chat?after=0") as chat:
            texts = [m["text"] for m in json.loads(chat.read())]
        assert texts == ["hello", "flagged scripted-peer-1 at 1:15: stuck"]
        assert json.loads((tmp_path / "feed" / "flags.jsonl").read_text())["note"] == "stuck"
        refused_paths = (
            "/notes.txt", "/../out/notes.txt", "/s/peer/../notes.txt", "/s/PEER/live.m3u8",
            "/replays/..%5Cnotes.txt", "/api/nothing", "/_app/../index.html",
            "/_app/%2e%2e/index.html", "/_app/version.txt",
        )  # fmt: skip
        for path in refused_paths:
            with pytest.raises(urllib.error.HTTPError) as refused:
                urllib.request.urlopen(base + path)
            assert refused.value.code == 404, path
    finally:
        server.shutdown()


def test_a_stream_that_stops_leaves_no_playlist_to_replay(tmp_path):
    for name in ("live.m3u8", "seg000001.ts", "latest.jpg"):
        (tmp_path / name).write_bytes(b"x")
    live.clear_stream(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["latest.jpg"], "what the screen showed last"


def test_the_watcher_says_when_no_game_is_being_recorded(tmp_path):
    runs = [str(tmp_path / "runs" / "*")]
    shown = live.watch(runs, out=tmp_path / "out", port=0, poll=0, ffmpeg="ffmpeg", rounds=1,
                       feed=tmp_path / "feed")  # fmt: skip
    assert not any(s["game"] for s in shown["stations"])
    assert (tmp_path / "out" / "manifest.webmanifest").exists()


def test_the_page_installs_as_an_app_with_its_icons(tmp_path):
    from PIL import Image

    from hoi4_arena.live.server import PAGE

    live.install_app(tmp_path)
    page = (PAGE / "index.html").read_text()
    assert 'rel="manifest"' in page and 'rel="apple-touch-icon"' in page
    assert "apple-mobile-web-app-capable" in page and "viewport-fit=cover" in page
    manifest = json.loads((tmp_path / "manifest.webmanifest").read_text())
    assert manifest["display"] == "standalone" and manifest["start_url"] == "/"
    for icon in manifest["icons"]:
        size = int(icon["sizes"].split("x")[0])
        assert Image.open(tmp_path / icon["src"]).size == (size, size)
    touch = Image.open(tmp_path / "apple-touch-icon.png")
    assert touch.size == (180, 180)
    # Blue on the left, Red on the right, the play mark white in the middle.
    blue, red = touch.getpixel((50, 90)), touch.getpixel((130, 90))
    assert blue[2] > blue[0] and red[0] > red[2] and min(touch.getpixel((90, 90))) > 200


def test_the_watcher_goes_on_past_a_round_that_fails(tmp_path, monkeypatch):
    from hoi4_arena.live import state

    calls = []

    def flaky(runs, now=None):
        calls.append(runs)
        if len(calls) == 1:
            raise PermissionError("held by a scanner for a moment")
        return {}

    monkeypatch.setattr(state, "live_games", flaky)
    live.watch(["runs/*"], out=tmp_path / "out", port=0, poll=0, ffmpeg="ffmpeg", rounds=2,
               feed=tmp_path / "feed")  # fmt: skip
    assert len(calls) == 2


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_replay_is_made_from_the_recording_on_request(tmp_path):
    game = tmp_path / "scripted-peer-1"
    game.mkdir()
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=5",
         "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv444p", str(game / "screen.mkv")],
        check=True,
    )  # fmt: skip
    (game / "manifest.json").write_text(json.dumps({"frames": 15}))
    replays = live.Replays(tmp_path / "replays", shutil.which("ffmpeg"), encoder="libx264")
    answer = replays.request(game)
    assert answer["state"] in ("working", "ready")
    deadline = time.monotonic() + 60
    while answer["state"] == "working" and time.monotonic() < deadline:
        time.sleep(0.2)
        answer = replays.request(game)
    assert answer == {"state": "ready", "url": "replays/scripted-peer-1.mp4"}
    assert replays.path("scripted-peer-1").read_bytes()[4:8] == b"ftyp"
    assert replays.request(tmp_path / "no-such-game")["state"] == "error"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_following_a_recording_makes_a_playlist_and_a_snapshot(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    video = tmp_path / "screen.mkv"
    subprocess.run(
        [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=5", "-t", "4",
         "-c:v", "libx264", "-pix_fmt", "yuv444p", "-f", "matroska", str(video)],
        check=True,
    )  # fmt: skip
    out = tmp_path / "out"
    out.mkdir()
    command = live.hls_command(ffmpeg, video, out, start=1.0, number=7, stall=1)
    subprocess.run(command, timeout=60, stdin=subprocess.DEVNULL)
    playlist = (out / "live.m3u8").read_text()
    assert "seg000007.ts" in playlist and "#EXT-X-DISCONTINUITY" in playlist
    assert (out / "seg000007.ts").stat().st_size > 0
    assert (out / "latest.jpg").read_bytes()[:2] == b"\xff\xd8"
    assert live.read_shared(out / "latest.jpg")[:2] == b"\xff\xd8"


def test_the_second_pc_s_view_is_cut_into_the_playlist_without_encoding_again(tmp_path):
    command = live.view_command("ffmpeg", tmp_path, number=12)
    joined = " ".join(command)
    assert "-f mpegts -i pipe:0" in joined and "-c:v copy" in joined
    assert command[command.index("-start_number") + 1] == "12"
    assert command[-1].endswith("latest.jpg")
    # One thread per stage: ffmpeg's own pools held ~900 MB for a 1-a-second snapshot.
    assert "-threads 1 -f mpegts -i pipe:0" in joined
    assert "-filter_threads 1 -threads 1 -vf fps=1" in joined


def _streams(path):
    probe = subprocess.run(
        [shutil.which("ffprobe") or "ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )  # fmt: skip
    return sorted(set(probe.stdout.split()))  # MPEG-TS lists them per program too


def _sound(on):
    """A tone's input and its AAC encoding (the worker's view with `audio`), or nothing."""
    return (["-f", "lavfi", "-i", "sine=frequency=440"], ["-c:a", "aac"]) if on else ([], [])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize("sound", [False, True])
def test_the_view_command_runs_on_a_stream_like_the_second_pc_s(tmp_path, sound):
    """ffmpeg takes every option, as a failed start would leave the page without a view,
    and the game's sound, where the stream has it, reaches the playlist and the pieces."""
    ffmpeg = shutil.which("ffmpeg")
    tone, aac = _sound(sound)
    stream = subprocess.run(
        [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=5", *tone,
         "-t", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "10", "-bf", "0", *aac,
         "-f", "mpegts", "pipe:1"],
        check=True, capture_output=True,
    ).stdout  # fmt: skip
    out, raw = tmp_path / "out", tmp_path / "raw"
    out.mkdir()
    raw.mkdir()
    done = subprocess.run(
        live.view_command(ffmpeg, out, number=3, raw=raw),
        input=stream, capture_output=True, timeout=60,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr.decode(errors="replace")
    assert "seg000003.ts" in (out / "live.m3u8").read_text()
    assert (out / "latest.jpg").read_bytes()[:2] == b"\xff\xd8"
    kinds = ["audio", "video"] if sound else ["video"]
    assert _streams(out / "seg000003.ts") == kinds
    assert _streams(next(raw.glob("*.ts"))) == kinds


def test_the_view_also_writes_raw_pieces_for_the_archive_when_asked(tmp_path):
    command = live.view_command("ffmpeg", tmp_path / "out", raw=tmp_path / "raw")
    at = command.index("segment")
    maps = ["-map", "0:v", "-map", "0:a?", "-c:v", "copy", "-c:a", "copy"]
    assert command[at - 9 : at - 1] == maps
    assert command[-1].endswith("%Y%m%d-%H%M%S.ts")
    assert "segment" not in live.view_command("ffmpeg", tmp_path / "out")


def test_a_game_s_pieces_are_used_once_they_cover_it_whole(tmp_path):
    def piece(began, written):
        path = tmp_path / f"{began}.ts"
        path.write_bytes(b"x")
        os.utime(path, (written, written))
        return (began, path)

    now = 10_000.0
    found = [piece(t, t + 30) for t in (1000, 1030, 1060, 1090)]
    # A game from 1010 to 1080: the piece begun at 1090 shows the view went on past it.
    assert [p[0] for p in live.covering(found, 1010, 1080, now)] == [1000, 1030, 1060]
    # Past the last piece's end, and it is still being written: not yet.
    assert live.covering(found[:3], 1010, 1080, 1085) is None
    # The view began after the game did: no archive (its start would be missing).
    assert live.covering(found[1:], 1010, 1080, now) is None
    # A hole where the view stopped mid-game.
    assert live.covering([found[0], found[2], found[3]], 1010, 1080, now) is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize("sound", [False, True])
def test_an_ended_game_is_archived_from_its_pieces_and_replayed_from_there(tmp_path, sound):
    ffmpeg = shutil.which("ffmpeg")
    raw = tmp_path / "raw" / "peer"
    raw.mkdir(parents=True)
    start = time.time() - 600
    tone, aac = _sound(sound)
    for k in range(3):
        began = time.strftime("%Y%m%d-%H%M%S", time.localtime(start + 4 * k))
        subprocess.run(
            [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=60", *tone,
             "-t", "4", "-c:v", "libx264", "-pix_fmt", "yuv420p", *aac, "-f", "mpegts",
             str(raw / f"{began}.ts")],
            check=True,
        )  # fmt: skip
        os.utime(raw / f"{began}.ts", (start + 4 * k + 4, start + 4 * k + 4))
    game = tmp_path / "run" / "scripted-peer-20260925-120000"
    game.mkdir(parents=True)
    manifest = {"recorder": {"started_unix": start + 1}, "frames": 40, "nominal_fps": 5}
    (game / "manifest.json").write_text(json.dumps(manifest))
    played = [{"game": game.name, "station": "peer", "path": str(game), "ended_unix": start + 9}]
    archive = live.Archive(tmp_path / "raw", tmp_path / "archive", ffmpeg, encoder="libx264")
    archive.step(played)
    deadline = time.time() + 60
    while archive.job and time.time() < deadline:
        time.sleep(0.1)
    made = archive.path(game.name)
    assert made is not None, archive.failed
    probe = subprocess.run(
        [shutil.which("ffprobe") or "ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=r_frame_rate", "-of", "json", str(made)],
        capture_output=True, text=True,
    )  # fmt: skip
    facts = json.loads(probe.stdout)
    assert abs(float(facts["format"]["duration"]) - 8) < 0.5, "the game's 8 s, cut from 12"
    assert facts["streams"][0]["r_frame_rate"] == "30/1"
    assert _streams(made) == (["audio", "video"] if sound else ["video"])
    # Its pieces go once no game needs them (all of them are older than RAW_KEEP here).
    archive.step(played, now=time.time() + live.archive.RAW_KEEP + 700)
    assert not list(raw.glob("*.ts"))


def test_the_view_steps_aside_while_a_new_worker_waits(tmp_path):
    peer = tmp_path / "peer.json"
    peer.write_text(json.dumps({"host": "second-pc"}))
    share = tmp_path / "share"
    share.mkdir()
    assert not live.update_pending(peer, share)
    (share / "hoi4-desktop-worker.exe.new").write_bytes(b"MZ")
    assert live.update_pending(peer, share)
    assert not live.update_pending(tmp_path / "missing.json")


def test_a_view_through_fleet_s_tunnel_never_looks_for_the_share(tmp_path, monkeypatch):
    """A pairing on this PC's loopback is fleet's worker service: no share to look in (a
    look at //127.0.0.1 would ask this PC's own file sharing every round)."""
    from pathlib import Path

    peer = tmp_path / "peer-fleet.json"
    peer.write_text(json.dumps({"host": "127.0.0.1", "port": 47941}))
    looked = []
    monkeypatch.setattr(Path, "exists", lambda self: looked.append(self) or True)
    assert not live.update_pending(peer)
    assert not looked


class FakeProcess:
    def __init__(self):
        self.stdin, self.returncode = io.BytesIO(), None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = -9


def fake_worker(monkeypatch, requests, refuse=False):
    from hoi4_arena import remote
    from hoi4_arena.desktop import DesktopError

    class Desk:
        def __init__(self, peer, attach=True, observer=False, wait=None):
            assert attach and observer  # Read-only, beside the recording's connection.
            assert wait == 0, "the round goes on: no waiting for a worker that is not there"
            self.streams = {}

        def request(self, op, timeout=10, **fields):
            requests.append((op, fields.get("action")))
            if refuse and fields.get("action") == "start":
                raise DesktopError("view_refused_for_observer")
            return {"view": fields.get("key"), "hz": fields.get("hz")}

        def close(self):
            requests.append(("close", None))

    monkeypatch.setattr(remote, "RemoteDesktop", Desk)
    processes = []

    def popen(command, **options):
        processes.append(FakeProcess())
        return processes[-1]

    from hoi4_arena.live import media

    monkeypatch.setattr(media.subprocess, "Popen", popen)
    return processes


def test_a_view_passes_the_worker_s_video_on_and_ends_with_it(tmp_path, monkeypatch):
    requests = []
    processes = fake_worker(monkeypatch, requests)
    view = live.PeerView("peer.json", "ffmpeg", tmp_path)
    view.start()
    assert requests == [("view", "start")] and view.running()
    view.deliver({"stream": "view", "data": 0, "payload": b"\x47video"})
    assert processes[0].stdin.getvalue() == b"\x47video"
    # The window closed (a relaunch): the view ends, and is started again later.
    view.deliver({"stream": "view", "end": {"reason": "window closed"}})
    assert not view.running() and not view.refused
    view.stop()
    assert requests[-2:] == [("view", "stop"), ("close", None)]


def test_a_worker_without_views_is_asked_no_more(tmp_path, monkeypatch):
    from hoi4_arena.desktop import DesktopError

    requests = []
    fake_worker(monkeypatch, requests, refuse=True)
    view = live.PeerView("peer.json", "ffmpeg", tmp_path)
    with pytest.raises(DesktopError):
        view.start()
    assert view.refused and not view.running()

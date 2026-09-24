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


def test_the_status_names_the_game_and_the_run_s_record_newest_first(tmp_path):
    run = tmp_path / "scripted-f12"
    game = recording(run, "scripted-peer-2", started_ago=95)
    entry = {"game": game.name, "arena": "arena-bay-v6", "started_as": "BLU"}
    (run / "live.json").write_text(json.dumps({**entry, "plan": {"variant": "tuned"}}))
    finished = [
        {"arena": "arena-bay-v6", "started_as": "RED", "winner": "RED", "seconds": 370},
        {"arena": "arena-bay-v6", "started_as": "BLU", "winner": "RED", "seconds": 434},
        {"arena": "arena-river-v6", "started_as": "RED", "winner": "timeout", "seconds": 600},
        {"arena": "arena-river-v6", "started_as": "BLU", "error": "start failed"},
    ]
    (run / "results-peer-1.json").write_text(json.dumps(finished))
    found = (game, json.loads((game / "manifest.json").read_text()))
    shown = live.status(found, [str(tmp_path / "*")])
    assert shown["live"] and shown["arena"] == "arena-bay-v6" and shown["side"] == "BLU"
    assert shown["plan"] == "tuned" and 94 <= shown["elapsed"] <= 97
    assert [g["result"] for g in shown["record"]] == ["timeout", "loss", "win"]
    # Between games: the latest run's record.
    idle = live.status(None, [str(tmp_path / "*")])
    assert not idle["live"] and idle["run"] == "scripted-f12" and len(idle["record"]) == 3


def test_the_server_gives_the_page_s_files_and_nothing_else(tmp_path):
    for name, body in {"index.html": "<p>", "live.m3u8": "#EXTM3U", "notes.txt": "x"}.items():
        (tmp_path / name).write_text(body)
    server = live.serve(tmp_path, port=0)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/") as page:
            assert page.read() == b"<p>"
        with urllib.request.urlopen(base + "/live.m3u8?123") as playlist:
            assert playlist.headers["Content-Type"] == "application/vnd.apple.mpegurl"
            assert playlist.headers["Cache-Control"] == "no-cache"
        head = urllib.request.Request(base + "/live.m3u8", method="HEAD")
        with urllib.request.urlopen(head) as probed:
            assert probed.headers["Content-Length"] == "7" and probed.read() == b""
        # The status comes from memory, not a file Windows may hold while it is replaced.
        server.status = {"live": True, "game": "scripted-peer-9"}
        with urllib.request.urlopen(base + "/status.json") as shown:
            assert json.loads(shown.read()) == server.status
        for path in ("/notes.txt", "/../live.m3u8", "/latest.jpg"):
            with pytest.raises(urllib.error.HTTPError) as refused:
                urllib.request.urlopen(base + path)
            assert refused.value.code == 404
    finally:
        server.shutdown()


def test_the_watcher_says_when_no_game_is_being_recorded(tmp_path):
    runs = [str(tmp_path / "runs" / "*")]
    shown = live.watch(runs, out=tmp_path / "out", port=0, poll=0, ffmpeg="ffmpeg", rounds=1)
    assert shown["live"] is False
    assert "HOI4 Live" in (tmp_path / "out" / "index.html").read_text()


def test_the_page_installs_as_an_app_with_its_icons(tmp_path):
    from PIL import Image

    live.install_app(tmp_path)
    page = (tmp_path / "index.html").read_text()
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
    server = live.serve(tmp_path, port=0)
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(base + "/manifest.webmanifest") as served:
            assert served.headers["Content-Type"] == "application/manifest+json"
        with urllib.request.urlopen(base + "/apple-touch-icon.png") as served:
            assert served.headers["Content-Type"] == "image/png"
    finally:
        server.shutdown()


def test_the_watcher_goes_on_past_a_round_that_fails(tmp_path, monkeypatch):
    calls = []

    def flaky(runs, now=None):
        calls.append(runs)
        if len(calls) == 1:
            raise PermissionError("held by a scanner for a moment")

    monkeypatch.setattr(live, "live_game", flaky)
    shown = live.watch(["runs/*"], out=tmp_path / "out", port=0, poll=0, ffmpeg="ffmpeg", rounds=2)
    assert len(calls) == 2 and shown["live"] is False


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

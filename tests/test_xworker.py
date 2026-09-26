"""The Linux worker (xworker): its rules match the Windows worker's, and the Desktop client
drives it over the same protocol. X itself is faked, so these run on any OS."""

import io
import json
import sys
import textwrap

import numpy as np
import pytest

from hoi4_arena import desktop, xworker
from hoi4_arena.actions import KEYS


class FakeX:
    """A 64x48 screen holding the game window at (8, 4), 32x24, with the input focus."""

    def __init__(self):
        self.events = []
        self.focus = 7

    def find(self, title):
        return 7

    def rect(self, window):
        return (8, 4, 32, 24) if window == 7 else None

    def screen_size(self):
        return 64, 48

    def grab(self, x, y, w, h):
        frame = np.zeros((h, w, 4), np.uint8)
        frame[:, :, 2] = 200  # red, in BGRA
        frame[:, :, 3] = 255
        return frame

    def pointer(self):
        return 18, 9

    def cursor(self):
        image = np.zeros((2, 2, 4), np.uint8)
        image[..., 0] = image[..., 3] = 255  # opaque blue, premultiplied
        return image, (0, 0), (18, 9)

    def focused(self):
        return self.focus

    def ancestors(self, window):
        return [window]

    def name(self, window):
        return "Hearts of Iron IV"

    def raise_and_focus(self, window):
        self.focus = window

    def motion(self, x, y):
        self.events.append(("motion", x, y))

    def button(self, button, down):
        self.events.append(("button", button, down))

    def key(self, keysym, down):
        self.events.append(("key", keysym, down))


def worker(tmp_path, **kw):
    out = io.BytesIO()
    w = xworker.Worker(FakeX(), mods=tmp_path / "mods", userdir=tmp_path / "user", out=out, **kw)
    return w, out


def ask(w, out, op, **kw):
    out.seek(0)
    out.truncate()
    w.handle(json.dumps({"op": op, "id": 1, **kw}).encode())
    header, _, payload = out.getvalue().partition(b"\n")
    reply = json.loads(header)
    assert len(payload) == reply.pop("bytes")
    return reply, payload


def test_every_key_of_the_vocabulary_and_of_setup_has_an_x_keysym():
    assert set(KEYS) <= xworker.MATCH_KEYS
    assert xworker.MATCH_KEYS | xworker.SETUP_KEYS <= set(xworker.KEYSYMS)
    assert xworker.KEYSYMS[0xC0] == 0x60 and xworker.KEYSYMS[0x41] == ord("a")


def test_events_follow_the_windows_workers_rules():
    ok = xworker.valid_event
    assert ok({"kind": "move", "x": 0.0, "y": 1.0}, False)
    assert not ok({"kind": "move", "x": 1.2, "y": 0.5}, False)
    assert ok({"kind": "button", "button": 1, "down": True}, False)
    assert not ok({"kind": "button", "button": 3, "down": True}, False)
    assert ok({"kind": "wheel", "delta": -120}, False)
    assert not ok({"kind": "wheel", "delta": 1320}, False)
    assert ok({"kind": "key", "vk": 0x41, "down": True}, False)
    # The console, space and escape only in setup.
    for vk in (0xC0, 0x20, 0x1B):
        assert not ok({"kind": "key", "vk": vk, "down": True}, False)
        assert ok({"kind": "key", "vk": vk, "down": True}, True)
    assert not ok({"kind": "key", "vk": 0x7B, "down": True}, True)  # F12


def test_arena_lines_keep_only_whole_arena_lines(tmp_path):
    text = (b"[1][x][effectbase.cpp:1783]: ARENA week  1:00, 4 January, 1936 BLU states 4\r\n"
            b"[2][x][other.cpp:1]: something else\n"
            b"[3][x][effectbase.cpp:1783]: ARENA start 12:00")  # fmt: skip
    lines, used = xworker.arena_lines(text)
    assert lines == ["week  1:00, 4 January, 1936 BLU states 4"]
    assert used == text.rfind(b"\n") + 1
    log = tmp_path / "game.log"
    log.write_bytes(text + b"\n")
    lines, offset = xworker.read_arena_log(log, 0)
    assert lines[-1] == "start 12:00" and offset == log.stat().st_size
    assert xworker.read_arena_log(log, offset) == ([], offset)
    # A rewritten (shorter) log is read from its start again.
    assert xworker.read_arena_log(log, offset + 100)[0][0].startswith("week")
    assert xworker.read_arena_log(tmp_path / "none.log", 5) == ([], 0)


def test_launch_arguments_and_the_mod_list(tmp_path):
    user = tmp_path / "user"
    assert xworker.launch_args(user) == ["-debug_mode", "-gdpr-compliant", f"-userdir={user}"]
    assert xworker.launch_args(user, "arenav4blu")[-1] == "-start_save=arenav4blu"
    with pytest.raises(xworker.WorkerError):
        xworker.launch_args(user, "../x")
    mod = tmp_path / "arena-12x8-v4"
    mod.mkdir()
    (mod / "descriptor.mod").write_text('name = "Arena"\n')
    xworker.write_mod_list(user, mod)
    assert (user / "mod" / "arena.mod").read_text().endswith(f'path="{mod.as_posix()}"\n')
    assert json.loads((user / "dlc_load.json").read_text())["enabled_mods"] == ["mod/arena.mod"]


def test_map_errors_are_reported_as_game_control_reports_them(tmp_path):
    log = tmp_path / "error.log"
    log.write_text("a map/definition.csv line\nfine\nMAP_ERROR two\n")
    report = xworker.map_error_report(log)
    assert report[0] == "== map errors in error.log: 2" and report[-1] == "== end of map errors"
    from hoi4_arena.ai_games import parse_map_errors

    assert parse_map_errors("\n".join(report))["count"] == 2


def test_capture_is_the_game_window_with_the_pointer_drawn_in(tmp_path):
    w, out = worker(tmp_path)
    meta, payload = ask(w, out, "capture", encoding="raw")
    assert (meta["width"], meta["height"]) == (32, 24)
    assert meta["cursor"] == [10, 5] and meta["foreground"] and meta["pointer_drawn"]
    frame = np.frombuffer(payload, np.uint8).reshape(24, 32, 4)
    assert tuple(frame[5, 10]) == (255, 0, 0, 255) and tuple(frame[0, 0]) == (0, 0, 200, 255)
    meta, payload = ask(w, out, "capture", regions=[[9, 4, 3, 2]])
    assert meta["full_bytes"] == 0 and meta["region_bytes"] == [24] and len(payload) == 24
    assert "error" in ask(w, out, "capture", views=[224, 224])[0]


def test_input_needs_arming_and_focus_and_is_let_go(tmp_path):
    w, out = worker(tmp_path)
    move = {"kind": "move", "x": 1.0, "y": 0.0}
    assert ask(w, out, "apply", events=[move])[0]["error"] == "input_not_armed_or_focus_lost"
    assert ask(w, out, "arm", mode="match")[0] == {"id": 1, "armed": True}
    press = {"kind": "button", "button": 1, "down": True}
    key = {"kind": "key", "vk": 0x10, "down": True}
    assert ask(w, out, "apply", events=[move, press, key])[0]["applied"] == 3
    assert w.x.events == [("motion", 39, 4), ("button", 3, True), ("key", 0xFFE1, True)]
    # The console key is setup's, not a match's.
    grave = {"kind": "key", "vk": 0xC0, "down": True}
    assert ask(w, out, "apply", events=[grave])[0]["error"] == "invalid_event_batch"
    ask(w, out, "release")
    assert ("button", 3, False) in w.x.events and ("key", 0xFFE1, False) in w.x.events
    assert not w.held_keys and not w.held_buttons
    ask(w, out, "arm", mode="setup")
    w.x.focus = 99  # something else took the focus
    assert "error" in ask(w, out, "apply", events=[grave])[0]
    assert "error" in ask(w, out, "arm", mode="setup")[0]
    reply, _ = ask(w, out, "focus")
    assert reply["foreground"] and w.x.focus == 7


def test_the_watchdog_lets_go_after_idle(tmp_path, monkeypatch):
    monkeypatch.setattr(xworker, "WATCHDOG_S", 0.05)
    w, out = worker(tmp_path)
    ask(w, out, "arm", mode="match")
    ask(w, out, "apply", events=[{"kind": "key", "vk": 0x41, "down": True}])
    import time

    time.sleep(0.4)
    assert not w.armed and ("key", ord("a"), False) in w.x.events


def test_an_observer_gives_no_input_and_runs_nothing(tmp_path):
    w, out = worker(tmp_path, observer=True)
    for op in ("arm", "focus", "launch", "quit"):
        assert ask(w, out, op)[0]["error"] == "observer_refuses_input_and_control"
    assert "width" in ask(w, out, "capture")[0]


def test_unknown_and_broken_requests_are_answered(tmp_path):
    w, out = worker(tmp_path)
    assert ask(w, out, "teleport")[0]["error"] == "unknown_operation"
    out.seek(0)
    w.handle(b"{not json")
    assert json.loads(out.getvalue().splitlines()[0])["error"] == "invalid_json"
    assert ask(w, out, "status")[0]["protocol"] == 1
    assert ask(w, out, "events")[0]["events"] == []
    assert ask(w, out, "launch", mod="../etc")[0] == {
        "id": 1,
        "exit": 1,
        "output": "invalid_mod_name",
    }


def test_the_desktop_client_drives_it_over_the_real_pipe(tmp_path):
    """Desktop starts a worker process and speaks to it; here the process is the X11
    worker over the fake screen."""
    script = textwrap.dedent(f"""
        import sys
        sys.path[:0] = {sys.path[:3]!r}
        sys.path.insert(0, {str(tmp_path)!r})
        from fake_x import FakeX
        from hoi4_arena import xworker
        from pathlib import Path
        w = xworker.Worker(FakeX(), mods=Path({str(tmp_path)!r}), userdir=Path({str(tmp_path)!r}))
        for line in sys.stdin.buffer:
            if line.strip():
                w.handle(line)
    """)
    source = open(__file__, encoding="utf8").read()
    fake = source[source.index("class FakeX") : source.index("def worker(")]
    (tmp_path / "fake_x.py").write_text("import numpy as np\n\n\n" + fake)
    with desktop.Desktop([sys.executable, "-c", script]) as desk:
        assert desk.attached["backend"] == "x11_getimage"
        frame = desk.capture(full=True)
        assert frame.rgb.shape == (24, 32, 3) and tuple(frame.rgb[0, 0]) == (200, 0, 0)
        desk.arm(setup=True)
        assert desk.apply([{"kind": "wheel", "delta": 240}])["applied"] == 1
        desk.release()
        assert desk.game_log(0) == ([], 0)
        assert desk.protocol() == 1


def test_desktop_starts_the_x11_worker_on_linux(monkeypatch):
    monkeypatch.setattr(desktop.sys, "platform", "linux")
    assert desktop.worker_command()[1:] == ["-m", "hoi4_arena.xworker"]
    monkeypatch.setattr(desktop.sys, "platform", "win32")
    assert desktop.worker_command()[0].endswith("hoi4-desktop-worker.exe")

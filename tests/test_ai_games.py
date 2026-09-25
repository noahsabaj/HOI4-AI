import random
import sys
import time

import numpy as np
import pytest

from hoi4_arena import ai_games
from hoi4_arena.ai_games import MAP_BOTTOM, MAP_TOP, OK_MATCH, Popups, arena_offset, pick_country
from hoi4_arena.vision import find_template

# Land tints measured at 1080p (vision.country_pixels) and the sea around them.
BLUE_LAND, RED_LAND, SEA = (120, 134, 145), (168, 145, 131), (33, 43, 61)


def _button():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (22, 95, 3), dtype=np.uint8)


def test_a_template_is_found_wherever_it_opens_and_nowhere_else():
    frame = np.full((1080, 1920, 3), 40, np.uint8)
    button = _button()
    frame[590:612, 1085:1180] = button
    x, y = find_template(frame, button, OK_MATCH)
    assert x == pytest.approx((1085 + 95 / 2) / 1920)
    assert y == pytest.approx((590 + 22 / 2) / 1080)
    assert find_template(np.full_like(frame, 40), button, OK_MATCH) is None


def test_a_popup_is_clicked_only_after_its_reading_time_and_only_once():
    now = [0.0]
    frame = np.full((540, 960, 3), 40, np.uint8)
    button = _button()
    frame[100:122, 200:295] = button
    popups = Popups([button], rng=random.Random(1), clock=lambda: now[0])
    popups.look(frame)
    assert popups.due() is None, "a player reads the popup first"
    now[0] = 4.0
    at = popups.due()
    assert at == pytest.approx(((200 + 47.5) / 960, 111 / 540))
    assert popups.due() is None


def test_the_arena_offset_points_from_the_screen_centre_to_the_arena_centre():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[:] = SEA
    # Blue's half left of the seam, Red's right, both shifted right of centre by 192 px.
    frame[300:700, 700:1152] = BLUE_LAND
    frame[300:700, 1152:1604] = RED_LAND
    down, right = arena_offset(frame)
    assert right == pytest.approx((1152 - 960) / 1920, abs=1e-3)
    assert down == pytest.approx((500 - 540) / 1080, abs=1e-3)
    # The HUD rows are ignored: land drawn only there is not an arena.
    chrome = np.zeros_like(frame)
    chrome[: MAP_TOP - 1] = BLUE_LAND
    chrome[1080 - MAP_BOTTOM + 1 :] = RED_LAND
    assert arena_offset(chrome) is None


def test_picking_a_country_clicks_its_land_until_its_flag_shows(monkeypatch):
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[:] = SEA
    frame[300:700, 700:1152] = BLUE_LAND
    frame[300:700, 1152:1604] = RED_LAND
    x0, y0, x1, y1 = ai_games.PICKER_FLAG
    frame[y0:y1, x0:x1] = (40, 60, 200)  # Blue, the default, is selected.
    clicks = []

    def click(desk, x, y):
        clicks.append((x, y))
        if 1152 <= x * 1920 < 1604:
            frame[y0:y1, x0:x1] = (200, 40, 40)

    monkeypatch.setattr(ai_games, "screen", lambda desk: frame)
    monkeypatch.setattr(ai_games, "click", click)
    monkeypatch.setattr(ai_games.time, "sleep", lambda s: None)
    monkeypatch.setattr(ai_games, "act", lambda desk, events, pause=0.15: None)
    assert ai_games.picked(frame) == "BLU"
    assert pick_country(None, "BLU") and not clicks, "already selected"
    assert pick_country(None, "RED")
    x, y = clicks[-1]
    assert 1152 <= x * 1920 < 1604 and 300 <= y * 1080 < 700
    assert ai_games.picked(frame) == "RED"
    # No Blue land and Blue not selected: the pick never takes.
    frame[300:700, 700:1152] = SEA
    assert not pick_country(None, "BLU")


def test_the_front_is_where_blue_land_meets_red():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[:] = SEA
    frame[300:700, 700:1152] = BLUE_LAND
    frame[300:700, 1152:1604] = RED_LAND
    xs = [x * 1920 for x, _ in ai_games.front_points(frame)]
    ys = [y * 1080 for _, y in ai_games.front_points(frame)]
    assert xs and min(xs) >= 1150 and max(xs) <= 1156
    assert min(ys) >= 300 and max(ys) < 700
    frame[300:700, 1152:1604] = SEA
    assert ai_games.front_points(frame) == []
    assert len(ai_games.land_points(frame)) == 400 * 452


def test_each_speed_is_played_as_both_countries_and_the_pcs_are_out_of_step():
    here = [ai_games.game_plan("here", i, [4, 5]) for i in range(4)]
    peer = [ai_games.game_plan("peer", i, [4, 5]) for i in range(4)]
    assert here == [("BLU", 4), ("RED", 4), ("BLU", 5), ("RED", 5)]
    assert peer == [("RED", 4), ("BLU", 4), ("RED", 5), ("BLU", 5)]


def test_each_arena_in_turn_is_played_as_both_countries():
    mods = ["a", "b"]
    games = [
        (ai_games.game_arena(mods, i), ai_games.game_plan("here", i, [5])[0]) for i in range(8)
    ]
    assert games[:4] == [("a", "BLU"), ("a", "RED"), ("b", "BLU"), ("b", "RED")]
    assert games[4:] == games[:4]
    # Accepted arenas take turns in every third pair; the main arena keeps two thirds.
    arenas = [ai_games.game_arena(["v4"], i, ["x", "y"]) for i in range(12)]
    assert arenas == ["v4"] * 4 + ["x"] * 2 + ["v4"] * 4 + ["y"] * 2
    latest = ai_games.latest_versions(["m/arena-plains-v1", "m/arena-bay-v2", "m/arena-plains-v2"])
    assert latest == ["m/arena-bay-v2", "m/arena-plains-v2"]


def test_an_arena_request_is_claimed_once_answered_and_accepted_once(tmp_path):
    import json
    import os

    queue = tmp_path / "arenas" / "queue"
    queue.mkdir(parents=True)
    (queue / "wide.json").write_text(json.dumps({"mod": "D:/mods/arena-wide"}))
    (queue / "broken.json").write_text("{")
    for path in queue.iterdir():
        os.utime(path, (1, 1))  # Long since written.
    request = ai_games.take_request(queue, "here")
    assert request["name"] == "wide" and request["mod"] == "D:/mods/arena-wide"
    assert request["claimed"].name == f"wide.here-{os.getpid()}.taken"
    assert request["claimed"].exists()
    # The other station finds nothing left, and a broken request is answered, not played.
    assert ai_games.take_request(queue, "peer") is None
    answered = json.loads((tmp_path / "arenas" / "results" / "broken.json").read_text())
    assert answered["accepted"] is False and "bad request" in answered["error"]
    assert not (queue / "broken.json").exists()
    # A request still being written waits.
    (queue / "fresh.json").write_text("{")
    assert ai_games.take_request(queue, "here") is None and (queue / "fresh.json").exists()
    for _ in range(2):
        ai_games.accept_arena(queue, "D:/mods/arena-wide")
    assert ai_games.accepted_arenas(queue) == ["D:/mods/arena-wide"]


def test_an_arena_passes_only_if_it_loads_starts_and_is_played_out(tmp_path):
    request = {"name": "wide", "mod": "D:/mods/arena-wide"}
    entry = {"station": "here", "started_as": "RED", "winner": "BLU", "seconds": 300}
    stages = {"loaded": True, "started": True, "shots": {"start": tmp_path / "start.png"}}
    result = ai_games.test_result(request, entry, tmp_path, stages)
    assert result["accepted"] and result["outcome"] == "BLU"
    assert result["screenshots"] == {"start": str(tmp_path / "start.png")}
    failed = {**entry, "winner": "timeout", "reason": "RuntimeError: the scripted player's setup"}
    assert not ai_games.test_result(request, failed, tmp_path, stages)["accepted"]
    unstarted = {"station": "here", "started_as": "RED", "error": "RuntimeError: no map"}
    stages = {"loaded": True, "started": False, "shots": {}}
    result = ai_games.test_result(request, unstarted, tmp_path, stages)
    assert not result["accepted"] and result["error"] == "RuntimeError: no map"


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows'")
def test_an_arena_kept_elsewhere_is_linked_into_the_mods_folder(tmp_path):
    mods = tmp_path / "mods"
    mods.mkdir()
    source = tmp_path / "elsewhere" / "arena-wide"
    source.mkdir(parents=True)
    (source / "descriptor.mod").write_text('name = "wide"')
    ai_games.local_mod(source, mods)
    assert (mods / "arena-wide" / "descriptor.mod").read_text() == 'name = "wide"'
    ai_games.local_mod(source, mods)  # Linked already: nothing to do.
    other = tmp_path / "other" / "arena-wide"
    other.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="already called arena-wide"):
        ai_games.local_mod(other, mods)


def test_the_second_pc_is_lent_out_until_the_evaluation_is_done(tmp_path):
    import json
    import os
    from unittest.mock import Mock

    root = tmp_path / "eval"
    (root / "queue").mkdir(parents=True)
    (root / "queue" / "bc-v3.json").write_text(json.dumps({"minutes": 20}))
    os.utime(root / "queue" / "bc-v3.json", (1, 1))
    reservation = ai_games.take_reservation(root)
    assert reservation["name"] == "bc-v3" and reservation["minutes"] == 20
    assert ai_games.take_reservation(root) is None
    station = Mock()
    station.name = "peer"
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds
        if clock[0] >= 600:  # The evaluation ends after ten minutes.
            (root / "done").mkdir(exist_ok=True)
            (root / "done" / "bc-v3.json").write_text("{}")

    assert ai_games.lend(station, root, reservation, clock=lambda: clock[0], sleep=sleep)
    station.quit.assert_called_once()
    assert json.loads((root / "granted" / "bc-v3.json").read_text())["minutes"] == 20
    assert 600 <= clock[0] < 620 and not reservation["claimed"].exists()
    # One that never says it is done gets the PC for its minutes and 15 more.
    (root / "queue" / "slow.json").write_text(json.dumps({"minutes": 5}))
    os.utime(root / "queue" / "slow.json", (1, 1))
    clock[0] = 0.0
    slow = ai_games.take_reservation(root)
    assert not ai_games.lend(
        station,
        root,
        slow,
        clock=lambda: clock[0],
        sleep=lambda s: None or clock.__setitem__(0, clock[0] + s),
    )
    assert clock[0] >= 20 * 60


def test_start_saves_are_named_per_arena_and_side():
    saves = ai_games.parse_saves(["arena-12x8-v4:BLU:arenav4blu", "arena-12x8-v4:RED:arenav4red"])
    assert saves == {("arena-12x8-v4", "BLU"): "arenav4blu", ("arena-12x8-v4", "RED"): "arenav4red"}
    with pytest.raises(ValueError):
        ai_games.parse_saves(["arena-12x8-v4:GRN:x"])


def test_the_opening_hours_end_paused_even_if_the_mark_blinks(monkeypatch):
    from unittest.mock import Mock

    game = {"paused": True, "presses": 0, "looks": 0}

    def fake_act(desk, events, pause=0.15):
        if any(e.get("vk") == 0x20 and e.get("down") for e in events):
            game["paused"] = not game["paused"]
            game["presses"] += 1

    def fake_screen(desk):
        game["looks"] += 1
        return game

    rules = Mock()
    # Paused, but the blinking mark shows only on every third look.
    rules.matches.side_effect = lambda name, g: g["paused"] and g["looks"] % 3 == 0
    monkeypatch.setattr(ai_games, "act", fake_act)
    monkeypatch.setattr(ai_games, "screen", fake_screen)
    monkeypatch.setattr(ai_games.time, "sleep", lambda s: None)
    ai_games.run_briefly(None, rules, 1.5)
    assert game["paused"] and game["presses"] == 2


def test_the_country_played_is_read_from_the_flag_at_the_top_left():
    frame = np.full((1080, 1920, 3), 30, np.uint8)
    assert ai_games.picked(frame, ai_games.TOP_FLAG) is None
    x0, y0, x1, y1 = ai_games.TOP_FLAG
    frame[y0:y1, x0:x1] = (28, 57, 114)  # Blue's flag, measured after loading its save.
    assert ai_games.picked(frame, ai_games.TOP_FLAG) == "BLU"
    frame[y0:y1, x0:x1] = (150, 40, 40)
    assert ai_games.picked(frame, ai_games.TOP_FLAG) == "RED"


def test_closeups_zoom_fully_in_over_each_country(monkeypatch, tmp_path):
    frame = np.full((1080, 1920, 3), 40, np.uint8)
    frame[300:700, 500:950] = BLUE_LAND
    frame[300:700, 950:1400] = RED_LAND
    moves, wheels = [], []

    def fake_act(desk, events, pause=0.15):
        for e in events:
            if e["kind"] == "move":
                moves.append((e["x"], e["y"]))
            elif e["kind"] == "wheel":
                wheels.append(e["delta"])

    monkeypatch.setattr(ai_games, "act", fake_act)
    monkeypatch.setattr(ai_games, "screen", lambda desk: frame)
    monkeypatch.setattr(ai_games, "recentre", lambda desk: True)
    monkeypatch.setattr(ai_games.time, "sleep", lambda s: None)
    shots = ai_games.closeups(None, tmp_path)
    assert len(shots) == 8 and all(p.exists() for p in shots)
    assert wheels.count(120) == 8 * ai_games.ZOOM_MAX
    # Four points in each country's land, and the pointer off the map for each view.
    points = [m for m in moves if m != (0.65, 0.012)]
    assert sum(x < 950 / 1920 for x, _ in points) == 4 and len(points) == 8


def test_a_claim_whose_recorder_died_is_offered_again(tmp_path):
    import os

    queue = tmp_path / "queue"
    queue.mkdir()
    mine = queue / f"ours.peer-{os.getpid()}.taken"
    mine.write_text("{}")
    # A process id that is not running (ids are multiples of 4 on Windows; 3 never is).
    (queue / "lost.peer-3.taken").write_text('{"minutes": 50}')
    (queue / "old.taken").write_text("{}")  # Before claims named their process.
    assert ai_games.reoffer(queue) == ["lost", "old"]
    assert mine.exists() and (queue / "lost.json").read_text() == '{"minutes": 50}'
    assert (queue / "old.json").exists()
    assert ai_games.alive(os.getpid()) and not ai_games.alive(3)


def test_a_popup_is_found_at_half_size_and_placed_at_full_size():
    rng = np.random.default_rng(3)
    frame = rng.integers(20, 60, (1080, 1920, 3), dtype=np.uint8)
    # Buttons are drawn in flat blocks, not noise: blocks survive the halving.
    button = np.kron(rng.integers(60, 255, (6, 24, 3)), np.ones((4, 4, 1))).astype(np.uint8)
    button = button[:22, :95]
    frame[611:633, 1301:1396] = button  # At an odd pixel, which halving blurs.
    popups = Popups([button], rng=random.Random(1), clock=lambda: 10.0)
    popups.look(frame)
    (x, y), _ = popups.pending
    assert abs(x * 1920 - (1301 + 47.5)) < 1 and abs(y * 1080 - (611 + 11)) < 1
    empty = Popups([button], rng=random.Random(1), clock=lambda: 10.0)
    empty.look(rng.integers(20, 60, (1080, 1920, 3), dtype=np.uint8))
    assert empty.pending is None


def test_a_start_through_the_menus_is_saved_for_the_next_games(tmp_path):
    name = ai_games.save_name("D:/mods/arena-marsh-v4", "RED")
    assert name == "arenamarshv4red"
    registry = tmp_path / "saves-peer.json"
    assert ai_games.known_saves(registry) == {}
    ai_games.remember_save(registry, "arena-marsh-v4", "RED", name)
    ai_games.remember_save(registry, "arena-marsh-v4", "BLU", "arenamarshv4blu")
    assert ai_games.known_saves(registry) == {
        ("arena-marsh-v4", "RED"): "arenamarshv4red",
        ("arena-marsh-v4", "BLU"): "arenamarshv4blu",
    }


def test_the_map_errors_are_read_from_the_report():
    report = "\n".join(
        [
            "== error.log (09/24/2026 02:00:00)",
            "[02:00:01][map.cpp:10]: something",
            "== map errors in error.log: 3",
            "[02:00:01][map.cpp:10]: MAP_ERROR: province 12 has no terrain",
            "  map/adjacencies.csv: line 4",
            "== end of map errors",
        ]
    )
    found = ai_games.parse_map_errors(report)
    assert found["count"] == 3 and len(found["examples"]) == 2
    assert found["examples"][0].startswith("[02:00:01]")
    assert ai_games.parse_map_errors("report:\nnothing") is None


def test_no_new_game_past_the_memory_limit(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    reading = {"memory": {"commit_mb": 33000, "commit_limit_mb": 34568}}
    observer = MagicMock()
    observer.__enter__.return_value.telemetry.return_value = reading
    import hoi4_arena.telemetry as telemetry

    monkeypatch.setattr(telemetry, "open_observer", lambda peer: observer)
    station = ai_games.Station("peer", "peer.json")
    reports = []

    class Desk:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def report(self):
            reports.append(1)
            return "report:\nnothing about the pagefile"  # A limit that cannot grow.

    monkeypatch.setattr(station, "connect", lambda attach=True: Desk())
    assert "would be 95% of the 34568 MB" in ai_games.memory_room(station)
    assert "would be 97% of the 34568 MB" in ai_games.memory_room(station, 500)
    assert ai_games.memory_room(ai_games.Station("here")) is None
    with pytest.raises(ai_games.MemoryStop):
        ai_games.check_memory(station, ai_games.GAME_MB)
    reading["memory"]["commit_mb"] = 27000  # No game running: room for one.
    ai_games.check_memory(station, ai_games.GAME_MB)
    assert len(reports) == 1, "the pagefile is read once"
    # A pagefile Windows manages, on a drive with room: the same PC has room for games.
    station = ai_games.Station("peer", "peer.json")
    monkeypatch.setattr(station, "pagefile", lambda: {"drive": "C:", "max_mb": None})
    reading["memory"].update(commit_mb=33000, total_mb=32520, available_mb=25000)
    reading["disks"] = [{"drive": "C:", "free_gb": 700.0, "total_gb": 1023.0}]
    ai_games.check_memory(station, ai_games.GAME_MB)


def test_a_start_save_is_loaded_in_game_only_where_its_name_is_calibrated(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_games, "SCREENS", tmp_path)
    from PIL import Image

    rng = np.random.default_rng(4)
    name = rng.integers(0, 255, (18, 80, 3), dtype=np.uint8)
    Image.fromarray(name).save(tmp_path / "save-arenav4red.png")
    assert ai_games.can_load("arenav4red") and not ai_games.can_load("arenamarshv6red")
    assert not ai_games.can_load(None)
    frame = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    frame[613:631, 865:945] = name
    x, y = ai_games.shown(frame, "save-arenav4red", 0.93)
    assert abs(x * 1920 - 905) < 1 and abs(y * 1080 - 622) < 1
    assert (
        ai_games.shown(rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8), "save-arenav4red")
        is None
    )


def test_the_menu_opens_through_a_win_s_conference_and_its_popups(tmp_path, monkeypatch):
    # A win leaves more than the four screens the first version cleared: a popup, the
    # peace conference, two popups after it, then the menu button opens the menu.
    state = ["popup", "conference", "popup", "popup", "map", "menu"]
    shows = {"conference-exit": "conference", "menu-load-game": "menu"}

    def click(desk, x, y):
        done = {
            "popup": (x, y) == (0.5, 0.6),
            "conference": (x, y) == ai_games.CONFIRM_OK,
            "map": (x, y) == ai_games.MENU_BUTTON,
        }
        if done.get(state[0]):
            state.pop(0)

    monkeypatch.setattr(ai_games, "screen", lambda desk: np.zeros((1080, 1920, 3), np.uint8))
    monkeypatch.setattr(ai_games, "shown", lambda rgb, name, threshold=0.9: (
        (0.5, 0.5) if shows.get(name) == state[0] else None))  # fmt: skip
    monkeypatch.setattr(ai_games, "find_template", lambda rgb, template, threshold: (
        (0.5, 0.6) if state[0] == "popup" else None))  # fmt: skip
    monkeypatch.setattr(ai_games, "click", click)
    monkeypatch.setattr(ai_games.time, "sleep", lambda s: None)
    assert ai_games.open_menu(None, [object()])
    assert state == ["menu"]
    # A menu that never opens: given up when the time is up, with the screen kept.
    state[:] = ["stuck"]
    clock = iter(range(0, 1000, 5))
    monkeypatch.setattr(ai_games.time, "monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="menu did not open"):
        ai_games.load_in_game(None, "arenav4blu", [object()], None, tmp_path / "g-start-failed.png")
    assert (tmp_path / "g-menu-failed.png").exists()


class ToyMap:
    """A 1080p view of a toy arena for the camera, 2400 x 800 world units: Blue's land to
    the left of FRONT_X, Red's to its right, sea around. Wheel notches zoom about the
    pointer, 1.105 times a notch (the arena fills half the screen at 0, a third of it the
    screen at 18, as measured), and an arrow key pans a screen a second. As in the game,
    the camera's centre stops at the map's edge: 600 units of sea around the land."""

    FRONT_X = 1200

    def __init__(self, cx, cy, zoom):
        self.cx, self.cy, self.zoom, self.pointer, self.down = cx, cy, zoom, (0.5, 0.5), {}
        self.seen = []  # (zoom, the front's x on screen) at each capture
        self.applied = []  # every input, logged by the recorder or not

    def scale(self):
        return 0.4 * 1.105**self.zoom

    def world(self, x, y):
        return self.cx + (x - 0.5) * 1920 / self.scale(), self.cy + (y - 0.5) * 1080 / self.scale()

    def arm(self, setup=False):
        pass

    def release(self):
        pass

    def focus(self):
        return True

    def apply(self, events):
        import time

        self.applied.extend(events)
        for e in events:
            if e["kind"] == "move":
                self.pointer = (e["x"], e["y"])
            elif e["kind"] == "wheel":
                wx, wy = self.world(*self.pointer)
                self.zoom = max(0, min(26, self.zoom + (1 if e["delta"] > 0 else -1)))
                self.cx = wx - (self.pointer[0] - 0.5) * 1920 / self.scale()
                self.cy = wy - (self.pointer[1] - 0.5) * 1080 / self.scale()
            elif e["kind"] == "key" and e["down"]:
                self.down[e["vk"]] = time.monotonic()
            elif e["kind"] == "key" and e["vk"] in self.down:
                seconds = time.monotonic() - self.down.pop(e["vk"])
                across = {0x25: -1, 0x27: 1}.get(e["vk"], 0)
                down = {0x26: -1, 0x28: 1}.get(e["vk"], 0)
                self.cx += across * seconds * 1920 / self.scale()
                self.cy += down * seconds * 1080 / self.scale()
            self.cx = min(3000, max(-600, self.cx))
            self.cy = min(1400, max(-600, self.cy))
        return {"t_ns": time.monotonic_ns()}  # As the worker replies, for Logged.

    def capture(self, full=True):
        from types import SimpleNamespace

        wx, _ = self.world(np.arange(1920) / 1920, 0)
        _, wy = self.world(0, np.arange(1080) / 1080)
        land = ((0 <= wy) & (wy < 800))[:, None] & ((0 <= wx) & (wx < 2400))[None]
        blue = land & (wx < self.FRONT_X)[None]
        rgb = np.empty((1080, 1920, 3), np.uint8)
        rgb[:] = SEA
        rgb[blue] = BLUE_LAND
        rgb[land & ~blue] = RED_LAND
        self.seen.append((self.zoom, 0.5 + (self.FRONT_X - self.cx) * self.scale() / 1920))
        return SimpleNamespace(rgb=rgb, meta={})


def test_the_camera_finds_the_front_and_keeps_it_in_the_middle():
    import threading

    # Close in over Blue's far west, where the front is off screen.
    world = ToyMap(cx=300, cy=400, zoom=16)
    stop = threading.Event()

    class NoPopups:
        def due(self):
            return None

    run = threading.Thread(target=ai_games.camera, args=(world, stop, "toy", NoPopups()),
                           kwargs={"rng": random.Random(1)})  # fmt: skip
    run.start()
    stop.wait(8)
    stop.set()
    run.join(10)
    assert not run.is_alive()
    low, high = ai_games.FRONT_ZOOM
    settled = [x for zoom, x in world.seen if low <= zoom <= high and abs(x - 0.5) < 0.12]
    assert settled, "the front, centred, at the zoom that shows its counters"
    assert max(zoom for zoom, _ in world.seen) < ai_games.ZOOM_TERRAIN, "never past the counters"
    zoom, x = world.seen[-1]
    assert 0 <= x <= 1, "the front on screen at the end"


def test_a_kick_knocks_the_camera_off_the_map_or_right_in(monkeypatch):
    monkeypatch.setattr(ai_games.time, "sleep", lambda s: None)
    kinds = {}
    for seed in range(12):
        world = ToyMap(cx=1200, cy=400, zoom=11)
        kinds[ai_games.kick_camera(world, random.Random(seed), 11)] = world.applied
    assert set(kinds) == {("edge", 11), ("close", ai_games.ZOOM_MAX)}
    edge = kinds[("edge", 11)]
    assert [e["kind"] for e in edge] == ["key", "key"] and edge[0]["vk"] in (0x25, 0x26, 0x27, 0x28)
    close = kinds[("close", ai_games.ZOOM_MAX)]
    assert close[0]["kind"] == "move" and len(close) == 1 + ai_games.ZOOM_MAX - 11


def test_the_camera_finds_the_front_again_after_a_kick_that_is_no_label():
    """The recorder knocks its camera astray through the desktop it does not log, so the
    recording shows the camera move with no input and then its way back (DART)."""
    import threading

    world = ToyMap(cx=1200, cy=400, zoom=11)
    logged = ai_games.Logged(world)
    stop, kicked = threading.Event(), []

    class NoPopups:
        def due(self):
            return None

    low, high = ai_games.FRONT_ZOOM

    def on_front():
        x = 0.5 + (world.FRONT_X - world.cx) * world.scale() / 1920
        return low <= world.zoom <= high and 0 <= x <= 1

    kwargs = {"rng": random.Random(3), "kicks": (4.0, 20.0), "kicked": kicked}
    kwargs["frame"] = lambda: len(world.seen)
    run = threading.Thread(target=ai_games.camera, args=(logged, stop, "toy", NoPopups()),
                           kwargs=kwargs)  # fmt: skip
    run.start()
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline and not (kicked and on_front()):
        time.sleep(0.05)
    found = bool(kicked) and on_front()
    stop.set()
    run.join(10)
    assert not run.is_alive() and kicked, "knocked astray at least once"
    first = kicked[0]
    assert first["kind"] in ("edge", "close") and first["from_frame"] <= first["to_frame"]
    assert found, "the front found again, at the zoom that shows its counters"
    labelled = [e["event"] for e in logged.take()]
    assert len(labelled) < len(world.applied), "the kick's inputs are no label"


def test_a_plan_line_is_no_front_and_battles_are_the_green_badges_on_it():
    import cv2

    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[:] = SEA
    frame[300:700, 700:1152] = BLUE_LAND
    # A battle plan's red line across Blue's land, as on 2026-09-24: not a front.
    frame[495:499, 750:1100] = RED_LAND
    assert ai_games.front_points(frame) == []
    frame[300:700, 1152:1604] = RED_LAND
    front = ai_games.front_points(frame)
    assert front and min(x for x, _ in front) * 1920 >= 1150
    # A battle badge on the front, and a green ring away from it.
    cv2.circle(frame, (1152, 450), 16, (3, 158, 3), 5)
    cv2.circle(frame, (800, 600), 16, (3, 158, 3), 5)
    battles = ai_games.battle_points(frame, front)
    assert len(battles) == 1 and abs(battles[0][0] * 1920 - 1152) < 3
    assert len(ai_games.battle_points(frame)) == 2
    # Counters crowd on the front, a lone one away from it.
    for x, y in ((1120, 400), (1170, 420), (1140, 470), (760, 650)):
        frame[y : y + 12, x : x + 22] = (40, 60, 220)
    counters = ai_games.counter_points(frame)
    assert len(counters) == 4
    x, y = ai_games.busiest(ai_games.near_front(counters, front))
    assert 1120 <= x * 1920 <= 1192 and 400 <= y * 1080 <= 482


def test_a_game_loaded_in_game_reads_the_log_from_its_end():
    # The worker returns at most a chunk a request: here 3 lines, and a 10-line log.
    log = [f"line {i}" for i in range(10)]

    class Desk:
        def game_log(self, offset=0):
            return log[offset : offset + 3], min(len(log), offset + 3)

    assert ai_games.log_end(Desk()) == 10
    log.clear()
    assert ai_games.log_end(Desk()) == 0

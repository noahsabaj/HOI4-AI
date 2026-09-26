"""The pieces that let a learned policy learn from the scripted player and play live."""

import json
from collections import deque

import numpy as np
import pytest
import torch
from test_dataset import _recording, needs_ffmpeg

from hoi4_arena import play
from hoi4_arena.actions import GRID, SLOTS, VOCAB, previous_actions, still_held, with_held
from hoi4_arena.dataset import VideoSessions, player_outcome, session_labels
from hoi4_arena.models import Policy
from hoi4_arena.privileged import DIM, NAMES, decision_states, state_rows
from hoi4_arena.train import presses, state_loss, state_r2

SPACE = {"kind": "key", "vk": 0x20, "down": True}
CLICK = {"kind": "button", "button": 0, "down": True}


def _scripted(root, **manifest):
    _recording(root, [1, 1], source="scripted", events=[(500_000_000, CLICK)])
    path = root / "manifest.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **manifest}))


def test_a_lead_in_of_none_keeps_the_first_seconds_of_a_game_as_labels(tmp_path):
    """The scripted player forms its army in the first 2.5 s, inside a clip's lead-in."""
    _scripted(tmp_path / "game")
    default = session_labels(tmp_path / "game", sources=("scripted",))
    early = session_labels(tmp_path / "game", sources=("scripted",), lead_in=0)
    assert not default["actions"][..., 0].any(), "the click at 0.5 s precedes every decision"
    assert early["decisions"][0] == 0
    assert VOCAB.index(CLICK) in early["actions"][2, :, 0]
    with pytest.raises(ValueError, match="clip"):
        VideoSessions(tmp_path, sources=("scripted",), lead_in=0, clips=True)


def test_a_key_the_harness_presses_is_dropped_from_the_labels_not_the_decision(tmp_path):
    _recording(tmp_path / "game", [1, 1], source="scripted", events=[(2_000_000_000, SPACE)])
    kept = session_labels(tmp_path / "game", sources=("scripted",))
    assert not kept["valid"][1], "space is outside the vocabulary: the decision is invalid"
    dropped = session_labels(tmp_path / "game", sources=("scripted",), drop_keys=(0x20,))
    assert dropped["valid"][1] and not dropped["actions"][1].any()


UP = {"kind": "key", "vk": 0x26, "down": True}


def test_the_old_camera_s_arrow_keys_are_dropped_from_recordings_made_before_it_changed():
    """Until #90 the recorder's camera panned at random; a policy that learned it walked its
    camera off the map live. A player's own recordings keep their arrows."""
    from hoi4_arena.dataset import ARROW_KEYS, camera_keys_dropped

    old = {"source": "scripted", "recorder": {"started_unix": 1000.0}}
    assert camera_keys_dropped(old, None) == ()
    assert camera_keys_dropped(old, 2000.0) == ARROW_KEYS
    assert camera_keys_dropped({**old, "recorder": {"started_unix": 3000.0}}, 2000.0) == ()
    assert camera_keys_dropped({"source": "ai"}, 2000.0) == ARROW_KEYS, "no start time: old"
    assert camera_keys_dropped({**old, "source": "human"}, 2000.0) == ()


@needs_ffmpeg
def test_training_drops_the_old_camera_per_recording(tmp_path):
    for name, started in (("old", 1000.0), ("new", 3000.0)):
        _recording(tmp_path / name, [1, 1], source="scripted", events=[(2_000_000_000, UP)])
        path = tmp_path / name / "manifest.json"
        manifest = {**json.loads(path.read_text()), "recorder": {"started_unix": started}}
        path.write_text(json.dumps(manifest))
    common = {"sources": ("scripted",), "length": 2, "burn_in": 1, "device": "cpu"}
    sessions = VideoSessions(tmp_path, camera_since=2000.0, **common).sessions
    pressed = {s["root"].name: (s["actions"][..., 0] == VOCAB.index(UP)).any() for s in sessions}
    assert pressed == {"old": False, "new": True}


def test_the_decisions_a_camera_kick_spans_weigh_nothing(tmp_path):
    """The recorder's camera kicks (ai_games.kick_camera) move the frames with no input."""
    _scripted(tmp_path / "game", camera_kicks=[{"kind": "edge", "from_frame": 10, "to_frame": 20}])
    labels = session_labels(tmp_path / "game", sources=("scripted",), lead_in=0)
    times, decisions = labels["times"], labels["decisions"]
    pushed = (decisions >= times[10]) & (decisions <= times[20])
    assert pushed.any() and (~pushed).any()
    assert (labels["weight"][pushed] == 0).all() and (labels["weight"][~pushed] == 1).all()
    assert labels["valid"][pushed].all(), "their windows stay, for the recovery after"


def test_a_lost_game_counts_for_its_loser_weight(tmp_path):
    _scripted(tmp_path / "won", winner="BLU", started_as="BLU", players=["BLU"])
    _scripted(tmp_path / "lost", winner="BLU", started_as="RED", players=["RED"])
    _scripted(tmp_path / "draw", winner="timeout", started_as="RED")
    assert player_outcome(json.loads((tmp_path / "won" / "manifest.json").read_text())) == "win"
    for name, expected in (("won", 1.0), ("lost", 0.25), ("draw", 0.25)):
        labels = session_labels(tmp_path / name, sources=("scripted",), loser_weight=0.25)
        assert labels["weight"] == pytest.approx(np.full(len(labels["weight"]), expected))


@needs_ffmpeg
def test_splits_json_chooses_the_held_out_games(tmp_path):
    _scripted(tmp_path / "a")
    _scripted(tmp_path / "b")
    common = {"sources": ("scripted",), "length": 2, "burn_in": 1, "device": "cpu"}
    assert len(VideoSessions(tmp_path, **common).sessions) == 2
    (tmp_path / "splits.json").write_text(json.dumps({"b": "validation"}))
    assert [s["root"].name for s in VideoSessions(tmp_path, **common).sessions] == ["a"]
    held = VideoSessions(tmp_path, split="validation", **common).sessions
    assert [s["root"].name for s in held] == ["b"]


def _day(tag, date, divisions, at, casualties=0.0):
    counts = " ".join(f"{s}={at.get(s, 0)}" for s in range(1, 17))
    return (
        f"day  {date} {tag} states 8 owned 8 divisions {divisions} surrender 0 strength 1"
        f" casualties {casualties} manpower 0 deployed 15 rifles 4.8 needed 4.8 at {counts}"
    )


def _log(root, entries):
    (root / "arena-log.jsonl").write_text(
        "\n".join(json.dumps({"frame": f, "line": line}) for f, line in entries) + "\n"
    )


def test_the_true_state_is_read_from_the_player_s_side_and_follows_control(tmp_path):
    entries = [
        (1, "player RED"),
        (5, _day("RED", "23:00, 1 January, 1936", 8, {15: 4, 16: 4})),
        (5, _day("BLU", "24:00, 1 January, 1936", 8, {7: 4, 8: 4})),
        (9, "control RED from BLU West 7  12:00, 9 March, 1936"),
        (9, _day("RED", "23:00, 9 March, 1936", 8, {7: 5, 16: 3}, casualties=2.0)),
        (9, _day("BLU", "24:00, 9 March, 1936", 7, {8: 7})),
    ]
    rows = state_rows([{"frame": f, "line": line} for f, line in entries], "RED")
    # A row after every report once both sides have reported: Red's report at frame 9
    # makes one with Blue's old report, Blue's then another.
    assert [frame for frame, _ in rows] == [5, 9, 9]
    first, last = np.array(rows[0][1]), np.array(rows[-1][1])
    held = NAMES.index("held_1")
    assert list(first[held : held + 16]) == [-1.0] * 8 + [1.0] * 8, "Red holds 9 to 16"
    assert last[held + 6] == 1.0, "Red took West 7"
    assert last[NAMES.index("own_casualties")] == pytest.approx(np.log1p(2.0))
    assert last[NAMES.index("enemy_divisions")] == pytest.approx(np.log1p(7))
    assert last[NAMES.index("own_at_7")] == pytest.approx(np.log1p(5))
    # Per decision: before the first report the game has not run, and the first stands.
    _log(tmp_path, entries)
    states = decision_states(tmp_path, {"players": ["RED"]}, np.array([0, 3, 4, 7, 8, 20]))
    assert states.shape == (6, DIM) and not np.isnan(states).any()
    for index, expected in ((0, first), (2, first), (3, first), (4, last), (5, last)):
        assert np.allclose(states[index], expected)
    assert np.isnan(decision_states(tmp_path, {}, np.array([0]))).all(), "no player, no state"


def test_the_state_loss_counts_only_known_targets_and_r2_reads_it_back():
    head = torch.nn.Linear(4, DIM)
    memory = torch.randn(3, 4)
    target = torch.full((3, DIM), float("nan"))
    assert float(state_loss(head, memory, target).detach()) == 0.0
    target[0] = 1.0
    loss = state_loss(head, memory, target)
    expected = (head(memory[:1]) - 1).square().mean()
    assert float(loss.detach()) == pytest.approx(float(expected.detach()), rel=1e-5)
    truth = torch.randn(50, DIM)
    report = state_r2(truth.clone(), truth)
    assert report["own"] == pytest.approx(1.0) and report["held"] == pytest.approx(1.0)
    assert state_r2(torch.zeros(50, DIM), truth)["own"] < 0.1


def test_a_decision_presses_when_any_slot_holds_a_key_or_button_down():
    actions = torch.zeros(3, SLOTS, 3, dtype=torch.long)
    actions[1, 3, 0] = VOCAB.index(CLICK)
    actions[2, 0, 0] = 1  # A move alone.
    assert presses(actions).tolist() == [False, True, False]


class _Desk:
    """A worker stand-in that keeps what it was asked to do."""

    def __init__(self):
        self.applied, self.armed = [], []

    def arm(self, setup=False):
        self.armed.append(setup)

    def apply(self, events):
        self.applied.extend(events)
        return {"t_ns": len(self.applied)}


def _token(kind, x=0, y=0):
    return [kind, x, y]


def test_the_dispatcher_notices_a_press_on_the_speed_control_and_keeps_its_events():
    desk = _Desk()
    dispatcher = play.Dispatcher(desk, clock=lambda: 0.0)
    plus_x = round(play.SPEED_UP[0] * (GRID - 1))
    plus_y = round(play.SPEED_UP[1] * (GRID - 1))
    action = np.zeros((SLOTS, 3), np.int64)
    action[0] = _token(1, plus_x, plus_y)
    action[1] = _token(VOCAB.index(CLICK))
    dispatcher.start(action, 0.0, (100.0, 100.0))
    dispatcher.join()
    assert dispatcher.speed_clicks == 1 and desk.armed == [False]
    taken = dispatcher.take()
    assert [e["event"]["kind"] for e in taken] == ["move", "button"]
    assert dispatcher.take() == []
    elsewhere = np.zeros((SLOTS, 3), np.int64)
    elsewhere[0] = _token(VOCAB.index(CLICK))
    dispatcher.start(elsewhere, 0.0, (500.0, 500.0))
    dispatcher.join()
    assert dispatcher.speed_clicks == 1, "a press away from + is not a start"
    dispatcher.close()


def test_the_referee_starts_the_game_after_the_policy_s_plus_or_the_setup_limit():
    now = [0.0]
    referee = play.Referee(setup_seconds=90, stall_seconds=8, clock=lambda: now[0])
    now[0] = 30.0
    assert referee.due() is None
    referee.clicked_speed_up()
    now[0] = 30.5
    assert referee.due() is None, "its other clicks on + land first"
    now[0] = 31.5
    assert referee.due() == "start"
    referee.started()
    now[0] = 35.0
    referee.saw_day()
    now[0] = 42.0
    assert referee.due() is None
    now[0] = 43.5
    assert referee.due() == "restart"
    referee.restarted()
    assert referee.due() is None and referee.restarts == 1
    lazy = play.Referee(setup_seconds=90, clock=lambda: now[0])
    now[0] += 91
    assert lazy.due() == "start", "a policy that never clicks + still gets its game"


def test_the_reservation_waits_for_the_grant_and_hands_the_pc_back(tmp_path, monkeypatch):
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        (tmp_path / "granted" / "trial.json").write_text("{}")

    monkeypatch.setattr(play.time, "sleep", sleep)
    play.reserve("trial", 30, root=tmp_path)
    assert json.loads((tmp_path / "queue" / "trial.json").read_text())["minutes"] == 30
    assert sleeps == [10]
    play.hand_back("trial", {"games": 2}, root=tmp_path)
    assert json.loads((tmp_path / "done" / "trial.json").read_text())["games"] == 2


class _Screen(torch.nn.Module):
    dim = 8
    reads_clip = False

    def forward(self, clip, quadrants):
        batch = quadrants.shape[0]
        pooled = quadrants.float().mean((1, 2, 3, 4))[:, None].expand(batch, self.dim)
        return pooled, pooled[:, :, None, None].expand(batch, self.dim, 4, 4).contiguous()


def test_a_memory_window_as_long_as_the_game_so_far_is_the_carried_memory(monkeypatch):
    """Recomputing the memory over the last N decisions equals carrying it, while the game
    is no longer than N, and forgets what came before once it is."""
    from hoi4_arena import runner
    from hoi4_arena.dataset import Views
    from hoi4_arena.runner import Actor

    def small_views(rgb, device="cpu", cursor=None):
        frame = torch.as_tensor(rgb)
        return Views(frame[:16, :16], torch.stack([frame[:16, :16]] * 4), frame[:16, :16])

    monkeypatch.setattr(runner, "views", small_views)

    torch.manual_seed(0)
    policy = Policy(_Screen(), memory_dim=16).eval().requires_grad_(False)

    def build(window):
        actor = Actor.__new__(Actor)
        actor.policy, actor.config, actor.deterministic = policy, {"objective": "bc"}, True
        actor.device, actor.hidden, actor.compiled, actor.speed = "cpu", None, False, 5
        actor.previous = np.zeros((SLOTS, 3), dtype=np.int64)
        actor.history = deque(maxlen=64)
        actor.recent = deque(maxlen=window) if window else None
        return actor

    carried, windowed, short = build(None), build(4), build(2)
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (64, 96, 3), dtype=np.uint8) for _ in range(4)]
    for t, rgb in enumerate(frames):
        carried.act(rgb, (t + 10) * 200_000_000, cursor=(5, 5))
        windowed.act(rgb, (t + 10) * 200_000_000, cursor=(5, 5))
        short.act(rgb, (t + 10) * 200_000_000, cursor=(5, 5))
    assert torch.allclose(carried.hidden, windowed.hidden, atol=1e-5)
    assert not torch.allclose(carried.hidden, short.hidden, atol=1e-3)
    # A lean actor (play-policy) returns the same action and no training sample, and keeps
    # no clip for an encoder that reads none.
    lean = build(4)
    lean.lean = True
    torch.manual_seed(1)
    action, sample = lean.act(frames[0], 10 * 200_000_000, cursor=(5, 5))
    torch.manual_seed(1)
    expected, full = build(4).act(frames[0], 10 * 200_000_000, cursor=(5, 5))
    assert sample is None and full is not None and not lean.history
    assert np.array_equal(action, expected)


class _Game:
    """A worker and a game stand-in: paused until space, then a day every call, then a win."""

    def __init__(self, win_after=6):
        from hoi4_arena.desktop import Frame

        self.Frame = Frame
        self.applied, self.setup_applied = [], []
        self.setup = False
        self.running = False
        self.days = 0
        self.win_after = win_after
        self.cursor = [10, 10]
        self.t = 0

    def capture(self, full=None, **_):
        self.t += 200_000_000
        meta = {"t_ns": self.t, "cursor": list(self.cursor), "foreground": True}
        meta.update(overflow=False, backend="dxgi_bgra", pointer_drawn=True)
        return self.Frame(np.zeros((54, 96, 3), np.uint8), meta, self.t)

    def arm(self, setup=False):
        self.setup = setup

    def release(self):
        pass

    def focus(self):
        return True

    def apply(self, events):
        (self.setup_applied if self.setup else self.applied).extend(events)
        for event in events:
            if event["kind"] == "key" and event["vk"] == 0x20 and event["down"]:
                assert self.setup, "space only from the harness, in setup mode"
                self.running = True
            if event["kind"] == "move":
                self.cursor = [round(event["x"] * 95), round(event["y"] * 53)]
        return {"t_ns": self.t}

    def game_log(self, offset=0):
        lines = []
        if self.running:
            self.days += 1
            lines.append(_day("BLU", f"24:00, {self.days} January, 1936", 8, {}))
            lines.append(_day("RED", f"24:00, {self.days} January, 1936", 8, {}))
            if self.days == self.win_after:
                lines.append("capitulated RED winner BLU 12:00, 9 January, 1936")
        return lines, offset + len(lines)


class _Actor:
    """Clicks + on its third decision, then only moves the pointer about."""

    digest = "test"

    def __init__(self):
        self.steps = 0

    def reset_episode(self):
        self.steps = 0

    def let_go(self, event=None):
        pass

    def act(self, rgb, t_ns, precomputed=None, cursor=None):
        self.steps += 1
        action = np.zeros((SLOTS, 3), np.int64)
        if self.steps == 3:
            action[0] = _token(
                1, round(play.SPEED_UP[0] * (GRID - 1)), round(play.SPEED_UP[1] * (GRID - 1))
            )
            action[2] = _token(VOCAB.index(CLICK))
            action[3] = _token(VOCAB.index({"kind": "button", "button": 0, "down": False}))
        else:
            action[0] = _token(1, 100 * (self.steps % 5), 200)
        return action, {}


@needs_ffmpeg
def test_a_policy_game_starts_when_the_policy_clicks_plus_and_ends_on_the_surrender(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("hoi4_arena.ai_games.time.sleep", lambda s: None)
    game = _Game()
    actor = _Actor()
    actor.sampling = {"temperature": 0.5, "pointer_temperature": 0.0, "deterministic": False}
    outcome, reason, manifest = play.play_policy_game(
        game, actor, tmp_path / "game", rules=None, country="BLU", codec="ffv1",
        cap_minutes=1, setup_seconds=30, after_surrender=0.5, snap_every=0.5,
    )  # fmt: skip
    assert reason is None and outcome == "BLU"
    saved = json.loads((tmp_path / "game" / "manifest.json").read_text())
    assert (saved["temperature"], saved["pointer_temperature"]) == (0.5, 0.0), "what it played at"
    assert manifest["source"] == "policy" and manifest["harness"] == {"starts": 1, "restarts": 0}
    assert manifest["complete"] and manifest["frames"] > 5
    assert manifest["presses"].get("b0") == 1, "its one click, on +"
    assert list((tmp_path / "game" / "snaps").glob("*.jpg")), "pictures for whoever follows it"
    assert any(e["kind"] == "key" and e["vk"] == 0x20 for e in game.setup_applied)
    assert not any(e["kind"] == "key" for e in game.applied), "the policy sent no keys"
    stamped = [json.loads(line) for line in (tmp_path / "game" / "arena-log.jsonl").open()]
    assert stamped and all("frame" in entry for entry in stamped)
    labels = session_labels(tmp_path / "game", sources=("policy",), lead_in=0)
    assert (labels["actions"][..., 0] == 1).any(), "its moves are its game's labels"


@needs_ffmpeg
def test_behaviour_cloning_on_scripted_games_learns_the_true_state_beside_the_actions(
    tmp_path, monkeypatch
):
    from test_unroll import _Screen

    import hoi4_arena.train as train

    data = tmp_path / "data"
    data.mkdir()
    for name in ("game", "held-out"):
        _recording(
            data / name,
            [8, 6],
            source="scripted",
            events=[(300_000_000, CLICK), (900_000_000, SPACE)],
        )
        path = data / name / "manifest.json"
        meta = {**json.loads(path.read_text()), "players": ["BLU"], "winner": "RED"}
        meta["orders"] = [{"frame": 4, "order": "army"}, {"frame": 30, "order": "run"}]
        path.write_text(json.dumps(meta))
        _log(data / name, [(3, _day("BLU", "24:00, 1 January, 1936", 8, {7: 4})),
                           (3, _day("RED", "24:00, 1 January, 1936", 8, {15: 4}))])  # fmt: skip
    (data / "splits.json").write_text(json.dumps({"game": "train", "held-out": "validation"}))
    monkeypatch.setattr(train, "build_encoder", lambda path, variant: _Screen())
    train.train_bc(
        data, "model", tmp_path / "out", sources=("scripted",), sequence=2, burn_in=1,
        workers=0, lead_in=0, drop_keys=(0x20,), state_weight=0.5, loser_weight=0.5,
        look_before_click=True, order_weight=0.2,
    )  # fmt: skip
    rows = [json.loads(line) for line in (tmp_path / "out" / "metrics.jsonl").open()]
    steps = [row for row in rows if "step" in row]
    assert steps and all(row["state"] > 0 and row["orders"] > 0 for row in steps)
    report = rows[-1]
    assert report["validation_nll"] > 0 and report["validation_presses"] >= 1
    assert set(report["validation_state_r2"]) >= {"own", "enemy", "at", "year"}
    assert 0 <= report["validation_next_order_accuracy"] <= 1
    assert (tmp_path / "out" / "state-head-0000.pt").exists()
    config = json.loads((tmp_path / "out" / "epoch-0000.json").read_text())["config"]
    assert config["lead_in"] == 0 and config["drop_keys"] == [0x20]
    # Fine-tuning from it starts from its weights and its read-outs.
    train.train_bc(
        data, "model", tmp_path / "tuned", sources=("scripted",), sequence=2, burn_in=1,
        workers=0, lead_in=0, drop_keys=(0x20,), state_weight=0.5, order_weight=0.2,
        look_before_click=True, init=tmp_path / "out" / "epoch-0000.pt", lr=1e-5,
    )  # fmt: skip
    tuned = json.loads((tmp_path / "tuned" / "epoch-0000.json").read_text())["config"]
    assert tuned["init"].endswith("epoch-0000.pt") and tuned["lr"] == 1e-5


def test_an_aimed_move_is_one_a_press_follows_before_any_other_move():
    from hoi4_arena.heatmap import aimed

    actions = np.zeros((6, SLOTS, 3), np.int64)
    actions[0, 1] = _token(1, 5, 5)
    actions[2, 0] = _token(VOCAB.index(CLICK))  # Pressed two decisions later.
    actions[3, 0] = _token(1, 9, 9)
    actions[4, 0] = _token(1, 7, 7)  # Moved on before any press.
    actions[5, 0] = _token(VOCAB.index(CLICK))
    assert aimed(actions, 0)
    assert not aimed(actions, 1), "no move"
    assert not aimed(actions, 3)
    assert aimed(actions, 4)


def test_the_next_order_is_the_first_stamped_after_the_frame_read():
    from hoi4_arena.privileged import ORDER_KINDS, decision_orders

    manifest = {
        "nominal_fps": 5,
        "orders": [{"frame": 30, "order": "army"}, {"frame": 10, "order": "run"},
                   {"frame": 40, "order": "retreat"}],
    }  # fmt: skip
    kinds, eta = decision_orders(manifest, np.array([0, 10, 29, 39, 45]))
    names = [ORDER_KINDS[k] for k in kinds]
    assert names == ["run", "army", "army", "other", "none"]
    assert eta[0] == pytest.approx(np.log1p(2.0)) and eta[2] == pytest.approx(np.log1p(0.2))
    assert np.isnan(eta[-1])
    kinds, eta = decision_orders({}, np.array([0, 1]))
    assert (kinds == -1).all() and np.isnan(eta).all(), "no orders, no targets"


def test_greedy_pointing_takes_the_likeliest_place_and_still_samples_the_kind():
    from hoi4_arena.models import CELL_DIM, ActionHead

    torch.manual_seed(0)
    head = ActionHead(memory_dim=8).eval()
    with torch.no_grad():
        head.kinds.bias.zero_()
        head.kinds.bias[1] = 50.0  # Always a move.
    memory, cells = torch.randn(1, 8), torch.randn(1, GRID, CELL_DIM)
    greedy = {tuple(head(memory, cells, point=True)[0][0, 0].tolist()) for _ in range(8)}
    sampled = {tuple(head(memory, cells)[0][0, 0].tolist()) for _ in range(8)}
    assert len(greedy) == 1 and len(sampled) > 1
    with torch.no_grad():
        head.kinds.bias[1] = 0.0
        head.kinds.bias[0] = 0.5
    kinds = {int(head(memory, cells, point=True)[0][0, 0, 0]) for _ in range(40)}
    assert len(kinds) > 1, "what to do is still sampled"


def test_training_holds_while_its_pause_file_is_there(tmp_path):
    from hoi4_arena.train import wait_while_paused

    assert not wait_while_paused(tmp_path)
    flag = tmp_path / "pause"
    flag.write_text("")
    polls = []

    def sleep(seconds):
        polls.append(seconds)
        if len(polls) == 3:
            flag.unlink()

    assert wait_while_paused(tmp_path, poll=2.0, sleep=sleep) and polls == [2.0] * 3


def test_a_frozen_tower_trains_nothing_of_the_encoder():
    from hoi4_arena.models import ScreenEncoder
    from hoi4_arena.train import train_blocks

    encoder = ScreenEncoder(pretrained=False)
    assert any(p.requires_grad for p in encoder.parameters())
    train_blocks(encoder, 0)
    assert not any(p.requires_grad for p in encoder.parameters())
    train_blocks(encoder, 1)
    trainable = [n for n, p in encoder.named_parameters() if p.requires_grad]
    last = len(encoder.model.blocks) - 1
    assert trainable and all(n.startswith(f"model.blocks.{last}.") for n in trainable)


@needs_ffmpeg
def test_the_tower_cache_reads_what_the_frozen_tower_reads(tmp_path, monkeypatch):
    """Cached and run, the tower gives the policy the same summary and cells, and a run
    trained from the cache refuses one made from another tower."""
    import hoi4_arena.models as models
    import hoi4_arena.train as train
    from hoi4_arena.dataset import batch_to_device
    from hoi4_arena.models import ScreenEncoder
    from hoi4_arena.tower_cache import cache_tower

    def small(path=None, variant="screen", **_):
        return ScreenEncoder(pretrained=False, size=(32, 64))

    monkeypatch.setattr(models, "build_encoder", small)
    monkeypatch.setattr(train, "build_encoder", small)
    data = tmp_path / "data"
    data.mkdir()
    for name in ("game", "held-out"):
        _scripted(data / name, players=["BLU"], winner="BLU")
    (data / "splits.json").write_text(json.dumps({"game": "train", "held-out": "validation"}))
    torch.manual_seed(0)
    policy = Policy(small(), memory_dim=512)
    checkpoint = tmp_path / "start.pt"
    torch.save(
        {"policy": policy.state_dict(), "config": {"model_path": "x", "variant": "screen"}},
        checkpoint,
    )
    report = cache_tower(data, checkpoint, tmp_path / "cache", device="cpu")
    assert report["recordings"] == 2 and report["frames"] == 80
    again = cache_tower(data, checkpoint, tmp_path / "cache", device="cpu")
    assert again["skipped"] == 2 and again["recordings"] == 0, "a finished recording stays"
    # A drive that would keep too little free sends the recordings to the spill folder.
    from hoi4_arena.tower_cache import tower_paths

    full = cache_tower(
        data, checkpoint, tmp_path / "full", device="cpu", spill=tmp_path / "spill",
        keep_free_gb=1e9,
    )  # fmt: skip
    assert full["spilled"] == 2
    assert tower_paths(tmp_path / "full", "game")["grid"].parent.parent == tmp_path / "spill"

    common = {"sources": ("scripted",), "length": 3, "burn_in": 1, "device": "cpu",
              "clips": False, "lead_in": 0, "shuffle": 0}  # fmt: skip
    plain = next(iter(VideoSessions(data, **common)))
    cached = next(iter(VideoSessions(data, tower=tmp_path / "cache", **common)))
    assert cached["tower_grid"].shape == (4, 768, 32, 32)
    policy.eval().requires_grad_(False)
    batch = batch_to_device(torch.utils.data.default_collate([plain]), "cpu")
    stored = batch_to_device(torch.utils.data.default_collate([cached]), "cpu")
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        run = policy.perceive_window(None, batch["quadrants"], batch["fovea"])
        read = policy.perceive_window(
            None, stored["quadrants"], stored["fovea"],
            tower=(stored["tower_summary"], stored["tower_grid"]),
        )  # fmt: skip
    for a, b in zip(run, read, strict=True):
        assert torch.allclose(a.float(), b.float(), atol=0.1, rtol=0.05)
    # Kept as int8 (half the space), it reads back within half a step of the scale, plus
    # bfloat16's rounding of both (a 2**-8 part of the value each).
    cache_tower(data, checkpoint, tmp_path / "small", device="cpu", int8=True)
    assert tower_paths(tmp_path / "small", "game")["scale"].exists()
    small_grid = next(iter(VideoSessions(data, tower=tmp_path / "small", **common)))["tower_grid"]
    step = cached["tower_grid"].float().abs().amax((-2, -1), keepdim=True) / 127
    bound = step * 0.5 + cached["tower_grid"].float().abs() * 2**-7
    assert ((small_grid.float() - cached["tower_grid"].float()).abs() <= bound).all()

    common = {"sources": ("scripted",), "sequence": 2, "burn_in": 1, "workers": 0, "lead_in": 0}
    train.train_bc(data, "model", tmp_path / "out", train_last=0, init=checkpoint,
                   tower_cache=tmp_path / "cache", **common)  # fmt: skip
    rows = [json.loads(line) for line in (tmp_path / "out" / "metrics.jsonl").open()]
    assert rows[-1]["validation_nll"] > 0
    with pytest.raises(ValueError, match="another tower"):
        train.train_bc(data, "model", tmp_path / "other", train_last=0,
                       tower_cache=tmp_path / "cache", **common)  # fmt: skip
    with pytest.raises(ValueError, match="frozen tower"):
        train.train_bc(data, "model", tmp_path / "unfrozen", init=checkpoint,
                       tower_cache=tmp_path / "cache", **common)  # fmt: skip


@needs_ffmpeg
def test_games_play_in_order_with_their_memory_started_once_a_game(tmp_path):
    from hoi4_arena.dataset import GameSequences

    for name in ("a", "b"):
        _scripted(tmp_path / name)
    sessions = VideoSessions(
        tmp_path, sources=("scripted",), length=4, burn_in=0, device="cpu", clips=False, lead_in=0
    )
    games = GameSequences(sessions, 4, 1, device="cpu", clips=False)
    batches = list(games)
    # 40 frames at 10 Hz are 20 decisions of 0.2 s, less the last: 4 whole windows a game.
    assert len(batches) == 8 == len(games)
    starts = [int(b["start"][0]) for b in batches]
    fresh = [bool(b["fresh"][0]) for b in batches]
    assert starts == [0, 4, 8, 12] * 2
    assert fresh == [True, False, False, False] * 2
    two = list(GameSequences(sessions, 4, 2, device="cpu", clips=False))
    assert len(two) == 4 and all(b["slot"].tolist() == [0, 1] for b in two)
    assert [b["fresh"].tolist() for b in two] == [[True, True]] + [[False, False]] * 3


@needs_ffmpeg
def test_a_carried_memory_trains_and_validates_game_by_game(tmp_path, monkeypatch):
    from test_unroll import _Screen

    import hoi4_arena.train as train

    data = tmp_path / "data"
    data.mkdir()
    for name in ("game", "other", "held-out"):
        _recording(
            data / name,
            [8, 6],
            source="scripted",
            events=[(300_000_000, CLICK), (900_000_000, SPACE)],
        )
    (data / "splits.json").write_text(
        json.dumps({"game": "train", "other": "train", "held-out": "validation"})
    )
    monkeypatch.setattr(train, "build_encoder", lambda path, variant: _Screen())
    train.train_bc(
        data, "model", tmp_path / "out", sources=("scripted",), sequence=4, workers=0,
        lead_in=0, look_before_click=True, carry=True,
    )  # fmt: skip
    rows = [json.loads(line) for line in (tmp_path / "out" / "metrics.jsonl").open()]
    steps = [row for row in rows if "step" in row]
    # Two games side by side, 4 windows each: 4 batches. Space is outside the vocabulary,
    # so its decision is invalid: seen by the memory, scored with weight 0.
    assert len(steps) == 4 and all(np.isfinite(row["bc"]) for row in steps)
    assert rows[-1]["validation_nll"] > 0 and rows[-1]["validation_decisions"] == 15
    config = json.loads((tmp_path / "out" / "epoch-0000.json").read_text())["config"]
    assert config["carry"] and config["burn_in"] == 0


class _TimedDesk(_Desk):
    """A worker of protocol 2: a decision's slots in one request, each at its offset."""

    def __init__(self):
        super().__init__()
        self.batches = []

    def protocol(self):
        return 2

    def apply(self, events, at_ms=None):
        self.batches.append((list(events), at_ms))
        self.applied.extend(events)
        if at_ms is None:
            return {"t_ns": 1}
        return {"t_ns": 9, "times_ns": [1000 + int(t) for t in at_ms]}


def test_a_timed_dispatch_sends_the_slots_in_one_request_at_their_offsets():
    desk = _TimedDesk()
    dispatcher = play.Dispatcher(desk, clock=lambda: 0.0, timed=True)
    plus = [1, round(play.SPEED_UP[0] * (GRID - 1)), round(play.SPEED_UP[1] * (GRID - 1))]
    action = np.zeros((SLOTS, 3), np.int64)
    action[1] = plus
    action[3] = _token(VOCAB.index(CLICK))
    dispatcher.start(action, 0.0, (100.0, 100.0))
    dispatcher.join()
    ((events, offsets),) = desk.batches
    assert [e["kind"] for e in events] == ["move", "button"] and events[1] == CLICK
    assert events[0]["x"] == pytest.approx(play.SPEED_UP[0], abs=1e-3)
    assert offsets == pytest.approx([25.0, 75.0])
    assert [e["t_ns"] for e in dispatcher.take()] == [1025, 1075], "each event at its own time"
    assert dispatcher.speed_clicks == 1
    dispatcher.start(np.zeros((SLOTS, 3), np.int64), 0.0, (5.0, 5.0))
    dispatcher.join()
    assert desk.batches[-1] == ([], None), "an idle decision still feeds the watchdog"
    dispatcher.close()


RIGHT_DOWN = {"kind": "key", "vk": 0x27, "down": True}
RIGHT_UP = {"kind": "key", "vk": 0x27, "down": False}


def test_a_key_held_past_its_limit_is_released_and_counted():
    # bc5 held Right for three minutes (2026-09-26); the scripted player never past 0.8 s.
    holds = play.Holds()
    holds.follow([RIGHT_DOWN, {"kind": "button", "button": 1, "down": True}])
    released = []
    for _ in range(25):
        holds.advance()
        released.append(holds.due())
    # The key goes up at the start of the fifth interval after its press (1 s), the right
    # button (a front being drawn) at the twentieth (4 s).
    assert released[4] == [RIGHT_UP] and not any(released[:4])
    assert released[19] == [{"kind": "button", "button": 1, "down": False}]
    assert holds.forced == {"key39": 1, "button1": 1} and not holds.since


def test_a_key_the_policy_releases_itself_is_left_alone():
    holds = play.Holds()
    holds.follow([RIGHT_DOWN])
    holds.advance()
    holds.follow([RIGHT_UP, RIGHT_DOWN])  # released and pressed again: held from here
    for _ in range(4):
        holds.advance()
        assert holds.due() == []
    holds.advance()
    assert holds.due() == [RIGHT_UP]
    holds.follow([RIGHT_DOWN])
    holds.clear()  # the harness let go of everything
    for _ in range(9):
        holds.advance()
        assert holds.due() == []


def test_a_timed_dispatch_releases_a_stuck_key_first_and_marks_it_the_harness_s():
    desk = _TimedDesk()
    dispatcher = play.Dispatcher(desk, clock=lambda: 0.0, timed=True)
    press = np.zeros((SLOTS, 3), np.int64)
    press[2] = _token(VOCAB.index(RIGHT_DOWN))
    click = np.zeros((SLOTS, 3), np.int64)
    click[0] = _token(VOCAB.index(CLICK))
    for action in [press, *[np.zeros((SLOTS, 3), np.int64)] * 4, click]:
        dispatcher.start(action, 0.0, (5.0, 5.0))
        dispatcher.join()
    events, offsets = desk.batches[-1]
    assert events == [RIGHT_UP, CLICK] and offsets == pytest.approx([0.0, 0.0])
    marks = [e.get("by") for e in dispatcher.take() if e["event"] in (RIGHT_UP, CLICK)]
    assert marks == ["harness", None]
    assert dispatcher.holds.forced == {"key39": 1}
    dispatcher.close()


def _actions(*tokens_per_decision):
    out = np.zeros((len(tokens_per_decision), SLOTS, 3), np.int64)
    for t, tokens in enumerate(tokens_per_decision):
        for slot, kind in tokens:
            out[t, slot] = (VOCAB.index(kind), 0, 0)
    return out


def test_the_previous_action_shows_a_key_still_held_until_it_is_released():
    down, up = VOCAB.index(RIGHT_DOWN), VOCAB.index(RIGHT_UP)
    # Right pressed at decision 0 and released at 3; the left button tapped at 1.
    actions = _actions(
        [(2, RIGHT_DOWN)], [(0, CLICK), (1, {**CLICK, "down": False})], [], [(0, RIGHT_UP)], []
    )
    plain = previous_actions(actions)
    assert (plain[0] == 0).all() and (plain[1:] == actions[:-1]).all(), "as it always was"
    held = previous_actions(actions, held=True)
    assert (held[1] == actions[0]).all(), "the press itself shows it"
    # Decisions 2 and 3: nothing of Right in the action before, so its press fills the
    # last empty slot; the tap's own slots stay as they were.
    assert held[2][SLOTS - 1, 0] == down and (held[2][: SLOTS - 1] == actions[1][: SLOTS - 1]).all()
    assert held[3][SLOTS - 1, 0] == down and (held[3][: SLOTS - 1] == 0).all()
    assert (held[4] == actions[3]).all() and held[4][0, 0] == up, "released: shown no more"


def test_what_is_held_follows_presses_and_releases_in_order():
    press = VOCAB.index(RIGHT_DOWN)
    (tap,) = _actions([(0, RIGHT_DOWN), (1, RIGHT_UP)])
    assert still_held((), tap) == ()
    (again,) = _actions([(0, RIGHT_UP), (1, RIGHT_DOWN)])
    assert still_held((press,), again) == (press,)
    (full,) = _actions([(s, {"kind": "key", "vk": 0x41 + s, "down": True}) for s in range(SLOTS)])
    assert (with_held(full, (press,)) == full).all(), "no empty slot: nothing added"


def test_the_live_actor_shows_what_it_holds_and_forgets_what_the_harness_released():
    from hoi4_arena.runner import Actor

    actor = Actor.__new__(Actor)
    actor.held, actor.held_previous = (), True
    press, (pressed, idle) = VOCAB.index(RIGHT_DOWN), _actions([(2, RIGHT_DOWN)], [])
    actor._remember(pressed)
    actor._remember(idle)
    assert actor.previous[SLOTS - 1, 0] == press and actor.held == (press,)
    actor.let_go(RIGHT_UP)  # play.Holds let it go
    actor._remember(idle)
    assert (actor.previous == 0).all() and actor.held == ()
    actor._remember(pressed)
    actor.let_go()  # the harness released every input
    assert actor.held == ()
    actor.held_previous = False
    actor._remember(pressed)
    actor._remember(idle)
    assert (actor.previous == idle).all(), "a checkpoint trained without it sees none"


def test_a_low_temperature_sharpens_what_to_do_and_leaves_scoring_alone():
    from hoi4_arena.models import CELL_DIM, ActionHead

    torch.manual_seed(0)
    head = ActionHead(memory_dim=8).eval()
    with torch.no_grad():
        head.kinds.bias.zero_()
        head.kinds.bias[0] = 1.0  # "none" likeliest, a move next.
        head.kinds.bias[1] = 0.5
    memory, cells = torch.randn(64, 8), torch.randn(64, GRID, CELL_DIM)
    with torch.no_grad():
        warm = (head(memory, cells)[0][:, 0, 0] == 0).float().mean()
        cold = (head(memory, cells, temperature=0.2)[0][:, 0, 0] == 0).float().mean()
        actions = torch.zeros(64, SLOTS, 3, dtype=torch.long)
        same = head(memory, cells, actions)[1], head(memory, cells, actions, temperature=0.2)[1]
    assert cold > warm, "the likeliest input gains"
    assert torch.equal(*same), "a demonstration's likelihood does not depend on it"


def test_a_reservation_made_ahead_is_not_made_twice(tmp_path, monkeypatch):
    (tmp_path / "granted").mkdir(parents=True)
    (tmp_path / "granted" / "early.json").write_text("{}")
    monkeypatch.setattr(play.time, "sleep", lambda s: None)
    play.reserve("early", 30, root=tmp_path)
    assert not (tmp_path / "queue" / "early.json").exists(), "granted already: no new request"


def test_what_the_memory_takes_in_does_not_grow_with_the_tower_s_loudness():
    """The Qwen3.5 tower's summary is about 50 in size. Unnormalized, training grew the
    fusion to about 25 and saturated every gate of the memory, which then never changed
    over a game: the policy acted the same everywhere. Normalized, a louder tower (or a
    fovea reader) leaves what the memory takes in as it was."""
    from hoi4_arena.models import fuse

    torch.manual_seed(0)
    policy = Policy(_Screen(), memory_dim=32)
    summary, centre = torch.randn(3, 8), torch.randn(3, 256)
    cells = torch.randn(3, GRID, 256)
    previous, speed = torch.zeros(3, SLOTS, 3, dtype=torch.long), torch.full((3,), 5)
    hidden = torch.zeros(3, 32)
    quiet = fuse(policy, summary, cells, centre, previous, speed, hidden)
    loud = fuse(policy, summary * 50 + 3, cells, centre * 20, previous, speed, hidden)
    assert torch.allclose(quiet, loud, atol=1e-4)


def test_a_game_s_milestones_count_the_scripted_player_s_steps():
    def at(t, **event):
        return {"t_ns": int(t * 1e9), "event": event}

    def move(t, x, y):
        return at(t, kind="move", x=x / 1919, y=y / 1079)

    def press(t, button=0):
        return at(t, kind="button", button=button, down=True)

    def key(t, vk):
        return at(t, kind="key", vk=vk, down=True)

    events = [
        move(1.0, 826, 58), key(1.2, 0x10), press(1.3),  # The alert, shift+clicked.
        move(2.0, 988, 1012), press(2.4),  # The create-army +.
        key(5.0, 0x5A), move(5.2, 900, 500), press(5.6),  # A front line.
        key(9.0, 0x58), move(9.2, 950, 520), press(9.6, button=1),  # An offensive.
        key(20.0, 0x51), move(20.2, 400, 400), press(20.5),  # Q, then a click elsewhere.
    ]  # fmt: skip
    counts = play.milestones(events)
    assert counts["alert"] == 1 and counts["plus"] == 1
    assert counts["front"] == 1 and counts["offensive"] == 1 and counts["q"] == 1
    assert counts["portrait"] == counts["law_slot"] == counts["confirm"] == 0


def test_decisions_that_act_weigh_more_than_waiting_and_the_camera():
    from hoi4_arena.dataset import acting

    actions = np.zeros((6, SLOTS, 3), np.int64)
    actions[0, 0] = _token(1, 5, 5)  # A move onto what decision 2 presses.
    actions[2, 1] = _token(VOCAB.index(CLICK))
    actions[4, 0] = _token(1, 9, 9)  # The camera looking about: no press follows.
    actions[5, 0] = _token(VOCAB.index({"kind": "wheel", "delta": 120}))
    assert acting(actions).tolist() == [True, False, True, False, False, False]


def test_a_save_rides_out_a_moment_s_lock_on_its_file(tmp_path, monkeypatch):
    from pathlib import Path

    from hoi4_arena import learning

    calls = []
    real = Path.replace

    def flaky(self, target):
        calls.append(target)
        if len(calls) < 3:
            raise PermissionError(32, "in use")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    monkeypatch.setattr(learning.time, "sleep", lambda s: None)
    (tmp_path / "a.tmp").write_text("new")
    learning.replace_patiently(tmp_path / "a.tmp", tmp_path / "a.pt")
    assert (tmp_path / "a.pt").read_text() == "new" and len(calls) == 3


def test_a_second_copy_of_a_run_refuses_its_folder_and_a_dead_owner_s_lock_is_taken(tmp_path):
    import os

    from hoi4_arena.learning import RunLock

    with RunLock(tmp_path):
        assert (tmp_path / "run.lock").read_text() == str(os.getpid())
        with pytest.raises(RuntimeError, match="one run per folder"):
            with RunLock(tmp_path):
                pass
    assert not (tmp_path / "run.lock").exists()
    (tmp_path / "run.lock").write_text("999999")  # A run killed without cleaning up.
    with RunLock(tmp_path):
        assert (tmp_path / "run.lock").read_text() == str(os.getpid())


def test_a_live_game_starts_only_with_commit_room_on_the_second_pc(monkeypatch):
    readings = iter([{"commit_mb": 36000, "commit_limit_mb": 38000},
                     {"commit_mb": 29000, "commit_limit_mb": 38000}])  # fmt: skip

    class Station:
        def pagefile(self):
            return None  # A limit that cannot grow.

        def connect(self, attach=True):
            class Desk:
                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    pass

                def telemetry(self, timeout=15):
                    return {"memory": next(readings)}

            return Desk()

    monkeypatch.setattr(play.time, "sleep", lambda s: None)
    assert play.room_for_a_game(Station(), tries=2), "room once the lender's game has closed"
    readings = iter([{"commit_mb": 36000, "commit_limit_mb": 38000}] * 2)
    assert not play.room_for_a_game(Station(), tries=2)


def test_the_pointer_s_parking_moves_are_found_and_its_clicks_kept():
    """The scripted player parks the pointer to read the screen; only those moves go."""
    from hoi4_arena.dataset import parking_moves

    def event(t, **fields):
        return {"t_ns": t, "event": fields}

    events = [
        event(1, kind="move", x=0.5, y=0.5),  # Parked: the next move comes first.
        event(2, kind="move", x=0.2, y=0.3),
        event(3, kind="move", x=0.5, y=0.5),  # Kept: something at the centre is clicked.
        event(4, kind="button", button=0, down=True),
        event(5, kind="button", button=0, down=False),
        event(6, kind="move", x=0.5, y=0.5),  # Parked: a full zoom out ignores the pointer.
        event(7, kind="wheel", delta=-120),
        event(8, kind="move", x=0.5, y=0.5),  # Kept: a zoom in closes in on the pointer.
        event(9, kind="wheel", delta=120),
        event(10, kind="button", button=1, down=True),
        event(11, kind="move", x=0.5, y=0.5),  # Kept: a drag, with a button held.
        event(12, kind="button", button=1, down=False),
        event(13, kind="move", x=0.65, y=0.012),  # Parked: the top bar's blank middle.
    ]
    assert parking_moves(events) == [0, 5, 12]


def test_the_setup_ends_at_the_run_order_and_can_weigh_more(tmp_path):
    from hoi4_arena.dataset import setup_end

    times = np.arange(10) * 200_000_000
    orders = [{"order": "army", "frame": 1}, {"order": "run", "frame": 4}]
    assert setup_end({"orders": orders}, times) == times[4]
    assert setup_end({}, times, seconds=1.0) == times[0] + 1_000_000_000
    _scripted(tmp_path / "game", orders=[{"order": "run", "frame": 20}])
    labels = session_labels(tmp_path / "game", sources=("scripted",), lead_in=0, setup_weight=3.0)
    early = labels["decisions"] < labels["times"][20]
    assert early.any() and (~early).any()
    assert labels["weight"][early] == pytest.approx(np.full(early.sum(), 3.0))
    assert labels["weight"][~early] == pytest.approx(np.full((~early).sum(), 1.0))


def test_the_setup_s_targets_are_the_moves_before_its_first_clicks():
    from hoi4_arena.heatmap import setup_targets

    def onto(x, y):
        return _token(1, round(x / 1919 * (GRID - 1)), round(y / 1079 * (GRID - 1)))

    actions = np.zeros((12, SLOTS, 3), np.int64)
    actions[1, 0], actions[1, 1] = onto(825, 57), _token(VOCAB.index(CLICK))  # The alert.
    actions[3, 0], actions[3, 1] = onto(988, 1013), _token(VOCAB.index(CLICK))  # The +.
    actions[5, 0] = _token(VOCAB.index({"kind": "key", "vk": 0x5A, "down": True}))  # Z.
    actions[6, 0], actions[6, 1] = onto(700, 500), _token(VOCAB.index(CLICK))  # The front.
    found = setup_targets(actions, 1920, 1080)
    assert set(found) == {"alert", "plus", "front"}, "the portrait was never clicked"
    assert found["alert"][:2] == (1, 0) and found["front"][:2] == (6, 0)
    assert abs(found["plus"][2][0] - 988) < 2 and abs(found["plus"][2][1] - 1013) < 2

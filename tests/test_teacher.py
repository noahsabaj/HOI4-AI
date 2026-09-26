import json

import numpy as np

from hoi4_arena.privileged import DIM as STATE_DIM
from hoi4_arena.teacher import (
    DIM,
    HISTORY,
    INTENTS,
    Teacher,
    decisions,
    held_out,
    intents_of,
    train,
)

AT = " ".join(f"{s}=0" for s in range(1, 17))


def _day(tag, day, divisions=8):
    return (
        f"day  24:00, {day} January, 1936 {tag} states 8 owned 8 divisions {divisions}"
        f" surrender 0 strength 1 casualties 0 manpower 1 deployed 15 rifles 4.8 needed 4.8"
        f" at {AT}"
    )


ORDERS = [
    {"frame": 10, "order": "army"},
    {"frame": 20, "order": "general"},
    {"frame": 30, "order": "front"},
    {"frame": 40, "order": "offensive", "attack": "near", "target_state": 11},
    {"frame": 45, "order": "run", "speed": 5},
    {"frame": 150, "order": "law", "law": "limited"},
    {"frame": 200, "order": "activate"},
    {"frame": 300, "order": "pause"},
    {"frame": 310, "order": "clear"},
    {"frame": 320, "order": "guard", "held": 0.2},
    {"frame": 330, "order": "front"},
    {"frame": 400, "order": "activate"},
]


def _recording(root, winner="BLU", orders=ORDERS, frames=420):
    root.mkdir(parents=True)
    manifest = {
        "players": ["BLU"], "started_as": "BLU", "winner": winner, "arena": "arena-marsh-v6",
        "orders": orders, "frames": frames, "nominal_fps": 5,
    }  # fmt: skip
    (root / "manifest.json").write_text(json.dumps(manifest))
    lines = [{"frame": 2, "line": "start  12:00, 1 January, 1936"}]
    for day, frame in enumerate(range(50, frames, 10), 1):
        lines.append({"frame": frame, "line": _day("RED", min(day, 28))})
        lines.append({"frame": frame, "line": _day("BLU", min(day, 28), 8 + day // 10)})
    (root / "arena-log.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    return root


def test_orders_become_intents_with_a_redraw_as_one():
    intents = [(frame, intent, target) for frame, intent, target, _ in intents_of(ORDERS)]
    assert [i for _, i, _ in intents] == [
        "form_army", "assign_general", "draw_front", "draw_offensive", "run", "set_law",
        "execute", "pause", "redraw", "execute",
    ]  # fmt: skip
    assert intents[3] == (40, "draw_offensive", 10)  # State 11, as an index.
    assert all(target == -1 for _, i, target in intents if i != "draw_offensive")


def test_a_recording_gives_one_row_a_second_with_the_history_before_each_intent(tmp_path):
    x, y, target, win = decisions(_recording(tmp_path / "game"))
    assert x.shape[1] == DIM and len(x) == len(y) == len(target) == len(win)
    assert not np.isnan(x).any() and (win == 1).all()
    acts = [INTENTS[i] for i in y if i]
    assert acts == [intent for _, intent, _, _ in intents_of(ORDERS)]
    history = x[:, STATE_DIM : STATE_DIM + len(HISTORY)]
    column = {name: history[:, i] for i, name in enumerate(HISTORY)}
    first = {INTENTS[y[i]]: i for i in range(len(y)) if y[i]}
    # Nothing is formed when the army is chosen; the front is drawn when the offensive is.
    assert column["army"][first["form_army"]] == 0
    assert column["front"][first["draw_offensive"]] == 1
    assert column["running"][first["run"]] == 0 and column["running"][first["set_law"]] == 1
    # The redraw cleared the front and the plan; the guard note is in the history after it.
    last = len(y) - 1 - list(y[::-1]).index(INTENTS.index("execute"))
    assert column["front"][last] == 1 and column["executing"][last] == 0
    assert column["guarded"][last] == 1
    # Long idle gaps are filled with waits a second apart.
    assert (y == 0).sum() > 30


def test_a_recording_without_orders_or_a_report_gives_nothing(tmp_path):
    assert decisions(_recording(tmp_path / "none", orders=[])) is None
    root = _recording(tmp_path / "unlogged")
    (root / "arena-log.jsonl").write_text("")
    assert decisions(root) is None


def test_held_out_is_a_fixed_share_by_name():
    names = [f"scripted-peer-20260924-{i:06d}" for i in range(2000)]
    share = np.mean([held_out(n) for n in names])
    assert 0.12 < share < 0.18
    assert held_out(names[7]) == held_out(names[7])


def test_the_teacher_trains_and_reports_on_held_out_games(tmp_path):
    roots = []
    i = 0
    while len({held_out(r) for r in roots}) < 2 or len(roots) < 4:
        roots.append(_recording(tmp_path / f"game-{i}", winner="BLU" if i % 2 else "RED"))
        i += 1
    report = train(roots, tmp_path / "out", epochs=2, width=32)
    assert report["fit_games"] + report["test_games"] == len(roots)
    assert 0 <= report["final"]["accuracy"] <= 1
    assert (tmp_path / "out" / "teacher.pt").exists()
    blind = Teacher(32, blind=True)
    x = np.random.default_rng(0).normal(size=(3, DIM)).astype(np.float32)
    import torch

    a = torch.from_numpy(x)
    b = a.clone()
    b[:, :STATE_DIM] = 0
    for one, two in zip(blind(a), blind(b)):
        assert torch.equal(one, two)


def test_the_teacher_chooses_an_intent_the_hand_takes():
    import torch

    from hoi4_arena import intents
    from hoi4_arena.teacher import ORDER_INTENT, choose

    assert set(INTENTS) == set(intents.INTENTS) - {"camera", "popup"}
    assert set(ORDER_INTENT.values()) <= set(INTENTS)
    torch.manual_seed(0)
    model = Teacher(16)
    x = np.zeros(DIM, np.float32)
    intent = choose(model, x)
    assert isinstance(intent, intents.Intent) and intent.name in INTENTS
    with torch.no_grad():
        model.intent.bias.fill_(-9.0)
        model.intent.bias[INTENTS.index("draw_offensive")] = 9.0
        model.intent.weight.zero_()
        model.target.weight.zero_()
        model.target.bias.copy_(torch.arange(16, dtype=torch.float32))
    assert choose(model, x).to_json() == {"intent": "draw_offensive"}
    assert choose(model, x, aim=True).to_json() == {"intent": "draw_offensive", "target_state": 16}
    drawn = {choose(model, x, np.random.default_rng(1), temperature=1.0).name for _ in range(5)}
    assert drawn == {"draw_offensive"}

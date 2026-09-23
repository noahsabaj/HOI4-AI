import json

import numpy as np

from hoi4_arena.state_value import (
    FEATURES,
    day_number,
    frame_values,
    game_examples,
    load_state_value,
    snapshots,
    state_id,
    train_state_value,
)


def _day(tag, day, divisions, surrender):
    at = " ".join(f"{s}={divisions if s in (7, 8, 15, 16) else 0}" for s in range(1, 17))
    return (
        f"day 24:00, {day} January, 1936 {tag} states 8 owned 8 divisions {divisions} surrender"
        f" {surrender} strength 1 casualties 0.1 manpower 0 deployed 14 rifles 4.7 needed 4.8"
        f" at {at}"
    )


def _game(root, winner, split="train", days=20):
    """A fake recorded game: the loser loses divisions and nears surrender day by day."""
    root.mkdir(parents=True)
    loser = "RED" if winner == "BLU" else "BLU"
    stamped = []
    for day in range(1, days + 1):
        for tag in ("BLU", "RED"):
            lost = day // 4 if tag == loser else 0
            surrender = round(day / days, 3) if tag == loser else 0
            stamped.append({"frame": 10 * day, "line": _day(tag, day, 4 - lost // 2, surrender)})
    stamped.insert(
        5, {"frame": 25, "line": f"control {winner} from {loser} West 3  1:00, 3 January, 1936"}
    )
    (root / "arena-log.jsonl").write_text("".join(json.dumps(s) + "\n" for s in stamped))
    manifest = {"winner": winner, "frames": 10 * days + 10, "split": split}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_names_and_dates_read_as_the_mod_writes_them():
    assert state_id("West 3") == 3 and state_id("East 3") == 11 and state_id("Kargopol") is None
    assert day_number("24:00, 4 January, 1936") == day_number("0:00, 5 January, 1936")
    assert day_number("12:00, 1 February, 1936") == 31.5


def test_snapshots_start_once_both_sides_report_and_follow_control(tmp_path):
    root = _game(tmp_path / "g", "BLU")
    rows = snapshots(
        [json.loads(line) for line in (root / "arena-log.jsonl").read_text().splitlines()]
    )
    assert len(rows) == 2 * 20 - 1
    assert all(x.shape == (FEATURES,) for _, x in rows)
    control = rows[-1][1][2 * 26 : 2 * 26 + 16]
    assert control[2] == 1.0  # West 3 is Blue's own state, taken back by Blue: still +1.
    x, target, frames = game_examples(root)
    assert len(x) == len(target) == len(frames) and (target > 0).all()


def test_the_predictor_learns_who_wins_from_the_state(tmp_path):
    roots = [_game(tmp_path / f"g{i}", "BLU" if i % 2 else "RED") for i in range(8)]
    roots += [_game(tmp_path / f"v{i}", "BLU" if i % 2 else "RED", "validation") for i in range(4)]
    report = train_state_value(roots, tmp_path / "state-value.pt", epochs=200)
    assert report["train_games"] == 8 and report["validation_games"] == 4
    # The last third of each game, when the loser is close to surrendering, is clear.
    assert report["sign_accuracy_by_third"][2] == 1.0
    model = load_state_value(tmp_path / "state-value.pt")
    values = frame_values(model, roots[1], 210)
    assert values.shape == (210,) and values[0] == 0 and values[-1] > 0
    assert np.all(frame_values(model, roots[0], 210)[-20:] < 0)

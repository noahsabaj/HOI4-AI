"""A training set: a folder of junctions to recordings, and its splits.json.

    python scripts/build_dataset.py C:/hoi4-data/scripted-v7 --drills --practice

Takes every usable scripted game on the main arena (v3/v4, since the first win,
20260923-222128) and on the seven v6 presets. Usable means complete, not salvaged, with
screen.mkv and arena-log.jsonl, and at least 100 frames: the 5-second games before #90 read
the last game's surrender and ended at once.
- `--drills` adds the complete setup drills (practice.drills).
- `--practice` adds the practice games (practice.practice), where only the coach's spans
  teach (dataset.session_labels); train those with `--sources scripted policy`.

Splits: the same 4 main-arena games as every set since v4 are held out for validation, one
game per preset is held out as test, and the rest train. A folder that is already in the set
keeps its junction. Built this way: v6 (2026-09-26, 218 games).
"""

import _winapi
import argparse
import json
from collections import defaultdict
from pathlib import Path

VALIDATION = {"20260923-222819", "20260923-223646", "20260923-225646", "20260923-234139"}
MAIN = {"arena-12x8-v3", "arena-12x8-v4"}
PRESETS = {
    f"arena-{n}-v6" for n in ("plains", "river", "passes", "marsh", "bay", "salient", "ford")
}
KINDS = {
    "scripted": "artifacts/scripted-*/scripted-*/manifest.json",
    "drills": "artifacts/drills/*/drill-*/manifest.json",
    "practice": "artifacts/learned/practice-*/practice-*/manifest.json",
}


def usable(folder, manifest, kind):
    if not manifest.get("complete") or manifest.get("salvaged"):
        return False
    if not (folder / "screen.mkv").exists() or not (folder / "arena-log.jsonl").exists():
        return False
    if kind == "practice":
        # Only the coach's spans teach; a practice game without one has nothing to give.
        return bool(manifest.get("coached"))
    return kind == "drills" or (manifest.get("frames") or 0) >= 100


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--drills", action="store_true")
    parser.add_argument("--practice", action="store_true")
    parser.add_argument(
        "--root", type=Path, default=Path("."), help="The checkout whose artifacts/ hold them"
    )
    args = parser.parse_args()
    kinds = ["scripted"] + [k for k in ("drills", "practice") if getattr(args, k)]
    games = {}
    for kind in kinds:
        for path in sorted(args.root.glob(KINDS[kind])):
            folder, manifest = path.parent, json.loads(path.read_text())
            arena = Path(str(manifest.get("arena") or "")).name
            if not usable(folder, manifest, kind) or arena not in MAIN | PRESETS:
                continue
            stamp = folder.name.split("-peer-")[-1]
            if kind == "scripted" and arena in MAIN and stamp < "20260923-222128":
                continue
            if folder.name in games:
                print("duplicate name, kept the first:", folder)
                continue
            games[folder.name] = (folder, arena, stamp, kind)
    held = {}
    for name, (_, arena, _, kind) in sorted(games.items()):
        if kind == "scripted" and arena in PRESETS and arena not in held:
            held[arena] = name
    splits = {
        name: "validation" if stamp in VALIDATION else "test" if name in held.values() else "train"
        for name, (_, _, stamp, _) in games.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    made = 0
    for name, (folder, *_) in games.items():
        link = args.output / name
        if not link.exists():
            _winapi.CreateJunction(str(folder), str(link))
            made += 1
    (args.output / "splits.json").write_text(json.dumps(dict(sorted(splits.items())), indent=2))
    count = defaultdict(lambda: defaultdict(int))
    for name, (_, arena, _, kind) in games.items():
        count[(arena, kind)][splits[name]] += 1
    print(f"{len(games)} recordings, {made} new links")
    for key in sorted(count):
        print(*key, dict(count[key]))


if __name__ == "__main__":
    main()

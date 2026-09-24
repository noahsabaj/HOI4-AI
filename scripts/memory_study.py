"""The memory comparison: every arm, five seeds, on one feature cache, and the verdict.

    python scripts/memory_study.py artifacts/features artifacts/memory-study

Arms (train_memory, the same decisions per update and passes for all):

    gru-16-reset   the way train-bc trains: 16 decisions after 2 of burn-in, from empty
    gru-16         16-decision windows, the memory carried through whole games
    gru-256        256-decision windows, carried
    gdn2-256       Gated DeltaNet-2, 256, carried
    mamba3-256     Mamba-3 SISO, 256, carried
    none-256       no memory: the floor

The rule, written before any run (2026-09-23). Each arm's score is the mean over seeds of
its held-out imitation loss with the memory carried from the start of each game, which
is how the policy plays; its noise is the larger of the two arms' standard deviations
over seeds. Training on long windows replaces the old way if gru-256 beats gru-16-reset
by more than twice that noise. A cell replaces the GRU if it beats gru-256 by more than
twice that noise, and its since-recentre probe (how long since the camera zoomed out,
the one target that needs 100 to 300 decisions of memory) is not worse by more than
twice its own noise. Otherwise the GRU stays: a new cell has to earn its place.

Added on 2026-09-24, before the rerun: the first study's GRUs were dead (the tower's
unnormalized summary saturated their gates, so their memory never moved), which no report
showed. Now an arm is invalid, and is not ranked, if the memory of any of its seeds reads
dead in its report (`memory_health`, features.memory_health: most units of the memory's
output never move over a held-out game). A run of the smaller first pass, which has no
gru-256, compares against the carried GRU it has, gru-16: the old training if gru-16 beats
gru-16-reset, and a cell against gru-16.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ARMS = {
    "gru-16-reset": {"memory": "gru", "window": 16, "carry": False, "burn_in": 2},
    "gru-16": {"memory": "gru", "window": 16},
    "gru-256": {"memory": "gru", "window": 256},
    "gdn2-256": {"memory": "gdn2", "window": 256},
    "mamba3-256": {"memory": "mamba3", "window": 256},
    "none-256": {"memory": "none", "window": 256},
}


def summarise(output, arms, seeds):
    rows = {}
    for arm in arms:
        reports = [
            json.loads(path.read_text())
            for seed in seeds
            if (path := Path(output) / f"{arm}-s{seed}" / "report.json").exists()
        ]
        if not reports:
            continue

        def stat(key, probe=False):
            values = [r["probes"].get(key, np.nan) if probe else r[key] for r in reports]
            return float(np.mean(values)), float(np.std(values))

        health = [r.get("memory_health") for r in reports]
        known = [h for h in health if h is not None]
        rows[arm] = {
            "seeds": len(reports),
            "dead_seeds": sum(h["dead"] for h in known) if known else None,
            "still_units": (float(np.mean([h["still_units"] for h in known])) if known else None),
            "std_over_time": (
                float(np.mean([h["std_over_time"] for h in known])) if known else None
            ),
            "nll": stat("validation_nll"),
            "nll_acting": stat("validation_nll_acting"),
            "nll_cleared": stat("validation_nll_cleared_every_18"),
            **{k: stat(k, True) for k in reports[0]["probes"]},
            "state_floats": reports[0]["state_floats"],
            "parameters": reports[0]["parameters"],
            "train_seconds": float(np.mean([r["train_seconds"] for r in reports])),
        }
    return rows


def beats(rows, challenger, incumbent, key="nll", lower=True):
    """Whether `challenger` beats `incumbent` on `key` by more than twice the noise."""
    (a, sa), (b, sb) = rows[challenger][key], rows[incumbent][key]
    margin = 2 * max(sa, sb)
    return (b - a if lower else a - b) > margin


def verdict(rows):
    lines = []
    invalid = {arm for arm, row in rows.items() if row.get("dead_seeds")}
    for arm in sorted(invalid):
        seeds = rows[arm]["seeds"]
        lines.append(
            f"{arm} is invalid: its memory reads dead in {rows[arm]['dead_seeds']} of {seeds} seeds"
        )
    # The carried GRU to compare against: gru-256, or in the first pass gru-16.
    gru = "gru-256" if "gru-256" in rows else "gru-16"

    def decided(*arms):
        if not set(arms) <= rows.keys():
            return False
        if set(arms) & invalid:
            lines.append(f"{' against '.join(arms)}: not ranked, an arm is invalid")
            return False
        return True

    if decided(gru, "gru-16-reset"):
        long = beats(rows, gru, "gru-16-reset")
        if gru == "gru-256":
            lines.append(f"long windows {'replace' if long else 'do not replace'} the old training")
        else:
            said = "replaces" if long else "does not replace"
            lines.append(f"a memory carried through games {said} the old training")
    for cell in ("gdn2-256", "mamba3-256"):
        if decided(cell, gru):
            better = beats(rows, cell, gru)
            recall_ok = not beats(rows, gru, cell, "since_recentre", lower=False)
            wins = better and recall_ok
            lines.append(f"{cell} {'replaces' if wins else 'does not replace'} the GRU ({gru})")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache")
    parser.add_argument("output")
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    from hoi4_arena.features import train_memory

    for seed in args.seeds:
        for arm in args.arms:
            out = Path(args.output) / f"{arm}-s{seed}"
            if (out / "report.json").exists():
                continue
            report = train_memory(args.cache, out, epochs=args.epochs, seed=seed, **ARMS[arm])
            print(
                arm,
                seed,
                round(report["validation_nll"], 4),
                report["train_seconds"],
                "s",
                flush=True,
            )
    rows = summarise(args.output, args.arms, args.seeds)
    for arm, row in rows.items():
        print(arm, json.dumps(row))
    for line in verdict(rows):
        print(line)
    (Path(args.output) / "summary.json").write_text(
        json.dumps({"rows": rows, "verdict": verdict(rows)}, indent=2)
    )


if __name__ == "__main__":
    main()

"""Tuning the scripted player's best plan by Bayesian optimisation.

Games are scarce and their results noisy: one PC plays about 9 an hour, and telling a plan
that wins 95% of games from one that wins 90% takes about 430 games of each. So the plans
tried in the explore share of a run's games (record-ai --tune) are chosen by Optuna's
GPSampler (stable since Optuna 5, 2026-09), which fits a Gaussian process to every game
so far, noise included, and asks for the plan most likely to beat the best seen. Each
game plays the plan it was given and reports its score back. The study lives in one
SQLite file, so it carries on across runs, and the best plan's earlier games can seed it
(seed).

The score is the enemy's surrender progress minus the player's when the game ends, from
the arena's daily lines: 1 for a win that cost nothing, -1 for a loss, and in between for
a game the time cap cut short. It says more than a win or a loss: on 2026-09-24 the best
plan's games on the main arena averaged +0.83, its exploring games +0.14.

The GP needs torch, which a recording otherwise never imports (about 1.5 GB for the whole
run), so each plan is asked for in a short-lived `hoi4-arena tune DB ask` (4-15 s,
between games); reporting a score needs only the SQLite file.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

# One study for the best plan's settings, pooled over the arenas a run plays.
STUDY = "best-plan"
# The daily line's country and its surrender progress (ArenaLog; arena v3 and later).
SURRENDER = re.compile(r"^day\s.*?\b(BLU|RED) states .*?\bsurrender ([0-9.]+)")


def space():
    """The best plan's settings the tuner may choose, as Optuna distributions.

    The rest of the best plan stays: a broad attack, redrawn while paused. From the games
    of 2026-09-23/24: holds under 90 s lost more; the attacks but broad lost far more;
    recruiting 4 slots won 1 of 6; among exploring plans, redraws under 60 s won 8 of 11
    and those at 60 s or more 5 of 17, and conscription only up to Service by Requirement
    won 6 of 8. The guard's share and the broad line's depth were never varied.
    """
    from optuna.distributions import (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )

    return {
        # Seconds the front holds before the attack (best_plan: 120-240).
        "wait": IntDistribution(90, 300),
        # Seconds between the attack's redraws (30-90).
        "redraw": IntDistribution(20, 90),
        # The share of the home land the enemy may hold before the army turns back (0.15).
        "guard": FloatDistribution(0.08, 0.3),
        # How far on into the enemy's land the broad offensive's line lies (1/3).
        "depth": FloatDistribution(0.15, 0.6),
        # How far up the conscription laws to go (All Adults Serve).
        "conscription": CategoricalDistribution(("extensive", "service", "all_adults")),
        # Training slots opened once the conscription goal is reached (none).
        "recruit": CategoricalDistribution((0, 2)),
    }


def game_score(game, country, winner):
    """The game's score for `country`: the enemy's surrender progress minus the player's
    at the end, from the daily lines of the game folder's arena-log.txt. A capitulation
    counts as full progress; a game with no daily lines scores its result alone."""
    enemy = "RED" if country == "BLU" else "BLU"
    last = {}
    log = Path(game) / "arena-log.txt"
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            match = SURRENDER.match(line)
            if match:
                last[match.group(1)] = float(match.group(2))
    own, other = last.get(country, 0.0), last.get(enemy, 0.0)
    if winner == country:
        other = 1.0
    elif winner == enemy:
        own = 1.0
    return round(other - own, 4)


def _study(path, sampler=None):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=STUDY,
        storage=f"sqlite:///{Path(path).resolve().as_posix()}",
        direction="maximize",
        sampler=sampler or optuna.samplers.RandomSampler(),
        load_if_exists=True,
    )


def ask(path):
    """A new trial's number and settings, from the Gaussian process (tune DB ask)."""
    import optuna

    study = _study(path, optuna.samplers.GPSampler())
    trial = study.ask(space())
    return {"trial": trial.number, "params": trial.params}


class Tuner:
    """The study behind a run's tuned plans, in the SQLite file at `path`.

    Opening it fails the trials left running: a recorder stopped mid-game never reported
    them, and only one recorder uses a study at a time.
    """

    def __init__(self, path):
        from optuna.trial import TrialState

        self.path = Path(path)
        self.study = _study(self.path)
        for trial in self.study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
            self.study.tell(trial.number, state=TrialState.FAIL)

    def propose(self, rng):
        """The best plan with the settings the GP asks for next: variant "tuned", and
        the trial's number, for report."""
        from .scripted import best_plan

        env = {**os.environ, "OMP_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": "-1"}
        done = subprocess.run(
            [sys.executable, "-m", "hoi4_arena", "tune", str(self.path), "ask"],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
            check=True,
        )
        asked = json.loads(done.stdout)
        plan = {**best_plan(rng), **asked["params"], "best": False, "variant": "tuned"}
        plan["trial"] = asked["trial"]
        return plan

    def report(self, plan, score):
        """The game's score for the plan's trial; None fails it (a game with no result)."""
        from optuna.trial import TrialState

        if score is None:
            self.study.tell(plan["trial"], state=TrialState.FAIL)
        else:
            self.study.tell(plan["trial"], float(score))


def plan_settings(plan):
    """A plan's settings in the tuner's space, or None if it lies outside it: the best
    plan since the guard (a broad attack, redrawn paused), whose settings have defaults
    where it never varied them."""
    from .scripted import BROAD_DEPTH

    if not plan.get("guard") or plan.get("attack") != "broad" or not plan.get("pause_redraw"):
        return None
    settings = {
        "wait": plan.get("wait"),
        "redraw": plan.get("redraw"),
        "guard": plan["guard"],
        "depth": plan.get("depth", BROAD_DEPTH),
        "conscription": plan.get("conscription"),
        "recruit": plan.get("recruit", 0),
    }
    for name, distribution in space().items():
        value = settings[name]
        if value is None:
            return None
        if hasattr(distribution, "choices"):
            if value not in distribution.choices:
                return None
        elif not distribution.low <= value <= distribution.high:
            return None
    return settings


def seed(path, roots, skip=()):
    """The best plan's finished games under `roots` (their manifests), added to the study
    as finished trials, once each; games on the `skip` arenas, tuned games, games shorter
    than 30 s (the broken loads of 2026-09-24) and games that ended in an error are left
    out. The number added. Safe beside a running recorder, whose trial it leaves be."""
    from optuna.trial import create_trial

    study = _study(path)
    seen = {t.user_attrs.get("game") for t in study.get_trials(deepcopy=False)}
    distributions = space()
    added = 0
    for root in roots:
        for manifest in sorted(Path(root).glob("*/manifest.json")):
            try:
                game = json.loads(manifest.read_text())
            except (OSError, ValueError):
                continue
            plan = game.get("plan") or {}
            arena = game.get("arena")
            arena = Path(arena.get("mod", "") if isinstance(arena, dict) else arena or "").name
            name = manifest.parent.name
            if name in seen or "trial" in plan or arena in skip:
                continue
            if (game.get("seconds") or 0) < 30 or game.get("reason"):
                continue
            winner, country = game.get("winner"), game.get("started_as")
            settings = plan_settings(plan)
            if settings is None or winner not in ("BLU", "RED", "timeout") or not country:
                continue
            value = game_score(manifest.parent, country, winner)
            study.add_trial(
                create_trial(
                    params=settings,
                    distributions=distributions,
                    value=value,
                    user_attrs={"game": name, "arena": arena, "seeded": True},
                )
            )
            added += 1
    return added


def show(path, top=5):
    """The study so far: its trials by state, the tuned games' mean score beside the
    seeded games', the best trials, and which settings matter most (PED-ANOVA)."""
    import optuna
    from optuna.trial import TrialState

    study = _study(path)
    trials = study.get_trials(deepcopy=False)
    done = [t for t in trials if t.state == TrialState.COMPLETE]
    tuned = [t.value for t in done if not t.user_attrs.get("seeded")]
    seeded = [t.value for t in done if t.user_attrs.get("seeded")]
    report = {
        "trials": {
            state.name.lower(): sum(t.state == state for t in trials) for state in TrialState
        },
        "tuned_mean": round(sum(tuned) / len(tuned), 3) if tuned else None,
        "seeded_mean": round(sum(seeded) / len(seeded), 3) if seeded else None,
        "best": [
            {"trial": t.number, "score": t.value, **t.params, "game": t.user_attrs.get("game")}
            for t in sorted(done, key=lambda t: -t.value)[:top]
        ],
    }
    try:
        report["importance"] = {
            name: round(value, 3)
            for name, value in optuna.importance.get_param_importances(study).items()
        }
    except (RuntimeError, ValueError):
        report["importance"] = None  # Too few finished trials.
    return report

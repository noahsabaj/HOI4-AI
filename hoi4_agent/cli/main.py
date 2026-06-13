"""argparse entrypoint: ``python -m hoi4_agent.cli.main <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import Config, load_config
from ..errors import AgentError, HaltAndFlag


def _build_live_ctx(cfg: Config, title: str):
    from ..brain.decide import Brain, build_backend
    from ..calibration import load_calibration
    from ..context import AgentContext
    from ..io import windows as win
    from ..perception.templates import TemplateStore

    locator, capture, inp = win.build_io(cfg.timing.action_dwell_ms)
    geo = locator.find(title)
    if geo is None:
        return None
    calib = load_calibration(cfg.paths.calibration)
    templates = TemplateStore.load_dir(cfg.paths.templates)
    brain = Brain(build_backend(cfg.llm))
    return AgentContext.build(
        config=cfg, geometry=geo, input=inp, capture=capture,
        calibration=calib, templates=templates, brain=brain,
    )


def cmd_run(cfg: Config, args) -> int:
    from ..controller.loop import run
    from ..playbook.loader import load_playbook
    from ..playbook.state import load_state
    from ..trace.writer import JsonlTraceWriter

    ctx = _build_live_ctx(cfg, args.title)
    if ctx is None:
        print("game window not found — is HOI4 running at the configured resolution?")
        return 1
    goals = load_playbook(cfg.paths.playbook)
    out = Path(cfg.paths.trace_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_path = str(out / "plan_state.json")
    state = load_state(state_path)
    with JsonlTraceWriter(out / "trace.jsonl", screenshot_dir=out / "frames") as w:
        try:
            final = run(ctx, goals, state, writer=w, state_path=state_path, max_cycles=args.max_cycles)
        except HaltAndFlag as e:
            print(f"HALTED: {e}\ntrace: {e.trace_ref}")
            return 2
    print("done. completed:", list(final.completed_goal_ids))
    return 0


def cmd_eval(cfg: Config, args) -> int:
    from ..brain.decide import Brain, build_backend
    from ..eval.corpus import load_corpus
    from ..eval.m0_perception import score_model

    corpus = load_corpus(cfg.paths.corpus)
    if not corpus:
        print(f"no corpus at {cfg.paths.corpus} (add labeled *.png + *.toml)")
        return 1
    report = score_model(corpus, Brain(build_backend(cfg.llm)))
    print(json.dumps(report, indent=2))
    return 0


def cmd_replay(cfg: Config, args) -> int:
    from ..brain.decide import Brain, build_backend
    from ..eval.replay import replay

    brain = Brain(build_backend(cfg.llm))

    def probe(img):
        d = brain.read_date(img)
        return d.to_str() if d else None

    print(json.dumps(replay(args.trace, probe), indent=2))
    return 0


def cmd_smoke(cfg: Config, args) -> int:
    from . import smoke

    return smoke.run_offline(cfg) if args.offline else smoke.run_live(cfg, args.title)


def cmd_calibrate(cfg: Config, args) -> int:
    from . import calibrate

    return calibrate.run(cfg, args.title)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="hoi4_agent")
    p.add_argument("--config", default="config/agent.toml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("smoke-test", help="validate I/O (live) or run controller with fakes (--offline)")
    s.add_argument("--offline", action="store_true")
    s.add_argument("--title", default="Hearts of Iron")
    s.set_defaults(func=cmd_smoke)

    s = sub.add_parser("calibrate", help="record ROIs + click-points -> calibration.toml")
    s.add_argument("--title", default="Hearts of Iron")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("eval", help="M0: score the model's perception on the corpus")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("replay", help="re-run saved trace frames through the model")
    s.add_argument("trace")
    s.set_defaults(func=cmd_replay)

    s = sub.add_parser("run", help="play Germany-1936 construction + research")
    s.add_argument("--title", default="Hearts of Iron")
    s.add_argument("--max-cycles", type=int, default=1000)
    s.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except AgentError as e:
        print(f"config error: {e}")
        return 1
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())

"""argparse entrypoint: ``python -m hoi4_agent.cli.main <command>``."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    geo = locator.find(title, (cfg.display.width, cfg.display.height))
    if geo is None:
        return None
    calib = load_calibration(cfg.paths.calibration)
    templates = TemplateStore.load_dir(cfg.paths.templates)
    brain = Brain(build_backend(cfg.llm))
    grounding_brain = Brain(build_backend(cfg.grounding)) if cfg.grounding is not None else None
    return AgentContext.build(
        config=cfg, geometry=geo, input=inp, capture=capture,
        calibration=calib, templates=templates, brain=brain, locator=grounding_brain,
    )


def cmd_run(cfg: Config, args) -> int:
    from ..controller.loop import run
    from ..io.backends import InputRecorder
    from ..playbook.loader import load_playbook, load_research_days
    from ..playbook.state import load_state
    from ..preflight import preflight
    from ..trace.writer import JsonlTraceWriter

    ctx = _build_live_ctx(cfg, args.title)
    if ctx is None:
        print("game window not found — is HOI4 running at the configured resolution?")
        return 1
    ctx.input = InputRecorder(ctx.input)  # journal every key/click into the trace
    goals = load_playbook(cfg.paths.playbook)

    errors, warnings = preflight(cfg, ctx.calibration, ctx.templates, goals)
    for warning in warnings:
        print(f"preflight WARN: {warning}")
    if errors:
        for err in errors:
            print(f"preflight ERROR: {err}")
        print(f"preflight: {len(errors)} error(s) — not starting the live loop.")
        return 1

    out = Path(cfg.paths.trace_dir)
    # One directory per run (trace + frames); plan state stays shared for resume.
    run_dir = out / time.strftime("run_%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = out / "plan_state.json"
    if args.fresh and state_path.exists():
        state_path.unlink()
        print("(--fresh: cleared plan_state.json)")
    state = load_state(state_path)
    with JsonlTraceWriter(run_dir / "trace.jsonl", screenshot_dir=run_dir / "frames") as w:
        try:
            final = run(ctx, goals, state, writer=w, state_path=str(state_path),
                        max_cycles=args.max_cycles,
                        research_days=load_research_days(cfg.paths.playbook))
        except HaltAndFlag as e:
            print(f"HALTED: {e}\ntrace: {e.trace_ref}")
            return 2
    print("done. completed:", list(final.completed_goal_ids))
    print(f"trace: {run_dir / 'trace.jsonl'}")
    return 0


def cmd_eval(cfg: Config, args) -> int:
    from ..brain.decide import Brain, build_backend
    from ..config import list_llm_profiles, load_config
    from ..eval.corpus import load_corpus
    from ..eval.m0_perception import score_model

    corpus = load_corpus(cfg.paths.corpus)
    if not corpus:
        print(f"no corpus at {cfg.paths.corpus} (add labeled *.png + *.toml)")
        return 1

    if not getattr(args, "all_profiles", False):
        report = score_model(corpus, Brain(build_backend(cfg.llm)))
        print(json.dumps(report, indent=2))
        return 0

    # The M0 shootout: score every configured profile on the same corpus.
    profiles = list_llm_profiles(args.config)
    if not profiles:
        print("no [llm.profiles.*] defined in config — nothing to compare")
        return 1
    rows = []
    for name in profiles:
        pcfg = load_config(args.config, llm_profile=name)
        print(f"scoring profile {name!r} ({pcfg.llm.model} via {pcfg.llm.backend})...")
        start = time.time()
        report = score_model(corpus, Brain(build_backend(pcfg.llm)))
        s_per_item = (time.time() - start) / max(report["total"], 1)
        per_task = "  ".join(
            f"{t}={v['accuracy']:.0%}" for t, v in sorted(report["per_task"].items())
        )
        rows.append((name, pcfg.llm.model, report["accuracy"], s_per_item, per_task))

    print(f"\n{'profile':<16} {'model':<26} {'accuracy':>8} {'s/item':>7}")
    for name, model, acc, s, per_task in rows:
        print(f"{name:<16} {model:<26} {acc:>8.1%} {s:>7.2f}")
        print(f"{'':<16} {per_task}")
    best = max(rows, key=lambda r: r[2])
    print(f"\nbest: {best[0]} ({best[2]:.1%}) — see [llm] comment in agent.toml for the switch rule")
    return 0


def cmd_replay(cfg: Config, args) -> int:
    from ..brain.decide import Brain, build_backend
    from ..calibration import load_calibration
    from ..eval.replay import replay

    brain = Brain(build_backend(cfg.llm))
    # Saved frames are full client captures; feed the model the calibrated date
    # crop (the same tight region the live loop reads), not the whole frame.
    date_frac = None
    try:
        date_frac = load_calibration(cfg.paths.calibration).rois.get("date")
    except AgentError:
        print("(no calibration found — probing full frames)")

    def probe(img):
        crop = img
        if date_frac is not None:
            fx0, fy0, fx1, fy1 = date_frac
            w, h = img.size
            crop = img.crop((round(fx0 * w), round(fy0 * h), round(fx1 * w), round(fy1 * h)))
        d = brain.read_date(crop)
        return d.to_str() if d else None

    print(json.dumps(replay(args.trace, probe), indent=2))
    return 0


def cmd_smoke(cfg: Config, args) -> int:
    from . import smoke

    return smoke.run_offline(cfg) if args.offline else smoke.run_live(cfg, args.title)


def cmd_calibrate(cfg: Config, args) -> int:
    from . import calibrate

    return calibrate.run(cfg, args.title)


def cmd_save_audit(cfg: Config, args) -> int:
    from ..eval.savefile import read_save_facts
    from ..trace.writer import JsonlTraceWriter

    facts = read_save_facts(args.save, country=args.country)
    print(f"save facts for {facts.country}:")
    print(f"  date:                {facts.date or '(not found)'}")
    print(f"  researching:         {', '.join(facts.researching) if facts.researching else '(not found)'}")
    print(f"  civilian factories:  {facts.civilian_factories if facts.civilian_factories is not None else '(not found)'}")
    print(f"  construction lines:  {facts.construction_lines if facts.construction_lines is not None else '(not found)'}")
    if facts.missing:
        print(f"  could not extract:   {', '.join(facts.missing)} (schema drift? inspect the save)")

    if args.trace:
        records = JsonlTraceWriter.read(args.trace)
        traced_date = next((r.date for r in reversed(records) if r.date), None)
        if traced_date is None or facts.date is None:
            print("trace cross-check: skipped (no date on one side)")
        elif traced_date == facts.date:
            print(f"trace cross-check: MATCH (both {traced_date})")
        else:
            print(f"trace cross-check: MISMATCH (trace {traced_date} vs save {facts.date})")
    return 0


def cmd_gui_import(cfg: Config, args) -> int:
    from ..calibration import dump_toml
    from ..gui_import import draft_calibration, load_elements

    elements = load_elements(args.path)
    print(f"parsed {len(elements)} positioned element(s)")
    calib, report = draft_calibration(elements, args.width, args.height)
    for roi, source in report["mapped"].items():
        print(f"  mapped {roi!r} <- {source}")
    for line in report["skipped"]:
        print(f"  skipped {line}")
    print(f"  still needing manual calibration: {', '.join(report['unmapped_rois']) or 'none'}")
    if args.out:
        Path(args.out).write_text(dump_toml(calib), encoding="utf-8")
        print(f"wrote DRAFT calibration to {args.out} — hand-verify with `calibrate` before use")
    if not report["mapped"]:
        print("no elements matched ELEMENT_MAP — extend hoi4_agent/gui_import.py with "
              "names from your game version's interface/*.gui")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="hoi4_agent")
    p.add_argument("--config", default="config/agent.toml")
    p.add_argument("--llm-profile", default=None,
                   help="override the [llm] profile from config (see [llm.profiles.*])")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("smoke-test", help="validate I/O (live) or run controller with fakes (--offline)")
    s.add_argument("--offline", action="store_true")
    s.add_argument("--title", default="Hearts of Iron")
    s.set_defaults(func=cmd_smoke)

    s = sub.add_parser("calibrate", help="record ROIs + click-points -> calibration.toml")
    s.add_argument("--title", default="Hearts of Iron")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("eval", help="M0: score the model's perception on the corpus")
    s.add_argument("--all-profiles", action="store_true",
                   help="score EVERY [llm.profiles.*] entry and print a comparison table")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("replay", help="re-run saved trace frames through the model")
    s.add_argument("trace")
    s.set_defaults(func=cmd_replay)

    s = sub.add_parser("save-audit", help="parse a text HOI4 save for ground-truth facts")
    s.add_argument("save", help="path to a non-ironman .hoi4 save file")
    s.add_argument("--country", default="GER")
    s.add_argument("--trace", default=None, help="trace.jsonl to cross-check against")
    s.set_defaults(func=cmd_save_audit)

    s = sub.add_parser("gui-import", help="EXPERIMENTAL: draft calibration from .gui interface files")
    s.add_argument("path", help=".gui file or directory containing them")
    s.add_argument("--out", default=None, help="write the draft calibration TOML here")
    s.add_argument("--width", type=int, default=2560)
    s.add_argument("--height", type=int, default=1440)
    s.set_defaults(func=cmd_gui_import)

    s = sub.add_parser("run", help="play Germany-1936 construction + research")
    s.add_argument("--title", default="Hearts of Iron")
    s.add_argument("--max-cycles", type=int, default=1000)
    s.add_argument("--fresh", action="store_true", help="discard persisted plan state and start over")
    s.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    try:
        cfg = load_config(args.config, llm_profile=args.llm_profile)
    except AgentError as e:
        print(f"config error: {e}")
        return 1
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())

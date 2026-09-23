from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Screen-only HOI4 research prototype")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Progress and diagnostics go to stderr; JSON results go to stdout.",
    )
    parser.add_argument("--traceback", action="store_true", help="Re-raise instead of exiting 1")
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Allow TF32 float32 matmuls. Measured worth nothing here and not free; "
        "see models.configure_precision.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--model", default="models/levjepa-large")
    bench.add_argument("--output", default="artifacts/benchmark")
    bench.add_argument("--iterations", type=int, default=30)
    bench.add_argument("--offline", action="store_true")
    record = sub.add_parser("record")
    record.add_argument("output")
    record.add_argument("--seconds", type=float, default=1200)
    record.add_argument("--hz", type=float, default=10)
    record.add_argument("--split", choices=["train", "validation", "test"])
    record.add_argument(
        "--codec",
        choices=["ffv1", "x264"],
        default="ffv1",
        help="ffv1 is lossless. x264 is visually lossless (CRF 18, 4:4:4) at about a "
        "hundredth of the size.",
    )
    record.add_argument(
        "--game-speed",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5],
        help="Speed the game is set to for the whole session. Written into the manifest. "
        "There is no default: a 1.6 s clip is 3.2 in-game hours at speed 2 and 16 at "
        "speed 4, and the speed bars are not read back.",
    )
    ai = sub.add_parser(
        "record-ai",
        help="Record AI-vs-AI arena games on this PC, the second PC, or both, until the "
        "time budget runs out",
    )
    ai.add_argument("output")
    ai.add_argument("--minutes", type=float, required=True, help="Total time budget.")
    ai.add_argument("--mod", default="artifacts/mods/arena-12x8-v1")
    ai.add_argument("--rules", default="artifacts/calibration-1080p/rules.json")
    ai.add_argument(
        "--ok-button",
        nargs="+",
        default=["artifacts/screens-1080p/ok-button.png"],
        help="1920x1080 crops of popup Ok buttons, clicked wherever they appear.",
    )
    ai.add_argument("--hz", type=float, default=5)
    ai.add_argument("--codec", choices=["ffv1", "x264"], default="x264")
    ai.add_argument("--cap-minutes", type=float, default=45)
    ai.add_argument("--peer", help="The second PC's pairing file, to record there too.")
    ai.add_argument("--peer-only", action="store_true", help="Leave this PC free.")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("source")
    prepare.add_argument("output")
    train = sub.add_parser("train-bc")
    train.add_argument("data")
    train.add_argument("output")
    train.add_argument("--model", default="models/levjepa-large")
    train.add_argument("--variant", choices=["large", "tiny"], default="large")
    train.add_argument("--student")
    train.add_argument("--auxiliary", choices=["none", "dense", "sparse"], default="none")
    train.add_argument("--objective", choices=["bc", "xm"], default="bc")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--sequence", type=int, default=8)
    train.add_argument("--burn-in", type=int, default=2)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--seed", type=int, default=42)
    distill = sub.add_parser("distill")
    distill.add_argument("data")
    distill.add_argument("output")
    distill.add_argument("--model", default="models/levjepa-large")
    distill.add_argument("--epochs", type=int, default=1)
    pairing = sub.add_parser("bundle-peer")
    pairing.add_argument("output")
    pairing.add_argument("--host", required=True)
    pairing.add_argument("--coordinator", required=True)
    pairing.add_argument(
        "--port",
        type=int,
        required=True,
        help="Worker port. Required rather than defaulted, because a published "
        "default is a detail of somebody's actual network.",
    )
    probe = sub.add_parser("probe-peer")
    probe.add_argument("config")
    capture = sub.add_parser("capture")
    capture.add_argument("output")
    capture.add_argument("--peer")
    template = sub.add_parser("template")
    template.add_argument("screenshot")
    template.add_argument("rules")
    template.add_argument("name")
    template.add_argument("--rect", nargs=4, type=int, required=True, metavar=("X", "Y", "W", "H"))
    template.add_argument(
        "--max-mae",
        type=int,
        default=5,
        help="How far this crop may drift and still match. The default of 5 is tighter "
        "than a live HUD holds still: measure the drift before trusting it.",
    )
    clock = sub.add_parser("clock", help="Calibrate the changing-clock ROI collection requires")
    clock.add_argument("screenshot")
    clock.add_argument("rules")
    clock.add_argument("--rect", nargs=4, type=int, required=True, metavar=("X", "Y", "W", "H"))
    minimap = sub.add_parser(
        "minimap", help="Calibrate the political-minimap crop the territory reward reads"
    )
    minimap.add_argument("screenshot")
    minimap.add_argument("rules")
    minimap.add_argument("--rect", nargs=4, type=int, required=True, metavar=("X", "Y", "W", "H"))
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("results")
    league_add = sub.add_parser("league-add", help="Register an immutable checkpoint in a league")
    league_add.add_argument("league")
    league_add.add_argument("checkpoint")
    league_sample = sub.add_parser(
        "league-sample", help="Sample one registered checkpoint. Does not start a match."
    )
    league_sample.add_argument("league")
    league_sample.add_argument("--seed", type=int, default=42)
    generation = sub.add_parser("generate-map")
    generation.add_argument("output")
    generation.add_argument("--game", required=True)
    generation.add_argument(
        "--undefended",
        choices=["BLU", "RED"],
        help="Field no divisions for this country. A diagnostic, not a playable arena: "
        "an empty front an AI never enters says it is not attacking at all.",
    )
    generation.add_argument("--columns-per-half", type=int)
    generation.add_argument("--rows", type=int)
    generation.add_argument("--state-columns", type=int)
    generation.add_argument("--state-rows", type=int)
    generation.add_argument(
        "--land-columns",
        type=int,
        help="Shrink each country to this many columns of the full grid, touching the "
        "seam. The rest is sea at the same province size.",
    )
    generation.add_argument(
        "--land-rows", type=int, help="Rows of land per country, centered vertically."
    )
    generation.add_argument(
        "--victory-points-on-border",
        action="store_true",
        help="Put every victory point on the border column. This does not capitulate a "
        "country: surrender is territorial. Kept as the measurement that showed that.",
    )
    inspection = sub.add_parser(
        "audit-map", help="Check a generated arena for references the engine cannot resolve"
    )
    inspection.add_argument("mod")
    collect = sub.add_parser("collect-pair")
    collect.add_argument("config")
    collect.add_argument("output")
    collect.add_argument("left_checkpoint")
    collect.add_argument("right_checkpoint")
    ppo = sub.add_parser("train-ppo")
    ppo.add_argument("rollouts")
    ppo.add_argument("checkpoint")
    ppo.add_argument("output")
    ppo.add_argument("--epochs", type=int, default=3)
    ppo.add_argument("--model-path")
    ppo.add_argument("--seed", type=int, default=42)
    ppo.add_argument("--burn-in", type=int, default=4)
    ppo.add_argument("--kl-limit", type=float, default=0.02)
    args = vars(parser.parse_args())
    command = args.pop("command")
    logging.basicConfig(
        level=getattr(logging, args.pop("log_level").upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    traceback = args.pop("traceback")
    tf32 = args.pop("tf32")
    try:
        from .models import configure_precision

        configure_precision(tf32)
        result = _dispatch(command, args)
    except KeyboardInterrupt:
        logging.getLogger(__name__).error("interrupted")
        raise SystemExit(130) from None
    except Exception as error:
        if traceback:
            raise
        logging.getLogger(__name__).error("%s: %s", type(error).__name__, error)
        raise SystemExit(1) from None
    if result is not None:
        print(json.dumps(result, indent=2))


def _report(problems):
    """One line per unresolved reference, then a short failure the exit code follows."""
    for problem in problems:
        logging.getLogger(__name__).error("arena: %s", problem)
    if problems:
        raise ValueError(f"{len(problems)} references the game cannot resolve")


def _dispatch(command, args):
    result = None
    if command == "benchmark":
        from .benchmark import benchmark

        benchmark(args.pop("model"), **args)
    elif command == "record":
        from .recording import record

        record(args.pop("output"), **args)
    elif command == "record-ai":
        from .ai_games import record_ai_games

        result = record_ai_games(args.pop("output"), **args)
    elif command == "prepare":
        from .dataset import prepare_session

        result = prepare_session(args["source"], args["output"])
    elif command == "train-bc":
        from .train import train_bc

        result = train_bc(args.pop("data"), args.pop("model"), args.pop("output"), **args)
    elif command == "distill":
        from .train import distill

        distill(args.pop("data"), args.pop("model"), args.pop("output"), **args)
    elif command == "bundle-peer":
        from .remote import bundle

        result = bundle(**args)
    elif command in {"probe-peer", "capture"}:
        from .desktop import Desktop
        from .remote import RemoteDesktop

        config = args.get("peer", args.get("config"))
        with RemoteDesktop(config) if config else Desktop() as desktop:
            if command == "capture":
                from PIL import Image

                frame = desktop.capture()
                path = Path(args["output"])
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(frame.rgb).save(path)
                result = {k: v for k, v in frame.meta.items() if k != "events"}
            else:
                result = {k: v for k, v in desktop.attached.items() if k != "payload"}
    elif command == "template":
        from .vision import add_template

        result = add_template(**args)
    elif command == "clock":
        from .vision import set_clock_rect

        result = set_clock_rect(**args)
    elif command == "minimap":
        from .vision import set_minimap_rect

        result = set_minimap_rect(**args)
    elif command == "evaluate":
        from .learning import paired_evaluation

        result = paired_evaluation(
            [
                json.loads(line)
                for line in Path(args["results"]).read_text().splitlines()
                if line.strip()
            ]
        )
    elif command == "league-add":
        from .learning import League

        league = League(args["league"])
        league.add(args["checkpoint"])
        result = {"entries": len(league.entries)}
    elif command == "league-sample":
        from .learning import League

        result = League(args["league"], seed=args["seed"]).sample()
    elif command == "generate-map":
        from .mapgen import audit, generate

        grid = {
            "columns_per_half": args["columns_per_half"],
            "rows": args["rows"],
            "state_columns": args["state_columns"],
            "state_rows": args["state_rows"],
            "land_columns": args["land_columns"],
            "land_rows": args["land_rows"],
        }
        result = generate(
            args["game"],
            args["output"],
            undefended=args["undefended"],
            victory_points_on_border=args["victory_points_on_border"],
            **{key: value for key, value in grid.items() if value is not None},
        )
        result["audit"] = audit(args["output"])
        # The map is on disk either way; a bad one must not exit zero, because the next
        # thing that reads it is the game, which crashes instead of reporting.
        _report(result["audit"]["problems"])
    elif command == "audit-map":
        from .mapgen import audit

        result = audit(args["mod"])
        _report(result["problems"])
    elif command == "collect-pair":
        from .runner import collect_pair

        result = collect_pair(
            args["config"], args["output"], args["left_checkpoint"], args["right_checkpoint"]
        )
    elif command == "train-ppo":
        from .runner import train_ppo

        result = train_ppo(**args)
    return result


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Screen-only HOI4 research prototype")
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
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("results")
    generation = sub.add_parser("generate-map")
    generation.add_argument("output")
    generation.add_argument("--game", required=True)
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
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = None
    if command == "benchmark":
        from .benchmark import benchmark

        benchmark(args.pop("model"), **args)
    elif command == "record":
        from .recording import record

        record(args.pop("output"), **args)
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
    elif command == "evaluate":
        from .learning import paired_evaluation

        result = paired_evaluation(
            [json.loads(line) for line in Path(args["results"]).read_text().splitlines()]
        )
    elif command == "generate-map":
        from .mapgen import generate

        result = generate(args["game"], args["output"])
    elif command == "collect-pair":
        from .runner import collect_pair

        result = collect_pair(
            args["config"], args["output"], args["left_checkpoint"], args["right_checkpoint"]
        )
    elif command == "train-ppo":
        from .runner import train_ppo

        result = train_ppo(**args)
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

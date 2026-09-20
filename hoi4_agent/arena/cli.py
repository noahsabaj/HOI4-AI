"""Arena CLI; no game or neural network imports until a command needs them."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .contracts import ArenaError, PROTOCOL_VERSION
from .diagnostics import REQUIRED_CAPABILITIES, inspect_installation, inspect_probe, write_json
from .probe import create_lua_probe, launch_isolated_probe

DEFAULT_GAME = Path("C:/Program Files (x86)/Steam/steamapps/common/Hearts of Iron IV")

# v0.1 slices register their own subcommands through add_commands(commands) -> handlers.
EXTENSIONS = ("hoi4_agent.arena.mod.generate", "hoi4_agent.arena.mod.custommap", "hoi4_agent.arena.vision", "hoi4_agent.arena.agents",
              "hoi4_agent.arena.lan", "hoi4_agent.arena.train")


def load_extensions(commands) -> dict:
    import importlib
    handlers: dict = {}
    for name in EXTENSIONS:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:  # optional dependency (e.g. Torch) or slice not installed
            print(f"arena: commands from {name} unavailable: {exc}")
            continue
        handlers.update(module.add_commands(commands))
    return handlers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HOI4 combat-learning arena (integration in development)")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="inspect installed ABI and scripting capabilities")
    inspect.add_argument("--game", type=Path, default=DEFAULT_GAME)
    inspect.add_argument("--output", type=Path, default=Path("artifacts/capabilities.json"))
    inspect.add_argument("--probe-profile", action="append", type=Path, default=[])
    probe = commands.add_parser("lua-probe", help="create an isolated Lua binding probe")
    probe.add_argument("--game", type=Path, default=DEFAULT_GAME)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--launch", action="store_true")
    probe.add_argument("--native-dll", type=Path, help="load the built native diagnostic DLL through Lua")
    bridge = commands.add_parser("bridge", help="query a native bridge's actual capabilities")
    bridge.add_argument("--pipe", required=True)
    model = commands.add_parser("model-info", help="report architecture and local Torch availability")
    model.add_argument("--width", type=int, default=128)
    evaluation = commands.add_parser("evaluation-report", help="summarize recorded paired real-game results")
    evaluation.add_argument("--games", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    pin = commands.add_parser("fingerprint", help="hash the exact executable, DLC configuration and enabled mod")
    pin.add_argument("--game", type=Path, default=DEFAULT_GAME)
    pin.add_argument("--mod", type=Path, required=True)
    pin.add_argument("--dlc-load", type=Path, required=True)
    pin.add_argument("--output", type=Path, required=True)
    handlers = load_extensions(commands)
    args = parser.parse_args(argv)
    try:
        if args.command in handlers:
            return int(handlers[args.command](args) or 0)
        if args.command == "inspect":
            report = inspect_installation(args.game)
            report["runtime_probes"] = [inspect_probe(profile) for profile in args.probe_profile]
            write_json(args.output, report)
            print(json.dumps({"report": str(args.output.resolve()), "gate": report["integration_gate"],
                              "version": report["game_version"], "blockers": report["blockers"]}, indent=2))
            return 2 if report["integration_gate"] != "passed" else 0
        if args.command == "lua-probe":
            command = create_lua_probe(args.game, args.output, args.native_dll)
            print(json.dumps({"launch_command": command, "profile": str(args.output.resolve())}, indent=2))
            if args.launch:
                result = launch_isolated_probe(args.game, args.output, command)
                print(json.dumps({**result, "launcher_settings_restored": True}, indent=2))
            return 0
        if args.command == "bridge":
            from .protocol import NamedPipeTransport
            transport = NamedPipeTransport(args.pipe)
            try:
                response = transport.exchange({"version": PROTOCOL_VERSION, "id": 1,
                                               "method": "hello", "payload": {}})
                print(json.dumps(response, indent=2))
                result = response.get("result", {})
                return 0 if (result.get("backend") == "hoi4_native" and
                    all(result.get("capabilities", {}).get(cap) is True for cap in REQUIRED_CAPABILITIES)) else 2
            finally:
                transport.close()
        if args.command == "model-info":
            import torch
            from .policy import RecurrentPolicy
            policy = RecurrentPolicy(args.width)
            print(json.dumps({"parameters": sum(p.numel() for p in policy.parameters()),
                              "width": policy.width, "torch": torch.__version__,
                              "cuda_available": torch.cuda.is_available(),
                              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                              "gameplay_verified": False}, indent=2))
            return 0
        if args.command == "evaluation-report":
            from .evaluation import EvaluationGame, improvement_report
            rows = json.loads(args.games.read_text(encoding="utf-8"))
            report = improvement_report([EvaluationGame(**row) for row in rows])
            write_json(args.output, report)
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "fingerprint":
            from .fingerprint import fingerprint
            result = asdict(fingerprint(args.game, args.mod, args.dlc_load))
            write_json(args.output, result)
            print(json.dumps(result, indent=2))
            return 0
    except (ArenaError, OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        print(f"arena: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

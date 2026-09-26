"""The learned policy's decision latency, measured the way it decides live, with a gate.

A decision is `runner.Actor.act` in lean mode, as play-policy, practice and drills run it
(`hoi4-arena on-peer`): the worker's uint8 views in (global view, four quadrants, fovea),
the action out on the host. It is timed on fixed recorded frames (`frames`, below), one
decision after another and paced at the live 5 Hz, on the second PC's GPU while its HOI4
plays, which is the condition the policy decides in.

    python scripts/bench_decide.py frames --recording <game folder> --output bench/frames.npz
    python scripts/bench_decide.py tower4b --checkpoint <bc5 epoch-0000.pt> \
        --tower <folder with the 4B tower's model.safetensors and config.json>
    python scripts/bench_decide.py run --checkpoint <epoch-0000.pt> --label baseline \
        [--write-reference bench/reference-a.pt | --reference bench/reference-a.pt] \
        [--option name=json ...]

`run` prints one JSON report: p50 and p95 per decision back to back and paced, a split of
each decision's GPU time by stage (CUDA events on the modules that run eagerly), the GPU
kernel time and launches per decision (torch.profiler), the synchronizing calls per
decision (torch.cuda sync debug mode), memory, and the gate.

The gate replays the fixed frames from an empty memory with a fixed seed, sampling at
temperature 1 as live play does, and compares with a reference written by the unmodified
code (`--write-reference`):

- the sampled actions must be identical at every decision;
- the log-probability of each chosen action under every head (kind, cell, position in the
  cell, for each of the eight slots), teacher-forced through the head's own layers from
  the memory and cells the decision produced, must lie within 0.01 of the reference;
- the summed log-probability the decision itself returned, within 0.02.

It says `exact` when every compared tensor is bit-identical as well. `--option` passes
keyword arguments to Actor (JSON values), so an option added to Actor is measured
without changing this file; numeric options (lower precision, fp8) are "fast mode" and
reported apart. Decisions keep what the gate reads in `Actor.last` (cells, summed
log-probability, entropy) and the memory in `Actor.hidden`.

`ledger` appends a report to bench/decide-ledger.tsv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The gate's bounds. Fixed with the baseline: never loosen them to pass a change.
HEAD_BOUND = 0.01
SUM_BOUND = 0.02
GATE_SEED = 1234
INTERVAL_NS = 200_000_000
LEDGER_COLUMNS = [
    "when",
    "label",
    "tower",
    "options",
    "p50_ms",
    "p95_ms",
    "paced_p50_ms",
    "paced_p95_ms",
    "gpu_ms",
    "launches",
    "syncs",
    "gate",
    "exact",
    "max_head_diff",
    "peak_mib",
    "note",
]


def percentiles(times):
    ordered = sorted(times)
    return {
        "p50_ms": round(ordered[len(ordered) // 2] * 1000, 2),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] * 1000, 2),
        "mean_ms": round(sum(ordered) / len(ordered) * 1000, 2),
        "n": len(ordered),
    }


# --- fixed inputs ---------------------------------------------------------------------


def make_frames(recording, start, count, output):
    """`count` consecutive decisions of a recorded game as the worker hands them over."""
    import subprocess

    import numpy as np

    from hoi4_arena.dataset import views

    recording = Path(recording)
    manifest = json.loads((recording / "manifest.json").read_text())
    rows = [json.loads(line) for line in (recording / "frames.jsonl").read_text().splitlines()]
    w, h = manifest["width"], manifest["height"]
    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(recording / "screen.mkv"),
            "-vf",
            f"select='between(n,{start},{start + count - 1})'",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    globals_, quads, foveas, cursors, times = [], [], [], [], []
    for i in range(start, start + count):
        buffer = decoder.stdout.read(w * h * 3)
        if len(buffer) != w * h * 3:
            raise ValueError("the video ended early")
        frame = np.frombuffer(buffer, np.uint8).reshape(h, w, 3)
        seen = views(frame, cursor=rows[i]["cursor"])
        globals_.append(seen.global_view.numpy())
        quads.append(seen.quadrants.numpy())
        foveas.append(seen.fovea.numpy())
        cursors.append(rows[i]["cursor"])
        times.append(rows[i]["t_ns"])
    decoder.stdout.close()
    decoder.wait()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        global_view=np.stack(globals_),
        quadrants=np.stack(quads),
        fovea=np.stack(foveas),
        cursor=np.array(cursors),
        t_ns=np.array(times),
        source=str(recording.name),
        start=start,
    )
    print(f"{count} decisions of {recording.name} from frame {start} -> {output}")


def load_frames(path):
    import numpy as np

    data = np.load(path)
    return [
        (data["global_view"][i], data["quadrants"][i], data["fovea"][i])
        for i in range(len(data["global_view"]))
    ]


def make_tower4b(checkpoint, tower, output, tower_output):
    """The bc5 policy on the Qwen3.5-4B model's tower (#117's load_carried): the tower's
    own weights, stored in bfloat16 (live play halves them anyway), the layers sized to its
    width fresh, everything else carried."""
    import shutil

    import torch
    from safetensors.torch import load_file, save_file

    from hoi4_arena.learning import save_checkpoint
    from hoi4_arena.models import Policy, ScreenEncoder
    from hoi4_arena.train import load_carried

    tower, tower_output = Path(tower), Path(tower_output)
    tower_output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tower / "config.json", tower_output / "config.json")
    weights = load_file(str(tower / "model.safetensors"))
    save_file(
        {k: v.to(torch.bfloat16).contiguous() for k, v in weights.items()},
        str(tower_output / "model.safetensors"),
    )
    del weights
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = dict(saved["config"], model_path=tower_output.as_posix())
    torch.manual_seed(0)
    policy = Policy(
        ScreenEncoder(tower_output),
        latents=config.get("xm_latents", 0),
        look=config.get("look_before_click", False),
    )
    fresh = load_carried(policy, saved["policy"])
    state = {
        k: v.to(torch.bfloat16) if k.startswith("encoder.") else v
        for k, v in policy.state_dict().items()
    }

    class Holder(torch.nn.Module):
        def state_dict(self, *args, **kwargs):  # noqa: ARG002 - save_checkpoint's interface
            return state

    save_checkpoint(output, Holder(), config, provenance={"bench": "bc5 on the 4B tower"})
    print(f"{output}: fresh {fresh}")


# --- the measurement ------------------------------------------------------------------


def head_logps(head, memory, cells, actions, noise):
    """(B, SLOTS, 3): the log-probability of each chosen kind, cell and position in the
    cell, teacher-forced through the head's own layers, as ActionHead scores them."""
    import torch

    from hoi4_arena.actions import GRID, SLOTS
    from hoi4_arena.models import CELLS, categorical, log_prob

    b = memory.shape[0]
    rows = torch.arange(b, device=memory.device)
    state = torch.tanh(head.init(torch.cat([memory, noise.to(memory.dtype)], -1)))
    previous = torch.zeros(b, 64, device=memory.device, dtype=memory.dtype)
    moved = torch.zeros(b, dtype=torch.bool, device=memory.device)
    out = []
    for slot in range(SLOTS):
        state = head.cell(previous, state)
        logits = head.kinds(state).float()
        if head.look:
            logits = logits.masked_fill(moved[:, None] & head.press, float("-inf"))
        kinds = categorical(logits)
        where = torch.einsum("bnc,bc->bn", cells, head.query(state).to(cells.dtype))
        places = categorical(where.float() * head.scale + head.cell_bias.float())
        kind, x, y = actions[:, slot].unbind(-1)
        place = (y // CELLS) * CELLS + x // CELLS
        offset = (y % CELLS) * CELLS + x % CELLS
        chosen = cells[rows, place].to(state.dtype)
        fine = categorical(head.fine(torch.cat([state, chosen], -1)).float())
        out.append(
            torch.stack(
                [log_prob(kinds, kind), log_prob(places, place), log_prob(fine, offset)], -1
            )
        )
        moved = moved | (kind == 1)
        previous = head.embedding(kind) + head.xy(
            torch.stack([x, y], -1).to(memory.dtype) / (GRID - 1)
        )
    return torch.stack(out, 1)


def gate_pass(actor, frames):
    """Every fixed frame in order from an empty memory, seeded: what the gate compares."""
    import torch

    from hoi4_arena.runner import act_noise

    actor.reset_episode()
    torch.manual_seed(GATE_SEED)
    torch.cuda.manual_seed_all(GATE_SEED)
    head = getattr(actor.policy.actor, "_orig_mod", actor.policy.actor)
    record = {k: [] for k in ("action", "hidden", "cells", "logp", "entropy", "heads")}
    for i, views in enumerate(frames):
        action = actor.act(None, (i + 1) * INTERVAL_NS, precomputed=views)
        cells, logp, entropy = (t.detach().clone() for t in actor.last)
        hidden = actor.hidden.detach().clone()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.random.fork_rng(devices=[actor.device]):
                noise = act_noise(
                    actor.config["objective"], head.noise_dim, True, actor.device, None
                )
            chosen = torch.as_tensor(action[0] if isinstance(action, tuple) else action)
            heads = head_logps(head, hidden, cells, chosen[None].to(actor.device), noise)
        record["action"].append(chosen.cpu())
        record["hidden"].append(hidden.float().cpu())
        record["cells"].append(cells.float().cpu())
        record["logp"].append(logp.float().cpu())
        record["entropy"].append(entropy.float().cpu())
        record["heads"].append(heads.float().cpu())
    actor.reset_episode()
    return {k: torch.stack(v) for k, v in record.items()}


def compare(result, reference):
    import torch

    diff = {
        k: float((result[k] - reference[k]).abs().max())
        for k in ("hidden", "cells", "logp", "entropy", "heads")
    }
    same_actions = bool(torch.equal(result["action"], reference["action"]))
    agree = float((result["action"] == reference["action"]).all(-1).float().mean())
    exact = same_actions and all(torch.equal(result[k], reference[k]) for k in diff)
    passed = same_actions and diff["heads"] <= HEAD_BOUND and diff["logp"] <= SUM_BOUND
    return {
        "pass": passed,
        "exact": exact,
        "same_actions": same_actions,
        "slot_agreement": round(agree, 4),
        "max_diff": diff,
        "bounds": {"heads": HEAD_BOUND, "logp": SUM_BOUND},
    }


def split_pass(actor, frames, count):
    """Each stage's GPU time, from CUDA events on the modules that run eagerly (a stage run
    inside a CUDA graph or compiled away shows as missing), and the synchronizing calls."""
    import warnings

    import torch

    policy = actor.policy
    stages = {
        "tower": policy.encoder,
        "quadrant_reader": policy.details,
        "fovea_reader": policy.foveal,
        "memory": policy.memory,
    }
    marks = {}
    handles = []

    def mark(key):
        def hook(*_args, **_kwargs):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            marks.setdefault(key, []).append(event)

        return hook

    for name, module in stages.items():
        # A compiled module would trace the hooks into its graph (and recompile to do it).
        if module is None or getattr(module, "_compiled_call_impl", None) is not None:
            continue
        handles.append(module.register_forward_pre_hook(mark(name + ".start")))
        handles.append(module.register_forward_hook(mark(name + ".end")))
    rows = []
    try:
        for i in range(count):
            marks.clear()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            actor.act(None, (i + 1) * INTERVAL_NS, precomputed=frames[i % len(frames)])
            end.record()
            end.synchronize()
            row = {"decision": begin.elapsed_time(end)}
            try:  # a stage replayed inside a CUDA graph records no events
                for name in stages:
                    if name + ".start" in marks and name + ".end" in marks:
                        row[name] = sum(
                            a.elapsed_time(b)
                            for a, b in zip(marks[name + ".start"], marks[name + ".end"])
                        )
                if "tower.start" in marks:
                    row["to_tower"] = begin.elapsed_time(marks["tower.start"][0])
                if "memory.end" in marks:
                    row["after_memory"] = marks["memory.end"][-1].elapsed_time(end)
            except ValueError:
                pass
            rows.append(row)
    finally:
        for handle in handles:
            handle.remove()
    keys = sorted({k for row in rows for k in row})
    split = {}
    for k in keys:
        values = sorted(r[k] for r in rows if k in r)
        split[k] = round(values[len(values) // 2], 3)
    # Synchronizing calls per decision.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            for i in range(10):
                actor.act(None, (count + i + 1) * INTERVAL_NS, precomputed=frames[i % len(frames)])
        finally:
            torch.cuda.set_sync_debug_mode("default")
    syncs = [str(w.message).split("\n")[0][:80] for w in caught if "synchroniz" in str(w.message)]
    return split, len(syncs) / 10, sorted(set(syncs))


def profile_pass(actor, frames, count):
    """GPU kernel time and kernel launches per decision, and the heaviest kernels."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(count):
            actor.act(None, (i + 1) * INTERVAL_NS, precomputed=frames[i % len(frames)])
        torch.cuda.synchronize()
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    gpu_us = sum(e.device_time for e in kernels if not e.name.startswith("Memcpy"))
    copy_us = sum(e.device_time for e in kernels if e.name.startswith("Memcpy"))
    by_name = {}
    for e in kernels:
        by_name[e.name] = by_name.get(e.name, 0) + e.device_time
    top = sorted(by_name.items(), key=lambda kv: -kv[1])[:15]
    return {
        "gpu_ms": round(gpu_us / count / 1000, 3),
        "copy_ms": round(copy_us / count / 1000, 3),
        "launches": round(len(kernels) / count, 1),
        "top_kernels_ms": [[name[:90], round(us / count / 1000, 3)] for name, us in top],
    }


def run(args):
    import numpy as np
    import psutil
    import torch

    from hoi4_arena.runner import Actor

    options = {}
    for item in args.option:
        key, _, value = item.partition("=")
        options[key] = json.loads(value)
    frames = load_frames(args.frames)
    started = time.perf_counter()
    actor = Actor(args.checkpoint, args.model, game_speed=5, **options)
    actor.lean = True
    report = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "tower": actor.config["model_path"],
        "options": options,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "node": os.environ.get("FLEET_NODE"),
        "load_s": round(time.perf_counter() - started, 1),
        "compiled_head": bool(getattr(actor, "compiled", False)),
    }
    # Warm up in the gate's own context (a compile or a graph is paid here), then gate.
    started = time.perf_counter()
    gate_pass(actor, frames[:4])
    report["warm_s"] = round(time.perf_counter() - started, 1)
    result = gate_pass(actor, frames)
    if args.write_reference:
        Path(args.write_reference).parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, args.write_reference)
        report["gate"] = {"wrote": args.write_reference}
    if args.reference:
        report["gate"] = compare(result, torch.load(args.reference, weights_only=True))
    # Back to back.
    for i in range(args.warm):
        actor.act(None, (i + 1) * INTERVAL_NS, precomputed=frames[i % len(frames)])
    times = []
    clock = args.warm
    for i in range(args.steps):
        clock += 1
        start = time.perf_counter()
        actor.act(None, clock * INTERVAL_NS, precomputed=frames[i % len(frames)])
        times.append(time.perf_counter() - start)
    report["back_to_back"] = percentiles(times)
    # Paced at 5 Hz, as live: the GPU idles (or draws the game) between decisions.
    times = []
    tick = time.perf_counter()
    for i in range(args.paced):
        tick += INTERVAL_NS / 1e9
        clock += 1
        start = time.perf_counter()
        actor.act(None, clock * INTERVAL_NS, precomputed=frames[i % len(frames)])
        times.append(time.perf_counter() - start)
        time.sleep(max(0.0, tick - time.perf_counter()))
    report["paced"] = percentiles(times) if times else None
    if args.split:
        report["split_ms"], report["syncs_per_decision"], report["syncs"] = split_pass(
            actor, frames, 40
        )
        try:
            report["profile"] = profile_pass(actor, frames, 20)
        except Exception as error:  # noqa: BLE001 - the report says why
            report["profile"] = f"{type(error).__name__}: {error}"
    memory = psutil.Process().memory_info()
    report["peak_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    report["process_private_mib"] = round(getattr(memory, "private", 0) / 2**20)
    report["process_rss_mib"] = round(memory.rss / 2**20)
    report["frames"] = {"file": args.frames, "decisions": len(frames)}
    report["numpy"] = np.__version__
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text)


def ledger(args):
    path = Path(args.ledger)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="\n") as out:
        if new:
            out.write("\t".join(LEDGER_COLUMNS) + "\n")
        for name in args.reports:
            report = json.loads(Path(name).read_text())
            gate = report.get("gate") or {}
            profile = report.get("profile") if isinstance(report.get("profile"), dict) else {}
            row = {
                "when": time.strftime("%Y-%m-%d %H:%M"),
                "label": report["label"],
                "tower": Path(report["tower"]).name,
                "options": json.dumps(report["options"], separators=(",", ":")),
                "p50_ms": report["back_to_back"]["p50_ms"],
                "p95_ms": report["back_to_back"]["p95_ms"],
                "paced_p50_ms": (report.get("paced") or {}).get("p50_ms", ""),
                "paced_p95_ms": (report.get("paced") or {}).get("p95_ms", ""),
                "gpu_ms": profile.get("gpu_ms", ""),
                "launches": profile.get("launches", ""),
                "syncs": report.get("syncs_per_decision", ""),
                "gate": "ref" if "wrote" in gate else ("pass" if gate.get("pass") else "FAIL"),
                "exact": gate.get("exact", ""),
                "max_head_diff": (gate.get("max_diff") or {}).get("heads", ""),
                "peak_mib": report["peak_mib"],
                "note": args.note,
            }
            out.write("\t".join(str(row[c]) for c in LEDGER_COLUMNS) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    frames = sub.add_parser("frames", help="Fixed decisions from a recorded game")
    frames.add_argument("--recording", required=True)
    frames.add_argument("--start", type=int, default=600)
    frames.add_argument("--count", type=int, default=32)
    frames.add_argument("--output", default="bench/frames.npz")
    tower = sub.add_parser("tower4b", help="The policy on the Qwen3.5-4B model's tower")
    tower.add_argument("--checkpoint", required=True)
    tower.add_argument("--tower", required=True)
    tower.add_argument("--output", default="artifacts/bench/bc5-4b.pt")
    tower.add_argument("--tower-output", default="models/qwen3-vit-306m")
    bench = sub.add_parser("run", help="Time and gate the decision")
    bench.add_argument("--checkpoint", required=True)
    bench.add_argument("--model", default=None)
    bench.add_argument("--frames", default="bench/frames.npz")
    bench.add_argument("--label", default="run")
    bench.add_argument("--steps", type=int, default=300)
    bench.add_argument("--paced", type=int, default=150)
    bench.add_argument("--warm", type=int, default=30)
    bench.add_argument("--no-split", dest="split", action="store_false")
    bench.add_argument("--reference")
    bench.add_argument("--write-reference")
    bench.add_argument("--option", action="append", default=[], help="Actor keyword: name=json")
    bench.add_argument("--output")
    book = sub.add_parser("ledger", help="Append reports to the ledger")
    book.add_argument("reports", nargs="+")
    book.add_argument("--note", default="")
    book.add_argument("--ledger", default="bench/decide-ledger.tsv")
    args = parser.parse_args()
    if args.command == "frames":
        make_frames(args.recording, args.start, args.count, args.output)
    elif args.command == "tower4b":
        make_tower4b(args.checkpoint, args.tower, args.output, args.tower_output)
    elif args.command == "run":
        run(args)
    else:
        ledger(args)


if __name__ == "__main__":
    main()

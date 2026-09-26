"""A fixed benchmark of train-bc's throughput, with a correctness gate.

It trains the configuration the learned player's runs use (bc6, bc7: a carried memory over
windows of 32 decisions, two games side by side, the frozen tower read from its cache)
on a fixed slice of real games, from a fixed checkpoint and seed, through train.train_bc
itself: its `probe` marks each phase of a step, and stops the run after the timed steps.

    python scripts/bench_train.py prepare
    python scripts/bench_train.py run --label baseline --repeat 3 [--split] [knobs]
    python scripts/bench_train.py run --label ref --save-reference

Each repeat is a process of its own. It reports decisions trained a second over the timed
steps (after `--warmup` steps), where a step's time goes (loader wait, the copy to the
card, forward, backward, optimizer, and the logging's sync), the GPU's use sampled by
nvidia-smi, and its peak memory. `--split` synchronizes the card at each mark, so the
phases are real but the run is slower; without it only the loader wait is split out.

The gate: every step's losses and the trainable weights after the last step, against a
reference saved from the unmodified code (`--save-reference`). "exact" is bit for bit;
"close" is within the tolerances below, which were set from how far two runs of the
reference itself, and the reference with cuDNN's autotuner on, land apart; "differs" is
anything else. A knob that changes what is trained (the batch, the loader's workers,
which split the games among themselves) differs by design and is reported as such.

Knobs: --workers, --batch-size, --gpu-views, --gpu-memory (a fraction, or "none"),
--threads (the training process's), --worker-threads (each loader worker's),
--set KEY=JSON (any other train_bc option) and --patch module.attr=JSON (for example
torch.backends.cudnn.benchmark=true), set in the training process before it starts.

Results are appended to the ledger (--ledger, bench/training-ledger.tsv) when --label
is given.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _main_checkout():
    """The main checkout, where models/ and artifacts/ live, also from a git worktree."""
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        return Path(common).parent
    except (OSError, subprocess.CalledProcessError):
        return REPO


MAIN = _main_checkout()
# The slice: eight training games and one held out (train_bc builds its validation set
# even when it stops before using it), all with their tower cache on the fast drive.
TRAIN = [
    "scripted-peer-20260923-224820",
    "scripted-peer-20260923-231020",
    "scripted-peer-20260923-233317",
    "scripted-peer-20260923-235144",
    "scripted-peer-20260924-000016",
    "scripted-peer-20260924-001658",
    "scripted-peer-20260924-002518",
    "scripted-peer-20260924-011125",
]
VALIDATION = ["scripted-peer-20260923-234139"]
# bc6's options (scratchpad learned6_bc6.sh), less the epochs, resume and saving.
BC6 = {
    "sources": ("scripted",),
    "lead_in": 0,
    "drop_keys": (0x20,),
    "look_before_click": True,
    "state_weight": 0.5,
    "order_weight": 0.2,
    "train_last": 0,
    "press_weight": 4.0,
    "carry": True,
    "sequence": 32,
    "epochs": 1,
    "batch_size": 2,
    "workers": 1,
    "save_every": 1e9,
    "drop_parking": True,
    "setup_weight": 4.0,
    "held_previous": True,
}
CAMERA_SINCE = "20260924-131000"
# The gate's tolerances (see the module's docstring and the ledger's first rows).
LOSS_TOLERANCE = 5e-2  # largest relative difference of any step's loss
WEIGHT_TOLERANCE = 5e-2  # |w - w_ref| over |w_ref - w_init|, pooled over the weights
GROUP_TOLERANCE = 0.1  # the same for the group of weights that drifted most


def prepare(args):
    """The slice as a data folder of links to the recordings, with its own splits."""
    data = Path(args.work) / "data"
    data.mkdir(parents=True, exist_ok=True)
    source = Path(args.source)
    for name in TRAIN + VALIDATION:
        link = data / name
        if link.exists():
            continue
        target = (source / name).resolve()
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True,
                           capture_output=True)  # fmt: skip
        else:
            link.symlink_to(target, target_is_directory=True)
    splits = {name: "train" for name in TRAIN} | {name: "validation" for name in VALIDATION}
    (data / "splits.json").write_text(json.dumps(splits, indent=1))
    print(f"prepared {data}: {len(TRAIN)} training games, {len(VALIDATION)} held out")


class Probe:
    """train_bc's probe: times each step's phases and stops after the last one."""

    def __init__(self, warmup, steps, split, sampler):
        import torch

        self.torch, self.split, self.sampler = torch, split, sampler
        self.warmup, self.steps = warmup, warmup + steps
        self.first = None  # when the first batch arrived
        self.rows, self.times, self.decisions = [], [], []
        self.phases = []  # per step, {phase: seconds}
        self.current = {}
        self.last = time.perf_counter()
        self.final = None

    def mark(self, name):
        if self.split:
            self.torch.cuda.synchronize()
        self.current[name] = time.perf_counter()
        if self.first is None:
            self.first = self.current[name]

    def step(self, step, batch, row, modules):
        now = time.perf_counter()
        marks = [("loaded", "loader"), ("device", "device"), ("forward", "forward"),
                 ("backward", "backward"), ("optimizer", "optimizer")]  # fmt: skip
        phases, before = {}, self.last
        for mark, phase in marks:
            if mark in self.current:
                phases[phase] = self.current[mark] - before
                before = self.current[mark]
        phases["log"] = now - before
        self.phases.append(phases)
        self.current, self.last = {}, now
        self.rows.append(row)
        self.times.append(now)
        actions = batch["actions"]
        self.decisions.append(int(actions.shape[0] * actions.shape[1]))
        if len(self.rows) == self.warmup:
            self.sampler.start()
        if len(self.rows) >= self.steps:
            self.sampler.stop()
            self.final = trainable(modules)
            return True
        return False


class Sampler:
    """nvidia-smi's GPU use and memory, sampled every 100 ms while running."""

    def __init__(self):
        self.util, self.memory, self.process = [], [], None

    def start(self):
        try:
            self.process = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits", "-lms", "100"],
                stdout=subprocess.PIPE, text=True,
            )  # fmt: skip
        except OSError:
            return
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            try:
                util, memory = (float(v) for v in line.split(","))
            except ValueError:
                continue
            self.util.append(util)
            self.memory.append(memory)

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.kill()


def trainable(modules):
    return {
        f"{name}.{key}": value.detach().float().cpu().clone()
        for name, module in modules.items()
        for key, value in module.named_parameters()
        if value.requires_grad
    }


def _worker_threads(_worker_id, threads):
    import torch

    torch.set_num_threads(threads)


def one(args):
    """One repeat, in this process: prints its result as JSON on the last line."""
    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    from hoi4_arena import dataset, train
    from hoi4_arena.cli import local_time
    from hoi4_arena.models import configure_precision, limit_gpu_memory

    configure_precision(False)
    if args.gpu_memory != "none":
        limit_gpu_memory(float(args.gpu_memory))
    if args.worker_threads:
        # A function of this script, which a spawned worker imports by name.
        dataset.worker_threads = functools.partial(_worker_threads, threads=args.worker_threads)
    for item in args.patch:
        path, value = item.split("=", 1)
        parts = path.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            try:
                owner = importlib.import_module(".".join(parts[:cut]))
                break
            except ImportError:
                continue
        for part in parts[cut:-1]:
            owner = getattr(owner, part)
        setattr(owner, parts[-1], json.loads(value))
    options = dict(BC6)
    options.update(
        batch_size=args.batch_size,
        workers=args.workers,
        gpu_views=args.gpu_views,
        seed=args.seed,
        init=args.init,
        tower_cache=args.cache,
        camera_since=local_time(CAMERA_SINCE),
    )
    for item in args.set:
        key, value = item.split("=", 1)
        options[key] = json.loads(value)
    for key in ("sources", "drop_keys"):
        options[key] = tuple(options[key])
    work = Path(args.work)
    output = work / "runs" / f"{args.label or 'run'}-{os.getpid()}-{time.time_ns()}"
    sampler = Sampler()
    probe = Probe(args.warmup, args.steps, args.split, sampler)
    initial = {}
    original = train.Progress.start

    def start(self, modules, optimizer):
        initial.update(trainable(modules))
        return original(self, modules, optimizer)

    train.Progress.start = start
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    train.train_bc(work / "data", args.model, output, probe=probe, **options)
    total = time.perf_counter() - began
    if len(probe.rows) < args.warmup + args.steps:
        raise SystemExit(f"the slice ran out after {len(probe.rows)} steps")
    w = args.warmup
    seconds = probe.times[-1] - probe.times[w - 1]
    decisions = sum(probe.decisions[w:])
    phases = {}
    for key in ("loader", "device", "forward", "backward", "optimizer", "log"):
        values = [p[key] for p in probe.phases[w:] if key in p]
        if values:
            phases[key] = statistics.mean(values)
    result = {
        "decisions_per_s": decisions / seconds,
        "step_s": seconds / args.steps,
        "decisions_per_step": decisions / args.steps,
        "phases_s": phases,
        "gpu_util_mean": statistics.mean(sampler.util) if sampler.util else None,
        "gpu_util_samples": len(sampler.util),
        "gpu_memory_used_max_mib": max(sampler.memory) if sampler.memory else None,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
        if torch.cuda.is_available()
        else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20
        if torch.cuda.is_available()
        else None,
        "setup_s": probe.first - began,
        "total_s": total,
    }
    state = {"rows": probe.rows, "final": probe.final, "initial": initial}
    if args.save_reference:
        torch.save(state, work / "reference.pt")
        result["gate"] = "saved as the reference"
    else:
        result["gate"] = gate(state, work / "reference.pt")
    if args.keep_state:
        torch.save(state, Path(args.keep_state))
    print(json.dumps(result))


def gate(state, path):
    """How this run's losses and weights compare with the reference's.

    `loss_rel_max`: the largest relative difference of any loss of any step. `weight_drift`:
    how far the trained weights are from the reference's, over how far the reference's
    moved from where both started, pooled; `group_drift_max` the same for the group of
    weights that drifted most (a layer of the policy, or a head), so a part that stopped
    training cannot hide in the pool. `forward_exact`: the first step's losses, before
    any update, are the same bits.
    """
    import torch

    if not path.exists():
        return {"verdict": "no reference"}
    reference = torch.load(path, weights_only=False)
    rows, expected = state["rows"], reference["rows"]
    if len(rows) != len(expected) or state["final"].keys() != reference["final"].keys():
        return {"verdict": "differs", "why": "another number of steps or other weights"}
    final, start, mine = reference["final"], reference["initial"], state["final"]
    exact = rows == expected and all(torch.equal(mine[k], final[k]) for k in final)
    keys = [key for key in expected[0] if key not in ("epoch", "step")]
    loss = max(
        abs(a[key] - b[key]) / max(abs(b[key]), 1e-6)
        for a, b in zip(rows, expected)
        for key in keys
    )
    first = max(abs(rows[0][k] - expected[0][k]) / max(abs(expected[0][k]), 1e-6) for k in keys)
    groups = {}
    for k in final:
        group = ".".join(k.split(".")[:2])
        moved = (final[k] - start[k]).double().square().sum()
        apart = (mine[k] - final[k]).double().square().sum()
        total = groups.setdefault(group, [0.0, 0.0])
        total[0] += float(moved)
        total[1] += float(apart)
    drift = sum(a for _, a in groups.values()) / max(sum(m for m, _ in groups.values()), 1e-30)
    drift = drift**0.5
    per_group = {g: (a / m) ** 0.5 for g, (m, a) in groups.items() if m > 0}
    worst = max(per_group, key=per_group.get)
    close = (
        loss <= LOSS_TOLERANCE and drift <= WEIGHT_TOLERANCE and per_group[worst] <= GROUP_TOLERANCE
    )
    return {
        "verdict": "exact" if exact else "close" if close else "differs",
        "forward_exact": first == 0,
        "loss_rel_max": loss,
        "first_step_loss_rel": first,
        "weight_drift": drift,
        "group_drift_max": per_group[worst],
        "group": worst,
    }


def run(args):
    """`--repeat` repeats, each its own process, summarized and put in the ledger."""
    command = [sys.executable, __file__, "one", *sys.argv[2:]]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("PYTHONUNBUFFERED", "1")
    results = []
    for i in range(args.repeat):
        done = subprocess.run(command, env=env, capture_output=True, text=True, cwd=REPO)
        lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
        if done.returncode or not lines:
            sys.stderr.write(done.stdout[-3000:] + done.stderr[-6000:])
            raise SystemExit(f"repeat {i} failed")
        result = json.loads(lines[-1])
        results.append(result)
        print(json.dumps(result), flush=True)
        if args.save_reference:
            break
    rates = [r["decisions_per_s"] for r in results]
    summary = {
        "label": args.label,
        "decisions_per_s": statistics.mean(rates),
        "sd": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        "runs": [round(r, 2) for r in rates],
        "step_s": statistics.mean(r["step_s"] for r in results),
        "gpu_util": statistics.mean(r["gpu_util_mean"] or 0 for r in results),
        "peak_reserved_mib": max(r["peak_reserved_mib"] or 0 for r in results),
        "gates": sorted({r["gate"]["verdict"] if isinstance(r["gate"], dict) else r["gate"]
                         for r in results}),
    }  # fmt: skip
    print("SUMMARY " + json.dumps(summary))
    if args.label and args.ledger:
        ledger = Path(args.ledger)
        ledger.parent.mkdir(parents=True, exist_ok=True)
        head = ("when\tlabel\tknobs\tdecisions_per_s\tsd\truns\tstep_s\tgpu_util\t"
                "peak_reserved_mib\tphases_s\tgate\tdecision\tnote\n")  # fmt: skip
        if not ledger.exists():
            ledger.write_text(head)
        knobs = " ".join(sys.argv[2:])
        phases = {
            k: round(statistics.mean(r["phases_s"].get(k, 0) for r in results), 3)
            for k in results[0]["phases_s"]
        }
        gates = [r["gate"] for r in results]
        with ledger.open("a") as handle:
            handle.write("\t".join(str(x) for x in (
                time.strftime("%Y-%m-%d %H:%M"), args.label, knobs,
                round(summary["decisions_per_s"], 2), round(summary["sd"], 2),
                summary["runs"], round(summary["step_s"], 3), round(summary["gpu_util"], 1),
                round(summary["peak_reserved_mib"]), json.dumps(phases),
                json.dumps(gates), args.decision, args.note,
            )) + "\n")  # fmt: skip


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["prepare", "run", "one"])
    parser.add_argument("--work", default=str(REPO / "artifacts" / "bench-train"))
    parser.add_argument("--source", default="C:/hoi4-data/scripted-v6")
    parser.add_argument("--cache", default="C:/hoi4-cache/tower-v2s5")
    parser.add_argument("--model", default=str(MAIN / "models" / "qwen3-vit-88m"))
    parser.add_argument("--init", default=str(MAIN / "artifacts/learned/bc5/epoch-0000.pt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gpu-views", action="store_true")
    parser.add_argument("--gpu-memory", default="0.5")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--worker-threads", type=int, default=0)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--patch", action="append", default=[])
    parser.add_argument("--label")
    parser.add_argument("--note", default="")
    parser.add_argument("--decision", default="measured")
    parser.add_argument("--ledger", default=str(REPO / "bench" / "training-ledger.tsv"))
    parser.add_argument("--save-reference", action="store_true")
    parser.add_argument("--keep-state")
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "one": one}[args.command](args)


if __name__ == "__main__":
    main()

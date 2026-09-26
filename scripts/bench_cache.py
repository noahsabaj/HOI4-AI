"""A fixed benchmark of `hoi4-arena cache-tower`: frames a second through the whole build.

    python scripts/bench_cache.py DATA OUT --model TOWER_FOLDER --label NAME
        [--repeats 3] [--option key=value ...] [--reference REF] [--save-reference]
        [--profile] [--ledger bench/cache-ledger.tsv]

DATA holds `warm/` (a short recording, cached first so the timed runs start warm: CUDA
context, kernels chosen, anything compiled) and `timed/` (the recordings timed). Each
repeat caches `timed/` afresh into OUT through tower_cache.cache_tower, as the command
does, and its frames a second are the frames kept over the build's own `seconds` (from
the first recording to the last, after the tower is loaded). GPU use is sampled with
nvidia-smi through the timed runs.

--option passes keyword arguments on to cache_tower (a Python literal, or a string).
--reference compares the first repeat's cache with a reference cache (check_cache.py):
bit-identical, or within one int8 step, with the largest and RMS errors. --save-reference
keeps the first repeat's cache as REF instead. --profile times the stages apart on the
first timed recording: decode (ffmpeg into rgb24), views (the quadrants, on the GPU), the
tower's forward at the default batch, int8 quantisation with the copy back, and the write.

Each run appends a row to the ledger. Its output folders are deleted afterwards (not the
reference).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from check_cache import compare  # noqa: E402

from hoi4_arena import tower_cache  # noqa: E402

COLUMNS = [
    "when", "label", "commit", "options", "repeats", "fps_each", "fps_median", "gpu_util",
    "peak_gb", "cpu_s_per_frame", "gate", "identical", "grid_max_steps", "grid_rms_rel",
    "summary_max_steps", "notes",
]  # fmt: skip


class GpuSampler:
    """nvidia-smi's utilization and memory every 250 ms while running."""

    def __init__(self):
        self.rows, self.on = [], False
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-lms", "250"],
            stdout=subprocess.PIPE, text=True,
        )  # fmt: skip
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            if self.on:
                try:
                    util, mem = (float(x) for x in line.split(","))
                    self.rows.append((util, mem))
                except ValueError:
                    pass

    def stop(self):
        self.proc.kill()


def parse_options(pairs):
    options = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        try:
            options[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            options[key] = value
    return options


def commit():
    """The code's commit, or BENCH_COMMIT when the benchmark runs from a copy of it."""
    if os.environ.get("BENCH_COMMIT"):
        return os.environ["BENCH_COMMIT"]
    root = Path(__file__).resolve().parents[1]
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()  # fmt: skip
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "src"],
                           capture_output=True, text=True).stdout.strip()  # fmt: skip
    return head + ("+dirty" if dirty else "")


def profile(root, model, device="cuda", frames=96, batch=8):
    """Each stage timed alone on the first `frames` frames of recording `root`, ms a frame."""
    from hoi4_arena.dataset import normalize, parse_cursor, views
    from hoi4_arena.models import ScreenEncoder

    manifest = json.loads((root / "manifest.json").read_text())
    width, height = manifest["width"], manifest["height"]
    lines = (root / "frames.jsonl").read_text().splitlines()
    size = width * height * 3
    out = {}
    began = time.perf_counter()
    decoder = subprocess.Popen(
        [shutil.which("ffmpeg"), "-v", "error", "-i", str(root / "screen.mkv"),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, bufsize=0,
    )  # fmt: skip
    raw = []
    for _ in range(frames):
        buffer = bytearray(size)
        view, got = memoryview(buffer), 0
        while got < size:
            got += decoder.stdout.readinto(view[got:])
        raw.append(np.frombuffer(buffer, np.uint8).reshape(height, width, 3))
    decoder.kill()
    decoder.wait()
    out["decode_ms"] = (time.perf_counter() - began) / frames * 1e3
    torch.cuda.synchronize()
    began = time.perf_counter()
    quads = []
    for index, frame in enumerate(raw):
        cursor = parse_cursor(json.loads(lines[index]).get("cursor"))
        quads.append(views(frame, None, device=device, cursor=cursor).quadrants)
    torch.cuda.synchronize()
    out["views_ms"] = (time.perf_counter() - began) / frames * 1e3
    encoder = ScreenEncoder(model).to(device).eval().requires_grad_(False)
    autocast = {"device_type": "cuda", "dtype": torch.bfloat16}
    grids = []
    for rep in range(2):  # the first pass warms up
        torch.cuda.synchronize()
        began = time.perf_counter()
        for first in range(0, frames, batch):
            x = normalize(torch.stack(quads[first : first + batch])).permute(0, 1, 4, 2, 3)
            with torch.no_grad(), torch.autocast(**autocast):
                summary, grid = tower_cache.read_frozen(encoder, x)
            if rep:
                grids.append((summary, grid))
        torch.cuda.synchronize()
    out["tower_ms"] = (time.perf_counter() - began) / frames * 1e3
    began = time.perf_counter()
    quantised = []
    for summary, grid in grids:
        g = grid.float()
        scale = g.abs().amax((-2, -1)).clamp_min(1e-6) / 127
        values = (g / scale[..., None, None]).round().clamp(-127, 127).to(torch.int8)
        quantised.append((values.cpu().numpy(), scale.to(torch.float16).cpu().numpy(),
                          summary.to(torch.bfloat16).view(torch.int16).cpu().numpy()))  # fmt: skip
    out["quantise_ms"] = (time.perf_counter() - began) / frames * 1e3
    target = PROFILE_DIR
    target.mkdir(parents=True, exist_ok=True)
    dim = encoder.dim
    began = time.perf_counter()
    g = np.lib.format.open_memmap(target / "g.npy", "w+", np.int8, (frames, dim, 32, 32))
    s = np.lib.format.open_memmap(target / "s.npy", "w+", np.float16, (frames, dim))
    m = np.lib.format.open_memmap(target / "m.npy", "w+", np.int16, (frames, dim))
    first = 0
    for values, scale, summary in quantised:
        n = len(values)
        g[first : first + n], s[first : first + n], m[first : first + n] = values, scale, summary
        first += n
    g.flush(), s.flush(), m.flush()
    del g, s, m
    out["write_ms"] = (time.perf_counter() - began) / frames * 1e3
    shutil.rmtree(target, ignore_errors=True)
    return {k: round(v, 2) for k, v in out.items()}


PROFILE_DIR = Path(".")


def main():
    global PROFILE_DIR
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--save-reference", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "bench" / "cache-ledger.tsv",
    )
    args = parser.parse_args()
    options = parse_options(args.option)
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    PROFILE_DIR = out / "profile"
    checkpoint = out / "checkpoint.pt"
    torch.save(
        {"policy": {}, "config": {"model_path": args.model, "variant": "screen"}}, checkpoint
    )
    common = {"model_path": args.model, "keep_free_gb": 1.0, **options}
    tower_cache.cache_tower(args.data / "warm", checkpoint, out / "warm", **common)
    sampler = GpuSampler()
    fps, util, peak, cpu = [], [], [], []
    try:
        for repeat in range(args.repeats):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            sampler.rows.clear()
            sampler.on = True
            cpu0 = time.process_time()
            report = tower_cache.cache_tower(
                args.data / "timed", checkpoint, out / f"timed-{repeat}", **common
            )
            cpu.append((time.process_time() - cpu0) / report["kept"])
            sampler.on = False
            fps.append(report["kept"] / report["seconds"])
            util.append(
                statistics.mean(u for u, _ in sampler.rows) if sampler.rows else float("nan")
            )
            peak.append(
                torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            )
            print(f"repeat {repeat}: {report['kept']} frames in {report['seconds']:.1f} s, "
                  f"{fps[-1]:.2f} frames/s, GPU {util[-1]:.0f}%, peak {peak[-1]:.2f} GB", flush=True)  # fmt: skip
            if repeat:
                shutil.rmtree(out / f"timed-{repeat}", ignore_errors=True)
    finally:
        sampler.stop()
    result = {
        "gate": "",
        "identical": "",
        "grid_max_steps": "",
        "grid_rms_rel": "",
        "summary_max_steps": "",
    }
    if args.save_reference:
        if args.reference.exists():
            shutil.rmtree(args.reference)
        shutil.move(str(out / "timed-0"), str(args.reference))
        result["gate"] = "reference"
    elif args.reference:
        check = compare(args.reference, out / "timed-0")
        print("gate:", json.dumps(check), flush=True)
        result = {
            "gate": "pass" if check["passed"] else "FAIL",
            "identical": check["identical"],
            "grid_max_steps": check["grid_max_steps"],
            "grid_rms_rel": f"{check['grid_rms_rel']:.2e}",
            "summary_max_steps": check["summary_max_steps"],
        }
    notes = args.notes
    if args.profile:
        first = sorted(p.parent for p in (args.data / "timed").glob("*/manifest.json"))[0]
        split = profile(first, args.model)
        print("profile (ms a frame):", json.dumps(split), flush=True)
        notes = (notes + " " if notes else "") + "profile " + json.dumps(split)
    row = {
        "when": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "label": args.label,
        "commit": commit(),
        "options": json.dumps(options) if options else "",
        "repeats": args.repeats,
        "fps_each": " ".join(f"{f:.2f}" for f in fps),
        "fps_median": f"{statistics.median(fps):.2f}",
        "gpu_util": f"{statistics.mean(util):.0f}",
        "peak_gb": f"{max(peak):.2f}",
        "cpu_s_per_frame": f"{statistics.mean(cpu):.3f}",
        **result,
        "notes": notes,
    }
    print(json.dumps(row), flush=True)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    new = not args.ledger.exists()
    with args.ledger.open("a", encoding="utf-8", newline="\n") as ledger:
        if new:
            ledger.write("\t".join(COLUMNS) + "\n")
        ledger.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")
    shutil.rmtree(out, ignore_errors=True)
    return 0 if result["gate"] != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())

"""Two actors in one process, deciding in turn at the live 5 Hz, so both meet the same GPU
and game conditions: `diag_interleave.py a|b ticks 'optsA' 'optsB'` (opts: name=json;...)."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import torch  # noqa: E402
from bench_decide import INTERVAL_NS, load_frames, percentiles  # noqa: E402

from hoi4_arena.runner import Actor  # noqa: E402

tower, ticks = sys.argv[1], int(sys.argv[2])
checkpoint = {"a": "artifacts/learned/bc5/epoch-0000.pt", "b": "artifacts/bench/bc5-4b.pt"}[tower]
frames = load_frames("bench/frames.npz")


def options(text):
    out = {}
    for item in filter(None, text.split(";")):
        key, _, value = item.partition("=")
        out[key] = json.loads(value)
    return out


specs = sys.argv[3:]
actors = []
for spec in specs:
    actor = Actor(checkpoint, None, game_speed=5, **options(spec))
    actor.lean = True
    for i in range(20):
        actor.act(None, (i + 1) * INTERVAL_NS, precomputed=frames[i % 32])
    actors.append(actor)
torch.cuda.synchronize()
times = [[] for _ in actors]
tick = time.perf_counter()
for i in range(ticks):
    tick += INTERVAL_NS / 1e9
    k = i % len(actors)
    start = time.perf_counter()
    actors[k].act(None, (100 + i) * INTERVAL_NS, precomputed=frames[i % 32])
    times[k].append(time.perf_counter() - start)
    time.sleep(max(0.0, tick - time.perf_counter()))
for spec, t in zip(specs, times):
    print(f"{tower} [{spec or 'base'}] paced {percentiles(t)}", flush=True)
# And back to back, in turn.
times = [[] for _ in actors]
for i in range(ticks):
    k = i % len(actors)
    start = time.perf_counter()
    actors[k].act(None, (1000 + i) * INTERVAL_NS, precomputed=frames[i % 32])
    times[k].append(time.perf_counter() - start)
for spec, t in zip(specs, times):
    print(f"{tower} [{spec or 'base'}] b2b {percentiles(t)}", flush=True)

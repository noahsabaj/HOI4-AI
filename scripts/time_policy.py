"""Time the policy's decision step at today's sizes, for one actor and for two.

The last measurements (STATUS.md, Performance) predate the Qwen3.5 tower and the 16:9
views: one actor took 135 ms of a 200 ms tick with the game running, and two actors on one
GPU 243.6 ms, which does not fit. This times the same step now, on random pixels (timing
does not depend on what the screen shows), with the weights the live actor uses
(frozen ones in bfloat16, autocast), under inference mode:

- `perceive`: the Qwen3.5 tower on the tiled quadrants, the detail and fovea readers and
  the cells, for batch 1 and for batch 2 (two actors' frames in one pass);
- `step`: perceive, then the memory and the action head (eager, as when its CUDA graph
  cannot be compiled), batch 1 and 2;
- `graph`: the tower captured once as a CUDA graph and replayed. Replaying runs the same
  kernels, so the numbers it produces are the eager ones, which the test below checks.

    python scripts/time_policy.py --output artifacts/time-policy.json

Run it with the GPU otherwise idle: a training run beside it doubles every number.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hoi4_arena.actions import SLOTS  # noqa: E402
from hoi4_arena.dataset import DETAIL_SIZE, FOVEA_SIZE  # noqa: E402
from hoi4_arena.models import Policy, ScreenEncoder, halve_frozen  # noqa: E402


def percentiles(times):
    ordered = sorted(times)
    return {
        "p50_ms": round(ordered[len(ordered) // 2] * 1000, 2),
        "p95_ms": round(ordered[int(len(ordered) * 0.95)] * 1000, 2),
    }


def timed(fn, steps, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(steps):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return percentiles(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen3-vit-88m")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", default="artifacts/time-policy.json")
    args = parser.parse_args()
    torch.manual_seed(0)
    device = "cuda"
    policy = halve_frozen(Policy(ScreenEncoder(args.model))).to(device).eval()
    report = {"gpu": torch.cuda.get_device_name(), "steps": args.steps}
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    for batch in (1, 2):
        quadrants = torch.randn(batch, 4, 3, *DETAIL_SIZE, device=device)
        fovea = torch.randn(batch, 3, FOVEA_SIZE, FOVEA_SIZE, device=device)
        clip = torch.zeros(batch, 3, 1, 8, 8, device=device)  # the tower does not read it
        previous = torch.zeros(batch, SLOTS, 3, dtype=torch.long, device=device)
        speed = torch.full((batch,), 5, device=device)
        hidden = torch.zeros(batch, policy.memory_dim, device=device)

        def perceive():
            with torch.inference_mode(), autocast:
                return policy.perceive(clip, quadrants, fovea)

        def step():
            with torch.inference_mode(), autocast:
                h, _, _, cells = policy(clip, quadrants, fovea, previous, speed, hidden)
                return policy.actor(h, cells, deterministic=True)

        report[f"perceive_b{batch}"] = timed(perceive, args.steps)
        report[f"step_b{batch}"] = timed(step, args.steps)
        # The tower alone, eager and replayed from a CUDA graph.
        tower = policy.encoder
        with torch.inference_mode(), autocast:
            eager = tower(clip, quadrants)[1].float()

        def tower_eager():
            with torch.inference_mode(), autocast:
                return tower(clip, quadrants)

        report[f"tower_eager_b{batch}"] = timed(tower_eager, args.steps)
        try:
            static = quadrants.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.inference_mode(), autocast:
                for _ in range(3):
                    tower(clip, static)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.inference_mode(), autocast:
                out = tower(clip, static)
            graph.replay()
            torch.cuda.synchronize()
            report[f"tower_graph_b{batch}"] = timed(graph.replay, args.steps)
            report[f"tower_graph_b{batch}_max_diff"] = float((out[1].float() - eager).abs().max())
        except Exception as error:  # noqa: BLE001 - the report says why a graph failed
            report[f"tower_graph_b{batch}"] = f"{type(error).__name__}: {error}"
    # The live actor compiles its head into a CUDA graph (runner.Actor._compile_head).
    try:
        import triton

        report["triton"] = triton.__version__
        policy.actor.compile(mode="reduce-overhead", dynamic=False)
        for batch in (1, 2):
            quadrants = torch.randn(batch, 4, 3, *DETAIL_SIZE, device=device)
            fovea = torch.randn(batch, 3, FOVEA_SIZE, FOVEA_SIZE, device=device)
            clip = torch.zeros(batch, 3, 1, 8, 8, device=device)
            previous = torch.zeros(batch, SLOTS, 3, dtype=torch.long, device=device)
            speed = torch.full((batch,), 5, device=device)
            hidden = torch.zeros(batch, policy.memory_dim, device=device)

            def compiled_step():
                with torch.inference_mode(), autocast:
                    h, _, _, cells = policy(clip, quadrants, fovea, previous, speed, hidden)
                    return policy.actor(h, cells, deterministic=True)

            report[f"step_compiled_head_b{batch}"] = timed(compiled_step, args.steps, warm=20)
            # PPO compares the likelihood stored at collection with the one computed at the
            # update, so the compiled head must give the eager head's numbers exactly.
            # Module.compile() compiles in place; calling forward itself runs eagerly.
            eager_head = policy.actor.forward
            with torch.inference_mode(), autocast:
                h, _, _, cells = policy(clip, quadrants, fovea, previous, speed, hidden)
                picked, logp, _ = policy.actor(h, cells, deterministic=True)
                eager_picked, eager_logp, _ = eager_head(h, cells, deterministic=True)
                scored = policy.actor(h, cells, actions=eager_picked)[1]
                eager_scored = eager_head(h, cells, actions=eager_picked)[1]
            report[f"compiled_head_b{batch}_same_actions"] = bool(torch.equal(picked, eager_picked))
            report[f"compiled_head_b{batch}_logp_max_diff"] = float(
                max((logp - eager_logp).abs().max(), (scored - eager_scored).abs().max())
            )
    except Exception as error:  # noqa: BLE001
        report["step_compiled_head"] = f"{type(error).__name__}: {error}"
    report["peak_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

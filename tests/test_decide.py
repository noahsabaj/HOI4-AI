"""The live decision's speedups keep its numbers: the trimmed tower and the one-graph
decision are exact (scripts/bench_decide.py's gate, pinned here), and the gate's own
per-head scoring is the head's. Fast mode attends through fast_attention's Triton kernel,
compiled and inside the decision's CUDA graph, and the exact path never does."""

import copy
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import torch

from hoi4_arena import fast_attention
from hoi4_arena.actions import GRID, SLOTS
from hoi4_arena.fast import FastTower, apply_fast, lean_tower
from hoi4_arena.models import CELL_DIM, ActionHead, Policy, ScreenEncoder, halve_frozen
from hoi4_arena.runner import Actor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _count_triton_attention(monkeypatch):
    """The calls that reach fast_attention's Triton kernel from Python, counted."""
    calls = []
    kernel = getattr(fast_attention, "fast_attention", None)

    def counted(q, k, v):
        calls.append(q.shape)
        return kernel(q, k, v)

    monkeypatch.setattr(fast_attention, "fast_attention", counted, raising=False)
    return calls


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=needs_cuda)])
def test_the_trimmed_tower_computes_the_same_numbers(device, monkeypatch):
    """The exact path: timm's own tower and the trimmed one agree bit for bit, on the GPU
    too, where fast mode's Triton kernel is at hand and must not be reached."""
    calls = _count_triton_attention(monkeypatch)
    torch.manual_seed(0)
    tower = halve_frozen(ScreenEncoder(pretrained=False, size=(64, 128))).eval()
    tower.to(device).requires_grad_(False)
    trimmed = copy.deepcopy(tower)
    assert lean_tower(trimmed)
    quadrants = torch.randn(1, 4, 3, 32, 64, device=device)
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        for _ in range(2):  # the second call reads the cached position tables
            summary, grid = tower(None, quadrants)
            lean_summary, lean_grid = trimmed(None, quadrants)
            assert torch.equal(summary, lean_summary) and torch.equal(grid, lean_grid)
    assert not calls


def _trimmed_tower():
    torch.manual_seed(0)
    tower = halve_frozen(ScreenEncoder(pretrained=False, size=(64, 128))).eval()
    tower.cuda().requires_grad_(False)
    assert lean_tower(tower)
    return tower


@needs_cuda
def test_fast_mode_attends_through_the_triton_kernel(monkeypatch):
    """FastTower(attention="triton") sends every block's float16 attention to
    fast_attention's kernel; without it PyTorch's kernel computes it."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
    calls = _count_triton_attention(monkeypatch)
    plain = FastTower(_trimmed_tower(), half=True)
    fast = FastTower(_trimmed_tower(), half=True, attention="triton")
    blocks = fast.inner.model.blocks
    assert fast.attention == "triton" and all(b.attn.triton_attention for b in blocks)
    assert not any(getattr(b.attn, "triton_attention", False) for b in plain.inner.model.blocks)
    # timm's own attention (an untrimmed tower) is not reached, and FastTower says so.
    untrimmed = ScreenEncoder(pretrained=False, size=(64, 128))
    assert FastTower(untrimmed, attention="triton").attention is None
    quadrants = torch.randn(1, 4, 3, 64, 128, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        summary, grid = plain(None, quadrants)
        assert not calls
        fast_summary, fast_grid = fast(None, quadrants)
    assert len(calls) == len(blocks)
    for a, b in ((summary, fast_summary), (grid, fast_grid)):
        assert ((a.float() - b.float()).norm() / a.float().norm()) < 2e-2


def test_a_tower_of_another_shape_is_left_alone():
    assert not lean_tower(torch.nn.Linear(2, 2))


def test_the_gate_scores_each_head_as_the_head_does():
    from bench_decide import head_logps

    torch.manual_seed(0)
    head = ActionHead(memory_dim=32, look=True).eval()
    memory, cells = torch.randn(3, 32), torch.randn(3, GRID, CELL_DIM)
    noise = torch.zeros(3, head.noise_dim)
    with torch.no_grad():
        actions, logp, _ = head(memory, cells, noise=noise)
        heads = head_logps(head, memory, cells, actions, noise)
    move = (actions[..., 0] == 1).float()
    summed = heads[..., 0] + move * (heads[..., 1] + heads[..., 2])
    assert heads.shape == (3, SLOTS, 3)
    assert torch.allclose(summed.sum(1), logp, atol=1e-5)


def _actor(policy, **fields):
    actor = Actor.__new__(Actor)
    actor.policy, actor.config, actor.deterministic = policy, {"objective": "bc"}, False
    actor.device, actor.hidden, actor.compiled, actor.speed = "cuda", None, False, 5
    actor.previous = np.zeros((SLOTS, 3), dtype=np.int64)
    actor.history, actor.recent, actor.held, actor.held_previous = deque(maxlen=64), None, (), False
    actor.temperature, actor.pointer_temperature, actor.point = 1.0, 1.0, False
    actor.lean, actor.static = True, None
    for key, value in fields.items():
        setattr(actor, key, value)
    return actor


@needs_cuda
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
def test_a_decision_replayed_as_one_graph_is_the_eager_decision(fast, monkeypatch):
    """In fast mode too, its tower compiled as the live actor compiles it: the Triton
    attention is traced into inductor's graph, one op a block, and captured with the rest
    of the decision."""
    from hoi4_arena.dataset import DETAIL_SIZE, FOVEA_SIZE
    from hoi4_arena.layout import VIEW_SIZE

    torch.manual_seed(0)
    policy = Policy(ScreenEncoder(pretrained=False), look=True).eval()
    policy = halve_frozen(policy).cuda().requires_grad_(False)
    calls = [] if fast else _count_triton_attention(monkeypatch)
    traced = []
    if fast:
        from torch._inductor.compile_fx import compile_fx

        def inductor(graph, inputs):
            traced.extend(str(node.target) for node in graph.graph.nodes)
            return compile_fx(graph, inputs)

        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
        assert lean_tower(policy.encoder)
        assert apply_fast(policy, {"half": True, "attention": "triton"}) == {
            "half": True,
            "attention": "triton",
        }
        policy.encoder.inner.compile(backend=inductor, dynamic=False)
    rng = np.random.default_rng(0)
    frames = [
        (
            rng.integers(0, 256, (*VIEW_SIZE, 3), dtype=np.uint8),
            rng.integers(0, 256, (4, *DETAIL_SIZE, 3), dtype=np.uint8),
            rng.integers(0, 256, (FOVEA_SIZE, FOVEA_SIZE, 3), dtype=np.uint8),
        )
        for _ in range(4)
    ]
    if fast:
        # The compiling call's numbers differ from every later call's (Triton attention or
        # not, measured 2026-09-26): compile first, as the actor does before its first game.
        _actor(policy, graph=False).act(None, 200_000_000, precomputed=frames[0])
    runs = []
    for graph in (False, True):
        actor = _actor(policy, graph=graph)
        torch.manual_seed(7)
        record = []
        for t, views in enumerate(frames * 2):
            if t == len(frames):
                actor.reset_episode()  # a new game starts from an empty memory
            action, sample = actor.act(None, (t + 1) * 200_000_000, precomputed=views)
            cells, logp, entropy = actor.last
            record.append((action, logp.clone(), entropy.clone(), actor.hidden.clone()))
            assert sample is None
        runs.append(record)
        assert (actor.static is not None) == graph
    if fast:
        # Compiled once (the eager actor's first decision), and neither actor recompiled.
        ops = [target for target in traced if "hoi4_arena.fast_attention" in target]
        assert len(ops) == len(policy.encoder.inner.model.blocks)
        torch._dynamo.reset()
    else:
        assert not calls and not traced
    for eager, graphed in zip(*runs):
        assert np.array_equal(eager[0], graphed[0])
        for a, b in zip(eager[1:], graphed[1:]):
            assert torch.equal(a, b)


def test_practice_and_play_policy_take_fast_mode(monkeypatch):
    from hoi4_arena import cli

    for command, target in (
        ("practice", "hoi4_arena.practice.practice"),
        ("play-policy", "hoi4_arena.play.evaluate_policy"),
    ):
        seen = {}
        monkeypatch.setattr(target, lambda *a, **k: seen.update(k))
        monkeypatch.setattr("hoi4_arena.models.configure_precision", lambda tf32: None)
        monkeypatch.setattr("hoi4_arena.models.limit_gpu_memory", lambda fraction: None)
        argv = ["hoi4-arena", command, "c", "o", "--peer", "p.json", "--minutes", "5", "--fast"]
        monkeypatch.setattr(sys, "argv", argv)
        cli.main()
        assert seen["fast"] is True

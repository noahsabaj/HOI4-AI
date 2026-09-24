import json
import random

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from hoi4_arena import features
from hoi4_arena.actions import GRID, SLOTS
from hoi4_arena.features import (
    STILL,
    CachedGame,
    MemoryHead,
    _carried_batches,
    _GraphedUnroll,
    _Loader,
    _nll,
    _reset_batches,
    _to,
    camera_targets,
    evaluate,
    load_cache,
    memory_health,
    train_memories,
    train_memory,
)
from hoi4_arena.memory import (
    KINDS,
    GatedDeltaNet2Memory,
    GRUMemory,
    Mamba3Memory,
    build_memory,
    reset,
    state_size,
)
from hoi4_arena.models import CELL_DIM


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_cell_steps_and_a_new_game_empties_its_state(kind):
    torch.manual_seed(0)
    cell = build_memory(kind, 64)
    dim = cell.dim
    state = cell.initial(3, "cpu")
    for _ in range(3):
        out, state = cell(torch.randn(3, dim), state)
    assert out.shape == (3, dim) and torch.isfinite(out).all()
    cleared = reset(state, torch.tensor([True, False, True]))
    for part in cleared:
        assert not part[0].any() and not part[2].any()
    for before, after in zip(state, cleared, strict=True):
        assert torch.equal(before[1], after[1])


def test_the_gru_cell_loads_into_the_policys_gru():
    assert GRUMemory(16).state_dict().keys() == torch.nn.GRUCell(16, 16).state_dict().keys()


def test_gated_deltanet2_is_the_papers_matrix_recurrence():
    """S_t = (I - k (b*k)^T) Diag(exp g) S + k (w*v)^T, o = S^T q / sqrt(K), per head."""
    torch.manual_seed(0)
    cell = GatedDeltaNet2Memory(64, heads=2, head_dim=8)
    x = torch.randn(1, 64)
    S = torch.randn(1, 2, 8, 8)
    state = (S, *cell.initial(1, "cpu")[1:])
    got, (after, *_) = cell(x, state)
    h = cell.norm(x)

    def conv(proj, weight):  # an empty history: only the kernel's last tap sees x
        return torch.nn.functional.silu(proj(h) * weight[:, -1])

    q = torch.nn.functional.normalize(conv(cell.q_proj, cell.q_conv).view(2, 8), dim=-1)
    k = torch.nn.functional.normalize(conv(cell.k_proj, cell.k_conv).view(2, 8), dim=-1)
    v = conv(cell.v_proj, cell.v_conv).view(2, 8)
    g = -cell.A_log.exp()[:, None] * torch.nn.functional.softplus(
        cell.f_proj(h) + cell.dt_bias
    ).view(2, 8)
    b = cell.b_proj(h).sigmoid().view(2, 8)
    w = cell.w_proj(h).sigmoid().view(2, 8)
    for head in range(2):
        erase = torch.eye(8) - torch.outer(k[head], b[head] * k[head])
        expected = erase @ torch.diag(g[head].exp()) @ S[0, head] + torch.outer(
            k[head], w[head] * v[head]
        )
        torch.testing.assert_close(after[0, head], expected, atol=1e-5, rtol=1e-5)
    o = torch.einsum("hkv,hk->hv", after[0], q) * 8**-0.5
    o = o * torch.rsqrt(o.square().mean(-1, keepdim=True) + cell.eps) * cell.o_weight
    o = o * cell.g_proj(h).view(2, 8).sigmoid()
    torch.testing.assert_close(got, x + cell.o_proj(o.reshape(1, -1)), atol=1e-5, rtol=1e-5)


def test_mamba3_decays_its_state():
    """A is negative, so with nothing written the state shrinks by exp(A dt) per head.

    A sign slip (clamping before negating) once made A a tiny positive number, which
    matched the reference on a first step and diverged from the second.
    """
    torch.manual_seed(0)
    cell = Mamba3Memory(64, d_state=16, headdim=16)
    with torch.no_grad():
        cell.in_proj.weight.zero_()  # every projection 0: v = 0, so nothing is written
    angle, S, k, v = cell.initial(2, "cpu")
    S = torch.randn_like(S)
    _, (_, after, _, _) = cell(torch.randn(2, 64), (angle, S, k, v))
    # With the projections at 0, A = -1 and dt = softplus(dt_bias).
    alpha = (-torch.nn.functional.softplus(cell.dt_bias)).exp()
    torch.testing.assert_close(after, alpha[None, :, None, None] * S)
    assert (alpha < 1).all()
    assert state_size(cell.initial(1, "cpu")) == cell.heads * (cell.angles + 16 * 16 + 16 + 16)


def test_camera_targets_count_notches_and_time_the_recentre(tmp_path):
    second = 1_000_000_000
    events = [{"t_ns": 1 * second, "event": {"kind": "wheel", "delta": 120}}] * 1
    events += [{"t_ns": 2 * second + i, "event": {"kind": "wheel", "delta": 120}} for i in range(4)]
    events += [
        {"t_ns": 5 * second + i, "event": {"kind": "wheel", "delta": -120}} for i in range(30)
    ]
    events += [{"t_ns": 7 * second, "event": {"kind": "wheel", "delta": 120}}]
    rows = [{"t_ns": 0, "scripted_events": events}]
    (tmp_path / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    decisions = np.array([0, 3, 6, 8]) * second
    zoom, since = camera_targets(tmp_path, decisions)
    assert zoom.tolist() == [0, 5, 0, 1]
    assert np.isnan(since[:2]).all() and since[2:].tolist() == pytest.approx([1.0, 3.0])


def _cache(root, name, split, n, seed):
    rng = np.random.default_rng(seed)
    game = root / name
    game.mkdir(parents=True)
    np.save(game / "summary.npy", rng.standard_normal((n, 8)).astype(np.float16))
    np.save(game / "centre.npy", rng.standard_normal((n, CELL_DIM)).astype(np.float16))
    np.save(game / "cells.npy", rng.standard_normal((n, GRID, CELL_DIM)).astype(np.float16))
    actions = np.zeros((n, SLOTS, 3), np.int64)
    actions[::3, 0] = [1, 5, 7]
    np.savez(
        game / "labels.npz",
        actions=actions,
        valid=np.ones(n, bool),
        outcome=np.zeros(n, np.float32),
        speed=np.full(n, 5),
        pointer=rng.random((n, 2)).astype(np.float32),
        zoom=rng.integers(0, 26, n).astype(np.float32),
        since_recentre=rng.random(n).astype(np.float32) * 30,
        winner=np.float32(1 if seed % 2 else -1),
    )
    (game / "meta.json").write_text(json.dumps({"split": split, "decisions": n}))


def test_a_cached_piece_carries_the_previous_action_across_its_start(tmp_path):
    _cache(tmp_path, "a", "train", 10, 0)
    game = CachedGame(tmp_path / "a")
    piece = game.piece(3, 4)
    assert torch.equal(piece["previous"][0], torch.from_numpy(game.labels["actions"][2]))
    tail = game.piece(8, 4)
    assert tail["valid"].tolist() == [True, True, False, False]


@pytest.mark.parametrize("carry", [True, False])
def test_train_memory_runs_end_to_end_on_a_small_cache(tmp_path, carry):
    for i in range(3):
        _cache(tmp_path / "cache", f"t{i}", "train", 24, i)
    _cache(tmp_path / "cache", "v", "validation", 20, 7)
    report = train_memory(
        tmp_path / "cache",
        tmp_path / "out",
        memory="gdn2" if carry else "gru",
        window=8,
        carry=carry,
        burn_in=0 if carry else 2,
        decisions=16,
        epochs=1,
        device="cpu",
    )
    assert np.isfinite(report["validation_nll"]) and report["updates"] > 0
    assert {"zoom", "since_recentre", "winner", "pointer_5"} <= report["probes"].keys()
    health = report["memory_health"]
    assert {"std_over_time", "still_units", "one_step_from_empty", "dead"} <= health.keys()
    assert (health["saturated_gates"] is None) == carry  # gates are the GRU's (carry=False)
    head = MemoryHead(8, "gru")
    saved = torch.load(tmp_path / "out" / "head.pt", weights_only=True)["head"]
    if not carry:
        head.load_state_dict(saved)
    with pytest.raises(FileExistsError):
        train_memory(tmp_path / "cache", tmp_path / "out", device="cpu")


def _collated(rows, length, device):
    """A batch the way train_memory made one before `_Loader`: pieces, collated, moved."""
    pieces = [None if row is None else row[0].piece(row[1], length) for row in rows]
    template = next(p for p in pieces if p is not None)
    pieces = [p or {k: torch.zeros_like(v) for k, v in template.items()} for p in pieces]
    return _to(default_collate(pieces), device)


def _games(root, lengths, invalid=()):
    for i, n in enumerate(lengths):
        _cache(root, f"t{i}", "train", n, i)
    games = load_cache(root, "train")
    for game, step in invalid:
        games[game].labels["valid"][step] = False
    return games


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_the_loader_reads_the_batches_collated_pieces_made(tmp_path, device):
    games = _games(tmp_path, [19, 30, 11], invalid=[(1, 4), (1, 21)])
    carried = list(_carried_batches(games, 8, 4, random.Random(0)))
    reset = list(_reset_batches(games, 6, 2, 3, random.Random(1)))
    assert any(row is None for rows, _, _ in carried for row in rows)
    assert len(reset[-1][0]) < 3  # a short last batch, as in training
    loader = _Loader(games, device, readers=3)
    try:
        for plans, first in ((carried, 0), (reset, 2)):
            got = list(loader(iter(plans), first))
            assert len(got) == len(plans)
            for (rows, length, fresh), (batch, flags) in zip(plans, got, strict=True):
                expected = _collated(rows, length, device)
                for key, value in expected.items():
                    assert batch[key].dtype == value.dtype, key
                    assert torch.equal(batch[key], value), key
                assert flags.tolist() == fresh
                valid = expected["valid"][:, first:].flatten()
                assert torch.equal(batch["scored"], valid.nonzero().squeeze(1))
    finally:
        loader.close()


def test_the_loader_stops_cleanly_when_left_early(tmp_path):
    games = _games(tmp_path, [40, 40])
    loader = _Loader(games, "cpu", readers=2)
    batches = loader(_carried_batches(games, 4, 2, random.Random(0)))
    next(batches)
    batches.close()
    loader.close()


@torch.no_grad()
def _evaluate_before(head, games, device, clear_every=None):
    """evaluate as it was before `_Loader`: batch by batch, one decision at a time when
    clearing, every result copied to the host as it came."""
    head.eval()
    losses, active, outputs = [], [], []
    autocast = {"device_type": device, "dtype": torch.bfloat16, "enabled": device == "cuda"}
    for game in games:
        out, state = head.initial(1, device)
        kept = []
        for start in range(0, game.length, 128):
            batch = _to(default_collate([game.piece(start, 128)]), device)
            with torch.autocast(**autocast):
                outs, out, state = _unroll_before(head, batch, out, state, start, clear_every)
                nll, valid = _nll(head, outs, batch)
            kept.append(outs[0, : min(128, game.length - start)].float().cpu())
            losses.extend(nll.float().cpu().tolist())
            acting = (batch["actions"][:, :, :, 0] != 0).any(-1)[valid]
            active.extend(nll[acting].float().cpu().tolist())
        outputs.append(torch.cat(kept).numpy())
    head.train()
    return float(np.mean(losses)), float(np.mean(active)) if active else None, outputs


def _unroll_before(head, batch, out, state, start, clear_every):
    """One piece of evaluate before `_Loader`: stepped one decision at a time when clearing."""
    device = batch["summary"].device
    if not clear_every:
        outs, (out, state) = head.unroll(
            batch, out, state, torch.tensor([start == 0], device=device)
        )
        return outs, out, state
    parts = []
    for t in range(128):
        if (start + t) % clear_every == 0:
            out, state = head.initial(1, device)
        piece = {k: v[:, t : t + 1] for k, v in batch.items()}
        o, (out, state) = head.unroll(
            piece, out, state, torch.zeros(1, dtype=torch.bool, device=device)
        )
        parts.append(o)
    return torch.cat(parts, 1), out, state


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("kind", ["gru", "gdn2"])
@pytest.mark.parametrize("clear_every", [None, 18])
def test_evaluate_computes_what_it_did_before_the_loader(
    tmp_path, monkeypatch, device, kind, clear_every
):
    # Games of uneven lengths, and graphs captured at a shape's second coming, so these few
    # pieces replay them.
    lengths = [150, 70, 300, 40, 130, 270, 90]
    games = _games(tmp_path, lengths, invalid=[(0, 3), (1, 69), (4, 128)])
    monkeypatch.setattr(features._Unrolls, "CAPTURE_AT", 2)
    torch.manual_seed(0)
    head = MemoryHead(8, kind).to(device)
    before = _evaluate_before(head, games, device, clear_every)
    after = evaluate(head, games, device, clear_every)
    assert after[:2] == before[:2]
    for a, b in zip(after[2], before[2], strict=True):
        assert np.array_equal(a, b)


@CUDA
@pytest.mark.parametrize("kind", ["gru", "gdn2", "mamba3", "none"])
@pytest.mark.parametrize("grad_from", [0, 2])
def test_a_graphed_unroll_is_the_eager_one_to_the_bit(tmp_path, kind, grad_from):
    games = _games(tmp_path, [40, 40, 40])
    loader = _Loader(games, "cuda", readers=2)
    rows = [(games[0], 3), (games[1], 0), (games[2], 20)]
    batch, fresh = next(loader(iter([(rows, 10, [False, True, False])]), grad_from))
    loader.close()
    torch.manual_seed(0)
    head = MemoryHead(8, kind).cuda()
    autocast = {"device_type": "cuda", "dtype": torch.bfloat16}

    def step(unroll):
        outs, carried = unroll()
        with torch.autocast(**autocast):
            loss = _nll(head, outs, batch, grad_from)[0].mean()
        loss.backward()
        grads = [None if p.grad is None else p.grad.clone() for p in head.parameters()]
        head.zero_grad(set_to_none=True)
        return [outs.detach().clone(), carried[0].clone(), *(s.clone() for s in carried[1])], grads

    def eager():
        with torch.autocast(**autocast):
            return head.unroll(batch, out, state, fresh, grad_from)

    for start in ("empty", "carried"):
        out, state = head.initial(3, "cuda")
        if start == "carried":  # a bfloat16 memory, as after an update
            with torch.no_grad(), torch.autocast(**autocast):
                _, (out, state) = head.unroll(batch, out, state, fresh)
        graph = _GraphedUnroll(head, batch, out, state, fresh, grad_from, autocast)
        want = step(eager)
        for _ in range(2):  # a replay leaves nothing behind for the next
            got = step(lambda: graph(batch, out, state, fresh))
            assert all(torch.equal(a, b) for a, b in zip(got[0], want[0], strict=True))
            for a, b in zip(got[1], want[1], strict=True):
                assert (a is None) == (b is None)
                assert a is None or torch.equal(a, b)


def _run(tmp_path, name, **settings):
    out = tmp_path / name
    report = settings.pop("train")(tmp_path / "cache", out, **settings)
    head = torch.load(out / "head.pt", weights_only=True)["head"]
    return report, (out / "metrics.jsonl").read_text(), head


def _same(a, b):
    ra, rb = dict(a[0]), dict(b[0])
    ra.pop("train_seconds"), rb.pop("train_seconds")
    assert ra == rb and a[1] == b[1]
    assert a[2].keys() == b[2].keys() and all(torch.equal(a[2][k], b[2][k]) for k in a[2])


def _small_cache(tmp_path):
    for i in range(3):
        _cache(tmp_path / "cache", f"t{i}", "train", 30 + 9 * i, i)
    _cache(tmp_path / "cache", "v", "validation", 21, 7)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("carry", [True, False])
def test_cells_trained_together_learn_what_each_learns_alone(tmp_path, device, carry):
    _small_cache(tmp_path)
    settings = {"window": 8, "carry": carry, "burn_in": 0 if carry else 2, "decisions": 16}
    settings.update(epochs=3, seed=3, device=device)
    kinds = ["gru", "gdn2", "none"]
    together = train_memories(
        tmp_path / "cache", [tmp_path / f"all-{k}" for k in kinds], kinds, **settings
    )
    for kind, report in zip(kinds, together, strict=True):
        out = tmp_path / f"all-{kind}"
        joint = (report, (out / "metrics.jsonl").read_text(), torch.load(out / "head.pt")["head"])
        alone = _run(tmp_path, f"one-{kind}", train=train_memory, memory=kind, **settings)
        _same(joint, alone)


@CUDA
@pytest.mark.parametrize("kind", ["gru", "mamba3"])
@pytest.mark.parametrize("carry", [True, False])
def test_training_from_cuda_graphs_changes_no_bit(tmp_path, monkeypatch, kind, carry):
    _small_cache(tmp_path)
    settings = {"window": 8, "carry": carry, "burn_in": 0 if carry else 2, "decisions": 16}
    settings.update(epochs=4, seed=1, device="cuda", memory=kind, train=train_memory)
    captured = []
    capture = features._GraphedUnroll.__init__

    def counted(self, *args, **kwargs):
        captured.append(torch.is_grad_enabled())
        capture(self, *args, **kwargs)

    monkeypatch.setattr(features._GraphedUnroll, "__init__", counted)
    graphs = _run(tmp_path, "graphs", graphs=True, **settings)
    assert True in captured and False in captured  # training and evaluation both replayed
    captured.clear()
    _same(graphs, _run(tmp_path, "eager", graphs=False, **settings))
    assert not captured


def test_memory_health_calls_a_saturated_gru_dead(tmp_path):
    """The study's GRUs were dead and nothing said so: a large constant input (the tower's
    unnormalized summary) held every gate at its bound, so the memory never moved. Here a
    large constant bias does the same, whatever the inputs are normalized to."""
    games = _games(tmp_path, [60, 45])
    torch.manual_seed(0)
    alive = MemoryHead(8, "gru")
    dead = MemoryHead(8, "gru")
    dead.load_state_dict(alive.state_dict())
    with torch.no_grad():
        dead.memory.bias_ih.copy_(50 * torch.randn_like(dead.memory.bias_ih).sign())
    healthy, saturated = memory_health([alive, dead], games, "cpu")
    assert saturated["dead"] and not healthy["dead"]
    assert saturated["still_units"] > 0.9 > 0.1 > healthy["still_units"]
    assert saturated["std_over_time"] < STILL < healthy["std_over_time"]
    assert min(saturated["saturated_gates"].values()) > 0.9
    assert max(healthy["saturated_gates"].values()) < 0.1
    # Dead, a step from an empty memory lands where the carried one does.
    assert saturated["one_step_from_empty"] < 0.01 < healthy["one_step_from_empty"]

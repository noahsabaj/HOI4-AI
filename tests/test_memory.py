import json

import numpy as np
import pytest
import torch

from hoi4_arena.actions import GRID, SLOTS
from hoi4_arena.features import CachedGame, MemoryHead, camera_targets, train_memory
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
    head = MemoryHead(8, "gru")
    saved = torch.load(tmp_path / "out" / "head.pt", weights_only=True)["head"]
    if not carry:
        head.load_state_dict(saved)
    with pytest.raises(FileExistsError):
        train_memory(tmp_path / "cache", tmp_path / "out", device="cpu")

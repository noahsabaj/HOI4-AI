import numpy as np
import torch

from hoi4_arena.actions import GRID, SLOTS
from hoi4_arena.dataset import presses_after_move
from hoi4_arena.heatmap import RAMP, colour, render
from hoi4_arena.models import CELL_DIM, CELLS, PRESSES, ActionHead, pointer_blob


def _head(look=False, seed=0):
    torch.manual_seed(seed)
    return ActionHead(memory_dim=32, look=look).eval()


def _inputs(n, seed=1):
    generator = torch.Generator().manual_seed(seed)
    memory = torch.randn(n, 32, generator=generator)
    cells = torch.randn(n, CELLS * CELLS, CELL_DIM, generator=generator)
    return memory, cells


def _moves(points):
    actions = torch.zeros(len(points), SLOTS, 3, dtype=torch.long)
    actions[:, 0] = torch.tensor([[1, x, y] for x, y in points])
    return actions


def test_the_blob_covers_the_two_nearest_cells_and_its_profiles_sum_to_one():
    # Left of a cell's middle, right of it, and at both edges of the screen.
    near, mass, profile = pointer_blob(torch.tensor([40.0, 50.0, 3.0, 1020.0]), sigma=4.0)
    assert near.tolist() == [[1, 0], [1, 2], [0, 0], [31, 31]]
    # A neighbour clamped onto the point's own cell at an edge counts nothing.
    assert mass[2, 1] == 0 and mass[3, 1] == 0
    assert torch.allclose(profile[:, 0].sum(-1), torch.ones(4))
    # 40 is 8 lattice units, two sigmas, from cell 0: a small but real share lands there.
    assert 0.01 < mass[0, 1] / mass[0].sum() < 0.05


def test_a_narrow_blob_scores_exactly_like_the_one_point_it_covers():
    head = _head()
    memory, cells = _inputs(3)
    actions = _moves([(100, 200), (511, 512), (1023, 0)])
    exact = head(memory, cells, actions)[1]
    narrow = head(memory, cells, actions, sigma=0.05)[1]
    assert torch.allclose(exact, narrow, atol=1e-4)


def test_a_wide_blob_trains_the_fine_head_of_every_cell_it_covers():
    head = _head()
    memory, cells = _inputs(1)
    cells.requires_grad_(True)
    # On the edge between two cells: exact scoring reads only the target cell's features
    # through the fine head; a blob also reads its neighbour's.
    for sigma, touched in ((0.0, 1), (6.0, 2)):
        cells.grad = None
        score = head(memory, cells, _moves([(31, 500)]), sigma=sigma)[1]
        assert torch.isfinite(score).all()
        score.sum().backward()
        column = cells.grad[0].view(CELLS, CELLS, CELL_DIM)[500 // CELLS]
        # Cells 0 and 1 of the target's row: the softmax over cells touches every cell a
        # little, so count the ones whose features also went through the fine head.
        fine_touched = (column[:2].abs().sum(-1) > 10 * column[2:].abs().sum(-1).max()).sum()
        assert fine_touched == touched


def test_a_head_that_looks_before_clicking_never_presses_after_a_move():
    memory, cells = _inputs(256)
    for look in (True, False):
        head = _head(look)
        with torch.no_grad():
            head.kinds.bias.zero_()
            head.kinds.bias[1] = 3.0
            head.kinds.bias[PRESSES] = 3.0
        actions, logp, entropy = head(memory, cells)
        after = presses_after_move(actions.numpy())
        assert after.any() != look
        assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()
    # A demonstration that presses after a move cannot be its label: session_labels
    # leaves such decisions out under look_before_click.
    bad = _moves([(5, 5)])
    bad[0, 1, 0] = PRESSES[0]
    assert presses_after_move(bad.numpy()).tolist() == [True]
    assert torch.isinf(_head(True)(memory[:1], cells[:1], bad)[1]).all()
    assert torch.isfinite(_head(False)(memory[:1], cells[:1], bad)[1]).all()


def test_the_pointer_map_is_the_heads_own_distribution_over_the_screen():
    head = _head()
    with torch.no_grad():
        head.kinds.bias.zero_()
        head.kinds.bias[1] = 20.0  # Always a move, so the greedy action has a position.
    memory, cells = _inputs(1)
    kind_p, grid = head.pointer_map(memory, cells, _moves([(0, 0)]), slot=0, top=16)
    assert grid.shape == (GRID, GRID)
    assert abs(float(grid.sum()) - 1) < 1e-4 and abs(float(kind_p.sum()) - 1) < 1e-5
    # The head's greedy first move is its likeliest cell, then its likeliest position in
    # that cell: the map must peak there within that cell.
    greedy = head(memory, cells, deterministic=True)[0][0, 0]
    assert int(greedy[0]) == 1
    x, y = int(greedy[1]), int(greedy[2])
    block = grid[
        y // CELLS * CELLS : (y // CELLS + 1) * CELLS, x // CELLS * CELLS : (x // CELLS + 1) * CELLS
    ]
    inner_y, inner_x = np.unravel_index(int(block.argmax()), block.shape)
    assert (inner_x, inner_y) == (x % CELLS, y % CELLS)
    assert np.unravel_index(int(grid.reshape(32, 32, 32, 32).sum((1, 3)).argmax()), (32, 32)) == (
        y // CELLS,
        x // CELLS,
    )


def test_the_drawing_keeps_the_frame_and_runs_cold_to_hot():
    assert np.allclose(colour(np.array([0.0, 1.0])), RAMP[[0, -1]])
    rgb = np.zeros((108, 192, 3), np.uint8)
    grid = np.zeros((GRID, GRID), np.float32)
    grid[512, 512] = 1
    out = render(rgb, grid, np.full(len(RAMP) + 200, 0.001), target=(96.0, 54.0))
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert out[54, 96].sum() > 0

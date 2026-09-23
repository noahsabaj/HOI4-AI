import pytest
import torch

from hoi4_arena.actions import GRID, SLOTS, VOCAB
from hoi4_arena.dataset import DETAIL_SIZE, FOVEA_SIZE, QUADRANTS, VIEW_SIZE
from hoi4_arena.models import (
    CELL_DIM,
    CELLS,
    ActionHead,
    DetailEncoder,
    Policy,
    screen_cells,
    xm_loss,
)


def _cells(batch, dim=CELL_DIM, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, GRID, dim, generator=generator)


def test_a_likelihood_replays_with_the_same_latent_and_ignores_inactive_xy():
    torch.manual_seed(1)
    actor = ActionHead(memory_dim=16)
    memory, cells, noise = torch.randn(2, 16), _cells(2), torch.randn(2, 16)
    action, old, _ = actor(memory, cells, noise=noise)
    _, new, entropy = actor(memory, cells, action, noise)
    assert torch.allclose(old, new)
    assert torch.isfinite(entropy).all()
    assert (action[:, :, 1:][action[:, :, 0] != 1] == 0).all()
    (-new.mean()).backward()
    assert actor.init.weight.grad.abs().sum() > 0


def test_cell_and_offset_are_exactly_the_action_lattice():
    """A move's x and y are cell * CELLS + offset: every lattice point, and only once."""
    assert CELLS * CELLS == GRID
    x = torch.arange(GRID)
    cell, offset = x // CELLS, x % CELLS
    assert torch.equal(cell * CELLS + offset, x)


def categorical_reference(actor, memory, cells, noise, actions):
    """Score `actions` through torch.distributions, slot by slot, the way the head should.

    An independent replay: the head builds no Distribution objects because validating
    one synchronizes with the GPU, and it must still produce exactly their numbers. Sums
    over slots use the same stack-then-reduce as the head, since adding eight floats in a
    Python loop reduces in a different order and could differ in the last bits.
    """
    from torch.distributions import Categorical

    rows = torch.arange(memory.shape[0])
    state = torch.tanh(actor.init(torch.cat([memory, noise], -1)))
    previous = torch.zeros(memory.shape[0], 64, dtype=memory.dtype)
    logps, entropies = [], []
    for slot in range(SLOTS):
        state = actor.cell(previous, state)
        kind, x, y = actions[:, slot].unbind(-1)
        place = (y // CELLS) * CELLS + x // CELLS
        offset = (y % CELLS) * CELLS + x % CELLS
        kinds = Categorical(logits=actor.kinds(state).float())
        where = torch.einsum("bnc,bc->bn", cells, actor.query(state).to(cells.dtype))
        places = Categorical(logits=where.float() * actor.scale + actor.cell_bias.float())
        chosen = cells[rows, place].to(state.dtype)
        fine = Categorical(logits=actor.fine(torch.cat([state, chosen], -1)).float())
        move = (kind == 1).float()
        logps.append(kinds.log_prob(kind) + move * (places.log_prob(place) + fine.log_prob(offset)))
        entropies.append(kinds.entropy() + kinds.probs[:, 1] * (places.entropy() + fine.entropy()))
        previous = actor.embedding(actions[:, slot, 0]) + actor.xy(
            actions[:, slot, 1:].to(memory.dtype) / (GRID - 1)
        )
    return torch.stack(logps, 1).sum(1), torch.stack(entropies, 1).sum(1)


def _moves(batch, seed):
    """Actions where every slot is a move, so the pointer terms are all exercised."""
    generator = torch.Generator().manual_seed(seed)
    xy = torch.randint(0, GRID, (batch, SLOTS, 2), generator=generator)
    return torch.cat([torch.ones(batch, SLOTS, 1, dtype=torch.long), xy], -1)


def test_the_likelihood_is_bit_for_bit_the_categorical_it_stands_for():
    """The distribution objects are not built; the numbers they produce must not change.

    A PPO ratio is a difference of two of these log-likelihoods, and a systematic shift
    in one of them is indistinguishable from a policy update that never happened. Once
    with ordinary logits, once with the heads amplified until normalized logits fall past
    -100, the regime where Categorical's clamp does anything at all.
    """
    torch.manual_seed(5)
    actor = ActionHead(memory_dim=16)
    memory, cells, noise = torch.randn(4, 16), _cells(4), torch.randn(4, actor.noise_dim)
    for scale in (1.0, 400.0):
        with torch.no_grad():
            for layer in (actor.kinds, actor.query, actor.fine[-1]):
                layer.weight.mul_(scale)
        for actions in (actor(memory, cells, noise=noise)[0], _moves(4, 1)):
            _, logp, entropy = actor(memory, cells, actions, noise)
            expected_logp, expected_entropy = categorical_reference(
                actor, memory, cells, noise, actions
            )
            assert torch.equal(logp, expected_logp), (logp - expected_logp).abs().max().item()
            assert torch.equal(entropy, expected_entropy)


def test_the_head_scores_in_float32_even_when_its_weights_are_bfloat16():
    """Pins the float casts, which are easy to drop and quiet to lose.

    Without them the normalization runs in bfloat16 and lands about 0.05 away.
    """
    torch.manual_seed(6)
    actor = ActionHead(memory_dim=16).to(torch.bfloat16)
    memory = torch.randn(4, 16, dtype=torch.bfloat16)
    cells = _cells(4).to(torch.bfloat16)
    noise = torch.randn(4, actor.noise_dim, dtype=torch.bfloat16)
    actions = _moves(4, 2)
    _, logp, entropy = actor(memory, cells, actions, noise)
    expected_logp, expected_entropy = categorical_reference(actor, memory, cells, noise, actions)
    assert logp.dtype == torch.float32 and entropy.dtype == torch.float32
    assert torch.equal(logp, expected_logp)
    assert torch.equal(entropy, expected_entropy)


def test_deterministic_takes_the_argmax_and_sampling_stays_stochastic():
    torch.manual_seed(0)
    actor = ActionHead(memory_dim=8)
    memory, cells = torch.randn(2, 8), _cells(2)
    greedy = [actor(memory, cells, deterministic=True)[0] for _ in range(3)]
    assert all(torch.equal(greedy[0], other) for other in greedy[1:])
    sampled = torch.stack([actor(memory, cells)[0] for _ in range(12)])
    assert not torch.equal(sampled[0], sampled[-1]), "sampling must still be stochastic"


def test_the_greedy_path_takes_the_mode_of_each_step_it_conditions_on():
    torch.manual_seed(3)
    actor = ActionHead(memory_dim=8)
    memory, cells = torch.randn(2, 8), _cells(2)
    noise = torch.zeros(2, actor.noise_dim)
    action, _, _ = actor(memory, cells, noise=noise, deterministic=True)
    rows = torch.arange(2)
    state = torch.tanh(actor.init(torch.cat([memory, noise], -1)))
    previous = torch.zeros(2, 64)
    for slot in range(SLOTS):
        state = actor.cell(previous, state)
        kind = actor.kinds(state).argmax(-1)
        where = torch.einsum("bnc,bc->bn", cells, actor.query(state))
        place = (where * actor.scale + actor.cell_bias).argmax(-1)
        offset = actor.fine(torch.cat([state, cells[rows, place]], -1)).argmax(-1)
        move = (kind == 1).long()
        x = ((place % CELLS) * CELLS + offset % CELLS) * move
        y = ((place // CELLS) * CELLS + offset // CELLS) * move
        expected = torch.stack([kind, x, y], -1)
        assert torch.equal(action[:, slot], expected)
        previous = actor.embedding(kind) + actor.xy(expected[:, 1:].float() / (GRID - 1))


def test_the_head_learns_to_point_at_what_the_cells_show():
    """Pointing is tied to what is at each place, not to a memorized position.

    One cell per sample carries a marker, at a random place each time. After a short
    training run the head moves onto the marked cell of screens it has never seen. The
    old head produced x and y from one pooled vector, with no link to where anything was.
    """
    torch.manual_seed(11)
    dim = 16
    actor = ActionHead(memory_dim=8, cell_dim=dim)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-3)
    marker = torch.randn(dim)

    def screens(batch, seed):
        generator = torch.Generator().manual_seed(seed)
        cells = 0.3 * torch.randn(batch, GRID, dim, generator=generator)
        target = torch.randint(0, GRID, (batch,), generator=generator)
        cells[torch.arange(batch), target] += marker
        x = (target % CELLS) * CELLS + CELLS // 2
        y = (target // CELLS) * CELLS + CELLS // 2
        actions = torch.zeros(batch, SLOTS, 3, dtype=torch.long)
        actions[:, 0] = torch.stack([torch.ones_like(x), x, y], -1)
        return cells, actions, target

    for step in range(150):
        cells, actions, _ = screens(32, step)
        loss = -actor(torch.zeros(32, 8), cells, actions)[1].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    cells, _, target = screens(64, 10_000)
    action, _, _ = actor(torch.zeros(64, 8), cells, deterministic=True)
    x, y = action[:, 0, 1], action[:, 0, 2]
    hit = (y // CELLS) * CELLS + x // CELLS == target
    assert (action[:, 0, 0] == 1).all()
    assert hit.float().mean() > 0.9, f"pointed at the marked cell {hit.float().mean():.0%}"


def test_xm_scores_with_the_cells():
    torch.manual_seed(2)
    actor = ActionHead(memory_dim=8)
    loss = xm_loss(actor, torch.randn(3, 8), _cells(3), _moves(3, 4), candidates=2)
    assert loss.shape == (3,) and torch.isfinite(loss).all()


def test_screen_cells_tile_the_quadrants_in_screen_order():
    maps = torch.zeros(1, QUADRANTS, 1, 4, 4)
    for quadrant in range(QUADRANTS):
        maps[0, quadrant] = quadrant + 1
    cells = screen_cells(maps)[0, 0]
    half = CELLS // 2
    assert cells.shape == (CELLS, CELLS)
    assert cells[0, 0] == 1 and cells[0, half] == 2 and cells[half, 0] == 3 and cells[-1, -1] == 4


def test_the_detail_reader_keeps_a_spatial_map_at_stride_sixteen():
    maps = DetailEncoder()(torch.zeros(2, QUADRANTS, 3, DETAIL_SIZE, DETAIL_SIZE))
    assert maps.shape == (2, QUADRANTS, CELL_DIM, DETAIL_SIZE // 16, DETAIL_SIZE // 16)


class _Encoder(torch.nn.Module):
    """Stands in for the video encoder: a summary token and a 14 x 14 patch grid."""

    dim = 8

    def forward(self, clip, quadrants=None):
        batch = clip.shape[0]
        pooled = clip.mean((1, 2, 3, 4))[:, None]
        return pooled.expand(batch, self.dim), pooled[:, :, None, None].expand(
            batch, self.dim, 14, 14
        )


def test_the_screen_encoder_reads_the_tiled_quadrants_as_one_screen():
    """Same outputs as the video encoder: a summary and a patch grid the cells resize."""
    from hoi4_arena.models import ScreenEncoder

    torch.manual_seed(9)
    encoder = ScreenEncoder(size=64, pretrained=False).eval()
    quadrants = torch.randn(2, QUADRANTS, 3, 32, 32)
    with torch.no_grad():
        summary, grid = encoder(None, quadrants)
        moved = quadrants.clone()
        moved[:, 0] += 3
        _, other = encoder(None, moved)
    assert summary.shape == (2, encoder.dim) and grid.shape == (2, encoder.dim, 4, 4)
    # What the quadrants show reaches the grid the cells are built from.
    assert not torch.allclose(grid, other)
    # Only the last two blocks train.
    trainable = {n.split(".")[1] for n, p in encoder.model.named_parameters() if p.requires_grad}
    assert trainable == {str(len(encoder.model.blocks) - 2), str(len(encoder.model.blocks) - 1)}
    with pytest.raises(FileNotFoundError, match="weights"):
        ScreenEncoder("no/such/folder")
    policy = Policy(encoder, memory_dim=16)
    assert policy.fusion.in_features == encoder.dim + 3 * CELL_DIM + 64 + 32


@pytest.mark.parametrize("speed", [1, 4])
def test_the_policy_reads_every_view_and_the_speed(speed):
    torch.manual_seed(4)
    policy = Policy(_Encoder(), memory_dim=32)
    clip = torch.randn(2, 3, 8, VIEW_SIZE, VIEW_SIZE)
    quadrants = torch.randn(2, QUADRANTS, 3, DETAIL_SIZE, DETAIL_SIZE)
    fovea = torch.randn(2, 3, FOVEA_SIZE, FOVEA_SIZE)
    previous = torch.zeros(2, SLOTS, 3, dtype=torch.long)
    speeds = torch.full((2,), speed)
    hidden, value, summary, cells = policy(clip, quadrants, fovea, previous, speeds)
    assert hidden.shape == (2, 32) and value.shape == (2,)
    assert summary.shape == (2, 8) and cells.shape == (2, GRID, CELL_DIM)
    other, _, _, _ = policy(clip, quadrants, fovea, previous, torch.full((2,), 6 - speed))
    assert not torch.allclose(hidden, other), "the speed must reach the memory"
    action, logp, _ = policy.actor(hidden, cells)
    assert action.shape == (2, SLOTS, 3) and logp.shape == (2,)
    assert len(VOCAB) > 1


def test_recomputing_steps_gives_the_same_loss_and_gradients():
    """Checkpointing trades memory for a second forward pass; it must not change the math."""
    from hoi4_arena.models import InverseDynamics
    from hoi4_arena.train import unroll

    torch.manual_seed(12)
    batch = {
        "clips": torch.randn(2, 3, 3, 8, 16, 16),
        "quadrants": torch.randn(2, 3, QUADRANTS, 3, 32, 32),
        "fovea": torch.randn(2, 3, 3, 16, 16),
        "previous": torch.zeros(2, 3, SLOTS, 3, dtype=torch.long),
        "speed": torch.full((2, 3), 4),
    }
    grads = []
    for recompute in (False, True):
        torch.manual_seed(13)
        policy = Policy(_Encoder(), memory_dim=16)
        memory, values, _, cells = unroll(policy, batch, burn_in=1, checkpoint=recompute)
        (memory.sum() + values.sum() + cells.mean()).backward()
        grads.append(
            torch.cat([p.grad.flatten() for p in policy.parameters() if p.grad is not None])
        )
    assert torch.allclose(grads[0], grads[1], atol=1e-6)
    grads = []
    for recompute in (False, True):
        torch.manual_seed(14)
        model = InverseDynamics(_Encoder(), memory_dim=16)
        context, cells = model(
            batch["clips"], batch["quadrants"], batch["fovea"], batch["speed"], checkpoint=recompute
        )
        (context.sum() + cells.mean()).backward()
        grads.append(
            torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        )
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_the_gpu_memory_cap_is_a_fraction_and_optional():
    from hoi4_arena.models import limit_gpu_memory

    assert limit_gpu_memory(None) is None
    if not torch.cuda.is_available():
        assert limit_gpu_memory(0.9) is None
        return
    with pytest.raises(ValueError, match="fraction"):
        limit_gpu_memory(1.5)
    total = torch.cuda.get_device_properties(0).total_memory / 2**20
    try:
        cap = limit_gpu_memory(0.5)
        assert abs(cap - total / 2) < 2
        # Past the cap the allocator refuses instead of letting the driver spill.
        with pytest.raises(torch.OutOfMemoryError):
            torch.empty(int(total * 0.6 * 2**20), dtype=torch.uint8, device="cuda")
    finally:
        limit_gpu_memory(1.0)

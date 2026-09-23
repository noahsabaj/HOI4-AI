"""Reading a window's frames at once, frozen blocks without a graph, learns what stepping did.

Each training step used to run the policy one decision at a time and recompute each
decision whole in the backward pass, the encoder's frozen blocks included. Perception
does not depend on the memory, so the window is now read in chunks and only the memory
steps. The references below are the old code, kept verbatim; the new must match them in
outputs and in every gradient.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hoi4_arena.actions import SLOTS
from hoi4_arena.dataset import QUADRANTS
from hoi4_arena.models import (
    InverseDynamics,
    Policy,
    ScreenEncoder,
    VideoEncoder,
    reads_clip,
)
from hoi4_arena.train import unroll


def unroll_by_step(policy, batch, burn_in=2, training=True, checkpoint=False):
    """train.unroll as it was: one whole decision at a time."""
    hidden = None
    memories, values, features, cells = [], [], [], []
    for t in range(batch["clips"].shape[1]):
        scored = training and t >= burn_in
        inputs = (
            batch["clips"][:, t],
            batch["quadrants"][:, t],
            batch["fovea"][:, t],
            batch["previous"][:, t],
            batch["speed"][:, t],
            hidden,
        )
        with torch.set_grad_enabled(scored):
            if scored and checkpoint:
                hidden, value, feature, cell = torch.utils.checkpoint.checkpoint(
                    policy, *inputs, use_reentrant=False
                )
            else:
                hidden, value, feature, cell = policy(*inputs)
        if t < burn_in:
            hidden = hidden.detach()
        else:
            memories.append(hidden)
            values.append(value)
            features.append(feature)
            cells.append(cell)
    return (
        torch.stack(memories, 1),
        torch.stack(values, 1),
        torch.stack(features, 1),
        torch.stack(cells, 1),
    )


def idm_by_step(model, clips, quadrants, fovea, speed):
    """InverseDynamics.forward as it was: one decision at a time."""
    b, steps = clips.shape[:2]
    previous = clips.new_zeros(b, SLOTS, 3, dtype=torch.long)
    hidden = clips.new_zeros(b, model.memory_dim)
    merged, cells = [], []
    for t in range(steps):
        seen, _, cell = model.trunk.observe(
            clips[:, t], quadrants[:, t], fovea[:, t], previous, speed[:, t], hidden
        )
        merged.append(seen)
        cells.append(cell)
    context, _ = model.context(torch.stack(merged, 1))
    return context, torch.stack(cells, 1)


class _Block(nn.Module):
    """Mixes tokens under the block-causal mask it is handed, as LeVJEPA's blocks do, and
    uses the other keyword arguments, so a stop that loses any of them changes the result."""

    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x, T=None, H_patches=None, W_patches=None, token_ids=None, attn_mask=None):
        weights = attn_mask[:, 0].float()
        mixed = (weights / weights.sum(-1, keepdim=True)) @ x
        return x + torch.tanh(self.fc(mixed)) * T / (H_patches * W_patches)


class _ViT(nn.Module):
    def __init__(self, dim=8, patch=4, depth=5):
        super().__init__()
        self.patch_size = patch
        self.patch_embed = nn.Conv3d(3, dim, (1, patch, patch), (1, patch, patch))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.blocks = nn.ModuleList(_Block(dim) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.out_layers = None

    def forward(self, x):
        steps, h, w = x.shape[2], x.shape[3] // self.patch_size, x.shape[4] // self.patch_size
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(len(x), -1, -1), x], 1)
        # The summary token sees every frame; each patch its own frame and those before.
        frame = torch.cat([torch.tensor([steps]), torch.arange(steps).repeat_interleave(h * w)])
        mask = (frame[:, None] >= frame[None, :])[None, None]
        for block in self.blocks:
            x = block(x, T=steps, H_patches=h, W_patches=w, token_ids=None, attn_mask=mask)
        return self.norm(x)


class _LeVJEPA(nn.Module):
    """The shape of LeVJEPA's Hugging Face wrapper, small enough for the CPU."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(embed_dim=8, patch_size=4)
        self.encoder = _ViT()

    def forward(self, pixel_values):
        return SimpleNamespace(last_hidden_state=self.encoder(pixel_values))


class _Plain(nn.Module):
    """An encoder with no frozen/tail split, like the tests' other stand-ins."""

    dim = 8

    def forward(self, clip, quadrants=None):
        pooled = clip.mean((1, 2, 3, 4))[:, None] + quadrants.mean((1, 2, 3, 4))[:, None]
        return pooled.expand(-1, self.dim), pooled[:, :, None, None].expand(-1, self.dim, 4, 4)


@pytest.fixture(scope="module")
def screen():
    torch.manual_seed(20)
    return ScreenEncoder(size=64, pretrained=False)


def _video(monkeypatch, train_last=2):
    import hoi4_arena.models as models

    torch.manual_seed(21)
    monkeypatch.setattr(models, "load_encoder", lambda path: _LeVJEPA())
    return VideoEncoder("levjepa", train_last=train_last)


@contextmanager
def _training(module, setting):
    """The requires_grad patterns training meets: the default last blocks, none, all.

    Put back afterwards, since the screen encoder is built once for the module."""
    before = [(p, p.requires_grad) for p in module.parameters()]
    if setting == "none":
        module.requires_grad_(False)
    elif setting == "all":
        module.requires_grad_(True)
    try:
        yield module
    finally:
        for parameter, flag in before:
            parameter.requires_grad_(flag)


def _batch(steps=4, seed=22):
    generator = torch.Generator().manual_seed(seed)
    return {
        "clips": torch.randn(2, steps, 3, 8, 16, 16, generator=generator),
        "quadrants": torch.randn(2, steps, QUADRANTS, 3, 32, 32, generator=generator),
        "fovea": torch.randn(2, steps, 3, 16, 16, generator=generator),
        "previous": torch.randint(0, 4, (2, steps, SLOTS, 3), generator=generator),
        "speed": torch.randint(1, 6, (2, steps), generator=generator),
    }


def _run(policy, fn):
    policy.zero_grad(set_to_none=True)
    memory, values, features, cells = fn()
    (memory.sum() + values.sum() + features.mean() + cells.mean()).backward()
    grads = {n: p.grad.clone() for n, p in policy.named_parameters() if p.grad is not None}
    return (memory, values, features, cells), grads


def _same(expected, actual):
    (outputs, grads), (new_outputs, new_grads) = expected, actual
    for a, b in zip(outputs, new_outputs, strict=True):
        assert a.shape == b.shape
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-5), (a - b).abs().max()
    assert grads.keys() == new_grads.keys()
    for name, grad in grads.items():
        assert torch.allclose(grad, new_grads[name], rtol=1e-4, atol=1e-5), name


@pytest.mark.parametrize("kind", ["screen", "video", "plain"])
@pytest.mark.parametrize("setting", ["default", "none", "all"])
def test_reading_the_window_at_once_learns_exactly_what_stepping_did(
    kind, setting, screen, monkeypatch
):
    if kind == "plain" and setting != "default":
        pytest.skip("nothing to freeze")
    encoder = {"screen": lambda: screen, "video": lambda: _video(monkeypatch), "plain": _Plain}
    with _training(encoder[kind](), setting) as encoder:
        _compare_unrolls(encoder)


def _compare_unrolls(encoder):
    torch.manual_seed(23)
    policy = Policy(encoder, memory_dim=16)
    batch = _batch()
    expected = _run(policy, lambda: unroll_by_step(policy, batch, burn_in=2))
    for checkpoint in (False, True):
        for chunk in (1, 3, 64):
            _same(
                expected,
                _run(
                    policy,
                    lambda: unroll(policy, batch, burn_in=2, checkpoint=checkpoint, chunk=chunk),
                ),
            )
    if not reads_clip(encoder):
        clipless = {k: v for k, v in batch.items() if k != "clips"}
        _same(expected, _run(policy, lambda: unroll(policy, clipless, burn_in=2)))
    # Validation, with no graph anywhere, and no burn-in at all.
    with torch.no_grad():
        for burn_in in (0, 2):
            old = unroll_by_step(policy, batch, burn_in, training=False)
            new = unroll(policy, batch, burn_in, training=False)
            for a, b in zip(old, new, strict=True):
                assert torch.allclose(a, b, rtol=1e-4, atol=1e-5)


def test_only_the_value_head_training_reads_the_whole_tower_without_a_graph(screen):
    """train-critic freezes everything but the value head."""
    torch.manual_seed(24)
    policy = Policy(screen, memory_dim=16)
    with _training(policy, "none"):
        policy.value.requires_grad_(True)
        batch = _batch()
        expected = _run(policy, lambda: unroll_by_step(policy, batch, checkpoint=True))
        _same(expected, _run(policy, lambda: unroll(policy, batch, checkpoint=True)))
        assert expected[1].keys() == {"value.weight", "value.bias"}


@pytest.mark.parametrize("kind", ["screen", "video"])
@pytest.mark.parametrize("setting", ["default", "none", "all"])
def test_the_split_encoder_is_the_whole_encoder(kind, setting, screen, monkeypatch):
    """`tail(frozen(x))` is `forward(x)` to the bit, whatever trains."""
    batch = _batch()
    clip, quadrants = batch["clips"][:, 0], batch["quadrants"][:, 0]
    with _training(screen if kind == "screen" else _video(monkeypatch), setting) as encoder:
        whole = encoder(clip, quadrants)
        state = encoder.frozen(clip, quadrants)
        parts = encoder.tail(state)
    for a, b in zip(whole, parts, strict=True):
        assert torch.equal(a, b)
    assert not any(t.requires_grad for t in state if torch.is_tensor(t))
    # Ten of the twelve blocks run before the tail by default, all when none train, and
    # none when the stem trains too.
    blocks = len(screen.model.blocks) if kind == "screen" else len(encoder.model.encoder.blocks)
    assert state[-1] == {"default": blocks - 2, "none": blocks, "all": None}[setting]


@pytest.mark.parametrize("kind", ["screen", "video", "plain"])
def test_the_inverse_model_reads_its_window_at_once_and_learns_the_same(kind, screen, monkeypatch):
    encoder = {"screen": lambda: screen, "video": lambda: _video(monkeypatch), "plain": _Plain}
    torch.manual_seed(25)
    model = InverseDynamics(encoder[kind](), memory_dim=16)
    batch = _batch(steps=3)
    views = batch["clips"], batch["quadrants"], batch["fovea"], batch["speed"]

    def run(fn):
        model.zero_grad(set_to_none=True)
        context, cells = fn()
        (context.sum() + cells.mean()).backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        return (context, cells), grads

    expected = run(lambda: idm_by_step(model, *views))
    for checkpoint in (False, True):
        for chunk in (1, 4):
            new = run(lambda: model(*views, checkpoint=checkpoint, chunk=chunk))
            (outputs, grads), (new_outputs, new_grads) = expected, new
            for a, b in zip(outputs, new_outputs, strict=True):
                assert torch.allclose(a, b, rtol=1e-4, atol=1e-5)
            assert grads.keys() == new_grads.keys()
            for name, grad in grads.items():
                assert torch.allclose(grad, new_grads[name], rtol=1e-4, atol=1e-5), name


def test_only_the_video_encoder_reads_clips(screen, monkeypatch):
    assert not reads_clip(screen)
    assert reads_clip(_video(monkeypatch))
    assert reads_clip(_Plain())


class _Screen(nn.Module):
    """Reads the quadrants only, like the Qwen3.5 tower, and says so."""

    dim, reads_clip = 8, False

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, clip, quadrants):
        assert clip is None, "clips were built for an encoder that does not read them"
        pooled = quadrants.mean((1, 2, 3, 4))[:, None] * self.scale
        return pooled.expand(-1, self.dim), pooled[:, :, None, None].expand(-1, self.dim, 4, 4)


@pytest.mark.parametrize("workers", [0, 2])
def test_behaviour_cloning_trains_from_worker_processes_without_clips(
    tmp_path, monkeypatch, workers
):
    import json
    import shutil

    from test_dataset import _recording

    import hoi4_arena.train as train

    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    (tmp_path / "data").mkdir()
    for name, split in (("game", "train"), ("held-out", "validation")):
        _recording(tmp_path / "data" / name, [8, 6])
        manifest = tmp_path / "data" / name / "manifest.json"
        manifest.write_text(json.dumps({**json.loads(manifest.read_text()), "split": split}))
    monkeypatch.setattr(train, "build_encoder", lambda path, variant: _Screen())
    train.train_bc(
        tmp_path / "data", "model", tmp_path / "out", sequence=2, burn_in=1, workers=workers
    )
    rows = [json.loads(line) for line in (tmp_path / "out" / "metrics.jsonl").open()]
    assert sum("step" in row for row in rows) == 2  # Four windows, two a batch.
    assert rows[-1]["validation_nll"] > 0
    assert (tmp_path / "out" / "epoch-0000.pt").exists()

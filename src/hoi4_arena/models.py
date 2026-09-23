from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .actions import GRID, SLOTS, VOCAB
from .benchmark import load_encoder, verify_model_source

# The pointer is placed in two steps: one of CELLS x CELLS cells of the screen, then one of
# CELLS x CELLS positions inside that cell. That is exactly the GRID x GRID lattice the
# action tokens already use (cell times CELLS plus offset, on each axis), so recorded labels
# and checkpoints keep their meaning. At 1080p a cell is 60 x 34 px and a step inside it
# about 1.9 x 1.1 px.
CELLS = 32
assert CELLS * CELLS == GRID
# Channels of every spatial feature the pointer head scores.
CELL_DIM = 256
# Game speeds 1 to 5, and 0 for a recording that predates the field.
SPEEDS = 6


class VideoEncoder(nn.Module):
    """The global clip encoder. Returns the summary token and the last frame's patch grid.

    The encoder attends block-causally over time, so the last frame's patch tokens have
    seen the whole clip. Keeping them, and not only the summary token, is what lets the
    pointer head score places on the screen: a single pooled vector had to encode every
    position the policy might click.

    It reads the last `frames` of the clip. Probed on recorded games (2026-09-23), LeVJEPA
    told the camera's motion apart best from four 448x256 frames in sequence (64.6%, where
    eight gave 59.1% and the Qwen3.5 tower's four side by side 54.4%), so that is the
    default, and why the inverse dynamics model uses it.
    """

    def __init__(self, model_path, variant="large", train_last=2, frames=4):
        super().__init__()
        if variant == "large":
            self.model = load_encoder(model_path)
        elif variant == "tiny":
            from transformers import AutoConfig, AutoModel

            verify_model_source(model_path)
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
            config.embed_dim, config.depth, config.num_heads = 192, 12, 3
            self.model = AutoModel.from_config(config, trust_remote_code=True)
        else:
            raise ValueError(variant)
        self.dim = self.model.config.embed_dim
        self.patch = self.model.config.patch_size
        self.variant, self.frames = variant, frames
        if variant == "large" and train_last >= 0:
            self.model.requires_grad_(False)
            if train_last:
                for block in self.model.encoder.blocks[-train_last:]:
                    block.requires_grad_(True)
                self.model.encoder.norm.requires_grad_(True)

    def forward(self, clip, quadrants=None):
        """`quadrants` is ignored: this encoder reads the global clip."""
        # All input frames are <= current observation time. No token dropping for control.
        clip = clip[:, :, -self.frames :]
        tokens = self.model(pixel_values=clip).last_hidden_state
        h, w = clip.shape[-2] // self.patch, clip.shape[-1] // self.patch
        # Patches are ordered time-major after the summary token, so the last h*w tokens
        # are the newest frame.
        grid = tokens[:, -h * w :].transpose(1, 2).reshape(tokens.shape[0], self.dim, h, w)
        return tokens[:, 0], grid


class ScreenEncoder(nn.Module):
    """The vision tower of Qwen3.5-0.8B, reading the four detail quadrants as one screen.

    Chosen on 2026-09-23 from every vision encoder released in 2026 that we could find,
    by probing each, frozen, on real frames of our recorded games. Characters of 10-12 px
    were drawn onto the game's own pixels and a linear read-out had to name them (chance
    2.8%). At 896 px this tower named 58.4% and found the pointer in 93%; the best
    general-purpose encoders managed 9-17% (LingBot-Vision, EUPE, TIPSv2), an OCR
    specialist 46-49% more slowly (MonkeyOCRv2-B), and Gemma 4's tower 4.4%. It was
    trained with a vision-language model on documents, screenshots and GUIs, which is the
    difference. 37 ms at 896 px on the 4060 Ti in bfloat16, against 63.5 for LeVJEPA.

    It sees one frame, the quadrants tiled back into a 16:9 screen of `size` (height,
    width); the policy's memory carries time. Probed again on 2026-09-23 it read 58.0% at
    1152x640 against 56.5% from the 896 square it had been fed, in 32 ms against 36. Returns the patch grid's mean as the summary, and the grid. Weights are
    timm's `qwen3_vit_88m_enc.qwen3_5_0_8b` (Apache-2.0), kept locally: `model_path` is
    the folder holding its `model.safetensors`, and nothing is fetched at run time.
    """

    ARCH = "qwen3_vit_88m_enc"

    def __init__(self, model_path=None, *, size=(640, 1152), train_last=2, pretrained=True):
        super().__init__()
        import timm

        if pretrained:
            weights = Path(model_path)
            weights = weights / "model.safetensors" if weights.is_dir() else weights
            if not weights.is_file():
                raise FileNotFoundError(f"No screen encoder weights at {weights}")
            self.model = timm.create_model(
                self.ARCH, pretrained=True, pretrained_cfg_overlay={"file": str(weights)}
            )
        else:
            self.model = timm.create_model(self.ARCH, pretrained=False)
        self.dim, self.variant = self.model.embed_dim, "screen"
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        if train_last >= 0:
            self.model.requires_grad_(False)
            for block in list(self.model.blocks)[len(self.model.blocks) - train_last :]:
                block.requires_grad_(True)
        # The views arrive normalized with ImageNet statistics; this tower expects mean
        # 0.5 and std 0.5 per channel, that is [-1, 1].
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[:, None, None], False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[:, None, None], False)

    def forward(self, clip, quadrants):
        top = torch.cat([quadrants[:, 0], quadrants[:, 1]], -1)
        bottom = torch.cat([quadrants[:, 2], quadrants[:, 3]], -1)
        screen = torch.cat([top, bottom], -2)
        if tuple(screen.shape[-2:]) != self.size:
            screen = F.interpolate(screen.float(), self.size, mode="area")
        screen = ((screen.float() * self.std + self.mean) - 0.5) / 0.5
        grid = self.model.forward_features(screen.to(self.model.patch_embed.proj.weight.dtype))
        grid = grid.permute(0, 3, 1, 2)  # (B, h, w, C) to (B, C, h, w)
        return grid.mean((-2, -1)), grid


def build_encoder(model_path, variant="large", **kwargs):
    """The encoder a checkpoint names: "large" or "tiny" LeVJEPA, or "screen" (Qwen3.5 tower)."""
    if variant == "screen":
        return ScreenEncoder(model_path, **kwargs)
    return VideoEncoder(model_path, variant=variant, **kwargs)


class Stage(nn.Module):
    """A strided convolution and a residual convolution after it."""

    def __init__(self, inputs, outputs):
        super().__init__()
        self.down = nn.Conv2d(inputs, outputs, 3, 2, 1)
        self.norm = nn.GroupNorm(8, outputs)
        self.body = nn.Conv2d(outputs, outputs, 3, 1, 1)

    def forward(self, x):
        x = F.gelu(self.norm(self.down(x)))
        return x + F.gelu(self.body(x))


class DetailEncoder(nn.Module):
    """A convolutional reader for the quadrants and the fovea, at stride 16.

    It keeps the spatial map. The old reader pooled each 224 px tile to 2 x 2 before the
    policy saw it, which kept that something was there and threw away where.
    """

    def __init__(self, width=CELL_DIM):
        super().__init__()
        self.stages = nn.Sequential(Stage(3, 32), Stage(32, 64), Stage(64, 128), Stage(128, width))

    def forward(self, images):
        """(..., 3, H, W) to (..., width, H / 16, W / 16)."""
        lead = images.shape[:-3]
        maps = self.stages(images.reshape(-1, *images.shape[-3:]))
        return maps.reshape(*lead, *maps.shape[-3:])


def screen_cells(quadrant_maps):
    """Four quadrant maps pooled and tiled into one CELLS x CELLS map of the whole screen.

    (B, 4, C, h, w) in the order top-left, top-right, bottom-left, bottom-right, to
    (B, C, CELLS, CELLS). Each quadrant is one quarter of the cells.
    """
    half = CELLS // 2
    pooled = F.adaptive_avg_pool2d(quadrant_maps.flatten(0, 1), half)
    pooled = pooled.reshape(*quadrant_maps.shape[:3], half, half)
    top = torch.cat([pooled[:, 0], pooled[:, 1]], -1)
    bottom = torch.cat([pooled[:, 2], pooled[:, 3]], -1)
    return torch.cat([top, bottom], -2)


def gumbel_argmax(logits):
    """Draw a category with the same law as Categorical.sample, without the object.

    argmax(logits + Gumbel(0,1)) is exactly a categorical draw, and a standard Gumbel is
    minus the log of a unit exponential. The textbook -log(-log(U)) form is rejected on
    both of its logs: torch.rand returns an exact zero often enough to matter, which
    would make one category unreachable forever, and the inner log loses precision as U
    approaches one. exponential_ is drawn on the open interval, so one kernel and one log
    suffice and no clamp is needed.
    """
    return (logits - torch.empty_like(logits).exponential_().log_()).argmax(-1)


def categorical(logits):
    """Normalized log-probabilities, the way torch.distributions.Categorical does it.

    Subtracting a logsumexp and taking a log_softmax are the same function and not the
    same bits -- they disagree by up to 2e-6, which is enough to move a stored old_logp
    and with it a PPO ratio. So this is the slower two-kernel form deliberately, and
    `log_prob` and `entropy` below are bit-for-bit what Categorical returned.
    """
    return logits - logits.logsumexp(-1, keepdim=True)


def log_prob(normalized, values):
    return normalized.gather(-1, values[:, None]).squeeze(-1)


def entropy(normalized, probabilities):
    floor = torch.finfo(normalized.dtype).min
    return -(normalized.clamp(min=floor) * probabilities).sum(-1)


class ActionHead(nn.Module):
    """Autoregressive event slots with an exact, replayable conditional likelihood.

    Each slot picks an event kind. A move also picks where: first a cell of the screen,
    by scoring every cell's features against the slot's state, then a position inside
    that cell, from the chosen cell's own features. So a click is chosen by looking at what
    is at each place on the screen, and x and y are chosen together rather than as two
    independent numbers. The old head took x and y from two separate 1024-way outputs of
    one pooled vector.

    The likelihood is computed directly rather than through torch.distributions. That is
    not a stylistic preference: Distribution validates its arguments whenever __debug__
    is set, and each validation ends in `Tensor.__bool__` on a CUDA tensor, which is a
    host-device synchronization, paid for every head of every slot -- including on the
    teacher-forcing path, which never samples anything. The arithmetic below reproduces
    Categorical's exactly, and a test pins it bit-for-bit.
    """

    def __init__(self, memory_dim=512, noise_dim=16, cell_dim=CELL_DIM, latents=0):
        super().__init__()
        self.noise_dim = noise_dim
        # Learned latents for XM to choose among, in place of Gaussian noise (xm_loss).
        if latents:
            self.latents = nn.Parameter(torch.randn(latents, noise_dim))
        self.init = nn.Linear(memory_dim + noise_dim, 256)
        self.embedding = nn.Embedding(len(VOCAB), 64)
        self.xy = nn.Linear(2, 64)
        self.cell = nn.GRUCell(64, 256)
        self.kinds = nn.Linear(256, len(VOCAB))
        self.query = nn.Linear(256, cell_dim)
        # A learned preference over cells, before looking: popups open near the middle,
        # the speed and pause controls sit in one corner.
        self.cell_bias = nn.Parameter(torch.zeros(GRID))
        self.fine = nn.Sequential(nn.Linear(256 + cell_dim, 256), nn.GELU(), nn.Linear(256, GRID))
        self.scale = 1 / math.sqrt(cell_dim)

    def forward(self, memory, cells, actions=None, noise=None, deterministic=False):
        """Sample (or score, given `actions`) the eight slots.

        `cells` is (B, GRID, cell_dim): the screen's CELLS x CELLS map, row-major, as
        Policy builds it. Returns the actions, their summed log-likelihood and the summed
        entropy.

        The entropy of a slot is the kind's, plus, weighted by the chance of a move, the
        cell's and the position's within one cell. That last term is exact only for the
        cell the slot actually chose (the sampled or scored one; cell 0 when the slot is
        not a move). The expectation over all 1024 cells would need 1024 fine heads per
        slot, a million logits, for a regularizer.
        """
        b = memory.shape[0]
        rows = torch.arange(b, device=memory.device)
        if noise is None:
            noise = torch.zeros(b, self.noise_dim, device=memory.device, dtype=memory.dtype)
        state = torch.tanh(self.init(torch.cat([memory, noise], -1)))
        previous = torch.zeros(b, 64, device=memory.device, dtype=memory.dtype)
        result, logps, entropies = [], [], []
        for slot in range(SLOTS):
            state = self.cell(previous, state)
            # The float casts are load-bearing, not incidental: run in bfloat16 this same
            # normalization lands about 0.05 away, which is a large number to put inside
            # a ratio of likelihoods.
            kinds = categorical(self.kinds(state).float())
            where = torch.einsum("bnc,bc->bn", cells, self.query(state).to(cells.dtype))
            places = categorical(where.float() * self.scale + self.cell_bias.float())
            if actions is not None:
                kind, x, y = actions[:, slot].unbind(-1)
                place = (y // CELLS) * CELLS + x // CELLS
                offset = (y % CELLS) * CELLS + x % CELLS
            elif deterministic:
                kind, place = kinds.argmax(-1), places.argmax(-1)
            else:
                kind, place = gumbel_argmax(kinds), gumbel_argmax(places)
            chosen = cells[rows, place].to(state.dtype)
            fine = categorical(self.fine(torch.cat([state, chosen], -1)).float())
            if actions is None:
                offset = fine.argmax(-1) if deterministic else gumbel_argmax(fine)
            kind_p, place_p, fine_p = kinds.softmax(-1), places.softmax(-1), fine.softmax(-1)
            move = (kind == 1).float()
            lp = log_prob(kinds, kind) + move * (log_prob(places, place) + log_prob(fine, offset))
            expected = entropy(kinds, kind_p) + kind_p[:, 1] * (
                entropy(places, place_p) + entropy(fine, fine_p)
            )
            x = (place % CELLS) * CELLS + offset % CELLS
            y = (place // CELLS) * CELLS + offset // CELLS
            values = torch.stack([kind, x, y], -1) * torch.stack(
                [torch.ones_like(kind), move.long(), move.long()], -1
            )
            previous = self.embedding(values[:, 0]) + self.xy(
                values[:, 1:].to(memory.dtype) / (GRID - 1)
            )
            result.append(values)
            logps.append(lp)
            entropies.append(expected)
        return (
            torch.stack(result, 1),
            torch.stack(logps, 1).sum(1),
            torch.stack(entropies, 1).sum(1),
        )


class Policy(nn.Module):
    """Global clip, quadrants and fovea in; a recurrent memory, a value and the cells out.

    The cells are the screen as a CELLS x CELLS map: the quadrants' detail pooled into
    each cell, plus the video encoder's patch grid (which has seen the whole clip)
    resized onto the same map, plus a learned position. The action head scores them to
    place the pointer. The memory reads the whole screen through their mean, the pointer's
    surroundings through the fovea, and one place of its own choosing through attention.
    """

    def __init__(self, encoder, memory_dim=512, latents=0):
        super().__init__()
        self.encoder = encoder
        self.details = DetailEncoder()
        self.foveal = DetailEncoder()
        self.global_cells = nn.Conv2d(encoder.dim, CELL_DIM, 1)
        self.position = nn.Parameter(torch.zeros(1, CELL_DIM, CELLS, CELLS))
        self.cell_norm = nn.LayerNorm(CELL_DIM)
        self.previous_action = nn.Linear(SLOTS * 3, 64)
        self.speed = nn.Embedding(SPEEDS, 32)
        self.read = nn.Linear(memory_dim, CELL_DIM)
        # Summary token, mean cell, fovea, attention readout, previous action, speed.
        self.fusion = nn.Linear(encoder.dim + 3 * CELL_DIM + 64 + 32, memory_dim)
        self.memory = nn.GRUCell(memory_dim, memory_dim)
        self.actor = ActionHead(memory_dim, latents=latents)
        self.value = nn.Linear(memory_dim, 1)
        self.memory_dim = memory_dim

    def cells(self, grid, quadrants):
        """The (B, GRID, CELL_DIM) map the action head scores."""
        local = screen_cells(self.details(quadrants))
        broad = F.interpolate(
            self.global_cells(grid), (CELLS, CELLS), mode="bilinear", align_corners=False
        )
        cells = (local + broad + self.position).flatten(2).transpose(1, 2)
        return self.cell_norm(cells)

    def forward(self, clip, quadrants, fovea, previous, speed, hidden=None, reset=None):
        """One decision. `speed` is the game speed, 1 to 5 (0 unknown), per sample.

        Returns the new memory, the value logit, the summary token (for the auxiliary
        loss) and the cells (for the action head). The value is a logit of the scaled
        return: `learning.value_estimate` turns it into a return, and it is trained with
        binary cross-entropy (`learning.critic_loss`).
        """
        if hidden is None:
            hidden = clip.new_zeros(clip.shape[0], self.memory_dim)
        if reset is not None:
            hidden = hidden * (~reset.bool()).to(hidden.dtype)[:, None]
        merged, summary, cells = self.observe(clip, quadrants, fovea, previous, speed, hidden)
        hidden = self.memory(merged, hidden)
        return hidden, self.value(hidden).squeeze(-1), summary, cells

    def observe(self, clip, quadrants, fovea, previous, speed, hidden):
        """What one decision sees, fused into one vector, before the memory takes it in.

        `hidden` only steers the attention readout. Returns the fused vector, the summary
        token and the cells.
        """
        summary, cells, centre = self.perceive(clip, quadrants, fovea)
        merged = fuse(self, summary, cells, centre, previous, speed, hidden, clip.dtype)
        return merged, summary, cells

    def perceive(self, clip, quadrants, fovea):
        """What the screen shows, before any memory: the summary, the cells, the fovea."""
        summary, grid = self.encoder(clip, quadrants)
        return summary, self.cells(grid, quadrants), self.foveal(fovea).mean((-2, -1))


def fuse(module, summary, cells, centre, previous, speed, hidden, dtype=None):
    """One decision's inputs as one vector for the memory.

    The screen through its summary, its mean cell and the fovea, one place the memory
    chooses to read (attention over the cells, steered by `hidden`), the previous action
    and the game speed. `module` holds the layers: a Policy, or features.MemoryHead,
    which trains the same layers on cached perception. `dtype` is what the parts are
    joined in, the clip's for a Policy, as it always was.
    """
    dtype = dtype or summary.dtype
    scales = previous.new_tensor([len(VOCAB) - 1, GRID - 1, GRID - 1])
    prior = module.previous_action((previous / scales).flatten(1).to(dtype))
    attention = torch.einsum("bnc,bc->bn", cells, module.read(hidden).to(cells.dtype))
    weights = (attention.float() / math.sqrt(CELL_DIM)).softmax(-1).to(cells.dtype)
    readout = torch.einsum("bn,bnc->bc", weights, cells)
    merged = torch.cat(
        [summary, cells.mean(1), centre, readout, prior, module.speed(speed)], -1
    ).to(dtype)
    return F.gelu(module.fusion(merged))


class InverseDynamics(nn.Module):
    """Labels the inputs behind a stretch of video, seeing what came after each decision.

    The policy has to act on the past alone. The inverse dynamics model does not: it is
    shown each decision's clip shifted into the future, so the last frames already show
    what the input did, and the views of the frame one interval later, where a moved
    pointer has arrived. A two-way recurrence then runs over the window, so each decision
    also knows its neighbours. That makes it a much easier problem than acting, and the
    point: trained on recordings whose inputs are known, it labels video whose inputs are
    not (Baker et al., 2022, Video PreTraining), and the labelled video trains the policy.

    It reuses the policy's reader and action head, so its labels are in the same lattice,
    and its cells come from the later frame, where the pointer ends up.
    """

    def __init__(self, encoder, memory_dim=512):
        super().__init__()
        self.trunk = Policy(encoder, memory_dim)
        # The trunk's own recurrence and value are not used; the two-way recurrence is.
        self.trunk.memory = self.trunk.value = None
        self.context = nn.GRU(memory_dim, memory_dim // 2, batch_first=True, bidirectional=True)
        self.memory_dim = memory_dim

    @property
    def actor(self):
        return self.trunk.actor

    def forward(self, clips, quadrants, fovea, speed, checkpoint=False):
        """A window of decisions: (B, T, ...) views in, per-decision context and cells out.

        The previous action is not an input: it is what a neighbouring decision is being
        asked to label. Each decision is read on its own; with `checkpoint` (and gradients
        on) its activations are recomputed in the backward pass rather than kept, which a
        16-step window needs on an 8 GB card (see train.train_bc).
        """
        b, steps = clips.shape[:2]
        previous = clips.new_zeros(b, SLOTS, 3, dtype=torch.long)
        hidden = clips.new_zeros(b, self.memory_dim)
        merged, cells = [], []
        for t in range(steps):
            inputs = (clips[:, t], quadrants[:, t], fovea[:, t], previous, speed[:, t], hidden)
            if checkpoint and torch.is_grad_enabled():
                seen, _, cell = torch.utils.checkpoint.checkpoint(
                    self.trunk.observe, *inputs, use_reentrant=False
                )
            else:
                seen, _, cell = self.trunk.observe(*inputs)
            merged.append(seen)
            cells.append(cell)
        context, _ = self.context(torch.stack(merged, 1))
        return context, torch.stack(cells, 1)


def limit_gpu_memory(fraction: float | None):
    """Cap this process's GPU allocations at `fraction` of the card, so running out fails.

    On Windows the driver's default lets a CUDA process that fills the card spill into
    system memory instead of failing: measured on 2026-09-23, a training step that
    reached 7 GB of the 8 GB 4060 Ti took 34.6 s instead of about 2.6. The caching
    allocator refuses anything past the cap with an out-of-memory error, which is the
    failure that should happen. None leaves the card uncapped. Returns the cap in MiB.
    """
    if fraction is None or not torch.cuda.is_available():
        return None
    if not 0 < fraction <= 1:
        raise ValueError("the GPU memory fraction must be in (0, 1]")
    torch.cuda.set_per_process_memory_fraction(fraction)
    return int(torch.cuda.get_device_properties(0).total_memory * fraction / 2**20)


def configure_precision(tf32: bool = False):
    """Say out loud what float32 matmuls are allowed to do, instead of inheriting it.

    Torch leaves float32 matmuls at full precision and warns that the tensor cores are
    idle, which reads like a free speedup being declined. Measured, it is not one here:
    every matmul in this project runs inside torch.autocast in bfloat16, and autocast
    lowers matmul, addmm, linear and GRUCell regardless of what dtype reaches them. The
    two candidates that look like exceptions are not -- ActionHead casts the head's
    *output* to float32, after the Linear has already run in bfloat16, and rdmreg's
    explicit .float() sits inside the same autocast block. There is no float32 GEMM left
    for TF32 to accelerate, and a synthetic one at these shapes measures between 1.01x
    and 1.18x, all of it launch overhead at batch one.

    So the default is off, and the reason is not caution about the speedup. TF32 keeps
    ten mantissa bits, and on the shapes this project does use it moved results by up to
    2.1e-2 -- against a PPO log-likelihood, that is a phantom ratio larger than anything
    the optimizer is being asked to find. Zero measured gain is not worth that.

    Use set_float32_matmul_precision and not the newer
    torch.backends.cuda.matmul.fp32_precision: on torch 2.11 assigning that attribute
    leaves get_float32_matmul_precision() raising RuntimeError.
    """
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    return torch.get_float32_matmul_precision()


def halve_frozen(module):
    """Keep every parameter that carries no gradient in bfloat16.

    The live path shipped 1170 MiB of float32 weights to the card and then ran every
    matmul through autocast in bfloat16 regardless, so most of that was precision
    nothing ever read. Halving it recovers 530 MiB against a headroom gate the
    loaded-game run fails: 710 MiB free where 1024 is required.

    Only the frozen parameters, and the restriction is the whole point. Casting the
    trainable ones as well saves a further 55 MiB and moves a stored old_logp by
    1.02e-2, which is a systematic 1.01x PPO ratio on every sample before a single
    gradient step, because collection would then compute the likelihood differently from
    the update that scores it. Frozen weights cannot diverge that way -- load_policy
    serves both sides, so both run the same function -- and no optimizer reads them.
    """
    for parameter in module.parameters():
        if not parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    return module


def reprelu(value):
    soft = F.gelu(value)
    return value.relu().detach() + soft - soft.detach()


def rdmreg(z, sparse: bool, projections=256, shift=0.0):
    """LpWM distribution matching: sample axis 0, separate time axis, shared projections.

    Lower projection count is a hardware adaptation, shared by both experimental arms
    (LpWM used 1024 to 8192). The sparse target is ReLU(Laplace(shift, 1/sqrt 2)), the
    rectified generalized Gaussian with p = 1; a negative `shift` (LpWM swept 0, -1, -2)
    moves mass below zero, so more coordinates are exactly zero.
    """
    if z.shape[0] < 2:
        raise ValueError("RDMReg needs at least two independent sequences in the batch")
    z = z.float()
    directions = F.normalize(torch.randn(z.shape[-1], projections, device=z.device), dim=0)
    if sparse:
        target = (
            torch.distributions.Laplace(z.new_tensor(float(shift)), z.new_tensor(2**-0.5))
            .sample(z.shape)
            .relu()
        )
    else:
        target = torch.randn_like(z)
    actual = (z @ directions).sort(dim=0).values
    reference = (target @ directions).sort(dim=0).values
    return (actual - reference).square().mean()


def temporal_jaccard(z, valid, eps=1e-6):
    """LpWM's temporal Jaccard loss (Kuang et al., 2026, arXiv 2608.22764, eq. 8).

    One minus the soft Jaccard index, sum(min) / sum(max), of consecutive non-negative
    codes, averaged over pairs of valid steps. RDMReg shapes each step's codes alone, so
    without this the support follows whatever changes fastest; in LpWM that was the arm's
    motion, and here it would be the camera. With it, the support changed with contact
    instead (correlation with cube motion 0.21 to 0.80), at no cost in success.
    """
    a, b = z[:, :-1].float(), z[:, 1:].float()
    overlap = torch.minimum(a, b).sum(-1) / (torch.maximum(a, b).sum(-1) + eps)
    pairs = (valid[:, 1:] & valid[:, :-1]).float()
    return ((1 - overlap) * pairs).sum() / pairs.sum().clamp_min(1)


class PredictiveAuxiliary(nn.Module):
    """Training-only action-conditioned prediction; dense/sparse arms share capacity.

    `shift`, `temporal_jaccard` and `projections` are LpWM's options for the sparse arm
    (rdmreg, temporal_jaccard); their defaults keep the objective as it was.
    """

    def __init__(
        self,
        memory_dim=512,
        feature_dim=1024,
        latent_dim=384,
        mode="sparse",
        *,
        shift=0.0,
        temporal_jaccard=0.0,
        projections=256,
    ):
        super().__init__()
        if mode not in {"none", "dense", "sparse"}:
            raise ValueError(mode)
        if (shift or temporal_jaccard) and mode != "sparse":
            raise ValueError("the target shift and temporal Jaccard apply to sparse codes only")
        self.mode = mode
        self.shift, self.jaccard, self.projections = shift, temporal_jaccard, projections
        self.project = nn.Linear(feature_dim, latent_dim)
        self.context = nn.Linear(memory_dim, latent_dim)
        self.action = nn.Linear(SLOTS * 3, latent_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim)
        )

    def link(self, z):
        return reprelu(z) if self.mode == "sparse" else z

    def forward(self, memories, features, actions, valid):
        if self.mode == "none":
            return memories.sum() * 0
        z = self.link(self.project(features))
        scales = actions.new_tensor([len(VOCAB) - 1, GRID - 1, GRID - 1])
        action_features = self.action((actions / scales).flatten(-2).to(memories.dtype))
        prediction = self.link(
            self.predictor(torch.cat([self.context(memories[:, :-1]), action_features[:, :-1]], -1))
        )
        mask = valid[:, 1:] & valid[:, :-1]
        error = (prediction.float() - z[:, 1:].float()).square().mean(-1)
        prediction_loss = (error * mask).sum() / mask.sum().clamp_min(1)
        # Only regularize temporal positions with a fully valid independent batch.
        regular = z[:, valid.all(0)]
        reg = (
            rdmreg(regular, self.mode == "sparse", self.projections, self.shift)
            if regular.shape[1]
            else z.sum() * 0
        )
        loss = prediction_loss + 0.5 * reg
        if self.jaccard:
            loss = loss + self.jaccard * temporal_jaccard(z, valid)
        return loss


def xm_loss(actor, memories, cells, actions, candidates=5, form="hard"):
    """Explorative Modeling for the action head (Gladstone, Ji and Du, 2026, arXiv 2607.27372).

    Forward XM: explore K latents, train on the one whose actions best match the
    demonstration. The head's start state is conditioned on the latent, so different
    latents can specialise to different ways of acting from one state instead of
    averaging them. The latents are Gaussian draws, or, when the head has learned
    `latents`, each of those (as the paper's discrete XMDLM does), and then K is their
    number. `form`:

    - "hard", the paper's min over candidates: the best is chosen without a graph and
      re-evaluated with one, the paper's memory-saving mode.
    - "smooth", -log mean_k p(a | z_k): every candidate gets gradients, and it is exactly
      maximum likelihood of the K-candidate mixture; it differs from "hard" by at most
      log K.

    Returns the per-sample loss. The paper found autoregressive models, like this head,
    the hardest to improve: they are less limited by expressivity. It recommends
    sweeping K over 1, 2, 3, 5, then 8 and 12.
    """
    b = memories.shape[0]
    rows = torch.arange(b, device=memories.device)
    latents = getattr(actor, "latents", None)
    if latents is not None:
        noises = latents.to(memories.dtype)[:, None].expand(-1, b, -1)
    else:
        noises = torch.randn(
            candidates, b, actor.noise_dim, device=memories.device, dtype=memories.dtype
        )
    if form == "smooth":
        logps = torch.stack([actor(memories, cells, actions, noise=z)[1] for z in noises])
        return -(logps.logsumexp(0) - math.log(len(noises)))
    if form != "hard":
        raise ValueError(form)
    with torch.no_grad():
        losses = torch.stack([-actor(memories, cells, actions, noise=z)[1] for z in noises])
        best = losses.argmin(0)
    return -actor(memories, cells, actions, noise=noises[best, rows])[1]

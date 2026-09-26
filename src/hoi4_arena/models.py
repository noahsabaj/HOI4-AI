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
# Frames the vision tower reads in one call when training reads a whole window at once
# (Policy.perceive_window). With recomputation on, the backward pass rebuilds one chunk's
# trainable activations at a time, so this bounds that peak: four frames is twice what a
# step of batch 2 held before the window was read in one piece.
CHUNK = 4


def frozen_depth(blocks, stem):
    """How many leading blocks carry no gradient, or None when the stem itself trains.

    Read from requires_grad at call time rather than from `train_last`, because callers
    change it afterwards: train-critic freezes the whole policy but its value head, and
    then the whole tower runs without a graph.
    """
    if any(p.requires_grad for p in stem):
        return None
    depth = 0
    for block in blocks:
        if any(p.requires_grad for p in block.parameters()):
            break
        depth += 1
    return depth


def reads_clip(encoder):
    """Whether an encoder reads the global clip. The Qwen3.5 tower reads only the quadrants,
    so for it the clips need not be built, normalized or moved at all."""
    return getattr(encoder, "reads_clip", True)


class _Captured(Exception):
    """Raised by a pre-hook to stop a forward pass at the first block that trains."""


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

    Training reads it in two parts (`frozen`, then `tail`), so the blocks that do not
    train run once without a graph instead of again in the backward pass.
    """

    reads_clip = True

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
        return self._read(self.model(pixel_values=clip).last_hidden_state, clip)

    def _read(self, tokens, clip):
        h, w = clip.shape[-2] // self.patch, clip.shape[-1] // self.patch
        # Patches are ordered time-major after the summary token, so the last h*w tokens
        # are the newest frame.
        grid = tokens[:, -h * w :].transpose(1, 2).reshape(tokens.shape[0], self.dim, h, w)
        return tokens[:, 0], grid

    @torch.no_grad()
    def frozen(self, clip, quadrants=None):
        """The encoder up to its first block that trains, run without a graph.

        LeVJEPA's forward is one piece of reviewed remote code, so rather than rewrite its
        tokenizer, positions and block-causal mask here, it is run as it is and stopped at
        that block: a pre-hook records the tokens and keyword arguments the block was
        about to receive. `tail` continues from them. When the stem trains (the "tiny"
        student) the whole forward is left to `tail`.
        """
        clip = clip[:, :, -self.frames :]
        vit = self.model.encoder
        stem = [p for n, p in vit.named_parameters() if not n.startswith(("blocks.", "norm."))]
        depth = frozen_depth(vit.blocks, stem)
        if depth is None or getattr(vit, "out_layers", None) is not None:
            return clip, None, None
        target = vit.blocks[depth] if depth < len(vit.blocks) else vit.norm
        captured = {}

        def stop(module, args, kwargs):
            captured["inputs"] = args, kwargs
            raise _Captured

        handle = target.register_forward_pre_hook(stop, with_kwargs=True)
        try:
            self.model(pixel_values=clip)
        except _Captured:
            pass
        finally:
            handle.remove()
        return clip, captured["inputs"], depth

    def tail(self, state):
        """The rest of `frozen`'s forward, with a graph for the blocks that train."""
        clip, inputs, depth = state
        if inputs is None:
            return self(clip)
        vit = self.model.encoder
        (x, *rest), kwargs = inputs
        for block in vit.blocks[depth:]:
            x = block(x, *rest, **kwargs)
        return self._read(vit.norm(x), clip)


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
    reads_clip = False

    def __init__(self, model_path=None, *, size=(640, 1152), train_last=2, pretrained=True):
        super().__init__()
        import timm

        if pretrained:
            weights = Path(model_path)
            weights = weights / "model.safetensors" if weights.is_dir() else weights
            if not weights.is_file():
                raise FileNotFoundError(f"No screen encoder weights at {weights}")
            self.model = timm.create_model(
                tower_arch(weights.parent),
                pretrained=True,
                pretrained_cfg_overlay={"file": str(weights)},
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
        """`clip` is ignored: this encoder reads the quadrants."""
        return self._read(self.model.forward_features(self._screen(quadrants)))

    def _screen(self, quadrants):
        top = torch.cat([quadrants[:, 0], quadrants[:, 1]], -1)
        bottom = torch.cat([quadrants[:, 2], quadrants[:, 3]], -1)
        screen = torch.cat([top, bottom], -2)
        if tuple(screen.shape[-2:]) != self.size:
            screen = F.interpolate(screen.float(), self.size, mode="area")
        screen = ((screen.float() * self.std + self.mean) - 0.5) / 0.5
        return screen.to(self.model.patch_embed.proj.weight.dtype)

    @staticmethod
    def _read(grid):
        grid = grid.permute(0, 3, 1, 2)  # (B, h, w, C) to (B, C, h, w)
        return grid.mean((-2, -1)), grid

    @torch.no_grad()
    def frozen(self, clip, quadrants):
        """The tower up to its first block that trains, run without a graph.

        With the default two trainable blocks, ten of the twelve run here, once. Before,
        each training step was recomputed whole in the backward pass, frozen blocks and
        all. The steps are timm's forward_features, split in two (`tail` is the rest).
        """
        model = self.model
        screen = self._screen(quadrants)
        depth = frozen_depth(model.blocks, [model.pos_embed, *model.patch_embed.parameters()])
        if depth is None:
            return screen, None, None, None
        x = model.patch_embed(screen)
        hw = tuple(x.shape[1:3])
        x, rope = model._pos_embed(x)
        for block in model.blocks[:depth]:
            x = block(x, rope=rope)
        return x, rope, hw, depth

    def tail(self, state):
        """The rest of `frozen`'s forward, with a graph for the blocks that train."""
        x, rope, hw, depth = state
        if depth is None:
            return self._read(self.model.forward_features(x))
        for block in self.model.blocks[depth:]:
            x = block(x, rope=rope)
        return self._read(x.reshape(x.shape[0], *hw, -1))


def tower_arch(folder, default=ScreenEncoder.ARCH):
    """The timm model a tower's folder holds: its config.json's architecture and tag, as
    timm's own download of it has (the Qwen3.5-4B model's tower, 2026-09-26, reads the
    screen better than the 0.8B's: STATUS.md), or the 0.8B's for a folder without one."""
    config = Path(folder) / "config.json"
    if not config.is_file():
        return default
    import json

    meta = json.loads(config.read_text())
    tag = (meta.get("pretrained_cfg") or {}).get("tag")
    return f"{meta['architecture']}.{tag}" if tag else meta["architecture"]


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


# The kinds that press a mouse button down (actions.VOCAB), one per button.
PRESSES = [i for i, e in enumerate(VOCAB) if e and e["kind"] == "button" and e["down"]]


def pointer_blob(coordinate, sigma):
    """A Gaussian blob of width `sigma` lattice units around `coordinate`, on one axis.

    Returns the two cells nearest the point (its own, and the neighbour on the side it is
    closer to), each cell's share of the blob, and the blob within each cell over its
    CELLS positions, normalised. Two cells hold all but a sliver of it while `sigma` is
    under about a quarter of a cell: the cell beyond the point's own lies at least half a
    cell away. A neighbour clamped onto the point's own cell at the screen's edge counts
    nothing.
    """
    own = torch.div(coordinate, CELLS, rounding_mode="floor").long()
    side = torch.where(coordinate % CELLS < CELLS / 2, -1, 1)
    near = torch.stack([own, (own + side).clamp(0, CELLS - 1)], 1)
    positions = near[..., None].float() * CELLS + torch.arange(CELLS, device=near.device)
    weight = torch.exp(-0.5 * ((positions - coordinate[:, None, None]) / sigma) ** 2)
    counts = torch.stack([torch.ones_like(own), (near[:, 1] != own).long()], 1)
    weight = weight * counts[..., None]
    mass = weight.sum(-1)
    return near, mass, weight / mass.clamp_min(1e-12)[..., None]


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

    With `look`, a button is pressed only where the pointer already was when the decision
    began: after a move, a press waits for the next decision, whose fovea shows what the
    pointer landed on. A player moves to a button, sees it light up, then clicks; so does
    this, and a click is never made on something it has not looked at.
    """

    def __init__(self, memory_dim=512, noise_dim=16, cell_dim=CELL_DIM, latents=0, look=False):
        super().__init__()
        self.noise_dim = noise_dim
        self.look = look
        press = torch.zeros(len(VOCAB), dtype=torch.bool)
        press[PRESSES] = True
        self.register_buffer("press", press, persistent=False)
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

    def forward(
        self,
        memory,
        cells,
        actions=None,
        noise=None,
        deterministic=False,
        sigma=0.0,
        point=False,
        temperature=1.0,
    ):
        """Sample (or score, given `actions`) the eight slots.

        `cells` is (B, GRID, cell_dim): the screen's CELLS x CELLS map, row-major, as
        Policy builds it. Returns the actions, their summed log-likelihood and the summed
        entropy.

        `sigma` > 0, when scoring, scores each demonstrated move against a Gaussian blob
        of that width in lattice units around it instead of its one exact point: the
        cross-entropy with the blob, over the cells it covers and the positions inside
        each (`soft_pointer`). A click anywhere on a button is right, and one 3 px off
        the demonstrated pixel should not be scored as wrong as one across the screen.
        The returned score is then that negative cross-entropy, not a likelihood.

        `point`, when sampling, takes the likeliest place of a move (its cell, then the
        position inside it) while the kind is still sampled: greedy pointing, sampled
        acting. A sampled place lands on a wrong button as often as the head leaves mass
        there. The likelihood returned is then of the place taken, not of a sample.
        `temperature` below 1, when sampling, sharpens what to do: the likeliest input of
        each slot gains, and rarely chosen ones, such as a camera's aimless moves, fade.

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
        moved = torch.zeros(b, dtype=torch.bool, device=memory.device)
        for slot in range(SLOTS):
            state = self.cell(previous, state)
            # The float casts are load-bearing, not incidental: run in bfloat16 this same
            # normalization lands about 0.05 away, which is a large number to put inside
            # a ratio of likelihoods.
            logits = self.kinds(state).float()
            if self.look:
                logits = logits.masked_fill(moved[:, None] & self.press, float("-inf"))
            kinds = categorical(logits)
            where = torch.einsum("bnc,bc->bn", cells, self.query(state).to(cells.dtype))
            places = categorical(where.float() * self.scale + self.cell_bias.float())
            if actions is not None:
                kind, x, y = actions[:, slot].unbind(-1)
                place = (y // CELLS) * CELLS + x // CELLS
                offset = (y % CELLS) * CELLS + x % CELLS
            elif deterministic:
                kind, place = kinds.argmax(-1), places.argmax(-1)
            else:
                kind = gumbel_argmax(kinds / temperature if temperature != 1.0 else kinds)
                place = places.argmax(-1) if point else gumbel_argmax(places)
            chosen = cells[rows, place].to(state.dtype)
            fine = categorical(self.fine(torch.cat([state, chosen], -1)).float())
            if actions is None:
                offset = fine.argmax(-1) if deterministic or point else gumbel_argmax(fine)
            kind_p, place_p, fine_p = kinds.softmax(-1), places.softmax(-1), fine.softmax(-1)
            move = (kind == 1).float()
            moved = moved | (kind == 1)
            if actions is not None and sigma > 0:
                pointer = -self.soft_pointer(state, cells, rows, x, y, places, sigma)
            else:
                pointer = log_prob(places, place) + log_prob(fine, offset)
            lp = log_prob(kinds, kind) + move * pointer
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

    def soft_pointer(self, state, cells, rows, x, y, places, sigma):
        """Cross-entropy of the pointer with a Gaussian blob around the target (x, y).

        The blob is separable, so its share of each of the 2x2 cells nearest the target is
        the product of the two axes' shares, and inside each cell it is the outer product
        of the two axes' profiles. The fine head is run for all four cells, each scored
        against its part of the blob and weighted by that cell's share.
        """
        cx, mx, fx = pointer_blob(x.float(), sigma)
        cy, my, fy = pointer_blob(y.float(), sigma)
        ids = (cy[:, :, None] * CELLS + cx[:, None, :]).flatten(1)
        share = (my[:, :, None] * mx[:, None, :]).flatten(1)
        share = share / share.sum(-1, keepdim=True).clamp_min(1e-12)
        coarse = -(share * places.gather(-1, ids)).sum(-1)
        chosen = cells[rows[:, None], ids].to(state.dtype)
        wide = torch.cat([state[:, None].expand(-1, 4, -1), chosen], -1)
        fine = categorical(self.fine(wide).float())
        target = (fy[:, :, None, :, None] * fx[:, None, :, None, :]).flatten(3).flatten(1, 2)
        return coarse + (share * -(target * fine).sum(-1)).sum(-1)

    @torch.no_grad()
    def pointer_map(self, memory, cells, actions, slot, noise=None, top=16):
        """Where the head would put the pointer at `slot`, over the whole lattice.

        The slots before `slot` are teacher-forced to `actions`. Returns the probability
        of each kind there, and a (GRID, GRID) map (rows y, columns x) of where a move
        would go: each cell's probability times the position inside it, run exactly for
        the `top` most likely cells and spread evenly over the rest. For one sample.
        """
        b = memory.shape[0]
        if noise is None:
            noise = torch.zeros(b, self.noise_dim, device=memory.device, dtype=memory.dtype)
        state = torch.tanh(self.init(torch.cat([memory, noise], -1)))
        previous = torch.zeros(b, 64, device=memory.device, dtype=memory.dtype)
        moved = torch.zeros(b, dtype=torch.bool, device=memory.device)
        for step in range(slot + 1):
            state = self.cell(previous, state)
            if step == slot:
                break
            values = actions[:, step]
            moved = moved | (values[:, 0] == 1)
            previous = self.embedding(values[:, 0]) + self.xy(
                values[:, 1:].to(memory.dtype) / (GRID - 1)
            )
        logits = self.kinds(state).float()
        if self.look:
            logits = logits.masked_fill(moved[:, None] & self.press, float("-inf"))
        kind_p = logits.softmax(-1)[0]
        where = torch.einsum("bnc,bc->bn", cells, self.query(state).to(cells.dtype))
        place_p = (where.float() * self.scale + self.cell_bias.float()).softmax(-1)[0]
        heat = (place_p / (CELLS * CELLS))[:, None].expand(-1, CELLS * CELLS).clone()
        best = place_p.topk(top).indices
        chosen = cells[0, best].to(state.dtype)
        fine = self.fine(torch.cat([state.expand(top, -1), chosen], -1)).float().softmax(-1)
        heat[best] = place_p[best, None] * fine
        # (cell row, cell column, row inside, column inside) to lattice rows and columns.
        grid = heat.view(CELLS, CELLS, CELLS, CELLS).permute(0, 2, 1, 3).reshape(GRID, GRID)
        return kind_p, grid


class Policy(nn.Module):
    """Global clip, quadrants and fovea in; a recurrent memory, a value and the cells out.

    The cells are the screen as a CELLS x CELLS map: the quadrants' detail pooled into
    each cell, plus the video encoder's patch grid (which has seen the whole clip)
    resized onto the same map, plus a learned position. The action head scores them to
    place the pointer. The memory reads the whole screen through their mean, the pointer's
    surroundings through the fovea, and one place of its own choosing through attention.
    """

    def __init__(self, encoder, memory_dim=512, latents=0, look=False):
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
        self.actor = ActionHead(memory_dim, latents=latents, look=look)
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

        `clip` may be None for an encoder that does not read it (`reads_clip`). The fused
        inputs are joined in the quadrants' dtype, which is the clip's: both arrive
        normalized in float32.
        """
        if hidden is None:
            hidden = quadrants.new_zeros(quadrants.shape[0], self.memory_dim)
        if reset is not None:
            hidden = hidden * (~reset.bool()).to(hidden.dtype)[:, None]
        summary, cells, centre = self.perceive(clip, quadrants, fovea)
        hidden, value = self.recall(
            summary, cells, centre, previous, speed, hidden, quadrants.dtype
        )
        return hidden, value, summary, cells

    def recall(self, summary, cells, centre, previous, speed, hidden, dtype):
        """The memory's step: one decision's perception taken in. Returns the new memory
        and the value logit."""
        hidden = self.memory(
            fuse(self, summary, cells, centre, previous, speed, hidden, dtype), hidden
        )
        return hidden, self.value(hidden).squeeze(-1)

    def observe(self, clip, quadrants, fovea, previous, speed, hidden):
        """What one decision sees, fused into one vector, before the memory takes it in.

        `hidden` only steers the attention readout. Returns the fused vector, the summary
        token and the cells.
        """
        summary, cells, centre = self.perceive(clip, quadrants, fovea)
        merged = fuse(self, summary, cells, centre, previous, speed, hidden, quadrants.dtype)
        return merged, summary, cells

    def perceive(self, clip, quadrants, fovea):
        """What the screen shows, before any memory: the summary, the cells, the fovea."""
        summary, grid = self.encoder(clip, quadrants)
        return summary, self.cells(grid, quadrants), self.foveal(fovea).mean((-2, -1))

    def perceive_window(
        self, clips, quadrants, fovea, *, checkpoint=False, chunk=CHUNK, tower=None
    ):
        """`perceive` over a window of decisions: (B, T, ...) views in, (B, T, ...) out.

        Perception does not depend on the memory, so a window's frames need not wait for
        the recurrence: they go through `chunk` frames at a time. The encoder's frozen
        blocks run first, once and without a graph (the encoders' `frozen`); only what
        trains after them (`tail`, the cells and the fovea's reader) builds one. With
        `checkpoint` (and gradients on) that trainable part keeps just its inputs and is
        recomputed chunk by chunk in the backward pass, which is what lets a batch of 2
        fit 8 GB. The frozen blocks are never recomputed. A chunk never spans two windows,
        so it is a view of the batch, not a copy the recomputation would have to keep.
        `clips` may be None for an encoder that does not read them.

        `tower`, (summary, grid) per decision from a tower cache (tower_cache.py), stands
        in for a wholly frozen encoder: its summary as it is, its grid already resized to
        the cells' CELLS x CELLS (the 1x1 convolution and the bilinear resize commute), so
        no frame goes through the tower at all.
        """
        split = getattr(self.encoder, "frozen", None)
        windows = []
        for i in range(quadrants.shape[0]):
            parts = []
            for start in range(0, quadrants.shape[1], chunk):
                span = slice(start, start + chunk)
                clip = None if clips is None else clips[i, span]
                quads, centre = quadrants[i, span], fovea[i, span]
                if tower is not None:
                    state = ("cached", tower[0][i, span], tower[1][i, span])
                else:
                    state = split(clip, quads) if split else (clip, quads)
                if checkpoint and torch.is_grad_enabled():
                    parts.append(
                        torch.utils.checkpoint.checkpoint(
                            self._perceive_tail, state, quads, centre, use_reentrant=False
                        )
                    )
                else:
                    parts.append(self._perceive_tail(state, quads, centre))
            windows.append([torch.cat(x) for x in zip(*parts)])
        return tuple(torch.stack(x) for x in zip(*windows))

    def _perceive_tail(self, state, quadrants, fovea):
        if isinstance(state[0], str):  # ("cached", summary, grid)
            # Stored in bfloat16; taken in the views' dtype, as the tower's own output is
            # joined with them (under autocast the convolutions read bfloat16 either way).
            summary, grid = state[1].to(quadrants.dtype), state[2].to(quadrants.dtype)
        else:
            tail = getattr(self.encoder, "tail", None)
            summary, grid = tail(state) if tail else self.encoder(*state)
        return summary, self.cells(grid, quadrants), self.foveal(fovea).mean((-2, -1))


_ACTION_SCALES = {}


def action_scales(like):
    """The largest value of each action field, as a tensor like `like`, made once per device.

    Made from a Python list at every step it was a copy from pageable host memory, and
    such a copy first waits for everything queued on the device: one full stop per
    decision, so the host never ran ahead of the GPU.
    """
    key = (like.device, like.dtype)
    if key not in _ACTION_SCALES:
        _ACTION_SCALES[key] = like.new_tensor([len(VOCAB) - 1, GRID - 1, GRID - 1])
    return _ACTION_SCALES[key]


def fuse(module, summary, cells, centre, previous, speed, hidden, dtype=None):
    """One decision's inputs as one vector for the memory.

    The screen through its summary, its mean cell and the fovea, one place the memory
    chooses to read (attention over the cells, steered by `hidden`), the previous action
    and the game speed. `module` holds the layers: a Policy, or features.MemoryHead,
    which trains the same layers on cached perception. `dtype` is what the parts are
    joined in, the clip's for a Policy, as it always was.

    The summary and the fovea's reading are normalized first (a layer norm with no
    parameters). The Qwen3.5 tower's summary is the mean of its last block's raw output,
    about 50 in size; unnormalized, it drove the fusion to about 25 and every gate of the
    memory into saturation, so the memory stood still: over a whole game its state did not
    change (a standard deviation of 0.0 over time, 60-76% of its units at +-1, measured on
    2026-09-24 for the scripted-game and AI-game policies alike) and the policy chose what
    to do with the same probabilities at every decision. LeVJEPA's summary is already
    normalized, so for it this changes little.
    """
    dtype = dtype or summary.dtype
    summary = F.layer_norm(summary.float(), summary.shape[-1:]).to(summary.dtype)
    centre = F.layer_norm(centre.float(), centre.shape[-1:]).to(centre.dtype)
    scales = action_scales(previous)
    prior = module.previous_action((previous / scales).flatten(1).to(dtype))
    attention = torch.einsum("bnc,bc->bn", cells, module.read(hidden).to(cells.dtype))
    weights = (attention.float() / math.sqrt(CELL_DIM)).softmax(-1).to(cells.dtype)
    readout = torch.einsum("bn,bnc->bc", weights, cells)
    merged = torch.cat(
        [summary, cells.mean(1), centre, readout, prior, module.speed(speed)], -1
    ).to(dtype)
    return F.gelu(module.fusion(merged))


def sinusoid(steps, dim, device=None):
    """The fixed sine and cosine positions (Vaswani et al., 2017, arXiv 1706.03762)."""
    position = torch.arange(steps, device=device, dtype=torch.float32)[:, None]
    rate = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    table = torch.zeros(steps, dim, device=device)
    table[:, 0::2] = torch.sin(position * rate)
    table[:, 1::2] = torch.cos(position * rate[: dim // 2])
    return table


class WindowAttention(nn.Module):
    """Full two-way attention over a window of decisions, in place of the two-way GRU.

    The inverse dynamics model is non-causal and local: each label may use every decision
    in its window, before and after, and a window is 16 to 64 decisions, where full
    attention costs nothing next to reading the frames. VPT's IDM put a non-causal
    transformer over its window (Baker et al., 2022, arXiv 2206.11795), and D2E's
    Generalist-IDM (arXiv 2510.05684) likewise conditions each action on the context
    before and after it. A recurrence has to carry a neighbour's evidence through every
    step between; attention reads it directly.

    Pre-norm layers, 8 heads at the memory's width, dropout 0.1. Positions are the fixed
    sinusoids, not a learned table, so any window length is covered (`--sequence` 16, 32
    or 64) and nothing in the checkpoint depends on the length it was trained at.
    """

    def __init__(self, dim=512, layers=2, heads=8, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            4 * dim,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(
            layer, layers, norm=nn.LayerNorm(dim), enable_nested_tensor=False
        )

    def forward(self, x):
        """(B, T, dim) in, (B, T, dim) out; no mask, so every step sees every other."""
        steps, dim = x.shape[1:]
        return self.layers(x + sinusoid(steps, dim, x.device).to(x.dtype))


class InverseDynamics(nn.Module):
    """Labels the inputs behind a stretch of video, seeing what came after each decision.

    The policy has to act on the past alone. The inverse dynamics model does not: it is
    shown each decision's clip shifted into the future, so the last frames already show
    what the input did, and the views of the frame one interval later, where a moved
    pointer has arrived. A two-way `context` then runs over the window, so each decision
    also knows its neighbours: a two-way GRU ("gru", the first model's) or full attention
    ("transformer", WindowAttention, with `layers` layers). That makes it a much easier
    problem than acting, and the point: trained on recordings whose inputs are known, it
    labels video whose inputs are not (Baker et al., 2022, Video PreTraining), and the
    labelled video trains the policy.

    It reuses the policy's reader and action head, so its labels are in the same lattice,
    and its cells come from the later frame, where the pointer ends up.
    """

    def __init__(self, encoder, memory_dim=512, context="gru", layers=2):
        super().__init__()
        self.trunk = Policy(encoder, memory_dim)
        # The trunk's own recurrence and value are not used; the two-way context is.
        self.trunk.memory = self.trunk.value = None
        if context == "gru":
            self.context = nn.GRU(memory_dim, memory_dim // 2, batch_first=True, bidirectional=True)
        elif context == "transformer":
            self.context = WindowAttention(memory_dim, layers)
        else:
            raise ValueError(context)
        self.context_kind = context
        self.memory_dim = memory_dim

    @property
    def actor(self):
        return self.trunk.actor

    def forward(self, clips, quadrants, fovea, speed, checkpoint=False, chunk=CHUNK):
        """A window of decisions: (B, T, ...) views in, per-decision context and cells out.

        The previous action is not an input: it is what a neighbouring decision is being
        asked to label. Each decision is read on its own, with an empty memory steering
        its attention readout, so the whole window is read at once (Policy.perceive_window,
        which also says what `checkpoint` and `chunk` do; a 16-step window needs the
        recomputation on an 8 GB card). `clips` may be None for an encoder that does not
        read them.
        """
        b, steps = quadrants.shape[:2]
        summary, cells, centre = self.trunk.perceive_window(
            clips, quadrants, fovea, checkpoint=checkpoint, chunk=chunk
        )
        previous = quadrants.new_zeros(b * steps, SLOTS, 3, dtype=torch.long)
        hidden = quadrants.new_zeros(b * steps, self.memory_dim)
        merged = fuse(
            self.trunk,
            summary.flatten(0, 1),
            cells.flatten(0, 1),
            centre.flatten(0, 1),
            previous,
            speed.flatten(0, 1),
            hidden,
            quadrants.dtype,
        )
        merged = merged.unflatten(0, (b, steps))
        if self.context_kind == "gru":
            context, _ = self.context(merged)
        else:
            context = self.context(merged)
        return context, cells


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


def xm_loss(actor, memories, cells, actions, candidates=5, form="hard", sigma=0.0):
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
        logps = torch.stack(
            [actor(memories, cells, actions, noise=z, sigma=sigma)[1] for z in noises]
        )
        return -(logps.logsumexp(0) - math.log(len(noises)))
    if form != "hard":
        raise ValueError(form)
    with torch.no_grad():
        losses = torch.stack(
            [-actor(memories, cells, actions, noise=z, sigma=sigma)[1] for z in noises]
        )
        best = losses.argmin(0)
    return -actor(memories, cells, actions, noise=noises[best, rows], sigma=sigma)[1]

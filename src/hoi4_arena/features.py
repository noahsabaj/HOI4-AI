"""Memory trained on long windows over cached perception, and the cells compared on it.

A decision is 200 ms. The policy trained on 16 of them after 2 of burn-in from an empty
memory, so it never learned to use more than 3.6 s of the past, and never saw a memory
older than that, though in a game it carries one for the whole match. Training on long
windows through the whole policy is out of reach: the vision tower runs at every step.
So the perception is frozen after behaviour cloning and what it reads at each decision
is cached (`cache_features`); the memory, the action head and the layers that feed the
memory (models.fuse) are then trained on that cache (`train_memory`), where a window of
hundreds of decisions costs a few milliseconds a step.

Every cell in memory.py is trained the same way, at the same budget: the same decisions
per update and the same number of passes. What each is judged on, on held-out games:

- the imitation loss, with the memory carried from the start of each game, which is
  how the policy runs, and with it cleared every 18 decisions, which is how it trained;
- linear probes of what the memory holds that the screen does not show: the camera's
  zoom (counted from its wheel notches), how long since it last zoomed fully out (it
  does every 20 to 60 s, 100 to 300 decisions), where the pointer was 5, 25 and 100
  decisions ago, and which side goes on to win.

The probes read the memory's output with a ridge regression fitted on training games.
The cell with no memory is the floor: whatever it scores came from the screen alone.
"""

from __future__ import annotations

import json
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import default_collate

from .actions import SLOTS
from .dataset import _Stream, batch_to_device, cover_starts, session_labels
from .memory import build_memory, detach, reset, state_size
from .models import CELL_DIM, CELLS, SPEEDS, ActionHead, fuse, reads_clip

# How far back the pointer probes look, in decisions.
POINTER_LAGS = (5, 25, 100)
# The scripted camera's deepest zoom, in wheel notches (ai_games.ZOOM_MAX), and how many
# notches out in a row mark its recentre: it zooms out by ZOOM_MAX + 4 at once.
ZOOM_MAX = 26
RECENTRE_RUN = 20


def camera_targets(root, decisions):
    """The scripted camera's zoom and seconds since its last recentre, at each decision.

    Rebuilt from its wheel events: each notch moves the zoom one step, clamped as the game
    clamps it, and a run of RECENTRE_RUN notches out marks a recentre. NaN before the
    first recentre, whose time is unknown.
    """
    rows = [json.loads(line) for line in (Path(root) / "frames.jsonl").read_text().splitlines()]
    events = sorted(
        (e for row in rows for e in row.get("scripted_events", [])), key=lambda e: e["t_ns"]
    )
    wheel = [
        (e["t_ns"], 1 if e["event"]["delta"] > 0 else -1)
        for e in events
        if e["event"].get("kind") == "wheel"
    ]
    times = np.array([t for t, _ in wheel], np.int64)
    zoom_after, level = [], 0
    recentres, run, run_start = [], 0, None
    for t, step in wheel:
        level = min(ZOOM_MAX, max(0, level + step))
        zoom_after.append(level)
        if step < 0:
            run_start = t if run == 0 else run_start
            run += 1
            if run == RECENTRE_RUN:
                recentres.append(run_start)
        else:
            run = 0
    index = np.searchsorted(times, decisions, side="left") - 1
    zoom = np.where(index >= 0, np.array(zoom_after + [0])[index], 0).astype(np.float32)
    recentres = np.array(recentres, np.int64)
    last = np.searchsorted(recentres, decisions, side="right") - 1
    since = np.where(last >= 0, (decisions - recentres[np.maximum(last, 0)]) / 1e9, np.nan).astype(
        np.float32
    )
    return zoom, since


@torch.no_grad()
def cache_features(data, checkpoint, output, *, model_path=None, sources=("ai",), device=None):
    """What a behaviour-cloned policy's frozen perception reads, at every decision.

    For each complete recording from `sources`, in every split, writes to
    output/<recording>/: `summary` (N, encoder dim), `cells` (N, 1024, 256), `centre`
    (N, 256), all float16, and `labels.npz` with the actions, validity, outcome, speed,
    the pointer, the camera's zoom and time since its recentre. N counts the decisions
    whose frames exist. About 0.5 MB a decision, nearly all of it cells.
    """
    from .runner import load_policy

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(output)
    policy, config, digest = load_policy(checkpoint, model_path, device)
    policy.eval()
    autocast = {"device_type": device, "dtype": torch.bfloat16, "enabled": device == "cuda"}
    written = []
    for manifest_path in sorted(Path(data).glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("complete") or manifest.get("source") not in sources:
            continue
        root = manifest_path.parent
        target = output / root.name
        if (target / "labels.npz").exists():
            continue
        labels = session_labels(root, sources=sources)
        count = int(labels["readable"].sum())
        window = 16
        starts = cover_starts(labels, window)
        if not starts:
            continue
        target.mkdir(parents=True, exist_ok=True)
        summary = np.zeros((count, policy.encoder.dim), np.float16)
        centre = np.zeros((count, CELL_DIM), np.float16)
        cells = np.lib.format.open_memmap(
            target / "cells.npy", "w+", np.float16, (count, CELLS * CELLS, CELL_DIM)
        )
        stream = _Stream(labels, window, 0, device, starts=starts, clips=reads_clip(policy.encoder))
        began = time.monotonic()
        try:
            while (done := stream.advance()) is not None:
                for piece in done:
                    start = piece.pop("start")
                    batch = batch_to_device(default_collate([piece]), device)
                    clips = batch.get("clips")
                    with torch.autocast(**autocast):
                        seen = policy.perceive(
                            None if clips is None else clips[0],
                            batch["quadrants"][0],
                            batch["fovea"][0],
                        )
                    s, c, f = (x.float().cpu().numpy().astype(np.float16) for x in seen)
                    summary[start : start + window] = s
                    cells[start : start + window] = c
                    centre[start : start + window] = f
        finally:
            stream.close()
        cells.flush()
        del cells
        np.save(target / "summary.npy", summary)
        np.save(target / "centre.npy", centre)
        decisions = labels["decisions"][:count]
        frame_ids = labels["frame_ids"][:count]
        pointer = np.array([labels["cursors"][i] for i in frame_ids], np.float32)
        pointer /= np.array([manifest["width"], manifest["height"]], np.float32)
        zoom, since = camera_targets(root, decisions)
        np.savez(
            target / "labels.npz",
            actions=labels["actions"][:count],
            valid=labels["valid"][:count],
            outcome=labels["outcome"][:count],
            speed=np.full(count, labels["speed"], np.int64),
            pointer=pointer,
            zoom=zoom,
            since_recentre=since,
            winner=np.float32({"BLU": 1.0, "RED": -1.0}.get(manifest.get("winner"), np.nan)),
        )
        meta = {
            "recording": str(root.resolve()),
            "split": manifest["split"],
            "decisions": count,
            "checkpoint": digest,
            "seconds": round(time.monotonic() - began, 1),
        }
        (target / "meta.json").write_text(json.dumps(meta, indent=2))
        written.append(meta)
    return written


class CachedGame:
    """One recording's cached features, the cells read from disk only when used."""

    def __init__(self, root):
        root = Path(root)
        self.name = root.name
        self.meta = json.loads((root / "meta.json").read_text())
        self.summary = np.load(root / "summary.npy")
        self.centre = np.load(root / "centre.npy")
        self.cells = np.load(root / "cells.npy", mmap_mode="r")
        labels = np.load(root / "labels.npz")
        self.labels = {key: labels[key] for key in labels.files}
        self.length = len(self.summary)

    def piece(self, start, length):
        """Steps [start, start + length), padded past the end with invalid steps."""
        stop = min(self.length, start + length)
        n = stop - start

        def padded(array, dtype=None):
            out = np.zeros((length, *array.shape[1:]), dtype or array.dtype)
            out[:n] = array[start:stop]
            return torch.from_numpy(out)

        actions = self.labels["actions"]
        previous = np.zeros_like(actions[start:stop])
        previous[1:] = actions[start : stop - 1]
        if start:
            previous[0] = actions[start - 1]
        valid = np.zeros(length, bool)
        valid[:n] = self.labels["valid"][start:stop]
        prev = np.zeros((length, SLOTS, 3), np.int64)
        prev[:n] = previous
        return {
            "summary": padded(self.summary),
            "cells": padded(self.cells),
            "centre": padded(self.centre),
            "actions": padded(actions),
            "previous": torch.from_numpy(prev),
            "speed": padded(self.labels["speed"]),
            "valid": torch.from_numpy(valid),
        }


def load_cache(cache, split):
    games = [
        CachedGame(path.parent)
        for path in sorted(Path(cache).glob("*/meta.json"))
        if json.loads(path.read_text())["split"] == split
    ]
    if not games:
        raise ValueError(f"No cached {split} games in {cache}")
    return games


class MemoryHead(nn.Module):
    """Everything of a Policy after its perception: the layers models.fuse uses, the
    memory, the action head and the value. The names are Policy's, so a GRU trained here
    loads into a Policy with the perception it was cached from."""

    def __init__(self, summary_dim, memory="gru", memory_dim=512):
        super().__init__()
        self.previous_action = nn.Linear(SLOTS * 3, 64)
        self.speed = nn.Embedding(SPEEDS, 32)
        self.read = nn.Linear(memory_dim, CELL_DIM)
        self.fusion = nn.Linear(summary_dim + 3 * CELL_DIM + 64 + 32, memory_dim)
        self.memory = build_memory(memory, memory_dim)
        self.actor = ActionHead(memory_dim)
        self.value = nn.Linear(memory_dim, 1)
        self.memory_dim = memory_dim

    def initial(self, batch, device):
        return torch.zeros(batch, self.memory_dim, device=device), self.memory.initial(
            batch, device
        )

    def unroll(self, batch, out, state, fresh, grad_from=0):
        """Step through a batch of pieces. `fresh` marks samples starting a game here.

        Steps before `grad_from` only warm the memory. Returns the outputs of the rest,
        (B, T', memory_dim), and the carried (out, state).
        """
        out = out * (~fresh)[:, None].to(out.dtype)
        state = reset(state, fresh)
        outs = []
        for t in range(batch["summary"].shape[1]):
            with torch.set_grad_enabled(torch.is_grad_enabled() and t >= grad_from):
                merged = fuse(
                    self,
                    batch["summary"][:, t],
                    batch["cells"][:, t],
                    batch["centre"][:, t],
                    batch["previous"][:, t],
                    batch["speed"][:, t],
                    out,
                )
                out, state = self.memory(merged, state)
            if t < grad_from:
                out, state = out.detach(), detach(state)
            else:
                outs.append(out)
        return torch.stack(outs, 1), (out, state)


_STREAMS = {}


def _stream(role, device=None):
    """The CUDA stream kept for one role for the life of the process.

    A stream's first matrix product gives it a cuBLAS workspace of about 8 MiB, which
    PyTorch keeps as long as the process lives; streams made anew for each capture or run
    left 26 MiB behind every run (up to the 32 streams PyTorch pools, about 290 MiB).
    """
    device = torch.device(device or "cuda")
    key = (role, device.index or 0)
    if key not in _STREAMS:
        _STREAMS[key] = torch.cuda.Stream(device)
    return _STREAMS[key]


def _unroll_parameters(head):
    """The parameters MemoryHead.unroll reads, by name: models.fuse's layers and the memory."""
    names = ("previous_action", "speed", "read", "fusion", "memory")
    return {
        f"{name}.{key}": p for name in names for key, p in getattr(head, name).named_parameters()
    }


class _Unroll(nn.Module):
    """MemoryHead.unroll as a module's forward, for torch.func.functional_call."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, batch, out, state, fresh, grad_from):
        return self.head.unroll(batch, out, state, fresh, grad_from)


class _Replay(torch.autograd.Function):
    """A captured unroll as one autograd node: its forward and backward replay graphs."""

    @staticmethod
    def forward(ctx, graphed, *parameters):
        graphed.forward.replay()
        ctx.graphed = graphed
        return graphed.outs.detach()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        graphed = ctx.graphed
        graphed.grad.copy_(grad)
        graphed.backward.replay()
        return (None, *(None if g is None else g.detach() for g in graphed.grads))


class _GraphedUnroll:
    """MemoryHead.unroll for one shape of batch, captured as CUDA graphs.

    Stepping a memory through 256 decisions is thousands of small kernels, forward and
    backward, and launching them one by one from Python took longer than running them.
    A graph launches them all at once. It runs the very kernels the eager code runs, in
    the same order, with the same autocast casts, so the numbers are the same bits (a
    test pins it). Only the part after the unroll, whose shape depends on how many steps
    are valid, stays eager; its gradient enters the captured backward through `_Replay`.

    The captures read the parameters through new leaves that share their memory. A
    parameter's own gradient accumulator belongs to the stream it was made on, and one
    made by an eager step still held (by the last update's loss, say) would tie the
    capture to the default stream, which a capture may not do.

    `stream` is where the graphs are warmed up and captured. A captured matrix product
    keeps the cuBLAS workspace of the stream it was captured on, so graphs that may run
    at the same time (several runs, each on its own stream) must be captured on streams
    of their own: sharing one, they wrote each other's workspace and the results drifted
    (a stress test caught it in 4 of 24 runs; with one stream per run, none).
    """

    def __init__(
        self, head, batch, out, state, fresh, grad_from, autocast, inputs=None, stream=None
    ):
        named = _unroll_parameters(head)
        self.parameters = list(named.values())
        # `inputs` are buffers shared with other runs' graphs (`_Statics`).
        self.inputs = inputs or {key: batch[key].clone() for key in _STEPS}
        self.out, self.fresh = out.clone(), fresh.clone()
        self.state = tuple(s.clone() for s in state)
        self.training = torch.is_grad_enabled()
        call = _Unroll(head)

        def unroll():
            leaves = {name: p.detach().requires_grad_(p.requires_grad) for name, p in named.items()}
            with torch.autocast(**autocast):
                args = (self.inputs, self.out, self.state, self.fresh, grad_from)
                prefixed = {f"head.{name}": leaf for name, leaf in leaves.items()}
                return torch.func.functional_call(call, prefixed, args), list(leaves.values())

        side = stream or _stream("graphs 0")
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm up once, off the graph, as capture expects
            (outs, _), leaves = unroll()
            if self.training:
                torch.autograd.grad(outs, leaves, torch.zeros_like(outs), allow_unused=True)
            del outs, _, leaves
        torch.cuda.current_stream().wait_stream(side)
        self.forward = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.forward, stream=side):
            (self.outs, (self.last, self.last_state)), leaves = unroll()
        if self.training:
            self.grad = torch.empty_like(self.outs)
            self.backward = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.backward, pool=self.forward.pool(), stream=side):
                self.grads = torch.autograd.grad(self.outs, leaves, self.grad, allow_unused=True)
        # The replays need only the buffers.
        self.outs, self.last = self.outs.detach(), self.last.detach()
        self.last_state = tuple(s.detach() for s in self.last_state)

    def __call__(self, batch, out, state, fresh):
        for key in _STEPS:
            if batch[key] is not self.inputs[key]:
                self.inputs[key].copy_(batch[key])
        self.out.copy_(out)
        for static, value in zip(self.state, state, strict=True):
            static.copy_(value)
        self.fresh.copy_(fresh)
        if self.training:
            outs = _Replay.apply(self, *self.parameters)
        else:
            self.forward.replay()
            outs = self.outs
        # The carried memory lives in the graph's buffers until the next replay, which
        # first copies it into its inputs.
        return outs, (self.last, self.last_state)


class _Unrolls:
    """MemoryHead.unroll under autocast, from a CUDA graph for a shape that keeps coming.

    A shape (and the dtypes of the memory it starts from) is captured the `capture_at`-th
    time it comes: a capture waits for the whole device and costs about as much as a few
    eager updates. Training captures at the second (its shapes repeat hundreds of times,
    and an eager 256-step update of Gated DeltaNet-2 takes over a second), evaluation at
    the fifth (the short pieces between clears come only a few times). If the card has
    no room for a graph, that shape runs eagerly, which computes the same bits. `enabled`
    False (the CPU) always runs eagerly.
    """

    CAPTURE_AT = 5

    def __init__(self, head, autocast, enabled, statics=None, capture_at=None, owner=0):
        self.head, self.autocast, self.enabled = head, autocast, enabled
        self.statics, self.capture_at = statics, capture_at
        # Graphs that may replay beside another run's are captured on a stream of their own.
        self.stream = _stream(f"graphs {owner}") if enabled else None
        self.graphs, self.seen = {}, {}

    def __call__(self, batch, out, state, fresh, grad_from=0):
        key = (
            tuple(batch["summary"].shape[:2]),
            grad_from,
            out.dtype,
            tuple(s.dtype for s in state),
            torch.is_grad_enabled(),
        )
        graphed = self.graphs.get(key)
        self.seen[key] = self.seen.get(key, 0) + 1
        at = self.capture_at or self.CAPTURE_AT
        if key not in self.graphs and self.enabled and self.seen[key] >= at:
            shared = self.statics is not None and self.statics.holds(batch)
            try:
                with _CUDA:
                    graphed = _GraphedUnroll(
                        self.head,
                        batch,
                        out,
                        state,
                        fresh,
                        grad_from,
                        self.autocast,
                        inputs=batch if shared else None,
                        stream=self.stream,
                    )
            except torch.OutOfMemoryError:
                graphed = None
                torch.cuda.empty_cache()
            self.graphs[key] = graphed
        if graphed is not None:
            return graphed(batch, out, state, fresh)
        with torch.autocast(**self.autocast):
            return self.head.unroll(batch, out, state, fresh, grad_from)


def _to(batch, device):
    """A batch on `device`; the features in bfloat16 on the GPU, where autocast runs."""
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    for key in ("summary", "cells", "centre"):
        batch[key] = batch[key].to(torch.bfloat16 if device == "cuda" else torch.float32)
    return batch


def _nll(head, outs, batch, first=0, scored=None):
    """Per-step negative log-likelihood of the demonstrated actions, and the valid mask.

    `_Loader` batches carry the valid steps' flat positions (`scored`), found on the host:
    indexing with them is what indexing with the mask does, without first waiting for the
    device to say how many there are. `scored` may hold the valid steps' cells and
    actions already gathered (`_scored`), for heads that share a batch.
    """
    valid = batch["valid"][:, first:]
    index = batch.get("scored")
    if index is None:
        index = valid.flatten()
        if not index.any():
            return outs.sum() * 0, valid
    elif not len(index):
        return outs.sum() * 0, valid
    if scored is None:
        cells = batch["cells"][:, first:].flatten(0, 1)[index]
        actions = batch["actions"][:, first:].flatten(0, 1)[index]
    else:
        cells, actions = scored
    _, logp, _ = head.actor(outs.flatten(0, 1)[index], cells, actions)
    return -logp, valid


def _carried_batches(games, window, streams, rng):
    """Consecutive windows of whole games, `streams` games side by side.

    Each stream plays its games front to back, so the memory entering a window has seen
    everything before it in that game; `fresh` marks a stream's first window of a game.
    Yields the plan of each batch, (rows, length, fresh): a row is the (game, start) of a
    piece, or None for a stream with no game left, which is all zeros. `_Loader` reads it.
    """
    order = list(games)
    rng.shuffle(order)
    queues = [order[i::streams] for i in range(streams)]
    positions = [[g, 0] for g in (q.pop(0) if q else None for q in queues)]
    while any(game is not None for game, _ in positions):
        rows, fresh = [], []
        for i, (game, start) in enumerate(positions):
            if game is None:
                rows.append(None)
                fresh.append(True)
                continue
            rows.append((game, start))
            fresh.append(start == 0)
            start += window
            if start >= game.length:
                game, start = (queues[i].pop(0) if queues[i] else None), 0
            positions[i] = [game, start]
        yield rows, window, fresh


def _reset_batches(games, window, burn_in, batch_size, rng):
    """Windows cut anywhere, each from an empty memory after `burn_in` steps: the old way.

    Yields plans as `_carried_batches` does."""
    span = window + burn_in
    cuts = [(g, s) for g in games for s in range(0, g.length - span + 1, window)]
    rng.shuffle(cuts)
    for i in range(0, len(cuts), batch_size):
        group = cuts[i : i + batch_size]
        yield group, span, [True] * len(group)


# When training captures a batch shape as CUDA graphs (_Unrolls): at its second coming.
CAPTURE_TRAINING = 2
# Threads that read a batch's cells, and how many decisions one reads at a time (8 MB).
READERS = 8
PIECE = 16
# Held around CUDA calls off the main thread and around graph capture: while a graph is
# captured, no other thread may allocate or wait on the device.
_CUDA = threading.Lock()
_FLOATS = ("summary", "cells", "centre")
# What MemoryHead.unroll reads of a batch at each step.
_STEPS = ("summary", "cells", "centre", "previous", "speed")


class _Loader:
    """Batches of cached pieces, read ahead while the device trains on the one before.

    A batch is exactly what default_collate makes of each row's CachedGame.piece (a row
    of None is all zeros), moved as `_to` moves it: the same tensors, by other means.
    Before, one thread copied every piece out of the memory-mapped cells, then collated
    and copied it again, then copied it to the GPU from pageable memory, and the GPU
    waited through all three: 0.16 to 0.66 s of each update of about 0.55 GB. Here
    `READERS` threads read the rows straight from the cells file into pinned memory (on
    this PC a memory map copied on one thread gave 0.6-0.9 GB/s, file reads on 8 threads
    about 9), and the copy to the GPU runs on its own stream while the previous batch
    trains.
    """

    def __init__(self, games, device, readers=READERS):
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        # What `_to` gives the features: bfloat16 on "cuda", float32 anywhere else.
        self.floats = torch.bfloat16 if device == "cuda" else torch.float32
        first = games[0]
        self.fields = {
            "summary": (first.summary.shape[1:], first.summary.dtype),
            "cells": (first.cells.shape[1:], first.cells.dtype),
            "centre": (first.centre.shape[1:], first.centre.dtype),
            "actions": (first.labels["actions"].shape[1:], first.labels["actions"].dtype),
            "previous": ((SLOTS, 3), np.dtype(np.int64)),
            "speed": (first.labels["speed"].shape[1:], first.labels["speed"].dtype),
            "valid": ((), np.dtype(bool)),
        }
        self.pool = ThreadPoolExecutor(readers, thread_name_prefix="cells")
        self.local = threading.local()
        self.files = []
        self.buffers = [None, None]
        # Recorded where the device finished with each slot's last batch.
        self.done = [None, None]
        self.stream = _stream("copies", self.device) if self.cuda else None

    def close(self):
        self.pool.shutdown()
        for handle in self.files:
            handle.close()
        if self.cuda:
            self.stream.synchronize()  # no copy still reads the buffers let go below
        for slot in range(len(self.buffers)):
            self._release(slot)

    def _release(self, slot):
        held, self.buffers[slot] = self.buffers[slot], None
        if held is not None and self.cuda:
            torch.cuda.cudart().cudaHostUnregister(held["memory"].ctypes.data)

    def _host(self, slot, rows, length):
        """Slot's host buffers as numpy arrays of (rows, length, ...), pinned for the GPU.

        One allocation of exactly the bytes needed, page-locked in place (cudaHostRegister).
        Tensor.pin_memory would round each buffer up to a power of two: 604 MB of cells
        took 1 GB of locked memory.
        """
        need = {
            key: rows * length * int(np.prod(shape, dtype=np.int64))
            for key, (shape, _) in self.fields.items()
        }
        need["fresh"], need["scored"] = rows, rows * length
        dtypes = {key: dtype for key, (_, dtype) in self.fields.items()}
        dtypes.update(fresh=np.dtype(bool), scored=np.dtype(np.int64))
        held = self.buffers[slot]
        if held is None or any(held[key][1].size < n for key, n in need.items()):
            self._release(slot)
            sizes = {key: -(-need[key] * dtypes[key].itemsize // 4096) * 4096 for key in need}
            memory = np.empty(sum(sizes.values()), np.uint8)
            if self.cuda:
                torch.cuda.cudart().cudaHostRegister(memory.ctypes.data, memory.nbytes, 0)
            held, offset = {"memory": memory}, 0
            for key, dtype in dtypes.items():
                array = memory[offset : offset + need[key] * dtype.itemsize].view(dtype)
                held[key] = (torch.from_numpy(array), array)
                offset += sizes[key]
            self.buffers[slot] = held
        return {
            key: held[key][1][: need[key]].reshape(rows, length, *shape)
            for key, (shape, _) in self.fields.items()
        }

    def _cells(self, game, start, into):
        """Read decisions [start, start + len(into)) of a game's cells straight into `into`."""
        handles = getattr(self.local, "handles", None)
        if handles is None:
            handles = self.local.handles = {}
        cells = game.cells
        handle = handles.get(cells.filename)
        if handle is None:
            handle = handles[cells.filename] = open(cells.filename, "rb", buffering=0)
            self.files.append(handle)
        handle.seek(cells.offset + start * cells.strides[0])
        view = memoryview(into.reshape(-1).view(np.uint8))
        done = 0
        while done < len(view):
            got = handle.readinto(view[done:])
            if not got:
                raise ValueError(f"{cells.filename} ends early")
            done += got

    def _fill(self, host, rows, length):
        """Every row as CachedGame.piece makes it, into the batch's host buffers.

        The cells are read by the pool, PIECE decisions a task, so a batch of one long
        row reads as fast as one of many short ones. Cells past a row's real steps (a
        stream with no game left, or past a game's end) are neither read nor zeroed here:
        `_to_device` zeroes them where they land, so zeros never cross to the GPU.
        Returns each row's number of real steps.
        """
        counts, reads = [], []
        for row, where in enumerate(rows):
            if where is None:
                counts.append(0)
                for key, array in host.items():
                    if key != "cells":
                        array[row] = 0
                continue
            game, start = where
            stop = min(game.length, start + length)
            n = stop - start
            counts.append(n)
            labels = game.labels
            for key, source in (
                ("summary", game.summary),
                ("centre", game.centre),
                ("actions", labels["actions"]),
                ("speed", labels["speed"]),
                ("valid", labels["valid"]),
            ):
                host[key][row, :n] = source[start:stop]
                host[key][row, n:] = 0
            previous = host["previous"][row]
            previous[0] = labels["actions"][start - 1] if start else 0
            previous[1:n] = labels["actions"][start : stop - 1]
            previous[n:] = 0
            for k in range(0, n, PIECE):
                reads.append((game, start + k, host["cells"][row, k : min(n, k + PIECE)]))
        list(self.pool.map(lambda read: self._cells(*read), reads))
        return counts

    @staticmethod
    def _runs(counts, length):
        """Rows grouped into runs of whole rows, and single rows that are part real."""
        runs, row = [], 0
        while row < len(counts):
            end = row + 1
            if counts[row] in (0, length):
                while end < len(counts) and counts[end] == counts[row]:
                    end += 1
            runs.append((row, end, counts[row]))
            row = end
        return runs

    def _to_device(self, slot, host, counts, fresh, first):
        """The host buffers on the device, the features in `self.floats`.

        Also the flat positions of the valid steps from `first` on (`scored`, see `_nll`).
        """
        held = self.buffers[slot]
        held["fresh"][1][: len(fresh)] = fresh
        scored = np.flatnonzero(host["valid"][:, first:].reshape(-1))
        held["scored"][1][: len(scored)] = scored
        tensors = {key: held[key][0][: array.size].view(array.shape) for key, array in host.items()}
        tensors["scored"] = held["scored"][0][: len(scored)]
        flag = held["fresh"][0][: len(fresh)]
        length = tensors["cells"].shape[1]
        runs = self._runs(counts, length)
        if not self.cuda:
            batch = {
                key: t.to(self.floats) if key in _FLOATS else t.clone()
                for key, t in tensors.items()
                if key != "cells"
            }
            cells = torch.zeros(tensors["cells"].shape, dtype=self.floats)
            for start, end, n in runs:
                cells[start:end, :n] = tensors["cells"][start:end, :n]
            batch["cells"] = cells
            return batch, flag.clone(), None
        with torch.cuda.stream(self.stream):
            batch = {}
            for key, t in tensors.items():
                if key == "cells":
                    cells = torch.empty(t.shape, dtype=t.dtype, device=self.device)
                    for start, end, n in runs:
                        if n:
                            cells[start:end, :n].copy_(t[start:end, :n], non_blocking=True)
                        if n < length:
                            cells[start:end, n:].zero_()
                    t = cells
                else:
                    t = t.to(self.device, non_blocking=True)
                if key in _FLOATS and t.dtype.itemsize == self.floats.itemsize:
                    # float16 to bfloat16 in place, element by element: the same bits as
                    # t.to(bfloat16) (a test pins it) in half the memory.
                    t = t.view(self.floats).copy_(t)
                elif key in _FLOATS:
                    t = t.to(self.floats)
                batch[key] = t
            flag = flag.to(self.device, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.stream)
        return batch, flag, event

    def __call__(self, plans, first=0):
        """(batch, fresh) for each plan of `_carried_batches` or `_reset_batches`, in order.

        `first` is where the scored steps start (`_nll`)."""
        ready, free = queue.Queue(maxsize=1), queue.Queue()
        for slot in range(len(self.buffers)):
            free.put(slot)
        stop = threading.Event()

        def produce():
            try:
                for rows, length, fresh in plans:
                    slot = free.get()
                    if stop.is_set():
                        return
                    with _CUDA:
                        if self.done[slot] is not None:
                            # Its last batch was copied and trained on: the pinned buffers
                            # are free, and no more than two batches are on the device.
                            self.done[slot].synchronize()
                        host = self._host(slot, len(rows), length)
                    counts = self._fill(host, rows, length)
                    with _CUDA:
                        batch = self._to_device(slot, host, counts, fresh, first)
                    ready.put((slot, batch))
                ready.put(None)
            except BaseException as error:  # handed to the consumer, which raises it
                ready.put(error)

        worker = threading.Thread(target=produce, daemon=True)
        worker.start()
        held = None
        try:
            while (item := ready.get()) is not None:
                if isinstance(item, BaseException):
                    raise item
                slot, (batch, fresh, event) = item
                if event is not None:
                    current = torch.cuda.current_stream(self.device)
                    current.wait_event(event)
                    for t in (*batch.values(), fresh):
                        t.record_stream(current)
                # The slot before is free once the device is done with its batch, which
                # the event marks: the host may queue work far ahead of the device.
                if held is not None:
                    if self.cuda:
                        self.done[held] = torch.cuda.Event()
                        self.done[held].record(torch.cuda.current_stream(self.device))
                    free.put(held)
                held = slot
                yield batch, fresh
        finally:
            if held is not None and self.cuda:  # for the next call's producer to wait on
                self.done[held] = torch.cuda.Event()
                self.done[held].record(torch.cuda.current_stream(self.device))
            stop.set()
            free.put(0)  # wakes a producer waiting for a slot, which then stops
            while worker.is_alive():
                try:
                    ready.get(timeout=0.1)
                except queue.Empty:
                    pass
            worker.join()


def _like(memory, carried):
    """An empty memory in the types of a carried one, or as it is when none is known.

    MemoryHead.initial gives float32 zeros, while a memory carried out of the unroll is in
    bfloat16 where autocast made it so. Zeros are zeros in either type and the unroll
    computes the same bits from both, but a CUDA graph is captured for one set of types:
    this lets the first update of an epoch replay the graph of the rest.
    """
    if carried is None:
        return memory
    (out, state), (like_out, like_state) = memory, carried
    return out.to(like_out.dtype), tuple(
        s.to(like.dtype) for s, like in zip(state, like_state, strict=True)
    )


def _release_cuda(cuda):
    """Hand the card back the memory the caching allocator holds for nothing.

    The graphs' pools and the batches in flight leave freed blocks cached, of sizes the
    next phase may not reuse; runs made one after another in one process (the memory
    study) grew the reserved memory to 7.4 GB this way, and on this WDDM card memory past
    what the desktop leaves spills into system memory.
    """
    if cuda:
        torch.cuda.empty_cache()


def _on(stream):
    """`stream`, after everything queued so far on the current one (the one batches arrive
    on); the current stream itself when None."""
    if stream is None:
        return nullcontext()
    stream.wait_stream(torch.cuda.current_stream())
    return torch.cuda.stream(stream)


def _join(streams):
    """The current stream waits for all of `streams`: their work on this batch is done
    before anything the batch holds is freed or overwritten."""
    for stream in streams:
        if stream is not None:
            torch.cuda.current_stream().wait_stream(stream)


def _scored(batch, first):
    """The valid steps' cells and actions, gathered once for every head that scores them.

    What `_nll` gathers, with no copy of the steps from `first` on made first: when
    `first` is not 0 that slice is not contiguous, and flattening it copied the cells.
    """
    index = batch["scored"]
    if not first:
        return batch["cells"].flatten(0, 1)[index], batch["actions"].flatten(0, 1)[index]
    scored = batch["cells"].shape[1] - first
    rows, steps = index // scored, index % scored + first
    return batch["cells"][rows, steps], batch["actions"][rows, steps]


class _Statics:
    """The step inputs of every batch of one shape, in fixed buffers that all the runs'
    graphs read (`_GraphedUnroll`), so a batch is copied there once, not once a run.

    A shape gets its buffers when it comes often enough to be captured; before that a
    batch passes through as it is.
    """

    def __init__(self, capture_at=None):
        self.seen, self.buffers, self.capture_at = {}, {}, capture_at

    def __call__(self, batch):
        key = tuple(batch["summary"].shape[:2])
        self.seen[key] = self.seen.get(key, 0) + 1
        if self.seen[key] < (self.capture_at or _Unrolls.CAPTURE_AT):
            return batch
        held = self.buffers.get(key)
        if held is None:
            held = self.buffers[key] = {k: batch[k].clone() for k in _STEPS}
        else:
            for k in _STEPS:
                held[k].copy_(batch[k])
        return held

    def holds(self, batch):
        return self.buffers.get(tuple(batch["summary"].shape[:2])) is batch

    def clear(self):
        self.seen.clear()
        self.buffers.clear()


class _Trainee:
    """One run of train_memories: its head, optimizer and graphs, and its CUDA stream."""

    def __init__(self, output, memory, summary_dim, seed, lr, device, stream):
        torch.manual_seed(seed)  # the head starts as it would in a run of its own
        self.output, self.memory, self.stream = output, memory, stream
        self.head = MemoryHead(summary_dim, memory).to(device)
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr)
        self.unrolls = None
        self.steps = 0
        self.out = self.state = self.loss = None

    def on(self):
        return _on(self.stream)


def evaluate(head, games, device, clear_every=None, loader=None, graphs=True):
    """Imitation loss over whole held-out games, and each decision's memory output.

    The memory is carried from each game's start, or cleared every `clear_every`
    decisions (after which the step's output is the first after an empty memory, as in
    training from windows). Returns the mean loss, the loss on decisions with an input,
    and the outputs per game.

    Games are read in pieces of 128 decisions by a `_Loader` (`loader`, or one of its
    own), and everything stays on the device until the end, so nothing waits for it on
    the way. Between two clears the memory runs as one piece; stepped one decision at a
    time, with a no-op reset before each, it computed the same numbers.
    """
    return _evaluate([head], games, device, clear_every, loader, graphs)[0]


def _cat(parts, dtype):
    return torch.cat(parts).cpu() if parts else torch.zeros(0, dtype=dtype)


class _Evaluation:
    """What evaluating several heads keeps from one pass to the next: each head's graphs,
    since the second pass over games of the same shape can replay the first's."""

    def __init__(self, heads, device, graphs=True, streams=None):
        self.heads, self.device = heads, device
        self.streams = streams or [None] * len(heads)
        self.autocast = {
            "device_type": device,
            "dtype": torch.bfloat16,
            "enabled": device == "cuda",
        }
        enabled = graphs and device == "cuda"
        # A whole piece is copied once, before any head reads it, so the heads share it.
        # The pieces between clears are copied by each head on its own stream as it goes,
        # so each head has its own.
        self.shared, self.own = _Statics(), [_Statics() for _ in heads]
        self.carried = [
            _Unrolls(h, self.autocast, enabled, self.shared, owner=i) for i, h in enumerate(heads)
        ]
        self.cleared = [
            _Unrolls(h, self.autocast, enabled, s, owner=i)
            for i, (h, s) in enumerate(zip(heads, self.own, strict=True))
        ]
        self.enabled = enabled


@torch.no_grad()
def _evaluate(heads, games, device, clear_every=None, loader=None, graphs=True, streams=None):
    """`evaluate` for several heads over one read of the games, each head on its stream.

    `heads` may be an `_Evaluation`, which keeps the graphs between calls.
    """
    run = heads if isinstance(heads, _Evaluation) else _Evaluation(heads, device, graphs, streams)
    heads, streams, autocast = run.heads, run.streams, run.autocast
    for head in heads:
        head.eval()
    chunk = 128
    reader = loader or _Loader(games, device)
    plans = (
        ([(game, start)], chunk, [start == 0])
        for game in games
        for start in range(0, game.length, chunk)
    )
    batches = reader(plans)
    carried = torch.zeros(1, dtype=torch.bool, device=device)
    losses, outputs, acting = [[] for _ in heads], [[] for _ in heads], []
    memories = None
    try:
        for game in games:
            before, memories, kept = memories, [], [[] for _ in heads]
            for i, (head, stream) in enumerate(zip(heads, streams, strict=True)):
                with _on(stream):
                    memories.append(_like(head.initial(1, device), before and before[i]))
            for start in range(0, game.length, chunk):
                batch, fresh = next(batches)
                steps = run.shared(batch) if run.enabled and not clear_every else batch
                scored = _scored(batch, 0)
                for i, (head, stream) in enumerate(zip(heads, streams, strict=True)):
                    with _on(stream):
                        out, state = memories[i]
                        if clear_every:
                            parts, t = [], 0
                            while t < chunk:
                                if (start + t) % clear_every == 0:
                                    out, state = _like(head.initial(1, device), (out, state))
                                end = min(chunk, t + clear_every - (start + t) % clear_every)
                                piece = {key: batch[key][:, t:end] for key in _STEPS}
                                piece = run.own[i](piece) if run.enabled else piece
                                o, (out, state) = run.cleared[i](piece, out, state, carried)
                                parts.append(o.clone())  # a graph rewrites its output
                                t = end
                            outs = torch.cat(parts, 1)
                        else:
                            outs, (out, state) = run.carried[i](steps, out, state, fresh)
                        memories[i] = out, state
                        with torch.autocast(**autocast):
                            nll, _ = _nll(head, outs, batch, 0, scored)
                        kept[i].append(outs[0, : min(chunk, game.length - start)].float())
                        if nll.dim():
                            losses[i].append(nll.float())
                if len(batch["scored"]):
                    moved = (batch["actions"][:, :, :, 0] != 0).any(-1).flatten()
                    acting.append(moved[batch["scored"]])
                # The streams are joined before `scored`, made on this stream and read on
                # theirs, is freed: freed memory goes straight back to the stream it was
                # made on, whoever still reads it.
                _join(streams)
            for i in range(len(heads)):
                outputs[i].append(torch.cat(kept[i]))
    finally:
        batches.close()
        if loader is None:
            reader.close()
    acting = _cat(acting, torch.bool)
    results = []
    for head, losses_, outputs_ in zip(heads, losses, outputs, strict=True):
        losses_ = _cat(losses_, torch.float32)
        active = losses_[acting].tolist()
        head.train()
        results.append(
            (
                float(np.mean(losses_.tolist())),
                float(np.mean(active)) if active else None,
                [o.cpu().numpy() for o in outputs_],
            )
        )
    return results


# A unit whose output moves less than this over a held-out game (its standard deviation
# over time) never moves. A memory most of whose units never move is dead.
STILL = 1e-3


@torch.no_grad()
def memory_health(heads, games, device, loader=None):
    """Whether each head's memory is alive, over held-out games with the memory carried from
    each game's start, as `evaluate` runs it, one decision at a time.

    The memory study of 2026-09-24 compared GRUs whose memory was dead: the tower's summary,
    about 70 in size, saturated every gate, so over a whole game their output did not move
    (0.003 standard deviation over time, 89% of units never moving) and one step from an
    empty memory landed within 3% of the carried one. Nothing in the report showed it. Each
    head gets:

    - `std_over_time`: the memory output's standard deviation over the steps of a game, mean
      over units and games;
    - `still_units`: the fraction of units whose standard deviation over time is under STILL;
    - `one_step_from_empty`: how far one step taken from an empty memory lands from the same
      step taken from the carried one: mean |difference| over mean |output|, over steps;
    - `saturated_gates`: for the GRU, the fraction of reset and update gates below 0.01 or
      above 0.99 and of candidate values beyond +-0.99; None for other cells;
    - `dead`: whether most units never move.
    """
    autocast = {"device_type": device, "dtype": torch.bfloat16, "enabled": device == "cuda"}
    chunk = 128
    reader = loader or _Loader(games, device)
    plans = (
        ([(game, start)], chunk, [start == 0])
        for game in games
        for start in range(0, game.length, chunk)
    )
    batches = reader(plans)
    spread = [[] for _ in heads]
    distance = [[] for _ in heads]
    gated = [hasattr(head.memory, "gates") for head in heads]
    saturated = [{"reset": [], "update": [], "candidate": []} for _ in heads]
    try:
        for game in games:
            memories = [head.initial(1, device) for head in heads]
            outputs = [[] for _ in heads]
            for start in range(0, game.length, chunk):
                batch, _ = next(batches)
                for t in range(min(chunk, game.length - start)):
                    for i, head in enumerate(heads):
                        out, state = memories[i]
                        with torch.autocast(**autocast):
                            x = fuse(
                                head,
                                batch["summary"][:, t],
                                batch["cells"][:, t],
                                batch["centre"][:, t],
                                batch["previous"][:, t],
                                batch["speed"][:, t],
                                out,
                            )
                            if gated[i]:
                                gates = head.memory.gates(x, state)
                            out, state = head.memory(x, state)
                            empty, _ = head.memory(x, head.memory.initial(1, device))
                        memories[i] = out, state
                        outputs[i].append(out[0].float())
                        size = out.float().abs().mean().clamp_min(1e-12)
                        distance[i].append((out.float() - empty.float()).abs().mean() / size)
                        if gated[i]:
                            for name, value in gates.items():
                                if name == "candidate":
                                    edge = value.abs() > 0.99
                                else:
                                    edge = (value < 0.01) | (value > 0.99)
                                saturated[i][name].append(edge.float().mean())
            for i in range(len(heads)):
                std = torch.stack(outputs[i]).std(0)
                spread[i].append((std.mean(), (std < STILL).float().mean()))
    finally:
        batches.close()
        if loader is None:
            reader.close()
    health = []
    for i in range(len(heads)):
        still = float(torch.stack([s for _, s in spread[i]]).mean())
        health.append(
            {
                "std_over_time": float(torch.stack([m for m, _ in spread[i]]).mean()),
                "still_units": still,
                "one_step_from_empty": float(torch.stack(distance[i]).mean()),
                "saturated_gates": (
                    {name: float(torch.stack(v).mean()) for name, v in saturated[i].items()}
                    if gated[i]
                    else None
                ),
                "dead": still > 0.5,
            }
        )
    return health


def _probe_targets(game):
    labels = game.labels
    n = game.length
    targets = {
        "zoom": labels["zoom"][:n],
        "since_recentre": np.log1p(labels["since_recentre"][:n]),
        "winner": np.full(n, labels["winner"], np.float32),
    }
    for lag in POINTER_LAGS:
        past = np.full((n, 2), np.nan, np.float32)
        if lag < n:
            past[lag:] = labels["pointer"][: n - lag]
        targets[f"pointer_{lag}"] = past
    return targets


def probe(train_outputs, train_games, test_outputs, test_games, ridge=1.0):
    """Linear read-outs of the memory, fitted on training games, scored on held-out ones.

    R^2 for each target, and for the winner, how often the read-out's sign is right.
    """
    report = {}
    names = _probe_targets(train_games[0]).keys()
    for name in names:

        def stack(outputs, games):
            xs, ys = [], []
            for out, game in zip(outputs, games, strict=True):
                y = _probe_targets(game)[name]
                y = y.reshape(len(y), -1)
                keep = np.isfinite(y).all(1)
                xs.append(out[keep])
                ys.append(y[keep])
            return np.concatenate(xs), np.concatenate(ys)

        x, y = stack(train_outputs, train_games)
        tx, ty = stack(test_outputs, test_games)
        if not len(x) or not len(tx):
            continue
        mean_x, mean_y = x.mean(0), y.mean(0)
        xc = x - mean_x
        weights = np.linalg.solve(
            xc.T @ xc + ridge * len(x) * np.eye(x.shape[1]), xc.T @ (y - mean_y)
        )
        guess = (tx - mean_x) @ weights + mean_y
        if name == "winner":
            report[name] = float((np.sign(guess) == np.sign(ty)).mean())
        else:
            residual = ((ty - guess) ** 2).sum()
            total = ((ty - ty.mean(0)) ** 2).sum()
            report[name] = float(1 - residual / total)
    return report


def train_memory(cache, output, *, memory="gru", **settings):
    """Train a MemoryHead on cached features and report how it does on held-out games.

    `decisions` per update is fixed, so a longer window means fewer games side by side,
    never more data. With `carry`, windows run through whole games and the memory enters
    each one where the last left off; without it, each window starts from an empty memory
    after `burn_in` steps, as train_bc does. The settings are train_memories'.
    """
    return train_memories(cache, [output], [memory], **settings)[0]


def train_memories(
    cache,
    outputs,
    memories,
    *,
    window=256,
    carry=True,
    burn_in=0,
    decisions=1024,
    epochs=4,
    lr=1e-4,
    seed=0,
    probe_games=None,
    device=None,
    graphs=True,
):
    """train_memory for several cells at once: `memories[i]` is trained into `outputs[i]`.

    Every run sees the same batches, since they come from the same seed, so each batch is
    read once for all of them, and each run trains on its own CUDA stream, so one run's
    small kernels fill the gaps between another's. Each run computes exactly what
    train_memory computes for it alone, to the bit: the same head (the seed is set before
    each), the same batches, the same kernels on its own stream. `train_seconds` in each
    report is the wall time of the whole group.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    outputs = [Path(output) for output in outputs]
    if len(outputs) != len(memories):
        raise ValueError("one output per memory")
    for output in outputs:
        if (output / "report.json").exists():
            raise FileExistsError("Results are immutable")
    cuda = device == "cuda"
    rng = random.Random(seed)
    train, validation = load_cache(cache, "train"), load_cache(cache, "validation")
    autocast = {"device_type": device, "dtype": torch.bfloat16, "enabled": cuda}
    use_graphs = graphs and cuda
    statics = _Statics(CAPTURE_TRAINING)
    runs = []
    for i, (output, memory) in enumerate(zip(outputs, memories, strict=True)):
        stream = _stream(f"run {i}") if cuda and len(outputs) > 1 else None
        runs.append(_Trainee(output, memory, train[0].summary.shape[1], seed, lr, device, stream))
        # The unroll's autocast region is its own: it shares no weight with the action head.
        runs[-1].unrolls = _Unrolls(
            runs[-1].head, autocast, use_graphs, statics, CAPTURE_TRAINING, owner=i
        )
    streams = max(1, decisions // window)
    for output in outputs:
        output.mkdir(parents=True, exist_ok=True)
    loader = _Loader(train + validation, device)
    first = 0 if carry else burn_in
    began = time.monotonic()
    try:
        with ExitStack() as files:
            logs = [files.enter_context((o / "metrics.jsonl").open("a")) for o in outputs]
            for epoch in range(epochs):
                if carry:
                    plans = _carried_batches(train, window, streams, rng)
                else:
                    plans = _reset_batches(train, window, burn_in, streams, rng)
                for run in runs:
                    with run.on():
                        carried = None if run.out is None else (run.out, run.state)
                        run.out, run.state = _like(run.head.initial(streams, device), carried)
                for batch, fresh in loader(plans, first):
                    steps = statics(batch) if use_graphs else batch
                    scored = _scored(batch, first)
                    if steps is not batch:
                        del batch["cells"]  # the graphs read their copy: free this one now
                    for run in runs:
                        with run.on():
                            if not carry:
                                run.out, run.state = run.head.initial(len(fresh), device)
                            run.optimizer.zero_grad(set_to_none=True)
                            outs, (run.out, run.state) = run.unrolls(
                                steps, run.out, run.state, fresh, grad_from=first
                            )
                            with torch.autocast(**autocast):
                                run.loss = _nll(run.head, outs, batch, first, scored)[0].mean()
                    for run in runs:
                        with run.on():
                            if not torch.isfinite(run.loss):
                                raise FloatingPointError("Nonfinite memory objective")
                    for run in runs:
                        with run.on():
                            run.loss.backward()
                            torch.nn.utils.clip_grad_norm_(run.head.parameters(), 1.0)
                            run.optimizer.step()
                            run.out, run.state = run.out.detach(), detach(run.state)
                            run.steps += 1
                    for run, log in zip(runs, logs, strict=True):
                        with run.on():
                            nll = run.loss.item()
                        line = {"epoch": epoch, "step": run.steps, "nll": nll}
                        log.write(json.dumps(line) + "\n")
                    _join([run.stream for run in runs])
                for log in logs:
                    log.flush()
        trained = time.monotonic() - began
        for run in runs:
            run.unrolls = None  # frees the training graphs' memory
        statics.clear()
        _release_cuda(cuda)
        heads = [run.head for run in runs]
        evaluation = _Evaluation(heads, device, graphs, [run.stream for run in runs])
        carried = _evaluate(evaluation, validation, device, loader=loader)
        cleared = _evaluate(evaluation, validation, device, clear_every=18, loader=loader)
        probed = train[: probe_games or len(train)]
        on_train = _evaluate(evaluation, probed, device, loader=loader)
        del evaluation  # and its graphs, before the cache is let go
        healths = memory_health(heads, validation, device, loader=loader)
    finally:
        loader.close()
        _release_cuda(cuda)
    reports = []
    for run, (nll, nll_acting, outs), (nll_cleared, _, _), (_, _, train_outs), health in zip(
        runs, carried, cleared, on_train, healths, strict=True
    ):
        report = {
            "memory": run.memory,
            "window": window,
            "carry": carry,
            "burn_in": burn_in,
            "decisions_per_update": decisions,
            "epochs": epochs,
            "seed": seed,
            "updates": run.steps,
            "parameters": sum(p.numel() for p in run.head.memory.parameters()),
            "state_floats": state_size(run.head.memory.initial(1, "cpu")) or 0,
            "train_seconds": round(trained, 1),
            "validation_nll": nll,
            "validation_nll_acting": nll_acting,
            "validation_nll_cleared_every_18": nll_cleared,
            "probes": probe(train_outs, probed, outs, validation),
            "train_games": len(train),
            "validation_games": len(validation),
            "memory_health": health,
        }
        torch.save({"head": run.head.state_dict(), "report": report}, run.output / "head.pt")
        (run.output / "report.json").write_text(json.dumps(report, indent=2))
        reports.append(report)
    return reports

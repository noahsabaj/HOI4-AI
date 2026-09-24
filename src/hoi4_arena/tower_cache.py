"""The frozen vision tower's reading of every frame, computed once and kept on disk.

When the whole tower is frozen (train-bc --train-last 0), what it reads from a frame never
changes, yet training ran it for every frame of every epoch: about a third of a step. Here
it runs once per frame of each recording, and training reads the result instead
(train-bc --tower-cache). For each frame: the summary, as the policy uses it, and the
tower's patch grid already resized to the cells' CELLS x CELLS. The policy's 1x1
convolution and its bilinear resize are both linear and the resize's weights sum to one,
so resizing first and convolving after is the same map. 768 x 32 x 32 in bfloat16 is
1.5 MB a frame.

The cache belongs to one set of tower weights: its fingerprint is kept with it, and
training refuses a cache made from another tower. A recording is complete once its
`done.json` is written, so an interrupted build resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import normalize, parse_cursor, views
from .models import CELLS

GRID_FILE, SUMMARY_FILE, DONE = "tower-grid.npy", "tower-summary.npy", "done.json"
log = logging.getLogger(__name__)


def fingerprint(encoder):
    """A digest of the tower's weights, in float32 whatever they are kept in."""
    digest = hashlib.sha256()
    for name, value in sorted(encoder.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def read_frozen(encoder, quadrants):
    """(summary, grid resized to CELLS x CELLS) of normalized quadrants (B, 4, 3, H, W)."""
    state = encoder.frozen(None, quadrants)
    summary, grid = encoder.tail(state)
    grid = F.interpolate(grid.float(), (CELLS, CELLS), mode="bilinear", align_corners=False)
    return summary, grid


def as_bits(tensor):
    """bfloat16 values as int16, the dtype numpy can store them in."""
    return tensor.to(torch.bfloat16).view(torch.int16).cpu().numpy()


def from_bits(array):
    return torch.from_numpy(np.ascontiguousarray(array)).view(torch.bfloat16)


def cache_recording(encoder, root, target, device, *, batch=8, stamp=None):
    """Every frame of one recording through the frozen tower, into `target`."""
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    rows = [json.loads(line) for line in (Path(root) / "frames.jsonl").read_text().splitlines()]
    count, width, height = manifest["frames"], manifest["width"], manifest["height"]
    if len(rows) != count:
        raise ValueError(f"{root}: {len(rows)} frame rows for {count} frames")
    target.mkdir(parents=True, exist_ok=True)
    (target / DONE).unlink(missing_ok=True)
    grids = np.lib.format.open_memmap(
        target / GRID_FILE, "w+", np.int16, (count, encoder.dim, CELLS, CELLS)
    )
    summaries = np.lib.format.open_memmap(
        target / SUMMARY_FILE, "w+", np.int16, (count, encoder.dim)
    )
    decoder = subprocess.Popen(
        [shutil.which("ffmpeg"), "-v", "error", "-i", str(Path(root) / "screen.mkv"),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, bufsize=0,
    )  # fmt: skip
    size = width * height * 3
    autocast = {"device_type": torch.device(device).type, "dtype": torch.bfloat16}
    # Frames are read and cut into views on a thread of their own while the tower runs:
    # read one after the other, the pipe and the views took about 20 ms of each 60.
    frames = queue.Queue(maxsize=32)

    def read():
        try:
            for index in range(count):
                buffer = bytearray(size)
                view, got = memoryview(buffer), 0
                while got < size:
                    part = decoder.stdout.readinto(view[got:])
                    if not part:
                        raise ValueError(f"{root}: video ends at frame {index} of {count}")
                    got += part
                frame = np.frombuffer(buffer, np.uint8).reshape(height, width, 3)
                cursor = parse_cursor(rows[index].get("cursor"))
                frames.put((index, views(frame, None, device=device, cursor=cursor).quadrants))
        except Exception as error:  # noqa: BLE001 - handed to the tower's thread to raise.
            frames.put(error)
        frames.put(None)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    pending = []

    def flush():
        nonlocal pending
        if not pending:
            return
        quads = torch.stack([q for _, q in pending])
        with torch.no_grad(), torch.autocast(**autocast):
            summary, grid = read_frozen(encoder, normalize(quads).permute(0, 1, 4, 2, 3))
        first = pending[0][0]
        grids[first : first + len(pending)] = as_bits(grid)
        summaries[first : first + len(pending)] = as_bits(summary)
        pending = []

    try:
        while (item := frames.get()) is not None:
            if isinstance(item, Exception):
                raise item
            pending.append(item)
            if len(pending) == batch:
                flush()
        flush()
    finally:
        if decoder.poll() is None:
            decoder.kill()
        decoder.wait()
        reader.join(timeout=10)
    grids.flush()
    summaries.flush()
    (target / DONE).write_text(json.dumps({"frames": count, "tower": stamp}))
    return count


def cache_tower(
    data,
    checkpoint,
    output,
    *,
    model_path=None,
    device=None,
    sources=None,
    spill=None,
    keep_free_gb=30.0,
):
    """The frozen tower's reading of every frame of every recording in `data`.

    The tower is `checkpoint`'s, in the weights training reads (float32, not halved).
    Recordings already cached for this tower are skipped. With `spill`, a recording that
    would leave less than `keep_free_gb` free on the output's drive goes there instead:
    the fast drive takes what it can hold, and tower_paths finds the rest. Returns what
    was done.
    """
    from .models import Policy, build_encoder

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    encoder = build_encoder(model_path or config["model_path"], config["variant"])
    policy = Policy(encoder)
    policy.load_state_dict(saved["policy"])
    encoder = policy.encoder.to(device).eval().requires_grad_(False)
    stamp = fingerprint(encoder)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    note = {"tower": stamp, "checkpoint": str(checkpoint), "spill": str(spill) if spill else None}
    (output / "tower.json").write_text(json.dumps(note))
    done = skipped = frames = spilled = 0
    began = time.monotonic()
    for manifest_path in sorted(Path(data).glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("complete") or (sources and manifest.get("source") not in sources):
            continue
        name = manifest_path.parent.name
        found = tower_paths(output, name)
        if found is not None and found.get("tower") == stamp:
            skipped += 1
            continue
        size = manifest["frames"] * encoder.dim * (CELLS * CELLS + 1) * 2
        target = output / name
        if spill and shutil.disk_usage(output).free - size < keep_free_gb * 2**30:
            target = Path(spill) / name
            spilled += 1
        started = time.monotonic()
        frames += cache_recording(encoder, manifest_path.parent, target, device, stamp=stamp)
        seconds = max(time.monotonic() - started, 1e-6)
        log.info(
            "%s: %d frames in %.0f s, %.0f MB/s written to %s",
            name, manifest["frames"], seconds, size / seconds / 2**20, target.parent,
        )  # fmt: skip
        done += 1
    return {
        "tower": stamp,
        "recordings": done,
        "spilled": spilled,
        "skipped": skipped,
        "frames": frames,
        "seconds": round(time.monotonic() - began),
    }


def tower_paths(cache, root):
    """The cached files of a recording, checked complete; None when it has none.

    Looked for in the cache, then in the drive it spilled onto (tower.json's `spill`).
    """
    cache, name = Path(cache), Path(root).name
    places = [cache / name]
    note = cache / "tower.json"
    if note.exists():
        spill = json.loads(note.read_text()).get("spill")
        if spill:
            places.append(Path(spill) / name)
    for target in places:
        marker = target / DONE
        if marker.exists():
            return {
                "grid": target / GRID_FILE,
                "summary": target / SUMMARY_FILE,
                **json.loads(marker.read_text()),
            }
    return None

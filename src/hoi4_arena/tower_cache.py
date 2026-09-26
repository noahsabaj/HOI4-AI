"""The frozen vision tower's reading of every frame, computed once and kept on disk.

When the whole tower is frozen (train-bc --train-last 0), what it reads from a frame never
changes, yet training ran it for every frame of every epoch: about a third of a step. Here
it runs once per frame of each recording, and training reads the result instead
(train-bc --tower-cache). For each frame: the summary, as the policy uses it, and the
tower's patch grid already resized to the cells' CELLS x CELLS. The policy's 1x1
convolution and its bilinear resize are both linear and the resize's weights sum to one,
so resizing first and convolving after is the same map.

The grid is nearly all of it, so it is kept as int8 with a scale per frame and channel
(SCALE_FILE): 768 x 32 x 32 is 0.79 MB a frame for the Qwen3.5-0.8B model's tower, and
1024 x 32 x 32 is 1.05 MB for the 4B model's, half of bfloat16 (`int8=False`, which
caches built before 2026-09-26 are and which still read). Only the frames a decision can
read are kept (kept_frames, ROWS_FILE). A build plans every recording's drive and size
before it writes anything, and refuses one the drives cannot hold (plan_drives).

The cache belongs to one set of tower weights: its fingerprint is kept with it, and
training refuses a cache made from another tower. A recording is complete once its
`done.json` is written, so an interrupted build resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import PERIOD_NS, normalize, parse_cursor, views
from .models import CELLS

GRID_FILE, SUMMARY_FILE, DONE = "tower-grid.npy", "tower-summary.npy", "done.json"
# The grid as int8 with a scale per frame and channel (its largest magnitude over the
# cells / 127), half of bfloat16. The Qwen3.5-4B model's tower's cache for 218 games would
# be ~700 GB in bfloat16, more than the drives hold; read back from int8, its text and
# pointer probes scored as from bfloat16 (39.8% against 40.6% and 85.4% both, 2026-09-26).
SCALE_FILE = "tower-grid-scale.npy"
# Each frame's row in the arrays above, -1 for a frame no decision reads (kept_frames).
# A cache without it holds every frame, in order.
ROWS_FILE = "tower-rows.npy"
# What a .npy file's header takes, at most, beyond its data.
HEADER = 128
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


def as_int8(grid):
    """(int8 grid, float16 scale per frame and channel) of a (N, C, H, W) grid."""
    grid = grid.float()
    scale = grid.abs().amax((-2, -1)).clamp_min(1e-6) / 127
    values = (grid / scale[..., None, None]).round().clamp(-127, 127).to(torch.int8)
    return values.cpu().numpy(), scale.to(torch.float16).cpu().numpy()


def read_grid(grid, scale=None):
    """A cached grid as bfloat16: its bits (from_bits), or int8 values times their scale."""
    if scale is None:
        return from_bits(grid)
    values = torch.from_numpy(np.ascontiguousarray(grid)).float()
    factor = torch.from_numpy(np.ascontiguousarray(scale)).float()
    return (values * factor[..., None, None]).to(torch.bfloat16)


def kept_frames(times):
    """The frames any policy decision can read, of frames at `times` (ns): at each whole
    number of decision intervals after the first frame, the latest frame by then.

    dataset.session_labels puts its decisions at the first frame's time plus a whole
    lead-in plus whole intervals, and each reads the latest frame by its time, so for any
    lead-in these are all the frames it reads. A 5 Hz recording's jitter leaves ~6% of
    its frames between two decisions (scripted-v6: 346,664 of 370,121), a 10 Hz one half.
    A reader that looks elsewhere (the inverse dynamics model's detail shift) finds its
    frame missing and says so (dataset._Stream).
    """
    times = np.asarray(times, np.int64)
    marks = np.arange(times[0], times[-1] + 1, PERIOD_NS, dtype=np.int64)
    return np.unique(np.searchsorted(times, marks, side="right") - 1)


def frame_bytes(dim, int8=True):
    """What one kept frame of a tower `dim` wide takes: grid, summary, and the scales."""
    return dim * (CELLS * CELLS * (1 if int8 else 2) + 2 + (2 if int8 else 0))


def recording_bytes(frames, kept, dim, int8=True):
    """What a recording of `frames` frames, `kept` of them cached, takes on disk."""
    return kept * frame_bytes(dim, int8) + frames * 4 + 4 * HEADER


def frame_times(root):
    rows = (Path(root) / "frames.jsonl").read_text().splitlines()
    return np.array([json.loads(line)["t_ns"] for line in rows], np.int64), rows


def cache_recording(encoder, root, target, device, *, batch=8, stamp=None, int8=True):
    """The frames of one recording a decision can read (kept_frames) through the frozen
    tower, into `target`: the grid as int8 and its scales, or with `int8` off as bfloat16."""
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    times, lines = frame_times(root)
    count, width, height = manifest["frames"], manifest["width"], manifest["height"]
    if len(lines) != count:
        raise ValueError(f"{root}: {len(lines)} frame rows for {count} frames")
    kept = kept_frames(times)
    rows = np.full(count, -1, np.int32)
    rows[kept] = np.arange(len(kept), dtype=np.int32)
    target.mkdir(parents=True, exist_ok=True)
    (target / DONE).unlink(missing_ok=True)
    kind = np.int8 if int8 else np.int16
    grids = np.lib.format.open_memmap(
        target / GRID_FILE, "w+", kind, (len(kept), encoder.dim, CELLS, CELLS)
    )
    scales = None
    if int8:
        scales = np.lib.format.open_memmap(
            target / SCALE_FILE, "w+", np.float16, (len(kept), encoder.dim)
        )
    else:
        (target / SCALE_FILE).unlink(missing_ok=True)
    summaries = np.lib.format.open_memmap(
        target / SUMMARY_FILE, "w+", np.int16, (len(kept), encoder.dim)
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
                if rows[index] < 0:
                    continue
                frame = np.frombuffer(buffer, np.uint8).reshape(height, width, 3)
                cursor = parse_cursor(json.loads(lines[index]).get("cursor"))
                seen = views(frame, None, device=device, cursor=cursor).quadrants
                frames.put((int(rows[index]), seen))
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
        if int8:
            values, scale = as_int8(grid)
            grids[first : first + len(pending)] = values
            scales[first : first + len(pending)] = scale
        else:
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
    if scales is not None:
        scales.flush()
    np.save(target / ROWS_FILE, rows)
    done = {"frames": count, "kept": len(kept), "tower": stamp, "int8": int8}
    (target / DONE).write_text(json.dumps(done))
    return count


def _existing(path):
    """`path`, or the nearest folder above it that exists (to ask its drive about)."""
    path = Path(path).absolute()
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def _free(path):
    """Bytes free on `path`'s drive."""
    return shutil.disk_usage(_existing(path)).free


def _drive(path):
    """Which drive (volume) `path` is on."""
    return os.stat(_existing(path)).st_dev


def _partial(folder):
    """Bytes a recording's unfinished folder (no DONE) holds, which its build overwrites."""
    if not folder.is_dir() or (folder / DONE).exists():
        return 0
    return sum(f.stat().st_size for f in folder.iterdir() if f.is_file())


def plan_drives(todo, output, spill=None, keep_free_gb=30.0):
    """Each recording's folder, decided before anything is written: `todo` is (name, bytes)
    pairs. The output's drive takes all it can while keeping `keep_free_gb` free, and the
    rest goes to `spill` only when one is given; a build that does not fit raises, with
    the numbers, instead of filling a drive halfway through. A folder on the same drive as
    the output shares its space.

    Returns ({name: folder}, {folder: [recordings, bytes, free left after]}).
    """
    margin = keep_free_gb * 2**30
    places = [Path(output)] + ([Path(spill)] if spill else [])
    drive = {p: _drive(p) for p in places}
    free = {drive[p]: _free(p) for p in places}
    chosen, summary, left = {}, {p: [0, 0, 0] for p in places}, []
    for name, need in todo:
        for place in places:
            credit = _partial(place / name)
            if free[drive[place]] + credit - need >= margin:
                free[drive[place]] -= need - credit
                chosen[name] = place / name
                summary[place][0] += 1
                summary[place][1] += need
                break
        else:
            left.append((name, need))
    for place in places:
        summary[place][2] = free[drive[place]]
    if left:
        gb = 2**30
        total = sum(need for _, need in todo)
        drives = "; ".join(f"{p} has {_free(p) / gb:.1f} GB free" for p in places)
        more = "" if spill else ", or name a second drive for the rest with --spill"
        raise RuntimeError(
            f"the cache needs {total / gb:.1f} GB for {len(todo)} recordings and "
            f"{keep_free_gb:g} GB must stay free, but {drives}: "
            f"{len(left)} recordings ({sum(n for _, n in left) / gb:.1f} GB) have no room. "
            f"Free some space{more}, or lower --keep-free"
        )
    return chosen, summary


def cache_tower(data, checkpoint, output, *, dry_run=False, **options):
    """The tower cache (_cache_tower), one build at a time per cache folder (RunLock).
    `dry_run` plans the build and reports it without writing anything."""
    from .learning import RunLock

    if dry_run:
        return _cache_tower(data, checkpoint, output, dry_run=True, **options)
    with RunLock(output):
        return _cache_tower(data, checkpoint, output, **options)


def _cache_tower(
    data,
    checkpoint,
    output,
    *,
    model_path=None,
    device=None,
    sources=None,
    spill=None,
    keep_free_gb=30.0,
    int8=True,
    dry_run=False,
):
    """The frozen tower's reading of every frame a decision reads, of every recording in
    `data`.

    The tower is `checkpoint`'s, in the weights training reads (float32, not halved).
    Recordings already cached for this tower are skipped. Before any is cached the whole
    build's size is worked out and placed (plan_drives): on the output's drive, keeping
    `keep_free_gb` free, and whatever does not fit on `spill`, only when it is given;
    tower_paths finds them there. A build that does not fit is refused. Returns what was
    done. With `model_path` naming another tower than the checkpoint's, that tower is
    cached in its own pretrained weights (train.load_carried). `int8` off keeps the grid
    in bfloat16, twice the space.
    """
    from .models import Policy, build_encoder
    from .train import load_carried

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    encoder = build_encoder(model_path or config["model_path"], config["variant"])
    policy = Policy(encoder)
    load_carried(policy, saved["policy"])
    encoder = policy.encoder.eval().requires_grad_(False)
    stamp = fingerprint(encoder)
    output = Path(output)
    note_path = output / "tower.json"
    earlier = json.loads(note_path.read_text()).get("spill") if note_path.exists() else None
    if spill and earlier and Path(spill).absolute() != Path(earlier).absolute():
        raise ValueError(
            f"{output} already spilled onto {earlier}; a cache spans at most two folders"
        )
    todo, skipped, frames, kept = [], 0, 0, 0
    for manifest_path in sorted(Path(data).glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("complete") or (sources and manifest.get("source") not in sources):
            continue
        name = manifest_path.parent.name
        found = tower_paths(output, name)
        if found is not None and found.get("tower") == stamp:
            skipped += 1
            continue
        times, _ = frame_times(manifest_path.parent)
        count = len(kept_frames(times))
        need = recording_bytes(len(times), count, encoder.dim, int8)
        todo.append((name, manifest_path.parent, need))
        frames, kept = frames + len(times), kept + count
    total = sum(need for _, _, need in todo)
    gb = 2**30
    log.info(
        "cache-tower: %d recordings to cache (%d already are): %d of their %d frames, those "
        "a decision reads, at %.2f MB a frame (%s, tower %d wide), %.1f GB in all",
        len(todo), skipped, kept, frames, frame_bytes(encoder.dim, int8) / 2**20,
        "int8" if int8 else "bfloat16", encoder.dim, total / gb,
    )  # fmt: skip
    chosen, placed = plan_drives(
        [(name, need) for name, _, need in todo], output, spill, keep_free_gb
    )
    for place, (count, need, free) in placed.items():
        log.info(
            "cache-tower: %s takes %d recordings, %.1f GB, leaving %.1f GB free",
            place, count, need / gb, free / gb,
        )  # fmt: skip
    report = {
        "tower": stamp,
        "int8": int8,
        "width": encoder.dim,
        "frame_bytes": frame_bytes(encoder.dim, int8),
        "frames": frames,
        "kept": kept,
        "bytes": total,
        "places": {str(p): {"recordings": c, "bytes": n} for p, (c, n, _) in placed.items()},
        "skipped": skipped,
    }
    if dry_run:
        return {"planned": len(todo), **report}
    encoder = encoder.to(device)
    output.mkdir(parents=True, exist_ok=True)
    spilled_to = spill or earlier
    note = {
        "tower": stamp,
        "checkpoint": str(checkpoint),
        "spill": str(spilled_to) if spilled_to else None,
    }
    note_path.write_text(json.dumps(note))
    done = spilled = 0
    began = time.monotonic()
    margin = keep_free_gb * 2**30
    for name, root, need in todo:
        target = chosen[name]
        # Other writers share the drives: check again, and stop rather than fill one.
        room = _free(target) + _partial(target) - need
        if room < margin:
            raise RuntimeError(
                f"{target.parent} has {(room + need) / gb:.1f} GB free now, too little for "
                f"{name} ({need / gb:.1f} GB) and {keep_free_gb:g} GB to spare; stopped "
                f"after {done} recordings (the build resumes where it stopped)"
            )
        spilled += target.parent != output
        started = time.monotonic()
        cache_recording(encoder, root, target, device, stamp=stamp, int8=int8)
        seconds = max(time.monotonic() - started, 1e-6)
        log.info(
            "%s: %.0f s, %.0f MB/s written to %s",
            name, seconds, need / seconds / 2**20, target.parent,
        )  # fmt: skip
        done += 1
    return {
        **report,
        "recordings": done,
        "spilled": spilled,
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
            found = {
                "grid": target / GRID_FILE,
                "summary": target / SUMMARY_FILE,
                **json.loads(marker.read_text()),
            }
            if found.get("int8"):
                found["scale"] = target / SCALE_FILE
            if (target / ROWS_FILE).exists():
                found["rows"] = target / ROWS_FILE
            return found
    return None

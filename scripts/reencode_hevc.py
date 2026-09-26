"""Re-encode recordings' H.264 4:4:4 video as lossless HEVC 4:4:4, which the GPU decodes.

    python scripts/reencode_hevc.py ROOT [ROOT...] [--dry-run] [--keep-original]

Every recording under each ROOT (ROOT/*/manifest.json, dataset junctions followed) whose
screen.mkv is H.264 4:4:4 is encoded again on this PC's NVIDIA encoder, losslessly
(hevc_nvenc -tune lossless): its decoded frames are the H.264's, bit for bit, so labels,
tower caches and anything else made from the frames stay valid. It is written beside the
original as screen.hevc-partial.mkv, checked, and only then swapped in:

- the same number of packets with the same timestamps, so ffmpeg's reader yields the same
  frames in the same places;
- every decoded frame identical to the original's (4:4:4 planes, compared whole);

The original goes to screen.h264.mkv with --keep-original, else is deleted after the swap.
The manifest records the re-encode under "reencoded". Lossless HEVC of these recordings
is ~3.1x the H.264's size (2026-09-26, four recordings), so each file waits for three and
a half times its size free on its drive, plus --keep-free GB, or the run stops.

A recording is skipped when it is not finished (its manifest not complete, or its video
written in the last 10 minutes), when its video is already HEVC or not 4:4:4 YUV, or when
another process holds its video open (a training run reading it): Windows refuses the
rename, the new file is removed and the next run tries again. Safe to stop at any point
and run again: a leftover partial is removed, and a swap cut short between its two
renames is finished.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hoi4_arena import nvdec  # noqa: E402

PARTIAL = "screen.hevc-partial.mkv"
ORIGINAL = "screen.h264.mkv"
# Lossless 4:4:4 HEVC on NVENC, with the recordings' own keyframe spacing (20 s at 5 Hz)
# and no B-frames, so packets stay in capture order.
ENCODER = ["-c:v", "hevc_nvenc", "-preset", "p5", "-tune", "lossless", "-profile:v", "rext",
           "-pix_fmt", "yuv444p", "-bf", "0", "-g", "100"]  # fmt: skip
GROWTH = 3.5
FRESH = 600


def probe(path):
    """(codec, pix_fmt) of the video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,pix_fmt", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    stream = json.loads(out)["streams"][0]
    return stream["codec_name"], stream["pix_fmt"]


def packet_times(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return [line.strip() for line in out.splitlines() if line.strip()]


def identical(original, new, width, height):
    """Frames compared: (count, first differing frame or None)."""
    a = nvdec.FfmpegFrames(original, width, height, "yuv444p")
    b = nvdec.open_frames(new, width, height, yuv=True)
    n = 0
    try:
        while True:
            x, y = a.read(), b.read()
            if x is None or y is None:
                return n, (None if x is None and y is None else n)
            if not np.array_equal(x, y):
                return n, n
            n += 1
    finally:
        a.close()
        b.close()


def finish_swap(folder):
    """A swap cut short between its renames: the original moved, the new one not yet."""
    video, partial, original = folder / "screen.mkv", folder / PARTIAL, folder / ORIGINAL
    if not video.exists() and original.exists():
        manifest = json.loads((folder / "manifest.json").read_text())
        if partial.exists() and manifest.get("reencoded", {}).get("pending"):
            os.replace(partial, video)
        else:
            os.replace(original, video)
    if partial.exists():
        partial.unlink()


def write_manifest(folder, manifest):
    tmp = folder / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, folder / "manifest.json")


def reencode(folder, keep_original=False, keep_free=20.0, dry_run=False, log=print):
    """Re-encode one recording. Returns what happened, as a word."""
    folder = Path(folder)
    finish_swap(folder)
    video = folder / "screen.mkv"
    manifest = json.loads((folder / "manifest.json").read_text())
    if not manifest.get("complete") or time.time() - video.stat().st_mtime < FRESH:
        return "unfinished"
    if manifest.get("codec") not in nvdec.YUV444_CODECS:
        return "not-yuv444"
    codec, pix_fmt = probe(video)
    if codec == "hevc":
        return "already"
    if codec != "h264" or pix_fmt != "yuv444p":
        return "not-yuv444"
    size = video.stat().st_size
    free = shutil.disk_usage(folder).free
    if free < size * GROWTH + keep_free * 1e9:
        return "no-space"
    if dry_run:
        return "would"
    width, height = manifest["width"], manifest["height"]
    partial = folder / PARTIAL
    started = time.perf_counter()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(video), "-map", "0:v",
         "-map_metadata", "0", *ENCODER, "-fps_mode", "passthrough", str(partial)],
        check=True,
    )  # fmt: skip
    encoded = time.perf_counter() - started
    if packet_times(partial) != packet_times(video):
        partial.unlink()
        raise RuntimeError(f"{folder.name}: the re-encode's timestamps differ")
    count, differs = identical(video, partial, width, height)
    if differs is not None:
        partial.unlink()
        raise RuntimeError(f"{folder.name}: frame {differs} differs after the re-encode")
    after = partial.stat().st_size
    manifest["reencoded"] = {
        "from": codec,
        "to": "hevc",
        "encoder": " ".join(ENCODER),
        "lossless": True,
        "frames_checked": count,
        "bytes_before": size,
        "bytes_after": after,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pending": True,
    }
    try:
        os.replace(video, folder / ORIGINAL)
    except PermissionError:
        # Open elsewhere (a training run reading it): try again another time.
        partial.unlink()
        return "busy"
    write_manifest(folder, manifest)
    os.replace(partial, video)
    del manifest["reencoded"]["pending"]
    write_manifest(folder, manifest)
    if not keep_original:
        (folder / ORIGINAL).unlink()
    log(f"{folder.name}: {count} frames identical, {size / 1e6:.0f} -> {after / 1e6:.0f} MB, "
        f"{count / encoded:.0f} frames/s")  # fmt: skip
    return "done"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--dry-run", action="store_true", help="Say what would be re-encoded")
    ap.add_argument("--keep-original", action="store_true", help=f"Keep it as {ORIGINAL}")
    ap.add_argument("--keep-free", type=float, default=20.0, help="GB to leave on each drive")
    ap.add_argument("--limit", type=int, help="Re-encode at most this many recordings")
    a = ap.parse_args(argv)
    counts, done = {}, 0
    folders = [m.parent for root in a.roots for m in sorted(Path(root).glob("*/manifest.json"))]
    for folder in folders:
        if a.limit is not None and done >= a.limit:
            break
        what = reencode(folder, a.keep_original, a.keep_free, a.dry_run)
        counts[what] = counts.get(what, 0) + 1
        done += what in ("done", "would")
        if what == "no-space":
            print(f"{folder.name}: not enough free space; stopping", file=sys.stderr)
            break
    print(json.dumps(counts))


if __name__ == "__main__":
    main()

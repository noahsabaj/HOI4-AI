"""Turn plain video of the game into a recording the inverse dynamics model can label.

A clip from a friend's recorder or a published video has frames and nothing else: no
input events, no capture times, no pointer position. The timing comes from the video's
own frame rate, and the pointer is found in the pixels by matching the game's pointer
images (saved from the worker with `hoi4-arena pointer`). What comes out has the same
layout as a recording made here, with source "video" and no inputs, ready for `label`.

Whether a video may be used is the user's decision; this does not fetch anything.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import recorded_speed
from .recording import split_for_session

# A match whose masked squared error is below this share of the worst possible is the
# pointer. Measured on synthetic frames it is exact at 0; kept loose enough for video
# compression, tight enough that map texture does not pass.
POINTER_MATCH = 0.02
# How far the pointer is searched for around where it was, in pixels, before the whole
# frame is searched. At 30 fps a hand-moved pointer rarely travels this far in a frame.
NEAR = 160


def save_pointer(desktop, path):
    """Save the pointer Windows is showing through `desktop`, with its hotspot beside it."""
    image, hotspot = desktop.pointer()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, "RGBA").save(path)
    path.with_suffix(".json").write_text(json.dumps({"hotspot": list(hotspot)}))
    return {"pointer": str(path), "size": list(image.shape[1::-1]), "hotspot": list(hotspot)}


def load_pointer(path):
    """An RGBA pointer image and its hotspot, as `save_pointer` wrote them."""
    path = Path(path)
    image = np.asarray(Image.open(path).convert("RGBA"))
    hotspot = tuple(json.loads(path.with_suffix(".json").read_text())["hotspot"])
    return image, hotspot


def find_pointer(rgb, pointers, near=None):
    """Where the pointer's hotspot is in `rgb`, as client pixels, or None.

    Each pointer image is matched where it is opaque only, so whatever is under its
    transparent corners does not count against it. With `near`, an (x, y) guess, the
    search looks around it first and falls back to the whole frame.
    """
    import cv2

    frame = np.ascontiguousarray(rgb)
    height, width = frame.shape[:2]
    windows = [(0, 0, width, height)]
    if near is not None:
        x, y = near
        box = (max(0, x - NEAR), max(0, y - NEAR), min(width, x + NEAR), min(height, y + NEAR))
        windows.insert(0, box)
    for left, top, right, bottom in windows:
        best = None
        for image, (hx, hy) in pointers:
            h, w = image.shape[:2]
            if right - left < w or bottom - top < h:
                continue
            mask = (image[:, :, 3] > 127).astype(np.float32)
            area = float(mask.sum())
            if not area:
                continue
            region = frame[top:bottom, left:right].astype(np.float32)
            template = image[:, :, :3].astype(np.float32)
            scores = cv2.matchTemplate(region, template, cv2.TM_SQDIFF, mask=np.dstack([mask] * 3))
            score, _, (mx, my), _ = cv2.minMaxLoc(scores)
            score /= area * 3 * 255**2
            if score < POINTER_MATCH and (best is None or score < best[0]):
                best = (score, left + mx + hx, top + my + hy)
        if best is not None:
            return int(best[1]), int(best[2])
    return None


def _probe(video):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("FFprobe is required to import video")
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-of", "json"]
        + ["-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames", str(video)],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    num, den = (int(v) for v in stream["avg_frame_rate"].split("/"))
    return stream["width"], stream["height"], num / den, int(stream["nb_read_frames"])


def import_video(video, output, *, pointers, game_speed, split=None):
    """Write `output` as a recording of `video`: its frames, their times, the pointer.

    Frames where no pointer is found keep the last position seen, and the manifest
    counts them. A video where the pointer is never found is refused: without it the
    fovea would be a guess.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to import video")
    speed = recorded_speed(game_speed)
    sprites = [load_pointer(p) for p in pointers]
    width, height, fps, count = _probe(video)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    decoder = subprocess.Popen(
        [ffmpeg, "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE,
    )
    rows, last, found = [], None, 0
    try:
        for index in range(count):
            buffer = decoder.stdout.read(width * height * 3)
            if len(buffer) != width * height * 3:
                raise ValueError("Video ended before the frame count it reported")
            frame = np.frombuffer(buffer, np.uint8).reshape(height, width, 3)
            seen = find_pointer(frame, sprites, near=last)
            if seen is not None:
                found += 1
                last = seen
            rows.append({"index": index, "t_ns": int(round(index / fps * 1e9)), "seen": seen})
    finally:
        if decoder.poll() is None:
            decoder.kill()
        decoder.wait()
    first = next((r["seen"] for r in rows if r["seen"] is not None), None)
    if first is None:
        shutil.rmtree(output)
        raise ValueError("The pointer was never found; save its images with `hoi4-arena pointer`")
    with (output / "frames.jsonl").open("w", encoding="utf8") as frames:
        position = first
        for row in rows:
            position = row.pop("seen") or position
            frames.write(json.dumps({**row, "cursor": list(position), "events": []}) + "\n")
    # The pixels are kept as they came, only rewrapped: training decodes with FFmpeg,
    # whatever the codec, and a re-encode would lose detail the model reads.
    remux = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(video), "-map", "0:v:0", "-c", "copy"]
        + [str(output / "screen.mkv")],
        capture_output=True,
    )
    if remux.returncode:
        raise RuntimeError(f"Could not remux {video}: {remux.stderr.decode(errors='replace')}")
    session = output.name
    manifest = {
        "schema": 1,
        "session_id": session,
        "split": split or split_for_session(session),
        "source": "video",
        "width": width,
        "height": height,
        "video_source": "imported",
        "nominal_fps": fps,
        **speed,
        "complete": True,
        "frames": count,
        "pointer_found": found,
        "privileged_state": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"frames": count, "pointer_found": found, "fps": fps}

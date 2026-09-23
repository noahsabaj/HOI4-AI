import json
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from hoi4_arena.dataset import session_labels
from hoi4_arena.video_import import find_pointer, import_video, load_pointer


def _arrow():
    """An 8x8 pointer: an opaque white triangle with a black edge, transparent elsewhere."""
    image = np.zeros((8, 8, 4), np.uint8)
    for y in range(8):
        image[y, : y + 1] = (255, 255, 255, 255)
        image[y, y] = (0, 0, 0, 255)
    return image, (0, 0)


def _frame(rng, at, arrow):
    frame = rng.integers(40, 120, (48, 64, 3), dtype=np.uint8)
    image, (hx, hy) = arrow
    x, y = at[0] - hx, at[1] - hy
    alpha = image[:, :, 3:] > 0
    patch = frame[y : y + 8, x : x + 8]
    frame[y : y + 8, x : x + 8] = np.where(alpha, image[:, :, :3], patch)
    return frame


def test_the_pointer_is_found_over_any_background_and_only_where_it_is():
    rng = np.random.default_rng(0)
    arrow = _arrow()
    frame = _frame(rng, (21, 13), arrow)
    assert find_pointer(frame, [arrow]) == (21, 13)
    # Searched near a stale guess first, then the whole frame.
    assert find_pointer(frame, [arrow], near=(60, 40)) == (21, 13)
    assert find_pointer(rng.integers(40, 120, (48, 64, 3), dtype=np.uint8), [arrow]) is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_video_becomes_a_recording_with_times_and_pointer_positions(tmp_path):
    rng = np.random.default_rng(1)
    arrow = _arrow()
    Image.fromarray(arrow[0], "RGBA").save(tmp_path / "arrow.png")
    (tmp_path / "arrow.json").write_text(json.dumps({"hotspot": [0, 0]}))
    assert load_pointer(tmp_path / "arrow.png")[1] == (0, 0)
    positions = [(10 + i, 10 + i // 2) for i in range(40)]
    frames = [_frame(rng, at, arrow) for at in positions]
    frames[20] = rng.integers(40, 120, (48, 64, 3), dtype=np.uint8)  # Pointer hidden.
    video = tmp_path / "clip.mkv"
    encoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "64x48"]
        + ["-r", "10", "-i", "pipe:0", "-c:v", "ffv1", str(video)],
        stdin=subprocess.PIPE,
    )
    encoder.stdin.write(b"".join(f.tobytes() for f in frames))
    encoder.stdin.close()
    assert encoder.wait(timeout=30) == 0
    result = import_video(video, tmp_path / "game", pointers=[tmp_path / "arrow.png"], game_speed=4)
    assert result == {"frames": 40, "pointer_found": 39, "fps": 10.0}
    rows = [json.loads(line) for line in (tmp_path / "game/frames.jsonl").read_text().splitlines()]
    assert [r["t_ns"] for r in rows[:3]] == [0, 100_000_000, 200_000_000]
    assert [tuple(r["cursor"]) for r in rows[:20]] == positions[:20]
    assert tuple(rows[20]["cursor"]) == positions[19], "a hidden pointer keeps its last place"
    manifest = json.loads((tmp_path / "game/manifest.json").read_text())
    assert manifest["source"] == "video" and manifest["game_speed"] == 4
    labels = session_labels(tmp_path / "game", sources=("video",))
    assert len(labels["decisions"]) == 10 and not labels["actions"].any()

"""The recorders and the worker's client load without torch (layout.py)."""

import subprocess
import sys


def test_recording_and_the_worker_client_import_without_torch():
    code = (
        "import sys\n"
        "import hoi4_arena.desktop, hoi4_arena.recording, hoi4_arena.remote\n"
        "import hoi4_arena.ai_games, hoi4_arena.telemetry\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_dataset_still_offers_the_moved_names():
    from hoi4_arena import dataset, layout

    for name in (
        "VIEW_SIZE", "DETAIL_SIZE", "FOVEA_SIZE", "QUADRANTS", "GAME_SPEED_SECONDS", "Views",
        "hw", "parse_cursor", "recorded_speed",
    ):  # fmt: skip
        assert getattr(dataset, name) is getattr(layout, name), name

"""scripts/check_cache.py, the gate a faster cache-tower must pass: a cache equal to the
reference, or within one int8 step of it."""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _gate():
    spec = importlib.util.spec_from_file_location("check_cache", ROOT / "scripts/check_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cache(folder, grid, scale, summary):
    gate = _gate()
    folder.mkdir(parents=True)
    np.save(folder / gate.GRID, grid)
    np.save(folder / gate.SCALE, scale.astype(np.float16))
    np.save(folder / gate.SUMMARY, summary)
    np.save(folder / gate.ROWS, np.arange(len(grid), dtype=np.int32))
    (folder / gate.DONE).write_text(json.dumps({"int8": True}))


def test_the_gate_passes_a_cache_within_one_int8_step_and_fails_one_beyond(tmp_path):
    gate = _gate()
    rng = np.random.default_rng(0)
    grid = rng.integers(-120, 120, (3, 4, 32, 32)).astype(np.int8)
    scale = np.full((3, 4), 0.5, np.float32)
    summary = (np.float32(rng.normal(size=(3, 4))).view(np.int32) >> 16).astype(np.int16)
    _cache(tmp_path / "ref" / "game", grid, scale, summary)
    shutil.copytree(tmp_path / "ref", tmp_path / "same")
    same = gate.compare(tmp_path / "ref", tmp_path / "same")
    assert same["identical"] and same["passed"] and same["grid_max_steps"] == 0

    near = grid.copy()
    near[0, 0, 0, 0] += 1
    _cache(tmp_path / "near" / "game", near, scale, summary)
    close = gate.compare(tmp_path / "ref", tmp_path / "near")
    assert not close["identical"] and close["passed"] and close["grid_max_steps"] == 1.0
    assert close["grid_rms"] > 0

    far = grid.copy()
    far[2, 3, 5, 7] += 2
    _cache(tmp_path / "far" / "game", far, scale, summary)
    assert not gate.compare(tmp_path / "ref", tmp_path / "far")["passed"]

    # The summary is held to the same channel's step: 0.5 here.
    moved = (np.float32(_bf16(summary)) + np.float32(0.75)).view(np.int32) >> 16
    _cache(tmp_path / "moved" / "game", grid, scale, moved.astype(np.int16))
    result = gate.compare(tmp_path / "ref", tmp_path / "moved")
    assert not result["passed"] and result["summary_max_steps"] > 1

    rows = tmp_path / "rows"
    shutil.copytree(tmp_path / "ref", rows)
    np.save(rows / "game" / gate.ROWS, np.arange(3, dtype=np.int32)[::-1].copy())
    with pytest.raises(ValueError, match="rows differ"):
        gate.compare(tmp_path / "ref", rows)


def _bf16(bits):
    return (bits.astype(np.int32) << 16).view(np.float32)

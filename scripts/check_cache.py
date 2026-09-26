"""Compare a tower cache with a reference cache of the same recordings (cache-tower's
output folders), for scripts/bench_cache.py or on its own:

    python scripts/check_cache.py REFERENCE CANDIDATE

It passes when every recording's rows are the same and every value reads back within one
int8 step of the reference: one step is the larger of the two caches' scales for that
frame and channel (the grid's largest magnitude over the cells / 127), for the grid and,
for the summary (the mean of the tower's grid over its patches), the same channel's step.
It reports whether the files are bit-identical, and the grid's and summary's largest
error in steps and their RMS error against the RMS of the reference's values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

GRID, SCALE, SUMMARY, ROWS, DONE = (
    "tower-grid.npy",
    "tower-grid-scale.npy",
    "tower-summary.npy",
    "tower-rows.npy",
    "done.json",
)


def _bf16(bits):
    """bfloat16 bits kept as int16 to float32."""
    return (bits.astype(np.int32) << 16).view(np.float32)


def _grid(folder):
    grid = np.load(folder / GRID, mmap_mode="r")
    if (folder / SCALE).exists():
        scale = np.load(folder / SCALE).astype(np.float32)
        return grid, scale
    # A bfloat16 cache: its values, and a step as int8 would have it.
    values = _bf16(np.asarray(grid))
    return values, np.abs(values).max((-2, -1)) / 127


def compare(reference, candidate, chunk=64):
    reference, candidate = Path(reference), Path(candidate)
    names = sorted(p.parent.name for p in reference.glob(f"*/{DONE}"))
    if not names:
        raise ValueError(f"{reference} holds no finished recording")
    out = {"recordings": len(names), "identical": True, "frames": 0}
    worst_grid = worst_summary = 0.0
    grid_err = grid_ref = summary_err = summary_ref = 0.0
    grid_count = summary_count = 0
    for name in names:
        a, b = reference / name, candidate / name
        if not (b / DONE).exists():
            raise ValueError(f"{b} is not a finished recording")
        files = [GRID, SUMMARY, ROWS] + ([SCALE] if (a / SCALE).exists() else [])
        for file in files:
            if (a / file).read_bytes() != (b / file).read_bytes():
                out["identical"] = False
        rows_a, rows_b = np.load(a / ROWS), np.load(b / ROWS)
        if not np.array_equal(rows_a, rows_b):
            raise ValueError(f"{name}: the kept rows differ")
        grid_a, scale_a = _grid(a)
        grid_b, scale_b = _grid(b)
        sum_a = _bf16(np.load(a / SUMMARY))
        sum_b = _bf16(np.load(b / SUMMARY))
        out["frames"] += len(grid_a)
        for first in range(0, len(grid_a), chunk):
            s = slice(first, first + chunk)
            sa, sb = scale_a[s], scale_b[s]
            va = np.asarray(grid_a[s], np.float32) * (
                sa[..., None, None] if grid_a.dtype == np.int8 else 1
            )
            vb = np.asarray(grid_b[s], np.float32) * (
                sb[..., None, None] if grid_b.dtype == np.int8 else 1
            )
            step = np.maximum(sa, sb)
            step = np.where(step > 0, step, 1e-12)
            diff = np.abs(va - vb)
            worst_grid = max(worst_grid, float((diff / step[..., None, None]).max()))
            grid_err += float((diff.astype(np.float64) ** 2).sum())
            grid_ref += float((va.astype(np.float64) ** 2).sum())
            grid_count += diff.size
            sdiff = np.abs(sum_a[s] - sum_b[s])
            worst_summary = max(worst_summary, float((sdiff / step).max()))
            summary_err += float((sdiff.astype(np.float64) ** 2).sum())
            summary_ref += float((sum_a[s].astype(np.float64) ** 2).sum())
            summary_count += sdiff.size
    out.update(
        grid_max_steps=round(worst_grid, 4),
        grid_rms=float(np.sqrt(grid_err / grid_count)),
        grid_rms_rel=float(np.sqrt(grid_err / max(grid_ref, 1e-30))),
        summary_max_steps=round(worst_summary, 4),
        summary_rms=float(np.sqrt(summary_err / summary_count)),
        summary_rms_rel=float(np.sqrt(summary_err / max(summary_ref, 1e-30))),
    )
    # One step, plus a little for float16 scales rounding the two caches' steps apart.
    out["passed"] = out["identical"] or (worst_grid <= 1.001 and worst_summary <= 1.001)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    print(json.dumps(result, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

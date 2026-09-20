"""Self-calibration, step 2: fit screen<-map homography from hovered province ids.

Each sample says "screen point (x,y) lies inside province P". Fit robustly (RANSAC affine on province
centroids, then refine a homography by minimising each sample's map-space distance to its province).
Writes camera JSON: homography (map px -> screen px), per-province screen centres, residual stats.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
GAME_MAP = Path("C:/Program Files (x86)/Steam/steamapps/common/Hearts of Iron IV/map")
samples_path, region_path, out_path = (Path(a) for a in sys.argv[1:4])
samples = [s for s in json.loads(samples_path.read_text()) if s.get("province")]
region = json.loads(region_path.read_text())
arena = {p["id"]: p for p in region["provinces"]}

rgb_to_id = {}
with open(GAME_MAP / "definition.csv", newline="") as handle:
    for row in csv.reader(handle, delimiter=";"):
        if row and row[0].isdigit():
            rgb_to_id[(int(row[1]), int(row[2]), int(row[3]))] = int(row[0])
xs = [p["map_x"] for p in arena.values()]
bitmap = Image.open(GAME_MAP / "provinces.bmp")
height = bitmap.height
# region json map_y may count from the bottom; detect by checking a known province's colour at both options.
probe = next(iter(arena.values()))


def province_at(px: float, py: float) -> int | None:
    return rgb_to_id.get(bitmap.getpixel((int(px), int(py))))


flip = province_at(probe["map_x"], probe["map_y"]) != probe["id"]
if flip and province_at(probe["map_x"], height - 1 - probe["map_y"]) != probe["id"]:
    raise SystemExit("cannot locate arena provinces in provinces.bmp")
ys = [(height - 1 - p["map_y"]) if flip else p["map_y"] for p in arena.values()]
x0, x1, y0, y1 = int(min(xs)) - 120, int(max(xs)) + 120, int(min(ys)) - 90, int(max(ys)) + 90
crop = np.asarray(bitmap.crop((x0, y0, x1, y1)).convert("RGB")).astype(np.int64)
ids = np.vectorize(lambda v: rgb_to_id.get(((v >> 16) & 255, (v >> 8) & 255, v & 255), 0))(
    (crop[..., 0] << 16) | (crop[..., 1] << 8) | crop[..., 2])
pixels = {pid: np.argwhere(ids == pid)[:, ::-1] + (x0, y0) for pid in {s["province"] for s in samples}}
samples = [s for s in samples if len(pixels[s["province"]])]
centroid = {pid: pts.mean(0) for pid, pts in pixels.items() if len(pts)}
screen = np.array([[s["x"], s["y"]] for s in samples], float)
target = np.array([centroid[s["province"]] for s in samples])
print(f"{len(samples)} usable samples over {len(centroid)} provinces; y flipped: {flip}")


def to_h(points: np.ndarray) -> np.ndarray:
    return np.hstack([points, np.ones((len(points), 1))])


def apply(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    q = to_h(points) @ h.T
    return q[:, :2] / q[:, 2:3]


def distance(h_screen_to_map: np.ndarray) -> np.ndarray:
    mapped = apply(h_screen_to_map, screen)
    return np.array([np.sqrt(((pixels[s["province"]] + 0.5 - m) ** 2).sum(1)).min()
                     for s, m in zip(samples, mapped)])


rng = np.random.default_rng(0)
best, best_inliers = None, -1
for _ in range(3000):
    pick = rng.choice(len(samples), 3, replace=False)
    try:
        affine = np.linalg.lstsq(to_h(screen[pick]), target[pick], rcond=None)[0].T
    except np.linalg.LinAlgError:
        continue
    h = np.vstack([affine, [0, 0, 1]])
    inliers = int((np.linalg.norm(apply(h, screen) - target, axis=1) < 6).sum())
    if inliers > best_inliers:
        best, best_inliers = h, inliers
h = best
keep = np.linalg.norm(apply(h, screen) - target, axis=1) < 8
print("ransac inliers", int(keep.sum()), "of", len(samples))
score = lambda m: float(np.clip(distance(m)[keep], 0, 10).__pow__(2).sum())  # noqa: E731
current, scale = score(h), np.array([[1e-3, 1e-3, 1.0], [1e-3, 1e-3, 1.0], [2e-7, 2e-7, 0.0]])
for iteration in range(4000):
    trial = h + rng.normal(size=(3, 3)) * scale * (0.3 if iteration > 2000 else 1.0)
    value = score(trial)
    if value < current:
        h, current = trial, value
residual = distance(h)
inside = residual < 0.75
print(f"refined: {int(inside.sum())}/{len(samples)} samples land inside their province; "
      f"median miss of the rest {np.median(residual[~inside]) if (~inside).any() else 0:.2f} map px")
for s, r in zip(samples, residual):
    if r > 3:
        print("  outlier", s["x"], s["y"], s["province"], f"{r:.1f} map px")
map_to_screen = np.linalg.inv(h)
centres = {}
for p in arena.values():
    my = (height - 1 - p["map_y"]) if flip else p["map_y"]
    sx, sy = apply(map_to_screen, np.array([[p["map_x"], my]]))[0]
    centres[str(p["id"])] = [round(float(sx), 1), round(float(sy), 1)]
probe_a, probe_b = apply(map_to_screen, np.array([[x0 + 100.0, y0 + 100.0], [x0 + 101.0, y0 + 100.0]]))
out_path.write_text(json.dumps({"schema_version": 1, "window": [1920, 1080], "screen_to_map": h.tolist(),
                                "map_to_screen": map_to_screen.tolist(), "map_y_flipped": bool(flip),
                                "province_centres": centres, "samples": len(samples),
                                "samples_inside": int(inside.sum()),
                                "screen_px_per_map_px": float(np.linalg.norm(probe_b - probe_a))}, indent=1))
print("wrote", out_path, "screen px per map px:", round(float(np.linalg.norm(probe_b - probe_a)), 2))

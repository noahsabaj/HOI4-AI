"""Does a video encoding keep what the policy reads? Measured against lossless frames.

    python scripts/codec_fidelity.py OUT --truth truth.mkv [--only NAME,...]
    python scripts/codec_fidelity.py OUT --build-truth SOURCE... (PNGs or lossless videos)

The truth is lossless 1080p video (ffv1, bgr0): the frames a recorder would have written,
held in the pixel format the worker hands its encoder (BGRA). Each candidate encodes the
truth with ffmpeg, so its colour conversion is the one a recording gets, and is decoded
with the training reader's exact command (dataset._Stream):
`ffmpeg -v error -i screen.mkv -f rawvideo -pix_fmt rgb24 pipe:1`. Frames stream through
two at a time, so any number fits in memory.

For each candidate: the size; PSNR over the whole frame, the top bar and the bottom panels
(where the small text is); the worst pixel error and its 99.9th percentile; every small
template in --templates wherever it shows in the truth, scored at the same spot on the
decoded frame (TM_CCOEFF_NORMED and TM_SQDIFF_NORMED, the two the recorder and the scripted
player use) and whether its best match moves; the screen rules' mean absolute error
(--rules); and encode and decode speed. Results go to OUT/results.json.

It runs on either PC: on the second one as a compute job (hoi4-arena job ... --kind
script -- codec_fidelity.py ...), where it measures that PC's video encoder.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

W, H = 1920, 1080
SIZE = W * H * 3

# The recordings' encoder until 2026-09-24, and the candidates to replace it.
CANDIDATES = {
    "x264-crf18": ["-c:v", "libx264", "-preset", "faster", "-crf", "18", "-pix_fmt", "yuv444p"],
    "x264-crf14": ["-c:v", "libx264", "-preset", "faster", "-crf", "14", "-pix_fmt", "yuv444p"],
}
for _qp in (10, 12, 14, 16, 18):
    CANDIDATES[f"h264nv-qp{_qp}"] = [
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p5",
        "-tune",
        "hq",
        "-profile:v",
        "high444p",
        "-pix_fmt",
        "yuv444p",
        "-rc",
        "constqp",
        "-qp",
        str(_qp),
        "-bf",
        "0",
        "-g",
        "100",
    ]
for _qp in (12, 14, 16, 18):
    CANDIDATES[f"hevcnv-qp{_qp}"] = [
        "-c:v",
        "hevc_nvenc",
        "-preset",
        "p5",
        "-tune",
        "hq",
        "-profile:v",
        "rext",
        "-pix_fmt",
        "yuv444p",
        "-rc",
        "constqp",
        "-qp",
        str(_qp),
        "-bf",
        "0",
        "-g",
        "100",
    ]
# The slowest preset spends more of the encoder on each frame for fewer bits at the same
# QP; B-frames save bits between frames. Both cost nothing on this PC's CPU.
for _qp in (12, 14, 16):
    CANDIDATES[f"h264nv-p7-qp{_qp}"] = [
        "-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-profile:v", "high444p",
        "-pix_fmt", "yuv444p", "-rc", "constqp", "-qp", str(_qp), "-bf", "0", "-g", "100",
    ]  # fmt: skip
    CANDIDATES[f"h264nv-p7-bf3-qp{_qp}"] = [
        "-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-profile:v", "high444p",
        "-pix_fmt", "yuv444p", "-rc", "constqp", "-qp", str(_qp), "-bf", "3",
        "-b_ref_mode", "middle", "-g", "100",
    ]  # fmt: skip
CANDIDATES["h264nv-lossless"] = [
    "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "lossless", "-profile:v", "high444p",
    "-pix_fmt", "yuv444p", "-bf", "0", "-g", "100",
]  # fmt: skip


def reader(path, threads=0):
    """Decoded rgb24 frames of `path`, the way training reads a recording."""
    extra = ["-threads", str(threads)] if threads else []
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", *extra, "-i", str(path), "-f", "rawvideo", "-pix_fmt",
         "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, bufsize=SIZE,
    )  # fmt: skip


def frames(proc):
    while True:
        buf = proc.stdout.read(SIZE)
        if len(buf) < SIZE:
            proc.wait()
            return
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)


def build_truth(out, sources, hold=3):
    """truth.mkv (ffv1, bgr0) and names.json from PNG screenshots (each held `hold`
    frames, so the encoders see still screens too) and lossless videos."""
    out.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "bgra",
         "-video_size", f"{W}x{H}", "-framerate", "5", "-i", "pipe:0", "-c:v", "ffv1",
         "-level", "3", "-slices", "4", "-pix_fmt", "bgr0", str(out / "truth.mkv")],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    names = []

    def put(rgb, name):
        bgra = np.empty((H, W, 4), np.uint8)
        bgra[..., :3] = rgb[..., ::-1]
        bgra[..., 3] = 255
        enc.stdin.write(bgra.tobytes())
        names.append(name)

    for source in map(Path, sources):
        if source.suffix.lower() == ".png":
            with Image.open(source) as im:
                if im.size != (W, H):
                    continue
                rgb = np.asarray(im.convert("RGB"))
            for k in range(hold):
                put(rgb, f"{source.name}#{k}")
        else:
            for i, f in enumerate(frames(reader(source))):
                put(f, f"{source.stem}:{i:04d}")
    enc.stdin.close()
    enc.wait()
    (out / "names.json").write_text(json.dumps(names))
    return names


def templates(folder):
    """Every small template (a UI element, not a full screenshot) in `folder`."""
    out = {}
    for p in sorted(Path(folder).glob("*.png")):
        with Image.open(p) as im:
            if im.size[0] >= W // 2 or min(im.size) < 6:
                continue
            out[p.stem] = np.asarray(im.convert("RGB"))
    return out


def rules(path):
    """The screen rules' rectangles and reference crops, by name."""
    if not path:
        return {}
    path = Path(path)
    spec = json.loads(path.read_text())
    out = {}
    for name, rule in spec["rules"].items():
        crop = path.parent / rule["template"]
        if crop.exists():
            out[name] = (rule["rect"], np.asarray(Image.open(crop).convert("RGB")).astype(np.int16))
    return out


def held(name):
    """A repeat of the frame before it: a screenshot's second or third copy."""
    return name.rsplit("#", 1)[-1] in ("1", "2") and "#" in name


def sightings(truth, names, temps, cache):
    """{frame: [(template, x, y, ccoeff, sqdiff)]} wherever a template shows in the truth."""
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    found = {}
    for i, f in enumerate(frames(reader(truth, 4))):
        if held(names[i]):
            continue
        for name, t in temps.items():
            cc = cv2.matchTemplate(f, t, cv2.TM_CCOEFF_NORMED)
            _, best, _, (x, y) = cv2.minMaxLoc(cc)
            if best >= 0.85:
                patch = f[y : y + t.shape[0], x : x + t.shape[1]]
                sq = cv2.matchTemplate(patch, t, cv2.TM_SQDIFF_NORMED)
                found.setdefault(i, []).append((name, x, y, float(best), float(sq[0, 0])))
    cache.write_text(json.dumps(found))
    return found


def compare(truth, path, names, found, temps, rule_set):
    sums = {"all": 0.0, "top": 0.0, "bottom": 0.0}
    hist = np.zeros(256, np.int64)
    worst, n = 0, 0
    worst_cc = worst_sq = mae_delta = 0.0
    moved = []
    pairs = zip(frames(reader(truth, 4)), frames(reader(path, 4)), strict=False)
    for i, (t, d) in enumerate(pairs):
        n += 1
        e = cv2.absdiff(t, d)
        worst = max(worst, int(e.max()))
        hist += np.bincount(e.ravel(), minlength=256)
        sq = e.astype(np.float32) ** 2
        sums["all"] += float(sq.mean())
        sums["top"] += float(sq[:80].mean())
        sums["bottom"] += float(sq[H - 120 :].mean())
        for name, x, y, cc0, sq0 in found.get(i, []):
            tm = temps[name]
            cc = cv2.matchTemplate(d, tm, cv2.TM_CCOEFF_NORMED)
            _, _, _, best_at = cv2.minMaxLoc(cc)
            patch = d[y : y + tm.shape[0], x : x + tm.shape[1]]
            sqv = float(cv2.matchTemplate(patch, tm, cv2.TM_SQDIFF_NORMED)[0, 0])
            worst_cc = max(worst_cc, abs(float(cc[y, x]) - cc0))
            worst_sq = max(worst_sq, abs(sqv - sq0))
            if tuple(best_at) != (x, y):
                moved.append(f"{name}@{names[i]}")
        if not held(names[i]):
            for rect, tmpl in rule_set.values():
                x, y, w, h = rect
                a = np.abs(t[y : y + h, x : x + w].astype(np.int16) - tmpl).mean()
                b = np.abs(d[y : y + h, x : x + w].astype(np.int16) - tmpl).mean()
                mae_delta = max(mae_delta, abs(float(b - a)))
    db = {k: float("inf") if v == 0 else 10 * np.log10(255**2 / (v / n)) for k, v in sums.items()}
    cdf = np.cumsum(hist) / hist.sum()
    return {
        "frames": n,
        "psnr": round(db["all"], 2),
        "psnr_top": round(db["top"], 2),
        "psnr_bottom": round(db["bottom"], 2),
        "max_err": worst,
        "p999_err": int(np.searchsorted(cdf, 0.999)),
        "template_ccoeff_change": round(worst_cc, 4),
        "template_sqdiff_change": round(worst_sq, 4),
        "templates_moved": moved,
        "rule_mae_change": round(mae_delta, 2),
    }


def decode_seconds(path):
    """The training reader's decode, alone, output discarded."""
    start = time.perf_counter()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24",
         "pipe:1"],
        stdout=subprocess.DEVNULL, check=True,
    )  # fmt: skip
    return time.perf_counter() - start


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--truth", help="Lossless truth video (default OUT/truth.mkv)")
    ap.add_argument("--names", help="Its frame names (default beside the truth)")
    ap.add_argument("--build-truth", nargs="+", metavar="SOURCE")
    ap.add_argument("--templates", default="artifacts/screens-1080p")
    ap.add_argument("--rules", default="artifacts/calibration-1080p/rules.json")
    ap.add_argument("--only", nargs="+", default=[], help="Candidate names (default: all)")
    ap.add_argument("--threads", type=int, default=4, help="Encoder and OpenCV threads")
    a = ap.parse_args(argv)
    cv2.setNumThreads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.build_truth:
        build_truth(out, a.build_truth)
    truth = Path(a.truth) if a.truth else out / "truth.mkv"
    names = json.loads(Path(a.names or truth.with_name("names.json")).read_text())
    print(f"{len(names)} truth frames", file=sys.stderr, flush=True)
    temps = templates(a.templates)
    rule_set = rules(a.rules)
    found = sightings(truth, names, temps, out / "sightings.json")
    kinds = sorted({s[0] for v in found.values() for s in v})
    print(f"{sum(map(len, found.values()))} template sightings: {kinds}", file=sys.stderr)
    wanted = a.only or list(CANDIDATES)
    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for name in wanted:
        path = out / f"{name}.mkv"
        start = time.perf_counter()
        rc = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(truth), "-an", *CANDIDATES[name],
             "-threads", str(a.threads), str(path)]
        ).returncode  # fmt: skip
        encode = time.perf_counter() - start
        if rc:
            print(f"{name}: encoder exit {rc}", file=sys.stderr, flush=True)
            continue
        row = {
            "candidate": name,
            "kbytes_per_frame": round(path.stat().st_size / len(names) / 1024, 1),
        }
        row.update(compare(truth, path, names, found, temps, rule_set))
        row["encode_fps"] = round(len(names) / encode, 1)
        row["decode_fps"] = round(len(names) / decode_seconds(path), 1)
        results[name] = row
        results_path.write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

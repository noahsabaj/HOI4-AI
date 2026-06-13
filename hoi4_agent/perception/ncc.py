"""Normalized cross-correlation with numpy (no opencv)."""

from __future__ import annotations

import numpy as np
from PIL import Image


def to_gray_f32(img) -> np.ndarray:
    """PIL image or array -> 2-D float32 grayscale."""
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("L"), dtype=np.float32)
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation (Pearson r) of two equal-shaped arrays.

    Returns 0.0 on shape mismatch or zero-variance input. Range [-1, 1].
    """
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size == 0 or a.size != b.size:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    da = float(np.sqrt((a * a).sum()))
    db = float(np.sqrt((b * b).sum()))
    if da == 0.0 or db == 0.0:
        return 0.0
    return float((a * b).sum() / (da * db))


def match_resized(roi: Image.Image, template_gray: np.ndarray) -> float:
    """NCC of an ROI crop against a stored template, resizing the ROI to match."""
    th, tw = template_gray.shape
    roi_gray = to_gray_f32(roi.resize((tw, th)))
    return ncc(roi_gray, template_gray)


def best_match(image, template) -> tuple[float, int, int]:
    """Slide ``template`` over ``image``; return (best_score, x, y).

    Naive O(image*template); intended for small ROIs only, never full frames.
    """
    img = to_gray_f32(image)
    tpl = to_gray_f32(template)
    ih, iw = img.shape
    th, tw = tpl.shape
    if th > ih or tw > iw:
        return (0.0, 0, 0)
    tpl_c = tpl - tpl.mean()
    tnorm = float(np.sqrt((tpl_c * tpl_c).sum()))
    if tnorm == 0.0:
        return (0.0, 0, 0)
    best = (-1.0, 0, 0)
    for y in range(ih - th + 1):
        for x in range(iw - tw + 1):
            win = img[y : y + th, x : x + tw]
            wc = win - win.mean()
            wn = float(np.sqrt((wc * wc).sum()))
            if wn == 0.0:
                continue
            score = float((wc * tpl_c).sum() / (wn * tnorm))
            if score > best[0]:
                best = (score, x, y)
    return best

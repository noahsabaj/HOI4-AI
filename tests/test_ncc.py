import sys

import numpy as np
import pytest
from PIL import Image

from hoi4_agent.perception import ncc


def test_no_opencv_dependency():
    assert "cv2" not in sys.modules


def test_ncc_identity_and_inverse():
    a = np.arange(64, dtype=np.float32).reshape(8, 8)
    assert ncc.ncc(a, a) == pytest.approx(1.0)
    assert ncc.ncc(a, -a) == pytest.approx(-1.0)


def test_ncc_shape_mismatch_and_zero_variance():
    a = np.ones((4, 4), dtype=np.float32)
    assert ncc.ncc(a, a) == 0.0  # zero variance
    assert ncc.ncc(np.arange(4.0), np.arange(9.0)) == 0.0  # mismatch


def test_ncc_noise_drops_score():
    a = np.arange(100, dtype=np.float32).reshape(10, 10)
    rng = np.random.default_rng(0)
    noisy = a + rng.normal(0, 60, a.shape).astype(np.float32)
    assert ncc.ncc(a, noisy) < 0.95


def test_best_match_locates_template():
    # A patterned image (variance) so NCC is well-defined; template cut from it.
    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, (20, 20)).astype("uint8")
    img = Image.fromarray(base)
    tpl_img = img.crop((6, 4, 14, 12))  # 8x6 region with internal structure
    score, x, y = ncc.best_match(img, tpl_img)
    assert score == pytest.approx(1.0, abs=1e-6)
    assert (x, y) == (6, 4)


def test_match_resized_zero_variance_is_zero():
    # uniform region -> zero variance -> NCC defined as 0 (not a false match)
    assert ncc.match_resized(Image.new("L", (30, 30), 128), np.full((10, 10), 128.0, np.float32)) == 0.0

import pytest

from hoi4_agent.errors import ResolutionError
from hoi4_agent.geometry import WindowGeometry


@pytest.mark.parametrize("left,top", [(0, 0), (100, 200)])
def test_norm_to_screen_center(left, top):
    g = WindowGeometry(1, left, top, 2560, 1440)
    assert g.norm_to_screen(g.full_crop(), 500, 500) == (left + 1280, top + 720)


def test_norm_roundtrip():
    g = WindowGeometry(1, 50, 75, 1920, 1080)
    crop = g.full_crop()
    sx, sy = g.norm_to_screen(crop, 640, 360)
    assert g.screen_to_norm(crop, sx, sy) == (640, 360)


def test_panel_crop_fraction():
    g = WindowGeometry(1, 0, 0, 1000, 1000)
    c = g.panel_crop((0.1, 0.2, 0.5, 0.6))
    assert (c.client_x0, c.client_y0, c.crop_w, c.crop_h) == (100, 200, 400, 400)


def test_monitor_offsets_by_crop():
    g = WindowGeometry(1, 10, 20, 2560, 1440)
    crop = g.panel_crop((0.0, 0.0, 0.5, 0.5))
    assert g.monitor(crop) == {"left": 10, "top": 20, "width": 1280, "height": 720}


def test_assert_resolution():
    g = WindowGeometry(1, 0, 0, 2560, 1440)
    g.assert_resolution(2560, 1440)  # no raise
    with pytest.raises(ResolutionError):
        g.assert_resolution(1920, 1080)

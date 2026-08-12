import numpy as np
from PIL import Image

from hoi4_agent.calibration import Calibration
from hoi4_agent.enums import PanelId
from hoi4_agent.geometry import WindowGeometry
from hoi4_agent.io.backends import FakeCapture
from hoi4_agent.perception.perceive import perceive
from hoi4_agent.perception.templates import TemplateStore
from hoi4_agent.perception.tiers import crop_roi, read_speed
from hoi4_agent.schemas import GameDate

GEO = WindowGeometry(1, 0, 0, 100, 100)
ROIS = {
    "date": (0.0, 0.0, 0.3, 0.1),
    "pause": (0.35, 0.0, 0.45, 0.1),
    "speed": (0.5, 0.0, 0.6, 0.1),
    "construction_panel": (0.0, 0.2, 0.4, 0.9),
    "research_panel": (0.6, 0.2, 1.0, 0.9),
    "event_popup": (0.4, 0.4, 0.6, 0.6),
    "pause_menu": (0.42, 0.25, 0.58, 0.4),
    "free_civ_slots": (0.0, 0.12, 0.2, 0.18),
    "idle_research_slots": (0.3, 0.12, 0.5, 0.18),
    "construction_queue": (0.6, 0.12, 0.9, 0.18),
}
CALIB = Calibration(width=100, height=100, rois=ROIS)


def _noise_img(seed=7):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (100, 100, 3), dtype=np.uint8), "RGB")


class _SpyReader:
    def __init__(self):
        self.calls = []

    def read_number(self, crop, field):
        self.calls.append(("number", field))
        return 5

    def read_date(self, crop):
        self.calls.append(("date", None))
        return GameDate(1936, 2, 3)


def _perceive(img, reader, templates=None, **kw):
    return perceive(
        capture=FakeCapture(img), geo=GEO, calib=CALIB,
        templates=templates or TemplateStore(), threshold=0.75, reader=reader, **kw,
    )


def test_gated_fields_not_read_outside_their_panel():
    # No templates -> panel classifies NONE -> slot/queue reads are suppressed.
    spy = _SpyReader()
    ws = _perceive(_noise_img(), spy)
    assert ws.open_panel is PanelId.NONE
    assert ws.free_civ_slots is None
    assert ws.idle_research_slots is None
    assert ws.construction_queue_len is None
    assert ws.date == GameDate(1936, 2, 3)  # ungated field still read
    assert all(kind == "date" for kind, _ in spy.calls)


def test_gated_fields_read_when_panel_matches():
    img = _noise_img()
    templates = TemplateStore()
    # The stored template IS the live crop -> NCC 1.0 -> panel = CONSTRUCTION.
    templates.add("construction_panel", crop_roi(img, GEO, ROIS["construction_panel"]))
    spy = _SpyReader()
    ws = _perceive(img, spy, templates)
    assert ws.open_panel is PanelId.CONSTRUCTION
    assert ws.free_civ_slots == 5
    assert ws.construction_queue_len == 5
    assert ws.idle_research_slots is None  # research-gated, research not open
    fields_read = [f for kind, f in spy.calls if kind == "number"]
    assert "idle_research_slots" not in fields_read


def test_fields_selects_exactly_what_is_read():
    spy = _SpyReader()
    ws = _perceive(_noise_img(), spy, fields={"date"})
    assert ws.date == GameDate(1936, 2, 3)
    assert ws.speed is None
    assert spy.calls == [("date", None)]

    spy2 = _SpyReader()
    ws2 = _perceive(_noise_img(), spy2, read_numbers=False)
    assert spy2.calls == []
    assert ws2.date is None


def test_popup_detected_independently_of_stronger_panel():
    # Both the construction panel AND an event popup match perfectly; the old
    # argmax reported only the panel and the blocking popup became invisible.
    img = _noise_img()
    templates = TemplateStore()
    templates.add("construction_panel", crop_roi(img, GEO, ROIS["construction_panel"]))
    templates.add("event_popup", crop_roi(img, GEO, ROIS["event_popup"]))
    ws = _perceive(img, _SpyReader(), templates, read_numbers=False)
    assert ws.open_panel is PanelId.CONSTRUCTION
    assert ws.event_popup is True  # detected DESPITE the panel winning the argmax
    assert ws.confidence["popup"] >= 0.99


def test_pause_menu_detected():
    img = _noise_img()
    templates = TemplateStore()
    templates.add("pause_menu", crop_roi(img, GEO, ROIS["pause_menu"]))
    ws = _perceive(img, _SpyReader(), templates, read_numbers=False)
    assert ws.pause_menu is True
    assert ws.open_panel is PanelId.NONE


def test_read_pause_near_tie_is_uncertain():
    from hoi4_agent.perception.tiers import read_pause

    img = _noise_img()
    pause_crop = crop_roi(img, GEO, ROIS["pause"])
    arr = np.asarray(pause_crop.convert("L"), dtype=np.float32)

    templates = TemplateStore()
    templates.add("pause_on", arr)
    templates.add("pause_off", arr + np.random.default_rng(1).normal(0, 2, arr.shape))
    paused, score = read_pause(img, GEO, CALIB, templates, threshold=0.75)
    assert paused is None  # both ~perfect: a coin flip must stay uncertain
    assert score >= 0.9

    templates2 = TemplateStore()
    templates2.add("pause_on", arr)
    templates2.add("pause_off", 255.0 - arr)  # clearly different
    paused2, _ = read_pause(img, GEO, CALIB, templates2, threshold=0.75)
    assert paused2 is True


def test_speed_is_template_classified_not_number_read():
    img = _noise_img()
    templates = TemplateStore()
    templates.add("speed_3", crop_roi(img, GEO, ROIS["speed"]))
    spy = _SpyReader()
    ws = _perceive(img, spy, templates, fields={"speed"})
    assert ws.speed == 3
    assert spy.calls == []  # never asked the numeric reader for speed

    value, score = read_speed(img, GEO, CALIB, templates, threshold=0.75)
    assert value == 3 and score >= 0.99
    none_value, _ = read_speed(img, GEO, CALIB, TemplateStore(), threshold=0.75)
    assert none_value is None


# --- the speed indicator, at a size where chevrons survive ---------------------
# The tiny shared GEO above crops speed to 10x10px, which destroys the very
# detail these tests are about, so they use their own frame.
SPEED_GEO = WindowGeometry(1, 0, 0, 1000, 400)
SPEED_ROI = (0.1, 0.1, 0.3, 0.22)  # exactly 200x48 client px
SPEED_CALIB = Calibration(width=1000, height=400, rois={"speed": SPEED_ROI})


def _chevron_widget(lit: int) -> np.ndarray:
    """The speed indicator as HOI4 draws it: one widget, `lit` of 5 chevrons on.

    Faithful in the way that matters. Most of the widget is chrome every speed
    shares, and adjacent speeds differ by one chevron, which is why whole-ROI
    NCC between them lands ~0.96 — matching the 0.9592 measured on the real
    speed_4/speed_5 captures.
    """
    h, w = 48, 200
    a = np.full((h, w), 40.0, dtype=np.float32)
    a[0:6, :] = a[h - 6 :, :] = 110.0
    a[:, 0:6] = a[:, w - 6 :] = 110.0
    a[h // 2 - 1 : h // 2 + 1, 6 : w - 6] = 95.0
    for i in range(5):
        x0 = 20 + i * 32
        a[10:18, x0 : x0 + 6] = 215.0 if i < lit else 70.0
    return a


def _speed_frame(lit: int) -> Image.Image:
    rng = np.random.default_rng(3)
    img = Image.fromarray(rng.integers(0, 255, (400, 1000, 3), dtype=np.uint8), "RGB")
    crop = SPEED_GEO.panel_crop(SPEED_ROI)
    patch = Image.fromarray(_chevron_widget(lit).astype(np.uint8), mode="L").convert("RGB")
    img.paste(patch, (crop.client_x0, crop.client_y0))
    return img


def _speed_templates(lits=range(1, 6)) -> TemplateStore:
    store = TemplateStore()
    for s in lits:
        store.add(f"speed_{s}", _chevron_widget(s))
    return store


def test_whole_roi_ncc_cannot_separate_adjacent_speeds():
    # The premise of the residual fix, asserted so it cannot silently lapse.
    from hoi4_agent.perception.ncc import ncc
    from hoi4_agent.perception.tiers import SPEED_MARGIN

    for s in range(1, 5):
        gap = 1.0 - ncc(_chevron_widget(s), _chevron_widget(s + 1))
        assert gap < SPEED_MARGIN, (
            f"speed_{s} vs speed_{s+1} leaves a whole-ROI margin of {gap:.4f}; "
            "gating SPEED_MARGIN on that score rejects real readings"
        )


def test_speed_is_read_from_residuals_so_every_speed_is_reachable():
    # The regression this replaces: gating SPEED_MARGIN on whole-ROI scores made
    # read_speed return None for speeds 4 and 5 against the real templates
    # (speed_4 vs speed_5 = 0.9592), so set_speed could never confirm reaching
    # the configured run_speed of 4. Residual matching separates them.
    templates = _speed_templates()
    for speed in range(1, 6):
        value, score = read_speed(_speed_frame(speed), SPEED_GEO, SPEED_CALIB,
                                  templates, threshold=0.75)
        assert value == speed, f"speed {speed} read as {value} (conf {score:.3f})"


def test_speed_stays_uncertain_when_residuals_cannot_separate():
    # Two templates that are genuinely identical carry no distinguishing signal,
    # so the guard must still refuse to pick one.
    templates = TemplateStore()
    templates.add("speed_3", _chevron_widget(3))
    templates.add("speed_4", _chevron_widget(3))  # same widget under two names
    value, _ = read_speed(_speed_frame(3), SPEED_GEO, SPEED_CALIB, templates, threshold=0.75)
    assert value is None


def test_speed_uncertain_when_the_roi_is_not_the_widget():
    value, _ = read_speed(_noise_img(), GEO, CALIB, _speed_templates(), threshold=0.75)
    assert value is None  # noise does not clear the whole-ROI gate


def test_speed_single_template_still_readable():
    # Nothing to center against and nothing to confuse: the argmax stands alone.
    value, _ = read_speed(_speed_frame(3), SPEED_GEO, SPEED_CALIB,
                          _speed_templates([3]), threshold=0.75)
    assert value == 3

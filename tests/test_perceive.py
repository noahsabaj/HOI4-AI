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


def test_speed_near_tie_is_uncertain_not_a_guess():
    # speed_1..speed_5 are the SAME widget with a different number of chevrons
    # lit, so they share even more pixels than pause_on/pause_off. A bare argmax
    # turned a coin-flip between adjacent speeds into a confident fact.
    from hoi4_agent.perception.ncc import to_gray_f32
    from hoi4_agent.perception.tiers import SPEED_MARGIN

    img = _noise_img()
    roi = crop_roi(img, GEO, ROIS["speed"])
    gray = to_gray_f32(roi)

    templates = TemplateStore()
    templates.add("speed_3", roi)  # exact match: scores ~1.0
    # A near-identical rival: the same pixels plus a whisper of noise, so it
    # lands just under speed_3 rather than distinctly below it.
    rng = np.random.default_rng(7)
    templates.add("speed_4", gray + rng.normal(0, 12.0, gray.shape).astype(np.float32))

    value, score = read_speed(img, GEO, CALIB, templates, threshold=0.75)
    assert score >= 0.75, "the rival must be a near-tie, not a low-confidence match"
    assert value is None, "a near-tie must read as uncertain"

    # Widen the gap past SPEED_MARGIN and the argmax is trustworthy again.
    clear = TemplateStore()
    clear.add("speed_3", roi)
    clear.add("speed_4", 255.0 - gray)  # inverted: scores about -1
    value2, score2 = read_speed(img, GEO, CALIB, clear, threshold=0.75)
    assert value2 == 3 and score2 >= 0.99
    assert SPEED_MARGIN > 0  # the guard is a real threshold, not a no-op

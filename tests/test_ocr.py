import pytest
from PIL import Image

from hoi4_agent.perception.digits import ChainReader
from hoi4_agent.perception.ocr import OcrReader
from hoi4_agent.schemas import GameDate

CROP = Image.new("RGB", (40, 12), (10, 10, 10))


def _engine(*texts: str):
    """Fake rapidocr-shaped engine: engine(ndarray) -> (results, elapse)."""

    def run(_img):
        if not texts:
            return None, 0.0
        return [((0, 0, 1, 1), t, 0.9) for t in texts], 0.0

    return run


class _V3Output:
    """Shape of rapidocr v2+/v3 results: an object with a .txts tuple."""

    def __init__(self, *txts: str) -> None:
        self.txts = tuple(txts)


def test_read_number_single_digit_run():
    assert OcrReader(engine=_engine("37")).read_number(CROP, "x") == 37
    assert OcrReader(engine=_engine("Queue: 12")).read_number(CROP, "x") == 12


def test_v3_output_shape_supported():
    assert OcrReader(engine=lambda _img: _V3Output("37")).read_number(CROP, "x") == 37
    assert OcrReader(engine=lambda _img: _V3Output()).read_number(CROP, "x") is None
    assert OcrReader(engine=lambda _img: _V3Output("1936.", "1.", "14")).read_date(CROP) is not None


def test_read_number_ambiguous_is_none():
    assert OcrReader(engine=_engine("12 34")).read_number(CROP, "x") is None  # two runs
    assert OcrReader(engine=_engine("no digits")).read_number(CROP, "x") is None
    assert OcrReader(engine=_engine()).read_number(CROP, "x") is None  # no detections


def test_read_date_parses_and_validates():
    assert OcrReader(engine=_engine("1936. 1. 14")).read_date(CROP) == GameDate(1936, 1, 14)
    assert OcrReader(engine=_engine("1936-13-01")).read_date(CROP) is None  # month 13
    assert OcrReader(engine=_engine("gibberish")).read_date(CROP) is None


def test_read_number_free_civ_slots_parses_x_of_xy():
    # the construction header renders "38/38  From trade: 2  Owned: 36" — the
    # wanted value is X of the first X/Y; other fields keep the one-run rule
    eng = _engine("38/38", "From trade: 2", "Owned: 36")
    assert OcrReader(engine=eng).read_number(CROP, "free_civ_slots") == 38
    assert OcrReader(engine=eng).read_number(CROP, "other_field") is None  # ambiguous
    assert OcrReader(engine=_engine("no pair 12")).read_number(CROP, "free_civ_slots") is None


def test_read_date_hoi4_topbar_format():
    # what the HOI4 top bar actually renders (incl. the clock, which is ignored)
    assert OcrReader(engine=_engine("12:00, 1 Jan, 1936")).read_date(CROP) == GameDate(1936, 1, 1)
    assert OcrReader(engine=_engine("04:00,", "11 Nov,", "1942")).read_date(CROP) == GameDate(1942, 11, 11)


def test_unavailable_without_engine(monkeypatch):
    import hoi4_agent.perception.ocr as ocr_mod

    monkeypatch.setattr(ocr_mod, "_load_engine", lambda: None)
    r = OcrReader()
    assert r.available() is False
    assert r.read_number(CROP, "x") is None
    assert r.read_date(CROP) is None


def test_chain_reader_order_and_none_skipping():
    calls = []

    class Tier:
        def __init__(self, name, value):
            self.name, self.value = name, value

        def read_number(self, crop, field):
            calls.append(self.name)
            return self.value

        def read_date(self, crop):
            calls.append(self.name)
            return None

    chain = ChainReader([None, Tier("a", None), Tier("b", 5), Tier("c", 9)])
    assert chain.read_number(CROP, "x") == 5
    assert calls == ["a", "b"]  # c never consulted; None entry skipped


def _rapidocr_importable() -> bool:
    # Import check only — constructing the engine (model load) belongs in the test.
    try:
        import rapidocr  # noqa: F401
        return True
    except ImportError:
        try:
            import rapidocr_onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False


@pytest.mark.skipif(
    not _rapidocr_importable(), reason="rapidocr not installed (optional [ocr] extra)"
)
def test_real_engine_smoke():
    from PIL import ImageDraw

    img = Image.new("RGB", (120, 40), (0, 0, 0))
    ImageDraw.Draw(img).text((10, 10), "42", fill=(255, 255, 255))
    # Real-engine sanity only: must not crash; a None (unreadable) result is acceptable.
    assert OcrReader().read_number(img, "x") in (42, None)

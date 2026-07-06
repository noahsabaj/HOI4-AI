import pytest
from PIL import Image

from hoi4_agent.brain.decide import Brain
from hoi4_agent.brain.llm import ScriptedBackend, encode_image
from hoi4_agent.brain.parse import coerce_enum, coerce_int, extract_json
from hoi4_agent.enums import GermanState, Tech
from hoi4_agent.errors import EnumError, ParseError, SchemaError

CROP = Image.new("RGB", (40, 20), (10, 10, 10))


def _brain(*responses):
    return Brain(ScriptedBackend(list(responses)))


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json("```json\n{\"a\": 2}\n```") == {"a": 2}
    assert extract_json('noise {"a": {"b": 1}} tail') == {"a": {"b": 1}}
    with pytest.raises(ParseError):
        extract_json("not json at all")
    with pytest.raises(SchemaError):
        extract_json("[1, 2, 3]")  # array, not object


def test_coercers():
    assert coerce_int("3") == 3
    assert coerce_int(3.0) == 3
    with pytest.raises(SchemaError):
        coerce_int(True)
    assert coerce_enum("ruhr", GermanState, list(GermanState), "state") is GermanState.RUHR
    with pytest.raises(EnumError):
        coerce_enum("atlantis", GermanState, list(GermanState), "state")


def test_read_number_ok_and_parsefail():
    assert _brain('{"value": 3}').read_number(CROP, "free") == 3
    assert _brain("garbage").read_number(CROP, "free") is None  # parse fail -> uncertain


def test_read_date_ok_and_sentinel():
    assert _brain('{"year":1936,"month":1,"day":1}').read_date(CROP).to_str() == "1936.01.01"
    assert _brain('{"year":-1,"month":0,"day":0}').read_date(CROP) is None


def test_read_date_out_of_range_is_none():
    # A hallucinated month/day must become uncertain, not an orderable wrong date.
    assert _brain('{"year":1936,"month":13,"day":1}').read_date(CROP) is None
    assert _brain('{"year":1936,"month":1,"day":40}').read_date(CROP) is None


def test_last_exchange_recorded_for_trace():
    b = _brain('{"value": 3}')
    assert b.last_raw is None
    b.read_number(CROP, "free_civ_slots")
    assert b.last_raw == '{"value": 3}'
    assert b.last_prompt is not None and "free_civ_slots" in b.last_prompt


def test_which_state_valid_and_invalid():
    assert _brain('{"state":"ruhr"}').which_state(CROP, list(GermanState)) is GermanState.RUHR
    with pytest.raises(EnumError):
        _brain('{"state":"atlantis"}').which_state(CROP, list(GermanState))


def test_which_state_prompt_names_the_actual_building():
    from hoi4_agent.brain.llm import ScriptedBackend
    from hoi4_agent.enums import BuildingType

    be = ScriptedBackend(['{"state":"ruhr"}'])
    Brain(be).which_state(CROP, list(GermanState), BuildingType.MILITARY_FACTORY)
    assert "military factory" in be.calls[0]["user"]

    be2 = ScriptedBackend(['{"state":"ruhr"}'])
    Brain(be2).which_state(CROP, list(GermanState))  # default stays civilian
    assert "civilian factory" in be2.calls[0]["user"]


def test_which_tech_and_yes_no():
    assert _brain('{"tech":"industry_1"}').which_tech(CROP, list(Tech)) is Tech.INDUSTRY_1
    assert _brain('{"answer":"yes"}').yes_no(CROP, "good?") is True
    assert _brain('{"answer":"no"}').yes_no(CROP, "good?") is False


def test_locate_valid_and_out_of_range():
    assert _brain('{"x": 640, "y": 320}').locate(CROP, "the Ruhr state") == (640, 320)
    with pytest.raises(SchemaError):
        _brain('{"x": 1200, "y": 0}').locate(CROP, "the Ruhr state")
    with pytest.raises(SchemaError):
        _brain('{"x": "left", "y": 0}').locate(CROP, "the Ruhr state")


def test_consult_recovery_validates_options():
    opts = ["press_escape", "press_map_mode", "none"]
    assert _brain('{"action":"press_escape"}').consult_recovery(CROP, opts) == "press_escape"
    with pytest.raises(EnumError):
        _brain('{"action":"reboot"}').consult_recovery(CROP, opts)


def test_encode_image_is_base64_str():
    s = encode_image(CROP)
    assert isinstance(s, str) and len(s) > 0


def test_backend_receives_one_image_and_schema():
    be = ScriptedBackend(['{"value": 1}'])
    Brain(be).read_number(CROP, "free")
    assert be.calls[0]["n_images"] == 1
    assert be.calls[0]["schema"]["properties"]["value"]["type"] == "integer"


def _openai_compat_chat(monkeypatch, message: dict) -> str | Exception:
    """Drive OpenAICompatBackend.chat against a canned /v1/chat/completions message."""
    from hoi4_agent.brain import openai_compat as oc

    class FakeResponse:
        status_code = 200
        text = "canned"

        def json(self):
            return {"choices": [{"message": message}]}

    monkeypatch.setattr(oc.requests, "post", lambda *a, **k: FakeResponse())
    be = oc.OpenAICompatBackend("http://localhost:1234", "holo-3.1-9b")
    return be.chat(images=[], system="s", user="u", schema={"type": "object"})


def test_openai_compat_returns_content(monkeypatch):
    msg = {"role": "assistant", "content": '{"value": 1}'}
    assert _openai_compat_chat(monkeypatch, msg) == '{"value": 1}'


def test_openai_compat_falls_back_to_reasoning_content(monkeypatch):
    # Thinking models under a schema grammar: content empty, JSON lands in
    # reasoning_content (never-closed <think> block).
    msg = {"role": "assistant", "content": "", "reasoning_content": '{"value": 1}'}
    assert _openai_compat_chat(monkeypatch, msg) == '{"value": 1}'


def test_openai_compat_raises_on_fully_empty_completion(monkeypatch):
    from hoi4_agent.errors import BrainError

    msg = {"role": "assistant", "content": "", "reasoning_content": ""}
    with pytest.raises(BrainError):
        _openai_compat_chat(monkeypatch, msg)

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


def test_which_state_valid_and_invalid():
    assert _brain('{"state":"ruhr"}').which_state(CROP, list(GermanState)) is GermanState.RUHR
    with pytest.raises(EnumError):
        _brain('{"state":"atlantis"}').which_state(CROP, list(GermanState))


def test_which_tech_and_yes_no():
    assert _brain('{"tech":"industry_1"}').which_tech(CROP, list(Tech)) is Tech.INDUSTRY_1
    assert _brain('{"answer":"yes"}').yes_no(CROP, "good?") is True
    assert _brain('{"answer":"no"}').yes_no(CROP, "good?") is False


def test_encode_image_is_base64_str():
    s = encode_image(CROP)
    assert isinstance(s, str) and len(s) > 0


def test_backend_receives_one_image_and_schema():
    be = ScriptedBackend(['{"value": 1}'])
    Brain(be).read_number(CROP, "free")
    assert be.calls[0]["n_images"] == 1
    assert be.calls[0]["schema"]["properties"]["value"]["type"] == "integer"

from pathlib import Path

import pytest
from PIL import Image

from hoi4_agent.enums import GermanState, Tech
from hoi4_agent.errors import ConfigError
from hoi4_agent.eval.corpus import load_corpus
from hoi4_agent.eval.m0_perception import score_model
from hoi4_agent.eval.replay import replay
from hoi4_agent.schemas import GameDate, TraceRecord
from hoi4_agent.trace.writer import JsonlTraceWriter


class FakeBrain:
    def read_number(self, crop, field):
        return 3

    def read_date(self, crop):
        return GameDate(1936, 1, 1)

    def which_state(self, crop, options):
        return GermanState.RUHR

    def which_tech(self, crop, options):
        return Tech.INDUSTRY_1

    def yes_no(self, crop, question):
        return True

    def locate(self, crop, instruction):
        return (500, 500)


def _img(path: Path):
    Image.new("RGB", (16, 16), (30, 30, 30)).save(path)


def _label(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def test_corpus_and_score(tmp_path):
    _img(tmp_path / "a.png")
    _label(tmp_path / "a.toml", 'task = "read_number"\nfield = "free_civ_slots"\nexpected = 3\n')
    _img(tmp_path / "b.png")
    _label(tmp_path / "b.toml", 'task = "which_state"\nexpected = "ruhr"\n')
    _img(tmp_path / "c.png")
    _label(tmp_path / "c.toml", 'task = "read_number"\nfield = "x"\nexpected = 99\n')  # FakeBrain says 3 -> wrong

    corpus = load_corpus(tmp_path)
    assert len(corpus) == 3
    report = score_model(corpus, FakeBrain())
    assert report["total"] == 3
    assert report["correct"] == 2
    assert report["accuracy"] == 2 / 3
    assert report["per_task"]["which_state"]["accuracy"] == 1.0
    assert len(report["mismatches"]) == 1


def test_load_corpus_missing_dir():
    assert load_corpus("/no/such/dir/xyz") == []


def test_corpus_rejects_unlabeled_png(tmp_path):
    # Regression: a stray PNG with no sidecar once scored as CORRECT and could
    # silently inflate M0 accuracy to 1.0. Ground truth is strict now.
    _img(tmp_path / "stray.png")
    with pytest.raises(ConfigError, match="stray"):
        load_corpus(tmp_path)


def test_corpus_rejects_bad_sidecars(tmp_path):
    _img(tmp_path / "a.png")
    _label(tmp_path / "a.toml", 'task = "not_a_task"\nexpected = 1\n')
    with pytest.raises(ConfigError, match="not_a_task"):
        load_corpus(tmp_path)
    _label(tmp_path / "a.toml", 'task = "read_number"\n')  # no expected
    with pytest.raises(ConfigError, match="expected"):
        load_corpus(tmp_path)


def test_locate_scored_by_distance_tolerance(tmp_path):
    _img(tmp_path / "near.png")
    _label(tmp_path / "near.toml",
           'task = "locate"\ninstruction = "the Ruhr state"\nexpected = [510, 520]\n')
    _img(tmp_path / "far.png")
    _label(tmp_path / "far.toml",
           'task = "locate"\ninstruction = "the Ruhr state"\nexpected = [600, 600]\n')
    _img(tmp_path / "tight.png")
    _label(tmp_path / "tight.toml",
           'task = "locate"\ninstruction = "x"\nexpected = [510, 520]\ntolerance = 5\n')

    report = score_model(load_corpus(tmp_path), FakeBrain())  # FakeBrain says (500, 500)
    assert report["per_task"]["locate"]["correct"] == 1  # near within default 25; others out
    assert report["per_task"]["locate"]["total"] == 3


def test_locate_labels_validated(tmp_path):
    _img(tmp_path / "a.png")
    _label(tmp_path / "a.toml", 'task = "locate"\nexpected = [1, 2]\n')  # no instruction
    with pytest.raises(ConfigError, match="instruction"):
        load_corpus(tmp_path)
    _label(tmp_path / "a.toml", 'task = "locate"\ninstruction = "x"\nexpected = 5\n')
    with pytest.raises(ConfigError, match="must be"):
        load_corpus(tmp_path)


def test_unknown_task_counts_as_wrong_not_correct(tmp_path):
    # Defense for direct callers that bypass load_corpus validation.
    from hoi4_agent.eval.corpus import CorpusItem

    png = tmp_path / "x.png"
    _img(png)
    item = CorpusItem(png, {"task": "bogus", "expected": None})
    report = score_model([item], FakeBrain())
    assert report["correct"] == 0 and report["total"] == 1
    assert report["mismatches"][0]["got"].startswith("ERROR:")


def test_replay(tmp_path):
    png = tmp_path / "frame_0.png"
    _img(png)
    trace = tmp_path / "trace.jsonl"
    with JsonlTraceWriter(trace) as w:
        w.append(TraceRecord(cycle=0, ts=1.0, verdict="ok", plan_step="a", pre_screenshot=str(png),
                             parsed_intent={"tool": "build_in_state", "state": "ruhr"}))
        w.append(TraceRecord(cycle=1, ts=2.0, verdict="ok", plan_step="b"))  # no screenshot -> skipped

    results = replay(trace, probe=lambda img: img.size)
    assert len(results) == 1
    assert results[0]["cycle"] == 0
    assert results[0]["replayed"] == (16, 16)
    assert results[0]["original"]["state"] == "ruhr"

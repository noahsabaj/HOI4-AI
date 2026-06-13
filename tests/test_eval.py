from pathlib import Path

from PIL import Image

from hoi4_agent.enums import GermanState, Tech
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

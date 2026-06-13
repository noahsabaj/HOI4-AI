from hoi4_agent.enums import GermanState, ToolName, Verdict
from hoi4_agent.schemas import Goal, GameDate, ToolResult, TraceRecord, WorldState
from hoi4_agent.trace.record import build_record
from hoi4_agent.trace.writer import JsonlTraceWriter


def test_writer_append_and_read(tmp_path):
    path = tmp_path / "trace.jsonl"
    r1 = TraceRecord(cycle=0, ts=1.0, verdict="ok", plan_step="a")
    r2 = TraceRecord(cycle=1, ts=2.0, verdict="failed", plan_step="b", error="BuildInStateError")
    with JsonlTraceWriter(path) as w:
        w.append(r1)
        w.append(r2)
    records = JsonlTraceWriter.read(path)
    assert records == [r1, r2]


def test_save_frame_disabled_returns_none(tmp_path):
    from PIL import Image

    w = JsonlTraceWriter(tmp_path / "t.jsonl")  # no screenshot_dir
    assert w.save_frame(Image.new("RGB", (4, 4)), "x.jpg") is None
    w.close()


def test_save_frame_writes(tmp_path):
    from PIL import Image

    w = JsonlTraceWriter(tmp_path / "t.jsonl", screenshot_dir=tmp_path / "frames")
    p = w.save_frame(Image.new("RGB", (4, 4)), "x.jpg")
    w.close()
    assert p is not None
    from pathlib import Path

    assert Path(p).is_file()


def test_build_record():
    goal = Goal(id="build_ruhr", tool=ToolName.BUILD_IN_STATE, state=GermanState.RUHR)
    pre = WorldState(date=GameDate(1936, 1, 1))
    result = ToolResult(ToolName.BUILD_IN_STATE, Verdict.OK, pre=pre, post=pre,
                        assertion="queue 0->1", retries=1, latency_s=0.5)
    rec = build_record(cycle=2, ts=9.0, result=result, goal=goal, mode="robust")
    assert rec.verdict == "ok"
    assert rec.plan_step == "build_ruhr"
    assert rec.date == "1936.01.01"
    assert rec.parsed_intent["tool"] == "build_in_state"
    assert rec.parsed_intent["state"] == "ruhr"
    assert rec.verification_question == "queue 0->1"
    assert rec.error is None
    # round-trips through JSON
    assert TraceRecord.from_dict(rec.to_dict()) == rec

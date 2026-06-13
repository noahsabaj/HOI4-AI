"""Build a TraceRecord from a cycle's data (TraceRecord itself lives in schemas)."""

from __future__ import annotations

from ..schemas import Goal, ToolResult, TraceRecord

__all__ = ["TraceRecord", "build_record"]


def build_record(
    *,
    cycle: int,
    ts: float,
    result: ToolResult,
    goal: Goal | None = None,
    mode: str = "robust",
    vlm_used: bool = False,
    pre_path: str | None = None,
    post_path: str | None = None,
    prompt: str | None = None,
    raw: str | None = None,
) -> TraceRecord:
    pre = result.pre
    intent_dict = goal.to_intent().to_dict() if goal else None
    return TraceRecord(
        cycle=cycle,
        ts=ts,
        verdict=result.verdict.value,
        date=pre.date.to_str() if (pre and pre.date) else None,
        plan_step=goal.id if goal else None,
        pre_screenshot=pre_path,
        prompt=prompt,
        raw_model_output=raw,
        parsed_intent=intent_dict,
        actions=(intent_dict,) if intent_dict else (),
        post_screenshot=post_path,
        verification_question=result.assertion,
        retries=result.retries,
        latency_s=result.latency_s,
        mode=mode,
        vlm_used=vlm_used,
        error=type(result.error).__name__ if result.error else None,
    )

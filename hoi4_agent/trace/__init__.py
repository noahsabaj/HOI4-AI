"""Observability: one replayable JSONL record per cycle + an offline reader.

The trace is the research product: pre/post screenshots, prompt, raw model output,
parsed intent, actions, verification verdict, retries, latency, mode — enough to
re-run the same frames through a new prompt/model offline (see ``eval/replay``).
"""

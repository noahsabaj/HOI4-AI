"""Typed tool API: precondition -> deterministic steps -> post-condition assertion.

Every tool returns a ``ToolResult`` whose verdict is OK / FAILED / UNCERTAIN —
never ``None``. Failures carry a typed error and the observed state.
"""

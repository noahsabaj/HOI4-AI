"""Externalized strategy: an ordered, idempotent goal queue + persisted progress.

The per-cycle question is narrow and reliable: "what is the next pending goal whose
precondition is visible right now?" — instead of re-deriving strategy each frame.
"""

"""Evaluation: M0 perception scoring (run FIRST, gates the live loop) + offline replay.

M0 measures whether the chosen model can READ HOI4 crops, decoupled from the game.
Replay re-runs saved trace frames through a new prompt/model without the game.
"""

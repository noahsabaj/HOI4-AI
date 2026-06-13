"""Deterministic perception: numpy NCC template matching + tiered screen reading.

T0 (state checks) and the cropping/NCC math are pure and deterministic. T1
(numbers/date) and T2 (decisions) delegate to an injected ``reader`` (the brain),
so this package has no dependency on the LLM layer.
"""

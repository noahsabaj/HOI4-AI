"""Physical event vocabulary; eight ordered 25 ms slots per policy decision.

Pointer positions are client-relative fractions quantized onto a square GRID x GRID
lattice, so the pixel pitch is anisotropic on a non-square screen: at 3840x2160 one step
is 3839/1023 = 3.75 px horizontally but 2159/1023 = 2.11 px vertically, giving up to
+/-1.9 px of horizontal round-trip error. Controls narrower than about four pixels cannot
be addressed exactly, and recorded human motion is re-quantized onto this lattice before
it becomes a behavioral-cloning label. Raising GRID widens the two position heads in
models.ActionHead and invalidates existing prepared datasets and checkpoints.
"""

from __future__ import annotations

import numpy as np

KEYS = [0x10, 0x11, *range(0x25, 0x29), *range(0x41, 0x5B)]
GRID = 1024
SLOTS = 8
PERIOD = 0.2
VOCAB = [None, {"kind": "move"}]
VOCAB += [{"kind": "button", "button": b, "down": d} for b in range(3) for d in [True, False]]
VOCAB += [{"kind": "wheel", "delta": d} for d in [-120, 120]]
VOCAB += [{"kind": "key", "vk": k, "down": d} for k in KEYS for d in [True, False]]


def decode(token):
    kind, x, y = map(int, token)
    if not 0 <= kind < len(VOCAB) or not 0 <= x < GRID or not 0 <= y < GRID:
        raise ValueError("Action outside vocabulary")
    event = VOCAB[kind]
    if event is None:
        return []
    if kind == 1:
        return [{"kind": "move", "x": x / (GRID - 1), "y": y / (GRID - 1)}]
    return [dict(event)]


def encode_event(event):
    if event["kind"] == "move":
        return [
            1,
            round(np.clip(event["x"], 0, 1) * (GRID - 1)),
            round(np.clip(event["y"], 0, 1) * (GRID - 1)),
        ]
    try:
        return [VOCAB.index(event), 0, 0]
    except ValueError:
        raise ValueError(f"Demonstration contains unsupported match input: {event}") from None


def encode_interval(events, start_ns, period=PERIOD):
    result = np.zeros((SLOTS, 3), dtype=np.int64)
    last_slot = -1
    last_kind = None
    last_timestamp = start_ns - 1
    for item in events:
        if item["t_ns"] < last_timestamp:
            raise ValueError("Input timestamps must be ordered")
        last_timestamp = item["t_ns"]
        t = (item["t_ns"] - start_ns) / 1e9
        if t < 0 or t >= period:
            continue
        event = item["event"]
        slot = min(SLOTS - 1, int(t / period * SLOTS))
        # Keep only the latest motion within a slot, never merge press/release edges.
        if slot <= last_slot and event["kind"] == "move" and last_kind == "move":
            result[last_slot] = encode_event(event)
            continue
        slot = max(slot, last_slot + 1)
        if slot >= SLOTS:
            raise ValueError("Input burst exceeds eight event slots; sample must be excluded")
        result[slot] = encode_event(event)
        last_slot, last_kind = slot, event["kind"]
    return result

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

# Append-only. Existing kind indices are checkpoints and prepared labels. Speed
# keys stay out: a match that can press +/- would make the recorded game speed a lie.
# Space and escape stay out too: both pause the game, which invalidates the match.
KEYS = [
    0x10,
    0x11,
    *range(0x25, 0x29),
    *range(0x41, 0x5B),
    0x09,
    0x0D,
    *range(0x30, 0x3A),
]
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
    if event["kind"] == "wheel":
        delta = event.get("delta", 0)
        if not isinstance(delta, int) or isinstance(delta, bool) or delta == 0:
            raise ValueError(f"Demonstration contains unsupported match input: {event}")
        # High-resolution wheels report multiples of a notch. One slot can carry one
        # notch; the sign is the part the policy can replay.
        event = {"kind": "wheel", "delta": 120 if delta > 0 else -120}
    try:
        return [VOCAB.index(event), 0, 0]
    except ValueError:
        raise ValueError(f"Demonstration contains unsupported match input: {event}") from None


# Presses of a key or button, by kind index, and for each release the press it ends.
PRESSES = frozenset(
    i for i, e in enumerate(VOCAB) if e and e["kind"] in ("key", "button") and e["down"]
)
RELEASES = {
    i: VOCAB.index({**e, "down": True})
    for i, e in enumerate(VOCAB)
    if e and e["kind"] in ("key", "button") and not e["down"]
}


def still_held(held, action):
    """The presses (kind indices, oldest first) still down after `action`, from `held`,
    those down before it."""
    down = list(held)
    for kind in (int(k) for k in np.asarray(action)[:, 0]):
        if kind in PRESSES and kind not in down:
            down.append(kind)
        elif kind in RELEASES and RELEASES[kind] in down:
            down.remove(RELEASES[kind])
    return tuple(down)


def with_held(action, held):
    """`action` as the next decision reads it (the previous action), with a press of each
    input still down (`held`, after it) in its last empty slots, unless the action shows
    that press itself.

    The policy sees only its previous action, so after one decision a key it holds is
    gone from what it sees: bc5 pressed Right and held it for three minutes (2026-09-26).
    The scripted player it copies releases a key the decision after pressing it, which
    is what "a press in the previous action" teaches; shown every decision while the key
    is down, the press keeps asking for its release.
    """
    out = np.array(action, dtype=np.int64, copy=True)
    shown = {int(k) for k in out[:, 0]}
    empty = [i for i in range(len(out)) if out[i, 0] == 0]
    for kind in held:
        if kind in shown or not empty:
            continue
        out[empty.pop()] = (kind, 0, 0)
    return out


def previous_actions(actions, held=False):
    """Each decision's previous action: the decision before's (nothing for the first), and
    with `held`, what is still held filled in (with_held)."""
    actions = np.asarray(actions)
    previous = np.zeros_like(actions)
    if len(actions) > 1:
        previous[1:] = actions[:-1]
    if held:
        down = ()
        for t in range(1, len(actions)):
            down = still_held(down, actions[t - 1])
            previous[t] = with_held(actions[t - 1], down)
    return previous


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

"""What a recording, the worker's client and training agree on, without torch.

The view sizes, the game speeds, the pointer and the speed record. They lived in dataset.py,
which imports torch: 4.2 s and about 1.5 GB of private memory in every recorder and control
process that only needed these few names. dataset.py imports them from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import torch

# Three views of each decision's frame, all area-averaged the same way (see `views`).
# Sizes are (height, width); the screen is 16:9 and so are the resized views, since
# squashing it into squares cost the encoders what they read (2026-09-23, STATUS.md):
#
# - The global view, the whole screen at VIEW_SIZE, is what the video encoder (LeVJEPA)
#   reads, frames in sequence. At 448x256 with four frames it read camera motion best.
# - The four quadrants at DETAIL_SIZE, tiled back into a 1152x640 screen for the Qwen3.5
#   tower and read by a small CNN. HOI4 is read from text, 10 px at 1080p: the tower named
#   58% of such characters at 1152x640 against 56.5% from the old 896 square, for less time.
# - The fovea, FOVEA_SIZE native pixels square, centred on the pointer, never resized:
#   whatever the pointer is over is seen at full resolution.
VIEW_SIZE = (256, 448)
DETAIL_SIZE = (320, 576)
FOVEA_SIZE = 224
QUADRANTS = 4
# Wall seconds of one in-game hour: the game's GAME_SPEED_SECONDS, speeds 1 through 5.
# Speed 1 was measured at 2.0 (48 s per in-game day) and speed 4 at 0.1. Speed 5 is 0
# because the simulation does not sleep, so that clip has no fixed game length. Eight
# frames at the decision interval are 1.6 s of wall time: 3.2 in-game hours at speed 2
# and 16 at speed 4. The policy is told the speed (Policy's speed input), so one model can
# learn from recordings made at different speeds.
GAME_SPEED_SECONDS = (2.0, 0.5, 0.2, 0.1, 0.0)


class Views(NamedTuple):
    """One frame as the policy sees it. All uint8, channels last."""

    global_view: torch.Tensor | None  # (*VIEW_SIZE, 3); None when not asked for
    quadrants: torch.Tensor  # (4, *DETAIL_SIZE, 3)
    fovea: torch.Tensor  # (FOVEA_SIZE, FOVEA_SIZE, 3)


def recorded_speed(value):
    """Manifest fields for the speed the operator set.

    There is no default. A recording that omits the speed cannot be assigned one
    afterwards, and the documented 2 is not what the long runs used.
    """
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise ValueError("game speed must be an integer from 1 to 5")
    seconds = GAME_SPEED_SECONDS[value - 1]
    return {"game_speed": value, "seconds_per_hour": None if seconds == 0 else seconds}


def parse_cursor(value):
    """Client-pixel pointer the fovea is centered on."""
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    ):
        return int(value[0]), int(value[1])
    raise ValueError("cursor must be two client-pixel integers")


def hw(size):
    """A view size as (height, width): an int is a square."""
    return (size, size) if isinstance(size, int) else tuple(int(v) for v in size)

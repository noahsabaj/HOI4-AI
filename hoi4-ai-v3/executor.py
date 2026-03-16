"""Input execution for HOI4-AI v3. Translates model actions to xdotool commands."""

import subprocess
import time


def scale_coordinates(
    model_x: int, model_y: int,
    model_width: int, model_height: int,
    window_info: dict
) -> tuple[int, int]:
    """Scale coordinates from model space (1280x720) to window-relative coordinates.

    xdotool mousemove --window takes coordinates relative to the window origin,
    so we only scale by the resolution ratio — no window offset needed.
    """
    win_x = int((model_x / model_width) * window_info["width"])
    win_y = int((model_y / model_height) * window_info["height"])
    return win_x, win_y


def execute_click(
    model_x: int, model_y: int,
    window_info: dict,
    model_width: int = 1280, model_height: int = 720
) -> None:
    """Move mouse to scaled window-relative coordinates and click."""
    win_x, win_y = scale_coordinates(
        model_x, model_y, model_width, model_height, window_info
    )
    wid = window_info["window_id"]
    # Activate window, move mouse relative to it, and click
    subprocess.run(["xdotool", "windowactivate", "--sync", wid], timeout=5)
    subprocess.run(
        ["xdotool", "mousemove", "--window", wid, str(win_x), str(win_y),
         "click", "--window", wid, "1"],
        timeout=5
    )


def execute_key(key: str, window_info: dict) -> None:
    """Send a keystroke to the game window."""
    subprocess.run(
        ["xdotool", "key", "--window", window_info["window_id"], key],
        timeout=5
    )


def escape_cleanup(window_info: dict, presses: int = 2) -> None:
    """Press Escape N times to close any open menus."""
    for _ in range(presses):
        execute_key("Escape", window_info)
        time.sleep(0.3)


def full_reset(window_info: dict) -> None:
    """Full reset: spam Escape, return to default map mode."""
    escape_cleanup(window_info, presses=3)
    execute_key("F1", window_info)
    time.sleep(0.5)


def dispatch_action(
    action: dict, window_info: dict,
    model_width: int = 1280, model_height: int = 720
) -> bool:
    """Execute a single action. Returns True if action is 'done'."""
    action_type = action.get("action", "")

    if action_type == "click":
        execute_click(action["x"], action["y"], window_info, model_width, model_height)
        return False

    elif action_type == "key":
        execute_key(action["key"], window_info)
        return False

    elif action_type == "done":
        return True

    return False

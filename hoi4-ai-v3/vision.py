"""Screenshot capture and window detection for HOI4-AI v3."""

import base64
import io
import subprocess

import mss
from PIL import Image, ImageDraw, ImageFont


def get_frame_extents(window_id: str) -> dict:
    """Get window frame/decoration sizes via xprop.

    Returns dict with left, right, top, bottom pixel offsets.
    The 'top' value is the title bar height.
    """
    try:
        result = subprocess.run(
            ["xprop", "-id", window_id, "_NET_FRAME_EXTENTS"],
            capture_output=True, text=True, timeout=5
        )
        # Output: _NET_FRAME_EXTENTS(CARDINAL) = left, right, top, bottom
        if "=" in result.stdout and "not found" not in result.stdout:
            values = result.stdout.split("=")[1].strip().split(",")
            extents = {
                "left": int(values[0].strip()),
                "right": int(values[1].strip()),
                "top": int(values[2].strip()),
                "bottom": int(values[3].strip()),
            }
            return extents
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
        pass
    return {"left": 0, "right": 0, "top": 0, "bottom": 0}


def find_game_window(title: str) -> dict | None:
    """Find the HOI4 game window by title. Returns window info dict or None."""
    try:
        result = subprocess.run(
            ["xdotool", "search", "--name", title],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None

        window_id = result.stdout.strip().split("\n")[0]

        # Use --shell for machine-parseable output
        geo = subprocess.run(
            ["xdotool", "getwindowgeometry", "--shell", window_id],
            capture_output=True, text=True, timeout=5
        )
        if geo.returncode != 0:
            return None

        # Parse KEY=VALUE lines
        vals = {}
        for line in geo.stdout.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                vals[k] = v

        if not all(k in vals for k in ("X", "Y", "WIDTH", "HEIGHT")):
            return None

        # Get frame extents to adjust for title bar / window decorations
        frame = get_frame_extents(window_id)

        # getwindowgeometry returns frame position; adjust to client area
        client_x = int(vals["X"]) + frame["left"]
        client_y = int(vals["Y"]) + frame["top"]
        client_w = int(vals["WIDTH"]) - frame["left"] - frame["right"]
        client_h = int(vals["HEIGHT"]) - frame["top"] - frame["bottom"]

        # If frame subtraction makes dimensions negative/zero, ignore the adjustment
        if client_w <= 0 or client_h <= 0:
            client_x = int(vals["X"])
            client_y = int(vals["Y"])
            client_w = int(vals["WIDTH"])
            client_h = int(vals["HEIGHT"])

        return {
            "window_id": window_id,
            "x": client_x,
            "y": client_y,
            "width": client_w,
            "height": client_h,
            "frame": frame,
        }
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def capture_screenshot(
    window_info: dict, target_width: int = 1280, target_height: int = 720,
    fmt: str = "jpeg"
) -> dict:
    """Capture the game window and return resized base64-encoded image."""
    monitor = {
        "left": window_info["x"],
        "top": window_info["y"],
        "width": window_info["width"],
        "height": window_info["height"],
    }

    with mss.mss() as sct:
        raw = sct.grab(monitor)
        # Use BGRX raw format (more efficient than .rgb which copies + converts)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

    img = img.resize((target_width, target_height), Image.LANCZOS)

    # Apply coordinate grid overlay
    img = draw_grid(img, step=100)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG" if fmt == "jpeg" else "PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "base64": encoded,
        "width": target_width,
        "height": target_height,
        "image": img,
    }


def draw_grid(img: Image.Image, step: int = 100) -> Image.Image:
    """Draw a coordinate grid overlay on the image.

    Draws semi-transparent lines every `step` pixels with coordinate labels.
    This helps the VLM output more accurate click coordinates.
    """
    # Work on a copy with alpha for semi-transparent lines
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img.size
    line_color = (255, 255, 0, 60)  # yellow, very transparent
    label_color = (255, 255, 0, 180)  # yellow, more visible for text

    # Try to get a small font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Vertical lines + top labels
    for x in range(step, w, step):
        draw.line([(x, 0), (x, h)], fill=line_color, width=1)
        draw.text((x + 2, 1), str(x), fill=label_color, font=font)

    # Horizontal lines + left labels
    for y in range(step, h, step):
        draw.line([(0, y), (w, y)], fill=line_color, width=1)
        draw.text((2, y + 1), str(y), fill=label_color, font=font)

    # Composite overlay onto original
    img_rgba = img.convert("RGBA")
    img_rgba = Image.alpha_composite(img_rgba, overlay)
    return img_rgba.convert("RGB")

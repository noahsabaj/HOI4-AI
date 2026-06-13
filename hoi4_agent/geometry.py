"""The single authoritative coordinate model.

All capture regions and click targets derive from one measured client rect, so
the v3 title-bar/frame drift and independent-axis aspect distortion cannot recur.
The model only ever emits normalized 0..1000 coordinates *on a crop*; this module
maps crop-normalized -> client px -> absolute screen px (and back).
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ResolutionError

NORM_SCALE = 1000  # model coordinate space is 0..1000 on the supplied crop


@dataclass(frozen=True, slots=True)
class CropRect:
    """A sub-region offset+size in CLIENT pixels (origin at the client top-left)."""

    client_x0: int
    client_y0: int
    crop_w: int
    crop_h: int


@dataclass(frozen=True, slots=True)
class WindowGeometry:
    """Game window geometry. ``screen_*`` is the client top-left in screen px."""

    hwnd: int
    screen_left: int
    screen_top: int
    client_w: int
    client_h: int

    # --- crops ---
    def full_crop(self) -> CropRect:
        return CropRect(0, 0, self.client_w, self.client_h)

    def panel_crop(self, frac: tuple[float, float, float, float]) -> CropRect:
        """Crop defined as fractions (fx0,fy0,fx1,fy1) of the client rect."""
        fx0, fy0, fx1, fy1 = frac
        x0 = int(round(fx0 * self.client_w))
        y0 = int(round(fy0 * self.client_h))
        x1 = int(round(fx1 * self.client_w))
        y1 = int(round(fy1 * self.client_h))
        return CropRect(x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    # --- capture region (absolute screen px) for mss ---
    def monitor(self, crop: CropRect | None = None) -> dict:
        c = crop or self.full_crop()
        return {
            "left": self.screen_left + c.client_x0,
            "top": self.screen_top + c.client_y0,
            "width": c.crop_w,
            "height": c.crop_h,
        }

    # --- model crop-normalized -> absolute screen px (for clicking) ---
    def norm_to_screen(self, crop: CropRect, nx: float, ny: float) -> tuple[int, int]:
        """Map (nx,ny) in 0..NORM_SCALE on ``crop`` to an absolute screen pixel.

        x and y are mapped against the *same* crop independently of each other,
        so there is no cross-axis distortion regardless of the crop's aspect.
        """
        client_x = crop.client_x0 + (nx / NORM_SCALE) * crop.crop_w
        client_y = crop.client_y0 + (ny / NORM_SCALE) * crop.crop_h
        sx = self.screen_left + int(round(client_x))
        sy = self.screen_top + int(round(client_y))
        return sx, sy

    # --- inverse: absolute screen px -> crop-normalized (for logging) ---
    def screen_to_norm(self, crop: CropRect, sx: int, sy: int) -> tuple[int, int]:
        client_x = sx - self.screen_left - crop.client_x0
        client_y = sy - self.screen_top - crop.client_y0
        nx = int(round(client_x / crop.crop_w * NORM_SCALE))
        ny = int(round(client_y / crop.crop_h * NORM_SCALE))
        return nx, ny

    def assert_resolution(self, width: int, height: int) -> None:
        if (self.client_w, self.client_h) != (width, height):
            raise ResolutionError((width, height), (self.client_w, self.client_h))

"""Byte-exact BMP/DDS writers and a strict BMP reader for the generated HOI4 map.

Pillow is deliberately NOT used for the map bitmaps: the game is picky about headers (``provinces.bmp``
must be 24-bit BITMAPINFOHEADER, the indexed maps must keep the vanilla palette ORDER) and the output
has to be byte-identical between builds. Layout written here, mirroring the vanilla files: 14-byte file
header, 40-byte BITMAPINFOHEADER, 256 BGRA palette entries for 8-bit files, bottom-up rows, no
compression, 2834 px/m (3780 for ``provinces.bmp``, as vanilla). Widths are multiples of 4 here, so
rows need no padding; the writer still pads for odd sizes (``trees.bmp``).

Palettes: with an install at hand the builder copies the vanilla palette bytes; ``FALLBACK_PALETTES``
holds the entries the generator actually uses (checked against vanilla 1.19.3) so tests run offline.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import ArenaError

# index -> (r, g, b); indices the generator may emit. Everything else is padded with black.
FALLBACK_PALETTES: dict[str, dict[int, tuple[int, int, int]]] = {
    "terrain": {0: (86, 124, 27), 1: (0, 86, 6), 6: (134, 84, 30), 9: (75, 147, 174), 14: (0, 0, 0),
                15: (8, 31, 130), 17: (132, 255, 0)},
    "rivers": {0: (0, 255, 0), 1: (255, 0, 0), 2: (255, 252, 0), 3: (0, 225, 255), 4: (0, 200, 255),
               5: (0, 150, 255), 6: (0, 100, 255), 7: (0, 0, 255), 8: (0, 0, 225), 9: (0, 0, 200),
               10: (0, 0, 150), 11: (0, 0, 100), 254: (122, 122, 122), 255: (255, 255, 255)},
    "trees": {0: (0, 0, 0)},
    "cities": {15: (8, 31, 130)},
}


@dataclass(frozen=True)
class Bmp:
    width: int
    height: int
    bits: int
    header_size: int
    compression: int
    palette: tuple[tuple[int, int, int], ...]  # RGB, () for 24-bit
    pixels: Any  # (h, w) uint8 indices or (h, w, 3) uint8 RGB, top row first


def grey_palette() -> list[tuple[int, int, int]]:
    return [(i, i, i) for i in range(256)]


def fallback_palette(name: str) -> list[tuple[int, int, int]]:
    known = FALLBACK_PALETTES[name]
    return [known.get(i, (0, 0, 0)) for i in range(256)]


def vanilla_palette(path: Path) -> list[tuple[int, int, int]]:
    """The palette of a vanilla 8-bit BMP, padded to 256 entries, in file order."""
    with path.open("rb") as stream:
        head = stream.read(14 + 124 + 1024)
    offset, header = struct.unpack("<I", head[10:14])[0], struct.unpack("<I", head[14:18])[0]
    if head[:2] != b"BM" or struct.unpack("<H", head[28:30])[0] != 8:
        raise ArenaError(f"{path} is not an 8-bit BMP")
    raw = head[14 + header:offset]
    entries = [(raw[i + 2], raw[i + 1], raw[i]) for i in range(0, len(raw) - 3, 4)][:256]
    return entries + [(0, 0, 0)] * (256 - len(entries))


def _header(width: int, height: int, bits: int, data: int, ppm: int) -> bytes:
    offset = 54 + (1024 if bits == 8 else 0)
    return (b"BM" + struct.pack("<IHHI", offset + data, 0, 0, offset)
            + struct.pack("<IiiHHIIiiII", 40, width, height, 1, bits, 0, data, ppm, ppm, 0, 0))


def write_bmp8(path: Path, indices: Any, palette: list[tuple[int, int, int]]) -> None:
    array = np.ascontiguousarray(indices, dtype=np.uint8)
    if array.ndim != 2 or len(palette) != 256:
        raise ArenaError("8-bit BMP needs a 2-D index array and 256 palette entries")
    height, width = array.shape
    stride = (width + 3) // 4 * 4
    rows = np.zeros((height, stride), dtype=np.uint8)
    rows[:, :width] = array[::-1]
    table = b"".join(bytes((b, g, r, 0)) for r, g, b in palette)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_header(width, height, 8, rows.size, 2834) + table + rows.tobytes())


def write_bmp24(path: Path, rgb: Any, ppm: int = 2834) -> None:
    array = np.ascontiguousarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ArenaError("24-bit BMP needs an (h, w, 3) RGB array")
    height, width, _ = array.shape
    stride = (width * 3 + 3) // 4 * 4
    rows = np.zeros((height, stride), dtype=np.uint8)
    rows[:, :width * 3] = array[::-1, :, ::-1].reshape(height, width * 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_header(width, height, 24, rows.size, ppm) + rows.tobytes())


def read_bmp(path: Path) -> Bmp:
    """Strict reader used by the validators: it reports what is in the file, it does not repair."""
    data = path.read_bytes()
    if len(data) < 54 or data[:2] != b"BM":
        raise ArenaError(f"{path.name}: not a BMP file")
    offset = struct.unpack("<I", data[10:14])[0]
    header, width, height, _, bits, compression = struct.unpack("<IiiHHI", data[14:34])
    if bits not in (8, 24, 32) or width <= 0 or height == 0:
        raise ArenaError(f"{path.name}: unsupported BMP ({bits} bit, {width}x{height})")
    palette: tuple[tuple[int, int, int], ...] = ()
    if bits == 8:
        raw = data[14 + header:offset]
        palette = tuple((raw[i + 2], raw[i + 1], raw[i]) for i in range(0, len(raw) - 3, 4))
    channels = bits // 8
    rows_n = abs(height)
    stride = (width * channels + 3) // 4 * 4
    body = np.frombuffer(data, dtype=np.uint8, count=stride * rows_n, offset=offset).reshape(rows_n, stride)
    body = body[:, :width * channels]
    if height > 0:
        body = body[::-1]
    pixels = body if bits == 8 else body.reshape(rows_n, width, channels)[:, :, 2::-1]
    return Bmp(width, rows_n, bits, header, compression, palette, np.ascontiguousarray(pixels))


# ---------------------------------------------------------------------------------------------
# DDS. Header fields mirror the vanilla files of the same name (see custommap.py for which).

def _dds_header(width: int, height: int, flags: int, pitch_or_size: int, pixel_format: bytes, caps: int = 0x1000,
                depth: int = 0, mips: int = 0) -> bytes:
    return (b"DDS " + struct.pack("<7I", 124, flags, height, width, pitch_or_size, depth, mips) + b"\0" * 44
            + pixel_format + struct.pack("<5I", caps, 0, 0, 0, 0))


def write_dds_argb(path: Path, rgba: Any) -> None:
    """Uncompressed A8R8G8B8, one mip level: the format of vanilla ``colormap_rgb_cityemissivemask_a.dds``."""
    array = np.ascontiguousarray(rgba, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ArenaError("ARGB DDS needs an (h, w, 4) RGBA array")
    height, width, _ = array.shape
    pixel_format = struct.pack("<2I4s5I", 32, 0x41, b"\0\0\0\0", 32, 0xFF0000, 0xFF00, 0xFF, 0xFF000000)
    flags = 0x100F  # caps, height, width, pitch, pixelformat: the vanilla file's flags (depth 1, mips 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_dds_header(width, height, flags, width * 4, pixel_format, depth=1, mips=1)
                     + array[:, :, [2, 1, 0, 3]].tobytes())


def write_dds_dxt5_flat(path: Path, width: int, height: int, rgba: tuple[int, int, int, int]) -> None:
    """A single-colour DXT5 texture, no mip chain: the format of vanilla ``colormap_water_*.dds``.
    A uniform block needs no encoder: both alpha endpoints = a, both colour endpoints = the 565 colour."""
    if width % 4 or height % 4:
        raise ArenaError("DXT5 needs dimensions that are multiples of 4")
    r, g, b, a = rgba
    c565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    block = bytes((a, a)) + b"\0" * 6 + struct.pack("<HHI", c565, c565, 0)
    blocks = (width // 4) * (height // 4)
    pixel_format = struct.pack("<2I4s5I", 32, 0x4, b"DXT5", 0, 0, 0, 0, 0)
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000  # caps, height, width, pixelformat, linearsize
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_dds_header(width, height, flags, blocks * 16, pixel_format) + block * blocks)


def read_dds_header(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if data[:4] != b"DDS " or len(data) < 128:
        raise ArenaError(f"{path.name}: not a DDS file")
    height, width = struct.unpack("<2I", data[12:20])
    pf_flags, fourcc, bits = struct.unpack("<I4sI", data[80:92])
    return {"width": width, "height": height, "fourcc": fourcc.rstrip(b"\0").decode("ascii"), "bits": bits,
            "pf_flags": pf_flags, "payload": len(data) - 128}

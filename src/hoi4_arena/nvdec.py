"""Recordings decoded by the GPU's video decoder, and the reader's colour conversion done
on the device, with the same pixels as the CPU reader.

The CPU reader (dataset._Stream) runs `ffmpeg -i screen.mkv -f rawvideo -pix_fmt rgb24`.
Measured on this PC (2026-09-26, 1080p recordings, CPU time of ffmpeg and the reader
together), a frame costs ~48 ms of CPU: ~10 ms decoding the H.264, ~20 ms converting its
4:4:4 YUV to RGB in swscale, and the rest moving 6 MB through the pipe. Two things take
that away:

- NVIDIA's decoder (NVDEC) reads HEVC 4:4:4 (Range Extensions), which it cannot do for
  H.264 4:4:4. PyNvVideoCodec decodes it into host memory at ~4.4 ms of CPU a frame,
  with the planes bit-for-bit what ffmpeg's decoder gives. The recorder writes HEVC 4:4:4
  since 2026-09-26, and scripts/reencode_hevc.py turns an older recording into it
  losslessly.
- The conversion is a table: swscale's yuv444p -> rgb24 maps each (Y, U, V) to one RGB
  (full chroma, no dither at 8 bits), so `rgb_table` asks this PC's own ffmpeg for all
  2^24 of them once (~0.8 s, 48 MB) and `to_rgb` looks every pixel up on the device, in
  ~1.2 ms a frame on the GPU. The frames are the ones ffmpeg's rgb24 output has, byte for
  byte (tests pin it); an H.264 file then skips only the conversion, decoded by ffmpeg
  as yuv444p.

Each decoder holds a CUDA context and its surfaces, ~0.2-0.3 GB of the card's memory
for the first in a process and ~0.1-0.2 GB for each more.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

# The recorders' codecs (the manifest's `codec`) whose video is 4:4:4 YUV, 8 bits: their
# frames can travel as the decoder's own planes and be converted by rgb_table.
YUV444_CODECS = frozenset({"x264", "nvenc", "nvenc-hevc", "x264-source"})


@functools.cache
def _module():
    """PyNvVideoCodec, or None where it cannot load (no wheel, no NVIDIA driver)."""
    try:
        import PyNvVideoCodec
    except Exception:
        return None
    return PyNvVideoCodec


def available():
    """Whether this process can decode on the GPU's video decoder."""
    return _module() is not None and torch.cuda.is_available()


class NvdecFrames:
    """A recording's frames as 4:4:4 YUV planes (3, H, W) uint8, from NVDEC into host
    memory. Only HEVC in 8-bit 4:4:4 at the given size: anything else is a ValueError,
    raised here for the first frame (open_frames then falls back to ffmpeg). The frames
    before `start` are decoded and dropped."""

    def __init__(self, path, width, height, start=0):
        nvc = _module()
        if nvc is None:
            raise ValueError("PyNvVideoCodec is not available")
        self.demuxer = nvc.CreateDemuxer(filename=str(path))
        if self.demuxer.GetNvCodecId() != nvc.cudaVideoCodec.HEVC:
            raise ValueError("not HEVC")
        self.decoder = nvc.CreateDecoder(
            gpuid=0,
            codec=self.demuxer.GetNvCodecId(),
            cudacontext=0,
            cudastream=0,
            usedevicememory=False,
        )
        self.format = nvc.Pixel_Format.YUV444
        self.shape = (3, height, width)
        self.packets = iter(self.demuxer)
        self.ready = []
        self.ended = False
        self.skip = start
        # The demuxer's chroma format is not to be trusted (it says 4:2:0 of 4:4:4), so the
        # first frame is decoded here and checked: 8-bit 4:4:4 planes at the manifest's size.
        self._fill()
        if self.ready and self.ready[0] is None:
            raise ValueError("not 8-bit 4:4:4 at the manifest's size")

    def _fill(self):
        while not self.ready and not self.ended:
            packet = next(self.packets, None)
            if packet is None:
                self.ended = True
                break
            # The decoder reuses its buffers: copy each frame out as it comes.
            for frame in self.decoder.Decode(packet):
                if self.skip:
                    self.skip -= 1
                    continue
                good = frame.format == self.format and tuple(frame.shape) == (
                    3 * self.shape[1],
                    self.shape[2],
                )
                self.ready.append(
                    np.from_dlpack(frame).reshape(self.shape).copy() if good else None
                )

    def read(self):
        """The next frame's planes, or None after the last."""
        self._fill()
        if not self.ready:
            return None
        frame = self.ready.pop(0)
        if frame is None:
            raise ValueError("a frame that is not 8-bit 4:4:4 at the recording's size")
        return frame

    def close(self):
        self.decoder = self.demuxer = self.packets = None


class FfmpegFrames:
    """A recording's frames through ffmpeg's pipe, as `pix_fmt` (yuv444p: its planes as
    (3, H, W); rgb24: (H, W, 3)), as the training reader has always run it, from frame
    `start` on."""

    def __init__(self, path, width, height, pix_fmt="rgb24", hwaccel=False, start=0):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg is required to read recordings")
        pre = ["-hwaccel", "cuda"] if hwaccel else []
        select = []
        if start:
            # The frames before it are decoded but never converted or piped. Selected by
            # their number, not a time, so no timestamp can land it a frame off; and
            # restarted at time zero, so the frames come out as they would from the start.
            select = ["-vf", f"select=gte(n\\,{start}),setpts=PTS-STARTPTS"]
        self.process = subprocess.Popen(
            [ffmpeg, "-v", "error", *pre, "-i", str(path), *select]
            + ["-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1"],
            stdout=subprocess.PIPE,
        )
        self.shape = (3, height, width) if pix_fmt == "yuv444p" else (height, width, 3)
        self.size = width * height * 3

    def read(self):
        buffer = self.process.stdout.read(self.size)
        if len(buffer) != self.size:
            return None
        return np.frombuffer(buffer, np.uint8).reshape(self.shape)

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()


@functools.cache
def codec_name(path):
    """The video stream's codec as ffprobe names it ("h264", "hevc", ...), or None."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name"]
        + ["-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def open_frames(path, width, height, yuv=False, gpu=None, start=0):
    """A reader of the recording at `path`, frame by frame from frame `start`: planes
    (3, H, W) with `yuv`, else RGB (H, W, 3), the same pixels either way once to_rgb has
    converted the planes.

    HEVC goes to the GPU's decoder when `gpu` (default: when this process can): planes
    through PyNvVideoCodec, RGB through ffmpeg's own CUDA decoder (also bit-exact, with
    the conversion still on the CPU). Anything else, or HEVC without a GPU, is ffmpeg on
    the CPU.
    """
    gpu = available() if gpu is None else gpu
    hevc = gpu and codec_name(Path(path)) == "hevc"
    if hevc and yuv:
        try:
            return NvdecFrames(path, width, height, start)
        except ValueError:
            pass
    return FfmpegFrames(
        path,
        width,
        height,
        "yuv444p" if yuv else "rgb24",
        hwaccel=hevc and not yuv and torch.cuda.is_available(),
        start=start,
    )


@functools.cache
def _table_bytes():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to build the colour table")
    every = np.arange(1 << 24, dtype=np.uint32)
    planes = np.stack([every >> 16, (every >> 8) & 255, every & 255]).astype(np.uint8)
    # A 4096x4096 picture holding every (Y, U, V) once, through the reader's conversion.
    out = subprocess.run(
        [ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv444p", "-s", "4096x4096"]
        + ["-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        input=planes.tobytes(),
        capture_output=True,
        check=True,
    ).stdout
    return out


_TABLES = {}


def rgb_table(device="cpu"):
    """ffmpeg's rgb24 for every yuv444p pixel, indexed by Y << 16 | U << 8 | V: (2^24, 3)
    uint8 on `device`, built once a process."""
    key = str(torch.device(device))
    if key not in _TABLES:
        table = torch.frombuffer(bytearray(_table_bytes()), dtype=torch.uint8)
        _TABLES[key] = table.view(1 << 24, 3).to(device)
    return _TABLES[key]


def to_rgb(planes, chunk=16):
    """uint8 YUV planes (N, 3, H, W) to the RGB (N, H, W, 3) ffmpeg's rgb24 gives, on the
    planes' device, `chunk` frames at a time (their int32 index is 8 MB a 1080p frame)."""
    table = rgb_table(planes.device)
    out = torch.empty(
        (planes.shape[0], *planes.shape[2:], 3), dtype=torch.uint8, device=planes.device
    )
    for first in range(0, planes.shape[0], chunk):
        p = planes[first : first + chunk].int()
        out[first : first + chunk] = table[(p[:, 0] << 16) | (p[:, 1] << 8) | p[:, 2]]
    return out

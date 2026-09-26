"""The GPU's video decoder and the device's colour conversion give the CPU reader's pixels."""

import shutil
import subprocess

import numpy as np
import pytest
import torch

from hoi4_arena import nvdec
from hoi4_arena.dataset import cursor_crop, cursor_crops

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"))]  # fmt: skip


def _video(path, frames, encoder):
    """frames (N, H, W, 3) RGB, encoded with `encoder`'s ffmpeg arguments."""
    n, h, w, _ = frames.shape
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24"]
        + ["-video_size", f"{w}x{h}", "-framerate", "5", "-i", "pipe:0", *encoder, str(path)],
        input=frames.tobytes(),
        check=True,
    )


def _read(reader):
    out = []
    while (frame := reader.read()) is not None:
        out.append(np.array(frame))
    reader.close()
    return np.stack(out)


def _noise(n, h, w):
    rng = np.random.default_rng(3)
    # Noise over smooth gradients: every channel's values, and sharp colour edges.
    ramp = np.linspace(0, 255, w, dtype=np.float32)[None, None, :, None]
    base = (ramp + np.arange(n, dtype=np.float32)[:, None, None, None] * 20) % 256
    return np.clip(base + rng.normal(0, 60, (n, h, w, 3)), 0, 255).astype(np.uint8)


@needs_ffmpeg
@pytest.mark.parametrize("device", DEVICES)
def test_the_planes_converted_on_the_device_are_ffmpeg_s_rgb(tmp_path, device):
    path = tmp_path / "x264.mkv"
    _video(path, _noise(6, 96, 128), ["-c:v", "libx264", "-crf", "30", "-pix_fmt", "yuv444p"])
    rgb = _read(nvdec.FfmpegFrames(path, 128, 96))
    planes = _read(nvdec.open_frames(path, 128, 96, yuv=True, gpu=False))
    assert planes.shape == (6, 3, 96, 128) and rgb.shape == (6, 96, 128, 3)
    converted = nvdec.to_rgb(torch.from_numpy(planes).to(device), chunk=4).cpu().numpy()
    assert np.array_equal(converted, rgb)
    # A resumed run's reader starts at a frame: the same frames from there on.
    later = _read(nvdec.open_frames(path, 128, 96, yuv=True, gpu=False, start=4))
    assert np.array_equal(later, planes[4:])


@needs_ffmpeg
def test_the_table_is_every_pixel_through_ffmpeg():
    table = nvdec.rgb_table("cpu")
    assert table.shape == (1 << 24, 3) and table.dtype == torch.uint8
    # BT.601 limited range: black and white at 16 and 235, grey chroma at 128.
    assert table[16 << 16 | 128 << 8 | 128].tolist() == [0, 0, 0]
    assert table[235 << 16 | 128 << 8 | 128].tolist() == [255, 255, 255]


@pytest.mark.parametrize("device", DEVICES)
def test_crops_on_the_device_are_cursor_crop(device):
    frames = torch.randint(0, 256, (5, 40, 60, 3), dtype=torch.uint8)
    cursors = torch.tensor([[0, 0], [30, 20], [59, 39], [-10, 5], [200, 200]])
    crops = cursor_crops(frames.to(device), cursors, 16).cpu()
    for frame, (x, y), crop in zip(frames.numpy(), cursors.tolist(), crops, strict=True):
        assert np.array_equal(crop.numpy(), cursor_crop(frame, x, y, 16))


def _hevc_encodes(tmp_path):
    probe = tmp_path / "probe.mkv"
    try:
        _video(probe, _noise(1, 144, 256), nvdec_encoder(0))
    except subprocess.CalledProcessError:
        return False
    return True


def nvdec_encoder(qp):
    return ["-c:v", "hevc_nvenc", "-preset", "p7", "-tune", "hq", "-profile:v", "rext",
            "-pix_fmt", "yuv444p", "-rc", "constqp", "-qp", str(qp), "-bf", "0"]  # fmt: skip


@needs_ffmpeg
@pytest.mark.skipif(not nvdec.available(), reason="needs PyNvVideoCodec and a GPU")
def test_hevc_decoded_on_the_gpu_is_what_ffmpeg_decodes(tmp_path):
    if not _hevc_encodes(tmp_path):
        pytest.skip("no HEVC 4:4:4 NVENC here")
    path = tmp_path / "hevc.mkv"
    _video(path, _noise(12, 144, 256), nvdec_encoder(14))
    assert nvdec.codec_name(path) == "hevc"
    reader = nvdec.open_frames(path, 256, 144, yuv=True)
    assert isinstance(reader, nvdec.NvdecFrames)
    planes = _read(reader)
    assert np.array_equal(planes, _read(nvdec.FfmpegFrames(path, 256, 144, "yuv444p")))
    later = nvdec.open_frames(path, 256, 144, yuv=True, start=5)
    assert isinstance(later, nvdec.NvdecFrames)
    assert np.array_equal(_read(later), planes[5:])
    # And RGB through ffmpeg's own CUDA decoder is the CPU's RGB.
    gpu = nvdec.open_frames(path, 256, 144)
    assert "-hwaccel" in gpu.process.args
    assert np.array_equal(_read(gpu), _read(nvdec.FfmpegFrames(path, 256, 144)))
    # H.264 4:4:4, which NVDEC cannot read, stays with ffmpeg.
    h264 = tmp_path / "h264.mkv"
    _video(h264, _noise(2, 144, 256), ["-c:v", "libx264", "-pix_fmt", "yuv444p"])
    reader = nvdec.open_frames(h264, 256, 144, yuv=True)
    assert isinstance(reader, nvdec.FfmpegFrames)
    reader.close()

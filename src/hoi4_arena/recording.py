from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from .desktop import Desktop


def split_for_session(session_id: str):
    bucket = int(hashlib.sha256(session_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


class Recorder:
    """Lossless native RGB frames; explicit frame index -> capture time, not nominal FPS."""

    def __init__(self, root, first, *, source="human", hz=15, session_id=None, split=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.id = session_id or str(uuid.uuid4())
        self.manifest = {
            "schema": 1,
            "session_id": self.id,
            "split": split or split_for_session(self.id),
            "source": source,
            "width": first.rgb.shape[1],
            "height": first.rgb.shape[0],
            "nominal_fps": hz,
            "complete": False,
            "frames": 0,
            "privileged_state": False,
        }
        self.events = (self.root / "frames.jsonl").open("w", encoding="utf8")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg is required for lossless recordings")
        self.log = (self.root / "ffmpeg.log").open("wb")
        self.encoder = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{self.manifest['width']}x{self.manifest['height']}",
                "-framerate",
                str(hz),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-threads",
                "4",
                str(self.root / "screen.mkv"),
            ],
            stdin=subprocess.PIPE,
            stderr=self.log,
        )
        self._manifest()

    def _manifest(self):
        temp = self.root / "manifest.tmp"
        temp.write_text(json.dumps(self.manifest, indent=2), encoding="utf8")
        temp.replace(self.root / "manifest.json")

    def append(self, frame, **extra):
        if frame.rgb.shape != (self.manifest["height"], self.manifest["width"], 3):
            raise ValueError("Resolution changed during recording")
        if frame.meta.get("overflow") or not frame.meta.get("foreground"):
            raise ValueError("Invalid capture")
        self.encoder.stdin.write(frame.rgb.tobytes())
        row = {
            "index": self.manifest["frames"],
            **frame.meta,
            "received_ns": frame.received_ns,
            **extra,
        }
        self.events.write(json.dumps(row) + "\n")
        self.events.flush()
        self.manifest["frames"] += 1

    def close(self, *, complete=True, reason=None, trailing_events=None):
        if trailing_events:
            (self.root / "trailing-events.json").write_text(json.dumps(trailing_events))
        self.encoder.stdin.close()
        rc = self.encoder.wait(timeout=60)
        self.events.close()
        self.log.close()
        self.manifest.update(complete=complete and rc == 0, reason=reason, encoder_exit=rc)
        self._manifest()


def record(root, seconds, hz=15, command=None, split=None):
    with Desktop(command) as desktop:
        first = desktop.capture()
        recorder = Recorder(root, first, hz=hz, split=split)
        start = time.monotonic()
        deadline = start
        reason = None
        try:
            recorder.append(first)
            while time.monotonic() - start < seconds:
                deadline += 1 / hz
                time.sleep(max(0, deadline - time.monotonic()))
                frame = desktop.capture()
                recorder.append(frame)
                if time.monotonic() - deadline > 1:
                    raise RuntimeError("Recording cannot maintain capture cadence")
        except (Exception, KeyboardInterrupt) as error:
            reason = str(error) or "interrupted"
        finally:
            tail = None
            try:
                tail = desktop.request("events")
                tail.pop("payload", None)
            except Exception as error:
                reason = reason or f"Final input drain failed: {error}"
            recorder.close(complete=reason is None, reason=reason, trailing_events=tail)
        print(json.dumps(recorder.manifest, indent=2))

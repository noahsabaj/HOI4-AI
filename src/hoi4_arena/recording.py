from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from .desktop import Desktop

log = logging.getLogger(__name__)


def audit_pixels(frame):
    """What to record for this frame.

    A worker that downscaled on the capture side never sent the native frame, so the
    audit video is the global view: the pixels the policy actually saw. The manifest
    says which, because a 224x224 video is not a substitute for a native recording
    when a human needs to review a match.
    """
    if frame.rgb is not None:
        return frame.rgb
    if frame.views is not None:
        return frame.views[0]
    raise ValueError("Capture carries no pixels to record")


def split_for_session(session_id: str):
    bucket = int(hashlib.sha256(session_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


class Recorder:
    """Lossless native RGB frames; explicit frame index -> capture time, not nominal FPS."""

    def __init__(self, root, first, *, source="human", hz=15, session_id=None, split=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.id = session_id or str(uuid.uuid4())
        pixels = audit_pixels(first)
        self.manifest = {
            "schema": 1,
            "session_id": self.id,
            "split": split or split_for_session(self.id),
            "source": source,
            "width": pixels.shape[1],
            "height": pixels.shape[0],
            "video_source": "full_frame" if first.rgb is not None else "global_view",
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
        pixels = audit_pixels(frame)
        if pixels.shape != (self.manifest["height"], self.manifest["width"], 3):
            raise ValueError("Resolution changed during recording")
        if frame.meta.get("overflow") or not frame.meta.get("foreground"):
            raise ValueError("Invalid capture")
        self.encoder.stdin.write(pixels.tobytes())
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
    """Write the manifest whatever happens, then fail loudly if the session is unusable.

    An incomplete recording is rejected by prepare_session, so exiting zero on a failed
    or interrupted run would hand the operator a session that can never be trained on.
    """
    with Desktop(command) as desktop:
        first = desktop.capture()
        recorder = Recorder(root, first, hz=hz, split=split)
        start = time.monotonic()
        deadline = start
        reason = None
        failure = None
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
            failure = error
            reason = str(error) or type(error).__name__
        finally:
            tail = None
            try:
                tail = desktop.request("events")
                tail.pop("payload", None)
            except Exception as error:
                reason = reason or f"Final input drain failed: {error}"
            try:
                lines = desktop.worker_log()
                if lines:
                    (recorder.root / "worker.log").write_text("\n".join(lines) + "\n")
            except Exception as error:  # noqa: BLE001 - diagnostics must never mask cleanup.
                log.warning("could not write worker.log: %s", error)
            recorder.close(complete=reason is None, reason=reason, trailing_events=tail)
            print(json.dumps(recorder.manifest, indent=2))
        # Recorder.close independently clears `complete` on a nonzero encoder exit without
        # setting a reason, so key the failure on the manifest rather than on `reason`.
        if not recorder.manifest["complete"]:
            if failure is not None:
                raise failure
            detail = reason or f"encoder exit {recorder.manifest['encoder_exit']}"
            raise RuntimeError(f"Recording incomplete: {detail}")

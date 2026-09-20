from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .desktop import Desktop

MODEL_ID = "galilai-group/LeVJEPA-VideoMix-Large"
MODEL_REVISION = "e831a0347737fcaa660b39c57d41c109de399845"


def gpu_memory():
    line = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()[0]
    total, used = map(int, line.split(","))
    return {"total_mib": total, "used_mib": used, "free_mib": total - used}


def verify_model_source(path):
    path = Path(path)
    if not (path / "reviewed-source.json").exists():
        raise RuntimeError("Review the pinned model source and write reviewed-source.json first")
    review = json.loads((path / "reviewed-source.json").read_text())
    if review["revision"] != MODEL_REVISION:
        raise ValueError("Unexpected model revision")
    for name in ("configuration_levjepa.py", "modeling_levjepa.py"):
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != review["reviewed_files"][name]:
            raise ValueError(f"Reviewed model source changed: {name}")


def load_encoder(path: str | Path):
    from transformers import AutoModel

    verify_model_source(path)
    # Only reviewed local custom code is executed; no remote fallback or mutable revision.
    return AutoModel.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)


def preprocess(rgb: np.ndarray, size=224):
    image = Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
    return (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
        [0.229, 0.224, 0.225]
    )[:, None, None]


def benchmark(model_path, output, iterations=30, offline=False, command=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "gameplay_verified": False,
        "mode": "offline_training" if offline else "live_inference",
    }
    report["baseline_memory"] = gpu_memory()
    try:
        model = load_encoder(model_path).cuda().to(torch.bfloat16)
        if offline:
            frame_path = output / "screen.png"
            if not frame_path.exists():
                raise RuntimeError("First run live capture to supply a real screenshot")
            x = preprocess(np.asarray(Image.open(frame_path).convert("RGB")))
            clip = x[None, :, None].expand(1, 3, 16, 224, 224).cuda().to(torch.bfloat16)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            # Two updates allocate optimizer moments and expose steady-state memory.
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss = model(pixel_values=clip).last_hidden_state.float().square().mean()
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            report["training_step_completed"] = True
            report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        else:
            model.eval()
            times, capture_times, memory = [], [], []
            with Desktop(command) as desktop:
                frames = []
                for _ in range(16):
                    f = desktop.capture()
                    frames.append(preprocess(f.rgb))
                    time.sleep(1 / 7.5)
                Image.fromarray(f.rgb).save(output / "screen.png")
                report["capture_shape"] = list(f.rgb.shape)
                report["capture_backend"] = desktop.attached["backend"]
                report["real_screen"] = True
                with torch.inference_mode():
                    for i in range(iterations + 5):
                        begin = time.perf_counter()
                        f = desktop.capture()
                        capture_end = time.perf_counter()
                        frames.append(preprocess(f.rgb))
                        frames = frames[-16:]
                        x = torch.stack(frames, 1)[None].cuda().to(torch.bfloat16)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        model(pixel_values=x)
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - start) * 1000
                        if i >= 5:
                            times.append(elapsed)
                            capture_times.append((capture_end - begin) * 1000)
                            memory.append(gpu_memory())
                        time.sleep(max(0, 0.2 - (time.perf_counter() - begin)))
            report.update(
                inference_ms=times,
                capture_ms=capture_times,
                memory=memory,
                inference_p95_ms=float(np.percentile(times, 95)),
                min_headroom_mib=min(x["free_mib"] for x in memory),
            )
            report["passes_encoder_gate"] = (
                report["inference_p95_ms"] < 200 and report["min_headroom_mib"] >= 1024
            )
            report["note"] = (
                "Encoder-only timing; policy, detail crops, and loaded combat require a separate runtime gate."
            )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["passes_encoder_gate"] = False
    filename = "training.json" if offline else "inference.json"
    (output / filename).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report

"""Train the inverse dynamics model on recordings with known inputs, and label video with it.

Video PreTraining (Baker et al., 2022) in this project's terms: a small amount of play
whose inputs were recorded teaches a model to infer inputs from video, and that model
labels video nobody recorded inputs for. The labels land in the recording as
`labels-idm.npz`, and `train-bc --sources idm` trains the policy on them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from .actions import SLOTS
from .dataset import (
    IDM_LABELS,
    VideoSessions,
    _Stream,
    batch_to_device,
    cover_starts,
    session_labels,
)
from .learning import file_hash, save_checkpoint
from .models import InverseDynamics, build_encoder

# How far ahead the model looks, in decision intervals. The clip ends four intervals
# (0.8 s) after the decision, so it spans 0.6 s before it and the effect after; the
# quadrants and fovea come from one interval later, where a moved pointer has landed.
CLIP_SHIFT = 4
DETAIL_SHIFT = 1
# Every source whose recordings carry inputs. Scripted camera inputs are real inputs, so
# they teach the model what a pan, a zoom and a click look like.
LABELLED = ("human", "ai")


def _score(model, batch, labels=None, deterministic=True, checkpoint=False):
    context, cells = model(
        batch["clips"], batch["quadrants"], batch["fovea"], batch["speed"], checkpoint=checkpoint
    )
    flat = context.flatten(0, 1), cells.flatten(0, 1)
    if labels is None:
        return model.actor(*flat, deterministic=deterministic)
    return model.actor(*flat, labels.flatten(0, 1))


def label_accuracy(predicted, actual):
    """How often the predicted event kind is right, per slot, and how far off moves land.

    Returns the fraction of slots whose kind matches and, over slots that are moves in
    both, the mean pointer error in lattice steps. Both are what decides whether a label
    is worth training on; a likelihood alone does not say.
    """
    kinds = predicted[..., 0] == actual[..., 0]
    moves = (predicted[..., 0] == 1) & (actual[..., 0] == 1)
    error = (predicted[..., 1:] - actual[..., 1:]).float().norm(dim=-1)
    return float(kinds.float().mean()), float(error[moves].mean()) if moves.any() else None


def train_idm(
    data,
    model_path,
    output,
    *,
    # LeVJEPA reading four 448x256 frames in sequence: it told the camera's inputs apart
    # best of the encoders probed (STATUS.md, "LeVJEPA, a second look"), and this model
    # runs offline, where its cost does not matter.
    variant="large",
    epochs=1,
    batch_size=2,
    sequence=16,
    seed=42,
    sources=LABELLED,
    recompute=True,
    context="gru",
    context_layers=2,
):
    """Train the inverse dynamics model on recordings whose inputs are known.

    `context` is what runs over the window: "gru", the two-way GRU, or "transformer",
    full two-way attention (models.WindowAttention) with `context_layers` layers.
    `sequence` is the window, in decisions: 16 (3.2 s), 32 or 64. A longer one lets a
    label read further from its decision. That matters at speed 5, where the simulation
    does not sleep and the screen can answer an input late, after the 0.8 s the shifted
    clip covers; a neighbour's frames may then be where its effect shows.
    """
    torch.manual_seed(seed)
    output = Path(output)
    if (output / "epoch-0000.pt").exists():
        raise FileExistsError("Checkpoints are immutable")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    common = {
        "length": sequence,
        "burn_in": 0,
        "sources": sources,
        "seed": seed,
        "device": device,
        "clip_shift": CLIP_SHIFT,
        "detail_shift": DETAIL_SHIFT,
    }
    dataset = VideoSessions(data, **common)
    validation = VideoSessions(data, split="validation", **common)
    model = InverseDynamics(
        build_encoder(model_path, variant), context=context, layers=context_layers
    ).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    config = {
        "kind": "idm",
        "variant": variant,
        "seed": seed,
        "sequence": sequence,
        "context": context,
        "context_layers": context_layers,
        "clip_shift": CLIP_SHIFT,
        "detail_shift": DETAIL_SHIFT,
        "model_path": str(Path(model_path).resolve()),
        "sources": list(sources),
    }
    output.mkdir(parents=True, exist_ok=True)
    autocast = {"device_type": device, "dtype": torch.bfloat16, "enabled": device == "cuda"}
    with (output / "metrics.jsonl").open("a") as log:
        for epoch in range(epochs):
            model.train()
            for step, batch in enumerate(DataLoader(dataset, batch_size=batch_size)):
                batch = batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(**autocast):
                    logp = _score(model, batch, batch["actions"], checkpoint=recompute)[1]
                    loss = -logp.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                log.write(json.dumps({"epoch": epoch, "step": step, "nll": loss.item()}) + "\n")
                log.flush()
            model.eval()
            nll, kinds, errors = [], [], []
            with torch.no_grad():
                for batch in DataLoader(validation, batch_size=batch_size):
                    batch = batch_to_device(batch, device)
                    with torch.autocast(**autocast):
                        nll.extend((-_score(model, batch, batch["actions"])[1]).float().tolist())
                        predicted = _score(model, batch)[0]
                    kind, error = label_accuracy(predicted, batch["actions"].flatten(0, 1))
                    kinds.append(kind)
                    if error is not None:
                        errors.append(error)
            log.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "validation_nll": float(np.mean(nll)),
                        "validation_kind_accuracy": float(np.mean(kinds)),
                        "validation_pointer_error": float(np.mean(errors)) if errors else None,
                    }
                )
                + "\n"
            )
            log.flush()
            save_checkpoint(
                output / f"epoch-{epoch:04d}.pt",
                model,
                config,
                optimizer=optimizer,
                provenance={"dataset": str(Path(data).resolve()), "gameplay_verified": False},
            )
    return config


def load_idm(checkpoint, model_path=None, device="cuda"):
    path = Path(checkpoint)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if file_hash(path) != metadata["sha256"]:
        raise ValueError("Checkpoint does not match its immutable manifest")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = saved["config"]
    if config.get("kind") != "idm":
        raise ValueError("Not an inverse dynamics checkpoint")
    encoder = build_encoder(model_path or config["model_path"], config["variant"])
    # A checkpoint from before the option has no "context": it is the two-way GRU.
    model = InverseDynamics(
        encoder, context=config.get("context", "gru"), layers=config.get("context_layers", 2)
    )
    model.load_state_dict(saved["policy"])
    return model.to(device).eval(), config, metadata["sha256"]


def label_recording(checkpoint, recording, *, model_path=None, window=None, device=None):
    """Write the model's inputs for every decision of `recording` to labels-idm.npz.

    Each label is the most likely input, slot by slot, with its log-likelihood kept as a
    confidence (`train-bc --idm-min-logp` and `--idm-weight` use it). Decisions the model
    could not see the frames after are marked invalid. The window defaults to the one the
    model trained on, so a label sees as far around it as the model learned to.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    recording = Path(recording)
    model, config, digest = load_idm(checkpoint, model_path, device)
    window = window or config.get("sequence", 16)
    manifest = json.loads((recording / "manifest.json").read_text())
    labels = session_labels(
        recording,
        sources=(manifest.get("source"),),
        clip_shift=config["clip_shift"],
        detail_shift=config["detail_shift"],
    )
    count = len(labels["decisions"])
    actions = np.zeros((count, SLOTS, 3), np.int64)
    logp = np.full(count, -np.inf, np.float32)
    stream = _Stream(labels, window, 0, device, starts=cover_starts(labels, window))
    autocast = {"device_type": device, "dtype": torch.bfloat16, "enabled": device == "cuda"}
    try:
        while (done := stream.advance()) is not None:
            for piece in done:
                start = piece.pop("start")
                batch = batch_to_device(default_collate([piece]), device)
                with torch.no_grad(), torch.autocast(**autocast):
                    predicted, score, _ = _score(model, batch)
                actions[start : start + window] = predicted.cpu().numpy()
                logp[start : start + window] = score.float().cpu().numpy()
    finally:
        stream.close()
    valid = labels["readable"] & np.isfinite(logp)
    np.savez(
        recording / IDM_LABELS,
        decisions=labels["decisions"],
        actions=actions,
        valid=valid,
        logp=logp,
        checkpoint=digest,
    )
    return {"decisions": count, "labelled": int(valid.sum()), "checkpoint": digest}

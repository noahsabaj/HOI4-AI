from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import VideoSessions, batch_to_device, window_loader
from .learning import Progress, save_checkpoint
from .models import (
    CHUNK,
    Policy,
    PredictiveAuxiliary,
    VideoEncoder,
    build_encoder,
    reads_clip,
    xm_loss,
)


def unroll(policy, batch, burn_in=2, training=True, checkpoint=False, chunk=CHUNK):
    """Run the policy over a batch of windows. Burn-in steps only warm the memory.

    Returns the scored steps' memories, values, summary features and cells, each with
    the time axis second.

    What the screen shows does not depend on the memory, so the window's frames are
    perceived first, `chunk` frames at a time (Policy.perceive_window), and only the
    memory then runs step by step. The encoder's frozen blocks run once, without a graph.
    Burn-in frames are read with no graph at all, and the memory they warm is detached,
    as when each step ran whole.

    `checkpoint` keeps only the inputs of the trainable part of perception (the
    encoder's last blocks, the detail and fovea readers) and recomputes it in the
    backward pass, chunk by chunk. Before, each step was recomputed whole, the frozen
    blocks included, which on the Qwen3.5 tower was most of a 3.2 s step.
    """
    clips = batch.get("clips")
    steps = batch["quadrants"].shape[1]
    dtype = batch["quadrants"].dtype
    seen = []
    for part, grad in ((slice(0, burn_in), False), (slice(burn_in, steps), training)):
        if part.start >= part.stop:
            continue
        with torch.set_grad_enabled(grad):
            seen.append(
                policy.perceive_window(
                    None if clips is None else clips[:, part],
                    batch["quadrants"][:, part],
                    batch["fovea"][:, part],
                    checkpoint=checkpoint,
                    chunk=chunk,
                )
            )
    summary, cells, centre = (torch.cat(x, 1) for x in zip(*seen))
    hidden = batch["quadrants"].new_zeros(summary.shape[0], policy.memory_dim)
    memories, values = [], []
    for t in range(steps):
        with torch.set_grad_enabled(training and t >= burn_in):
            hidden, value = policy.recall(
                summary[:, t],
                cells[:, t],
                centre[:, t],
                batch["previous"][:, t],
                batch["speed"][:, t],
                hidden,
                dtype,
            )
        if t < burn_in:
            hidden = hidden.detach()
        else:
            memories.append(hidden)
            values.append(value)
    return (
        torch.stack(memories, 1),
        torch.stack(values, 1),
        summary[:, burn_in:],
        cells[:, burn_in:],
    )


def imitation_score(policy, memory, cells, actions, objective, xm=None, sigma=0.0):
    """Per-step log-likelihood of the demonstrated actions, flattened over time.

    `xm` holds xm_loss's options (candidates, form) for the "xm" objective. `sigma` > 0
    scores each demonstrated move against a blob that wide around it (ActionHead).
    """
    memory, cells, actions = memory.flatten(0, 1), cells.flatten(0, 1), actions.flatten(0, 1)
    if objective == "xm":
        return -xm_loss(policy.actor, memory, cells, actions, **(xm or {}), sigma=sigma)
    if objective == "bc":
        return policy.actor(memory, cells, actions, sigma=sigma)[1]
    raise ValueError(objective)


def imitation_loss(score, weight):
    """The imitation loss from per-step log-likelihoods and each step's weight.

    The mean of the weighted negative log-likelihoods over every step, not their sum over
    the weights' sum: that would cancel a weight shared by the whole batch, and a batch of
    inferred labels alone would train as hard as recorded ones. With every weight 1 it is
    the plain mean.
    """
    return -(score * weight.flatten().to(score.dtype)).mean()


def presses(actions):
    """Per decision (N, SLOTS, 3), whether any slot presses a key or a mouse button."""
    from .actions import VOCAB

    down = torch.tensor(
        [bool(e) and e["kind"] in ("button", "key") and e["down"] for e in VOCAB],
        device=actions.device,
    )
    return down[actions[..., 0]].any(-1)


def state_r2(predicted, truth):
    """Explained variance of each group of true-state targets, over the known ones.

    Groups: each side's army numbers, who holds each state, divisions in each state, and
    the date. One minus the squared error over the variance around the mean, pooled over
    the group's targets; None for a group with no variance.
    """
    from .privileged import NAMES

    groups = {
        "own": [i for i, n in enumerate(NAMES) if n.startswith("own_") and "_at_" not in n],
        "enemy": [i for i, n in enumerate(NAMES) if n.startswith("enemy_") and "_at_" not in n],
        "held": [i for i, n in enumerate(NAMES) if n.startswith("held_")],
        "at": [i for i, n in enumerate(NAMES) if "_at_" in n],
        "year": [NAMES.index("year")],
    }
    report = {}
    for name, index in groups.items():
        p, t = predicted[:, index].float(), truth[:, index].float()
        known = torch.isfinite(t)
        if not known.any():
            continue
        t0 = torch.nan_to_num(t)
        count = known.sum(0).clamp_min(1)
        mean = (t0 * known).sum(0) / count
        variance = (((t0 - mean) ** 2) * known).sum()
        error = (((p - t0) ** 2) * known).sum()
        report[name] = round(float(1 - error / variance), 4) if float(variance) > 0 else None
    return report


def state_loss(head, memory, target):
    """Squared error of the true state (privileged.NAMES) read from the memory.

    Mean over the known targets only: a recording without the arena's daily reports has
    none, and then the loss is zero.
    """
    prediction = head(memory.float())
    known = torch.isfinite(target)
    if not known.any():
        return prediction.sum() * 0
    error = (prediction - torch.nan_to_num(target.float())).square()
    return (error * known).sum() / known.sum()


def train_bc(
    data,
    model_path,
    output,
    *,
    variant="screen",
    student=None,
    auxiliary="none",
    objective="bc",
    epochs=1,
    batch_size=2,
    sequence=8,
    burn_in=2,
    seed=42,
    sources=("human",),
    recompute=True,
    sparsity_shift=0.0,
    temporal_jaccard=0.0,
    projections=256,
    xm_candidates=5,
    xm_form="hard",
    xm_latents=0,
    idm_min_logp=None,
    idm_weight=1.0,
    advantage=False,
    pointer_sigma=0.0,
    look_before_click=False,
    workers=2,
    chunk=CHUNK,
    save_every=600.0,
    resume=False,
    lead_in=None,
    drop_keys=(),
    loser_weight=1.0,
    state_weight=0.0,
    lr=1e-4,
    init=None,
):
    """Behaviour cloning on recordings, read straight from their video.

    `sources` picks which recordings' inputs are demonstrations: "human" play, "ai"
    games' scripted camera and popup clicks (see ai_games), and "idm", inputs the inverse
    dynamics model labelled. Those are noisier than recorded ones: `idm_min_logp` drops
    the ones it was least sure of and `idm_weight` (0 < W <= 1) scales the rest's loss
    against a recorded label's (dataset.session_labels says why).

    `recompute` recomputes the trainable part of perception in the backward pass instead
    of keeping its activations (see `unroll`; `chunk` is how many frames it reads at
    once). Measured on the 4060 Ti with LeVJEPA, windows of 8 steps after 2 of burn-in
    (2026-09-23), when each whole step was recomputed: batch 1 took 1.0 s and 4.3 GB,
    checkpointed 1.6 s and 2.1 GB; batch 2 checkpointed took 1.3 s a window and 2.5 GB,
    and without it 34.6 s a step, because at 7 GB Windows moved GPU memory into system
    memory instead of failing.

    `workers` background processes decode the video and cut the windows while the GPU
    trains (see `window_loader`); 0 does it on this thread, the views on the GPU, in
    exactly the order training has always seen. Clips are read only for an encoder that
    reads them: the default Qwen3.5 tower reads the quadrants alone.

    `lead_in`, `drop_keys` and `loser_weight` pass to dataset.session_labels.
    `state_weight` > 0 adds the privileged-state loss: a linear read-out of the memory
    predicts the arena's true state at each decision (privileged.NAMES), from the
    arena log, weighted by it; the read-out is saved beside the policy and never used to
    act. `init` starts the policy from a checkpoint's weights (fine-tuning), `lr` sets
    the learning rate.
    """
    if not 0 < idm_weight <= 1:
        raise ValueError("idm_weight must be in (0, 1]")
    torch.manual_seed(seed)
    output = Path(output)
    if (output / "epoch-0000.pt").exists() and not resume:
        raise FileExistsError("Checkpoints are immutable")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = build_encoder(model_path, variant)
    common = {
        "length": sequence,
        "burn_in": burn_in,
        "sources": sources,
        "seed": seed,
        "device": device if not workers else "cpu",
        "clips": reads_clip(encoder),
        "idm_min_logp": idm_min_logp,
        "idm_weight": idm_weight,
        "advantage": advantage,
        "look_before_click": look_before_click,
        "lead_in": lead_in,
        "drop_keys": tuple(drop_keys),
        "loser_weight": loser_weight,
        "state": state_weight > 0,
    }
    dataset = VideoSessions(data, **common)
    validation = VideoSessions(data, split="validation", **common)
    # The regularizer needs two sequences. Plain behavior cloning can use a leftover one.
    loader = window_loader(
        dataset, batch_size, workers=workers, device=device, drop_last=auxiliary != "none"
    )
    if len(dataset) < (2 if auxiliary != "none" else 1):
        raise ValueError("Need at least two sequences for independent-batch regularization")
    if variant == "tiny":
        if not student:
            raise ValueError("Distill a student before training a compact policy")
        encoder.load_state_dict(
            torch.load(student, map_location="cpu", weights_only=True)["encoder"]
        )
    policy = Policy(encoder, latents=xm_latents, look=look_before_click)
    if init is not None:
        saved = torch.load(init, map_location="cpu", weights_only=True)
        policy.load_state_dict(saved["policy"])
    policy = policy.to(device)
    xm = {"candidates": xm_candidates, "form": xm_form}
    aux = PredictiveAuxiliary(
        feature_dim=encoder.dim,
        mode=auxiliary,
        shift=sparsity_shift,
        temporal_jaccard=temporal_jaccard,
        projections=projections,
    ).to(device)
    from .privileged import DIM as STATE_DIM

    state_head = torch.nn.Linear(policy.memory_dim, STATE_DIM).to(device)
    trained = [*policy.parameters(), *aux.parameters()]
    if state_weight > 0:
        trained += list(state_head.parameters())
    params = [p for p in trained if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    config = {
        "variant": variant,
        "auxiliary": auxiliary,
        "sparsity_shift": sparsity_shift,
        "temporal_jaccard": temporal_jaccard,
        "projections": projections,
        "xm_candidates": xm_latents or xm_candidates,
        "xm_form": xm_form,
        "xm_latents": xm_latents,
        "objective": objective,
        "seed": seed,
        "sequence": sequence,
        "burn_in": burn_in,
        "batch_size": batch_size,
        "model_path": str(Path(model_path).resolve()),
        "sources": list(sources),
        "idm_min_logp": idm_min_logp,
        "idm_weight": idm_weight,
        "advantage": advantage,
        "pointer_sigma": pointer_sigma,
        "look_before_click": look_before_click,
        "lead_in": lead_in,
        "drop_keys": list(drop_keys),
        "loser_weight": loser_weight,
        "state_weight": state_weight,
        "lr": lr,
        "init": str(Path(init).resolve()) if init else None,
    }
    output.mkdir(parents=True, exist_ok=True)
    progress = Progress(output, config, every=save_every, resume=resume)
    modules = {"policy": policy, "auxiliary": aux, "state_head": state_head}
    first_epoch, skip = progress.start(modules, optimizer)
    autocast = {"device_type": device, "dtype": torch.bfloat16, "enabled": device == "cuda"}
    with (output / "metrics.jsonl").open("a") as log:
        for epoch in range(first_epoch, epochs):
            policy.train()
            aux.train()
            # Workers iterate copies of the dataset, so its epoch is set here, not counted.
            dataset.epoch = epoch
            for step, batch in enumerate(loader):
                if epoch == first_epoch and step < skip:
                    continue  # Trained before the run was interrupted.
                batch = batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(**autocast):
                    memory, _, features, cells = unroll(
                        policy, batch, burn_in, checkpoint=recompute, chunk=chunk
                    )
                    actions = batch["actions"][:, burn_in:]
                    score = imitation_score(
                        policy, memory, cells, actions, objective, xm, sigma=pointer_sigma
                    )
                    bc = imitation_loss(score, batch["weight"][:, burn_in:])
                    predictive = aux(memory, features, actions, batch["valid"][:, burn_in:])
                    loss = bc + 0.1 * predictive
                    if state_weight > 0:
                        truth = state_loss(state_head, memory, batch["state"][:, burn_in:])
                        loss = loss + state_weight * truth
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                row = {
                    "epoch": epoch,
                    "step": step,
                    "bc": bc.item(),
                    "predictive": predictive.item(),
                    "loss": loss.item(),
                }
                if state_weight > 0:
                    row["state"] = truth.item()
                log.write(json.dumps(row) + "\n")
                log.flush()
                progress.tick(epoch, step + 1, modules, optimizer)
            policy.eval()
            validation_losses, acting, predicted, truths = [], [], [], []
            with torch.no_grad():
                for batch in window_loader(validation, batch_size, workers=workers, device=device):
                    batch = batch_to_device(batch, device)
                    with torch.autocast(**autocast):
                        memory, _, _, cells = unroll(
                            policy, batch, burn_in, training=False, chunk=chunk
                        )
                        labels = batch["actions"][:, burn_in:]
                        # For xm, the same choice among candidates the training loss
                        # makes, not a score of candidates the optimizer never saw.
                        score = imitation_score(policy, memory, cells, labels, objective, xm)
                        validation_losses.extend((-score).float().cpu().tolist())
                        acting.extend(presses(labels.flatten(0, 1)).cpu().tolist())
                        if state_weight > 0:
                            predicted.append(state_head(memory.float()).flatten(0, 1).cpu())
                            truths.append(batch["state"][:, burn_in:].flatten(0, 1).cpu())
            report = {
                "epoch": epoch,
                "validation_nll": sum(validation_losses) / len(validation_losses),
                "validation_decisions": len(validation_losses),
                "selection_requires_held_out_games": True,
            }
            pressed = [x for x, a in zip(validation_losses, acting, strict=True) if a]
            if pressed:
                # The decisions that press a key or a button: the orders themselves,
                # against the many that only wait or move the camera.
                report["validation_nll_presses"] = sum(pressed) / len(pressed)
                report["validation_presses"] = len(pressed)
            if predicted:
                report["validation_state_r2"] = state_r2(torch.cat(predicted), torch.cat(truths))
            log.write(json.dumps(report) + "\n")
            log.flush()
            if state_weight > 0:
                torch.save(state_head.state_dict(), output / f"state-head-{epoch:04d}.pt")
            save_checkpoint(
                output / f"epoch-{epoch:04d}.pt",
                policy,
                config,
                auxiliary=aux,
                optimizer=optimizer,
                provenance={"dataset": str(Path(data).resolve()), "gameplay_verified": False},
            )
            progress.save(epoch + 1, 0, modules, optimizer)
    progress.finish()
    return config


def train_critic(
    data,
    checkpoint,
    output,
    *,
    model_path=None,
    epochs=1,
    batch_size=2,
    sequence=8,
    burn_in=2,
    seed=42,
    trunk=False,
    recompute=True,
):
    """Pre-train a policy's critic on recorded AI games, whose winners are known.

    Each decision's target is the game's result from Blue's side, discounted by the time
    left (session_labels' `outcome`), scored with the same binary cross-entropy PPO uses.
    Self-play then starts with a critic that already tells a winning position from a
    losing one, instead of learning that from its own first, poor games; PACT pre-trained
    its critics the same way before comparing objectives. Only the value head trains
    unless `trunk`, so a behaviour-cloned policy acts exactly as before.

    Validation reports the loss, how often the predicted return has the winner's sign,
    and the separation: the mean scaled prediction in games Blue won minus in games it
    lost.
    """
    from .learning import critic_loss, save_checkpoint, scale_return, value_estimate
    from .runner import load_policy

    torch.manual_seed(seed)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Checkpoints are immutable")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, config, digest = load_policy(checkpoint, model_path, device)
    policy.train()
    if not trunk:
        policy.requires_grad_(False)
        policy.value.requires_grad_(True)
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    common = {
        "length": sequence,
        "burn_in": burn_in,
        "sources": ("ai",),
        "seed": seed,
        "clips": reads_clip(policy.encoder),
    }
    dataset = VideoSessions(data, device=device, **common)
    try:
        validation = VideoSessions(data, split="validation", device=device, **common)
    except ValueError:
        validation = []  # Few games may hash none into validation; the report says so.
    autocast = {"device_type": device, "dtype": torch.bfloat16, "enabled": device == "cuda"}
    history = []
    for epoch in range(epochs):
        for batch in DataLoader(dataset, batch_size=batch_size):
            batch = batch_to_device(batch, device)
            target = batch["outcome"][:, burn_in:]
            known = torch.isfinite(target)
            if not known.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(**autocast):
                _, values, _, _ = unroll(policy, batch, burn_in, checkpoint=recompute)
            loss = critic_loss(values[known], target[known]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite critic objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            history.append(loss.item())
    policy.eval()
    losses, right, wins, losses_side = [], [], [], []
    with torch.no_grad():
        for batch in DataLoader(validation, batch_size=batch_size):
            batch = batch_to_device(batch, device)
            target = batch["outcome"][:, burn_in:]
            known = torch.isfinite(target)
            if not known.any():
                continue
            with torch.autocast(**autocast):
                _, values, _, _ = unroll(policy, batch, burn_in, training=False)
            values, target = values[known].float(), target[known].float()
            losses.extend(critic_loss(values, target).tolist())
            right.extend((value_estimate(values).sign() == target.sign()).float().tolist())
            scaled = scale_return(value_estimate(values))
            wins.extend(scaled[target > 0].tolist())
            losses_side.extend(scaled[target < 0].tolist())
    report = {
        "train_steps": len(history),
        "train_loss_first": history[0] if history else None,
        "train_loss_last": history[-1] if history else None,
        "validation_loss": float(np.mean(losses)) if losses else None,
        "validation_sign_accuracy": float(np.mean(right)) if right else None,
        "validation_separation": (
            float(np.mean(wins) - np.mean(losses_side)) if wins and losses_side else None
        ),
        "trunk": trunk,
    }
    save_checkpoint(
        output,
        policy,
        config,
        optimizer=optimizer,
        provenance={
            "parent": digest,
            "critic_pretraining": str(Path(data).resolve()),
            **report,
            "gameplay_verified": False,
        },
    )
    return report


def distill(data, model_path, output, epochs=1, seed=42, sources=("human", "ai")):
    """Offline teacher only; the deployed student remains a single direct policy encoder.

    The student learns the teacher's summary token and its patch grid, since the policy
    now reads both. AI games count here: distillation needs pixels, not actions.
    """
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = VideoSessions(data, length=1, burn_in=0, sources=sources, seed=seed, device=device)
    teacher = VideoEncoder(model_path, train_last=0).to(device).eval()
    student = VideoEncoder(model_path, variant="tiny").to(device)
    projection = torch.nn.Linear(student.dim, teacher.dim).to(device)
    optimizer = torch.optim.AdamW([*student.parameters(), *projection.parameters()], lr=1e-4)
    for _ in range(epochs):
        for batch in DataLoader(dataset, batch_size=2):
            clip = batch_to_device(batch, device)["clips"][:, 0]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                with torch.no_grad():
                    target, target_grid = teacher(clip)
                summary, grid = student(clip)
                loss = torch.nn.functional.mse_loss(
                    projection(summary).float(), target.float()
                ) + torch.nn.functional.mse_loss(
                    projection(grid.flatten(2).transpose(1, 2)).float(),
                    target_grid.flatten(2).transpose(1, 2).float(),
                )
            loss.backward()
            optimizer.step()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"encoder": student.state_dict(), "teacher": model_path, "seed": seed, "trained": True},
        path,
    )

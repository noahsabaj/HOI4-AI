from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import VideoSessions, batch_to_device
from .learning import save_checkpoint
from .models import Policy, PredictiveAuxiliary, VideoEncoder, build_encoder, xm_loss


def unroll(policy, batch, burn_in=2, training=True, checkpoint=False):
    """Run the policy over a batch of windows. Burn-in steps only warm the memory.

    Returns the scored steps' memories, values, summary features and cells, each with
    the time axis second.

    `checkpoint` keeps only each step's inputs and outputs for the backward pass and
    recomputes the rest (torch.utils.checkpoint), trading one more forward pass per step
    for the activations of every step's trainable encoder blocks and detail reader.
    """
    hidden = None
    memories, values, features, cells = [], [], [], []
    for t in range(batch["clips"].shape[1]):
        scored = training and t >= burn_in
        inputs = (
            batch["clips"][:, t],
            batch["quadrants"][:, t],
            batch["fovea"][:, t],
            batch["previous"][:, t],
            batch["speed"][:, t],
            hidden,
        )
        with torch.set_grad_enabled(scored):
            if scored and checkpoint:
                hidden, value, feature, cell = torch.utils.checkpoint.checkpoint(
                    policy, *inputs, use_reentrant=False
                )
            else:
                hidden, value, feature, cell = policy(*inputs)
        if t < burn_in:
            hidden = hidden.detach()
        else:
            memories.append(hidden)
            values.append(value)
            features.append(feature)
            cells.append(cell)
    return (
        torch.stack(memories, 1),
        torch.stack(values, 1),
        torch.stack(features, 1),
        torch.stack(cells, 1),
    )


def imitation_score(policy, memory, cells, actions, objective):
    """Per-step log-likelihood of the demonstrated actions, flattened over time."""
    memory, cells, actions = memory.flatten(0, 1), cells.flatten(0, 1), actions.flatten(0, 1)
    if objective == "xm":
        return -xm_loss(policy.actor, memory, cells, actions)
    if objective == "bc":
        return policy.actor(memory, cells, actions)[1]
    raise ValueError(objective)


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
):
    """Behaviour cloning on recordings, read straight from their video.

    `sources` picks which recordings' inputs are demonstrations: "human" play, and "ai"
    games' scripted camera and popup clicks (see ai_games).

    `recompute` recomputes each step in the backward pass instead of keeping its
    activations. Measured on the 4060 Ti with LeVJEPA, windows of 8 steps after 2 of
    burn-in (2026-09-23): batch 1 took 1.0 s and 4.3 GB, checkpointed 1.6 s and 2.1 GB;
    batch 2 checkpointed took 1.3 s a window and 2.5 GB, and without it 34.6 s a step,
    because at 7 GB Windows moved GPU memory into system memory instead of failing.
    """
    torch.manual_seed(seed)
    output = Path(output)
    if (output / "epoch-0000.pt").exists():
        raise FileExistsError("Checkpoints are immutable")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = VideoSessions(
        data, length=sequence, burn_in=burn_in, sources=sources, seed=seed, device=device
    )
    validation = VideoSessions(
        data,
        split="validation",
        length=sequence,
        burn_in=burn_in,
        sources=sources,
        seed=seed,
        device=device,
    )
    # The regularizer needs two sequences. Plain behavior cloning can use a leftover one.
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=auxiliary != "none")
    if len(dataset) < (2 if auxiliary != "none" else 1):
        raise ValueError("Need at least two sequences for independent-batch regularization")
    encoder = build_encoder(model_path, variant)
    if variant == "tiny":
        if not student:
            raise ValueError("Distill a student before training a compact policy")
        encoder.load_state_dict(
            torch.load(student, map_location="cpu", weights_only=True)["encoder"]
        )
    policy = Policy(encoder).to(device)
    aux = PredictiveAuxiliary(
        feature_dim=encoder.dim,
        mode=auxiliary,
        shift=sparsity_shift,
        temporal_jaccard=temporal_jaccard,
        projections=projections,
    ).to(device)
    params = [p for p in [*policy.parameters(), *aux.parameters()] if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    config = {
        "variant": variant,
        "auxiliary": auxiliary,
        "sparsity_shift": sparsity_shift,
        "temporal_jaccard": temporal_jaccard,
        "projections": projections,
        "objective": objective,
        "seed": seed,
        "sequence": sequence,
        "burn_in": burn_in,
        "batch_size": batch_size,
        "model_path": str(Path(model_path).resolve()),
        "sources": list(sources),
    }
    output.mkdir(parents=True, exist_ok=True)
    autocast = {"device_type": device, "dtype": torch.bfloat16, "enabled": device == "cuda"}
    with (output / "metrics.jsonl").open("a") as log:
        for epoch in range(epochs):
            policy.train()
            aux.train()
            for step, batch in enumerate(loader):
                batch = batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(**autocast):
                    memory, _, features, cells = unroll(
                        policy, batch, burn_in, checkpoint=recompute
                    )
                    actions = batch["actions"][:, burn_in:]
                    bc = -imitation_score(policy, memory, cells, actions, objective).mean()
                    predictive = aux(memory, features, actions, batch["valid"][:, burn_in:])
                    loss = bc + 0.1 * predictive
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
                log.write(json.dumps(row) + "\n")
                log.flush()
            policy.eval()
            validation_losses = []
            with torch.no_grad():
                for batch in DataLoader(validation, batch_size=batch_size):
                    batch = batch_to_device(batch, device)
                    with torch.autocast(**autocast):
                        memory, _, _, cells = unroll(policy, batch, burn_in, training=False)
                        labels = batch["actions"][:, burn_in:]
                        if objective == "xm":
                            # The same best-of-K choice the training loss makes, not a
                            # log-sum-exp of candidates the optimizer never saw.
                            flat_memory, flat_cells = memory.flatten(0, 1), cells.flatten(0, 1)
                            scores = torch.stack(
                                [
                                    policy.actor(
                                        flat_memory,
                                        flat_cells,
                                        labels.flatten(0, 1),
                                        noise=torch.randn(
                                            len(flat_memory), policy.actor.noise_dim, device=device
                                        ),
                                    )[1]
                                    for _ in range(5)
                                ]
                            )
                            score = scores.max(0).values
                        else:
                            score = imitation_score(policy, memory, cells, labels, "bc")
                        validation_losses.extend((-score).float().cpu().tolist())
            log.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "validation_nll": sum(validation_losses) / len(validation_losses),
                        "selection_requires_held_out_games": True,
                    }
                )
                + "\n"
            )
            log.flush()
            save_checkpoint(
                output / f"epoch-{epoch:04d}.pt",
                policy,
                config,
                auxiliary=aux,
                optimizer=optimizer,
                provenance={"dataset": str(Path(data).resolve()), "gameplay_verified": False},
            )
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
    common = {"length": sequence, "burn_in": burn_in, "sources": ("ai",), "seed": seed}
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

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import Sessions, require_one_game_speed
from .learning import save_checkpoint
from .models import Policy, PredictiveAuxiliary, VideoEncoder, xm_loss


def unroll(policy, batch, burn_in=2, training=True):
    hidden = None
    memories, values, features = [], [], []
    for t in range(batch["clips"].shape[1]):
        context = torch.set_grad_enabled(training and t >= burn_in)
        with context:
            hidden, value, feature = policy(
                batch["clips"][:, t], batch["tiles"][:, t], batch["previous"][:, t], hidden
            )
        if t < burn_in:
            hidden = hidden.detach()
        else:
            memories.append(hidden)
            values.append(value)
            features.append(feature)
    return torch.stack(memories, 1), torch.stack(values, 1), torch.stack(features, 1)


def train_bc(
    data,
    model_path,
    output,
    *,
    variant="large",
    student=None,
    auxiliary="none",
    objective="bc",
    epochs=1,
    batch_size=2,
    sequence=8,
    burn_in=2,
    seed=42,
):
    torch.manual_seed(seed)
    output = Path(output)
    if (output / "epoch-0000.pt").exists():
        raise FileExistsError("Checkpoints are immutable")
    dataset = Sessions(data, length=sequence, burn_in=burn_in)
    validation = Sessions(data, split="validation", length=sequence, burn_in=burn_in)
    require_one_game_speed([dataset.game_speed, validation.game_speed])
    # The regularizer needs two sequences. Plain behavior cloning can use a leftover one.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=auxiliary != "none")
    if len(loader) == 0:
        raise ValueError("Need at least two sequences for independent-batch regularization")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = VideoEncoder(model_path, variant=variant)
    if variant == "tiny":
        if not student:
            raise ValueError("Distill a student before training a compact policy")
        encoder.load_state_dict(
            torch.load(student, map_location="cpu", weights_only=True)["encoder"]
        )
    policy = Policy(encoder).to(device)
    aux = PredictiveAuxiliary(feature_dim=encoder.dim, mode=auxiliary).to(device)
    params = [p for p in [*policy.parameters(), *aux.parameters()] if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    config = {
        "variant": variant,
        "auxiliary": auxiliary,
        "objective": objective,
        "seed": seed,
        "sequence": sequence,
        "burn_in": burn_in,
        "batch_size": batch_size,
        "model_path": str(Path(model_path).resolve()),
        "game_speed": dataset.game_speed,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.jsonl").open("a") as log:
        for epoch in range(epochs):
            policy.train()
            aux.train()
            for step, batch in enumerate(loader):
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"
                ):
                    memory, _, features = unroll(policy, batch, burn_in)
                    actions = batch["actions"][:, burn_in:]
                    flat_memory, flat_actions = memory.flatten(0, 1), actions.flatten(0, 1)
                    if objective == "xm":
                        bc = xm_loss(policy.actor, flat_memory, flat_actions).mean()
                    elif objective == "bc":
                        bc = -policy.actor(flat_memory, flat_actions)[1].mean()
                    else:
                        raise ValueError(objective)
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
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    with torch.autocast(
                        device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"
                    ):
                        memory, _, _ = unroll(policy, batch, burn_in, training=False)
                        memory = memory.flatten(0, 1)
                        labels = batch["actions"][:, burn_in:].flatten(0, 1)
                        if objective == "xm":
                            # The same best-of-K choice the training loss makes, not a
                            # log-sum-exp of candidates the optimizer never saw.
                            scores = torch.stack(
                                [
                                    policy.actor(
                                        memory,
                                        labels,
                                        noise=torch.randn(
                                            len(memory), policy.actor.noise_dim, device=device
                                        ),
                                    )[1]
                                    for _ in range(5)
                                ]
                            )
                            score = scores.max(0).values
                        else:
                            score = policy.actor(memory, labels)[1]
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


def distill(data, model_path, output, epochs=1, seed=42):
    """Offline teacher only; the deployed student remains a single direct policy encoder."""
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    torch.manual_seed(seed)
    dataset = Sessions(data, length=1, burn_in=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = VideoEncoder(model_path, train_last=0).to(device).eval()
    student = VideoEncoder(model_path, variant="tiny").to(device)
    projection = torch.nn.Linear(student.dim, teacher.dim).to(device)
    optimizer = torch.optim.AdamW([*student.parameters(), *projection.parameters()], lr=1e-4)
    for _ in range(epochs):
        for batch in DataLoader(dataset, batch_size=2, shuffle=True):
            clip = batch["clips"][:, 0].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                with torch.no_grad():
                    target = teacher(clip)
                features = student(clip)
                loss = torch.nn.functional.mse_loss(projection(features).float(), target.float())
            loss.backward()
            optimizer.step()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"encoder": student.state_dict(), "teacher": model_path, "seed": seed, "trained": True},
        path,
    )

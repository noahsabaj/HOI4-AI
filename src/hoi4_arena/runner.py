"""Direct recurrent-policy collection. Match weights are immutable until collection ends."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .actions import SLOTS
from .dataset import normalize, views
from .desktop import Desktop
from .environment import ArenaEnv, ArenaPair
from .learning import file_hash
from .models import Policy, VideoEncoder
from .recording import Recorder
from .remote import RemoteDesktop
from .vision import ScreenRules


def load_policy(checkpoint, model_path=None, device="cuda"):
    path = Path(checkpoint)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if file_hash(path) != metadata["sha256"]:
        raise ValueError("Checkpoint does not match its immutable manifest")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = saved["config"]
    encoder = VideoEncoder(model_path or config["model_path"], variant=config["variant"])
    policy = Policy(encoder)
    policy.load_state_dict(saved["policy"])
    return policy.to(device), config, metadata["sha256"]


class Actor:
    def __init__(self, checkpoint, model_path=None):
        self.policy, self.config, self.digest = load_policy(checkpoint, model_path)
        self.policy.eval().requires_grad_(False)
        self.hidden = None
        self.previous = np.zeros((SLOTS, 3), dtype=np.int64)
        self.history = deque(maxlen=64)

    def act(self, rgb, timestamp):
        global_view, tiles = views(rgb)
        self.history.append((timestamp, global_view))
        times = np.array([t for t, _ in self.history])
        desired = timestamp - np.arange(15, -1, -1) / 7.5
        ids = np.searchsorted(times, desired, side="right") - 1
        clip = np.stack([self.history[max(0, int(i))][1] for i in ids])
        before = (
            np.zeros(512, np.float32)
            if self.hidden is None
            else self.hidden[0].float().cpu().numpy()
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.hidden, value, _ = self.policy(
                normalize(clip).permute(3, 0, 1, 2)[None].cuda(),
                normalize(tiles).permute(0, 3, 1, 2)[None].cuda(),
                torch.from_numpy(self.previous)[None].cuda(),
                self.hidden,
            )
            noise = (
                torch.randn(1, 16, device="cuda")
                if self.config["objective"] == "xm"
                else torch.zeros(1, 16, device="cuda")
            )
            action, logp, _ = self.policy.actor(self.hidden, noise=noise)
        action = action[0].cpu().numpy()
        sample = {
            "clip": clip,
            "tiles": tiles,
            "hidden": before,
            "previous": self.previous.copy(),
            "action": action,
            "old_logp": logp.item(),
            "old_value": value.item(),
            "noise": noise[0].cpu().numpy(),
        }
        self.previous = action
        return action, sample


def collect_pair(config_path, output, left_checkpoint, right_checkpoint):
    """Prototype coordinator: both actors infer here; second PC supplies its own pixels.

    This mode must pass its measured throughput gate before unattended acceptance.
    Model weights are loaded once; neither actor gets its opponent's observations.
    """
    config = json.loads(Path(config_path).read_text())
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    actors = [
        Actor(left_checkpoint, config.get("model_path")),
        Actor(right_checkpoint, config.get("model_path")),
    ]
    environments = []
    recorders = []
    desktops = []
    pair = None
    manifest = {
        "complete": False,
        "valid": False,
        "checkpoint_sha256": [a.digest for a in actors],
        "scenario": config["scenario"],
        "pair_id": config["pair_id"],
        "screen_only": True,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    reason = None
    count = 0
    try:
        for side in ["left", "right"]:
            spec = config[side]
            desktop = RemoteDesktop(spec["peer"]) if spec.get("peer") else Desktop()
            desktops.append(desktop)
            environments.append(
                ArenaEnv(
                    desktop,
                    ScreenRules(spec["rules"]),
                    json.loads(Path(spec["recipe"]).read_text()),
                    seconds=config.get("seconds", 1800),
                )
            )
        pair = ArenaPair(*environments)
        starts = pair.reset()
        for index, env in enumerate(environments):
            recorder = Recorder(
                root / f"player-{index}" / "recording", env.last, source="policy", hz=5
            )
            recorder.append(env.last)
            recorders.append(recorder)
            env.recorder = recorder
        observations = [s[0] for s in starts]
        while True:
            actions = []
            samples = []
            for actor, rgb, env in zip(actors, observations, environments, strict=True):
                action, sample = actor.act(rgb, env.last.meta["t_ns"] / 1e9)
                actions.append(action)
                samples.append(sample)
            results = pair.step(actions)
            for i, (sample, result) in enumerate(zip(samples, results, strict=True)):
                _, reward, terminated, truncated, info = result
                # Time-limit draws are terminal game outcomes, not bootstrapped truncations.
                sample.update(
                    reward=reward,
                    terminal=terminated or truncated,
                    valid=info["valid"],
                    elapsed=info.get("elapsed_seconds", 0.0),
                )
                np.savez_compressed(root / f"player-{i}" / f"{count:06d}.npz", **sample)
            count += 1
            observations = [r[0] for r in results]
            if any(r[2] or r[3] for r in results):
                manifest.update(
                    valid=all(r[4]["valid"] for r in results),
                    outcomes=[r[4]["outcome"] for r in results],
                )
                break
    except (Exception, KeyboardInterrupt) as error:
        reason = f"{type(error).__name__}: {error}"
        raise
    finally:
        for desktop in desktops:
            try:
                desktop.close()
            except Exception as error:
                reason = reason or f"Worker cleanup failed: {error}"
        if pair is not None:
            pair.pool.shutdown(wait=True)
        for recorder in recorders:
            try:
                recorder.close(complete=reason is None, reason=reason)
            except Exception as error:
                reason = reason or f"Recording cleanup failed: {error}"
        manifest.update(complete=reason is None, steps=count, error=reason)
        if reason is not None:
            manifest["valid"] = False
        try:
            if [file_hash(p) for p in [left_checkpoint, right_checkpoint]] != manifest[
                "checkpoint_sha256"
            ]:
                manifest.update(valid=False, error="Checkpoint changed during match")
        except OSError as error:
            manifest.update(valid=False, error=f"Checkpoint unavailable after match: {error}")
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def replay_batch(files, device):
    rows = [dict(np.load(path)) for path in files]
    batch = {
        name: torch.as_tensor(np.stack([r[name] for r in rows]), device=device) for name in rows[0]
    }
    batch["clips"] = (
        (
            batch.pop("clip").float() / 255
            - batch["tiles"].new_tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        )
        / batch["tiles"].new_tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    ).permute(0, 4, 1, 2, 3)
    tiles = batch["tiles"].float() / 255
    batch["tiles"] = (
        (tiles - tiles.new_tensor([0.485, 0.456, 0.406])) / tiles.new_tensor([0.229, 0.224, 0.225])
    ).permute(0, 1, 4, 2, 3)
    return batch


def train_ppo(rollouts, checkpoint, output, epochs=3, sequence=8, burn_in=2, model_path=None):
    from .learning import gae, ppo_loss, save_checkpoint

    policy, config, digest = load_policy(checkpoint, model_path)
    policy.train()
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-5)
    episodes = []
    for manifest_path in sorted(Path(rollouts).glob("*/manifest.json")):
        meta = json.loads(manifest_path.read_text())
        if not meta["complete"] or not meta["valid"]:
            continue
        for player in range(2):
            if meta["checkpoint_sha256"][player] != digest:
                continue  # Historical opponents are not on-policy training data.
            files = sorted((manifest_path.parent / f"player-{player}").glob("*.npz"))
            scalars = []
            for path in files:
                with np.load(path) as archive:
                    scalars.append(
                        {
                            key: archive[key]
                            for key in ["reward", "old_value", "terminal", "valid", "elapsed"]
                        }
                    )
            if not scalars or not scalars[-1]["terminal"]:
                raise ValueError("Incomplete episode cannot enter PPO")

            def get(key):
                return torch.as_tensor(np.array([r[key] for r in scalars]))

            advantages, returns = gae(
                get("reward"),
                get("old_value"),
                torch.tensor(0.0),
                get("terminal"),
                get("valid"),
                get("elapsed"),
            )
            episodes.append((files, advantages.float(), returns.float()))
    if not episodes:
        raise ValueError("No completed valid on-policy episodes; collection must precede PPO")
    losses = []
    for _ in range(epochs):
        for files, advantages, returns in episodes:
            for start in range(0, len(files), sequence):
                begin = max(0, start - burn_in)
                end = min(start + sequence, len(files))
                batch = replay_batch(files[begin:end], "cuda")
                hidden = batch["hidden"][0:1].float()
                logps = []
                values = []
                entropies = []
                optimizer.zero_grad(set_to_none=True)
                for t in range(end - begin):
                    burn = begin + t < start
                    with (
                        torch.set_grad_enabled(not burn),
                        torch.autocast("cuda", dtype=torch.bfloat16),
                    ):
                        hidden, value, _ = policy(
                            batch["clips"][t : t + 1],
                            batch["tiles"][t : t + 1],
                            batch["previous"][t : t + 1],
                            hidden,
                        )
                        if burn:
                            hidden = hidden.detach()
                        else:
                            _, logp, entropy = policy.actor(
                                hidden, batch["action"][t : t + 1], noise=batch["noise"][t : t + 1]
                            )
                            logps.append(logp)
                            values.append(value)
                            entropies.append(entropy)
                offset = start - begin
                loss = ppo_loss(
                    torch.cat(logps),
                    batch["old_logp"][offset:].float(),
                    torch.cat(values),
                    returns[start:end].cuda(),
                    advantages[start:end].cuda(),
                    torch.cat(entropies),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite PPO objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
    save_checkpoint(
        output,
        policy,
        config,
        optimizer=optimizer,
        provenance={
            "parent": digest,
            "rollouts": str(Path(rollouts).resolve()),
            "episodes": len(episodes),
            "gameplay_verified": False,
        },
    )
    return {
        "episodes": len(episodes),
        "updates": len(losses),
        "mean_loss": float(np.mean(losses)),
        "selection_requires_held_out_games": True,
    }

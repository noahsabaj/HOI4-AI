"""Direct recurrent-policy collection. Match weights are immutable until collection ends."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .actions import SLOTS
from .dataset import normalize, views
from .desktop import Desktop
from .environment import ArenaEnv, ArenaPair
from .learning import file_hash
from .models import Policy, VideoEncoder, halve_frozen
from .recording import Recorder
from .remote import RemoteDesktop
from .vision import ScreenRules

log = logging.getLogger(__name__)


def seed_everything(seed, *, salt=None):
    """Seed every RNG the collection and PPO paths draw from.

    A salt keeps runs reproducible without making them identical: seeding every match
    from the same constant would replay one RNG stream across a whole league, so the
    collected matches would no longer be independent samples of the policy.
    """
    if salt is not None:
        seed = int.from_bytes(hashlib.sha256(f"{seed}:{salt}".encode()).digest()[:4], "big")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    return int(seed)


def ppo_exclusion(meta):
    """Why this rollout may not train, or None if it may.

    Greedy evaluation rollouts are not samples from the behavior policy: their stored
    old_logp is the likelihood of an argmax, so the PPO ratio would be meaningless.
    Evaluation data never trains.
    """
    if not meta.get("complete"):
        return "incomplete"
    if not meta.get("valid"):
        return "invalid"
    if meta.get("deterministic"):
        return "deterministic evaluation rollout, not on-policy"
    return None


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
    # Both collection and PPO come through here, so the frozen weights are halved in
    # both and the likelihood stays the same function on either side of a rollout.
    # Halve before the transfer, not after: casting on the card first allocates the whole
    # float32 model there and leaves the freed half as allocator cache, which counts
    # against the free-VRAM gate exactly as if it were still in use.
    return halve_frozen(policy).to(device), config, metadata["sha256"]


def act_noise(objective, width, deterministic, device):
    """The latent an actor conditions on for one decision.

    ActionHead conditions its entire start state on this vector, so an xm checkpoint that
    keeps drawing a fresh latent stays stochastic no matter what the categorical heads do.
    A deterministic actor must therefore pin the latent to the prior mean as well; taking
    the argmax alone would leave exactly the sampling variance the flag exists to remove.
    """
    if deterministic or objective != "xm":
        return torch.zeros(1, width, device=device)
    return torch.randn(1, width, device=device)


class Actor:
    def __init__(
        self, checkpoint, model_path=None, deterministic=False, device="cuda", compile_head=True
    ):
        self.policy, self.config, self.digest = load_policy(checkpoint, model_path, device)
        self.policy.eval().requires_grad_(False)
        # Evaluation runs take the argmax so paired_evaluation's bound is not inflated by
        # sampling noise the analysis does not model. Self-play collection must sample.
        self.deterministic = deterministic
        self.device = device
        self.hidden = None
        self.previous = np.zeros((SLOTS, 3), dtype=np.int64)
        self.history = deque(maxlen=64)
        self.compiled = compile_head and device == "cuda" and self._compile_head()

    def _compile_head(self):
        """Capture the action head into a CUDA graph, and prove it before trusting it.

        The head is eight fixed-length slots of very small tensors, so nearly all of its
        cost is launching them: 7.72 ms of work that measures 0.741 ms once replayed from
        a graph. The encoder is deliberately left alone. Compiling it is worth about
        24 ms, but it moves a stored old_logp by 4.6e-4 and there is no configuration
        that removes the gap -- inductor's no-grad graph and its grad graph differ by
        3.4e-4 from each other, so compiling PPO's side too does not fix it. Eager
        collection and eager update agree exactly today, and that property is worth more
        than the milliseconds. The head is a different case: compiled, it is bit-for-bit
        the eager head, so this costs nothing at all.

        Returns whether compilation took, because inductor needs Triton and a failure
        here must cost latency rather than the match.
        """
        try:
            self.policy.actor.compile(mode="reduce-overhead", dynamic=False)
            self._warm_head()
        except Exception as error:  # noqa: BLE001 - an uncompiled actor is still correct.
            log.warning("action head left uncompiled: %s: %s", type(error).__name__, error)
            self.policy.actor = getattr(self.policy.actor, "_orig_mod", self.policy.actor)
            return False
        return True

    def _warm_head(self):
        """Pay compilation now, in act's exact context, without touching the match RNG.

        Both halves matter. Dynamo guards on the tensor dispatch key set, so warming
        outside inference_mode compiles a graph the first live tick then discards, and
        the tick that was supposed to be cheap pays for two. And the warm-up samples:
        drawing from the match generator here would make a compiled actor produce a
        different action sequence than an uncompiled one from the same seed, which is
        exactly the reproducibility seed_everything exists to provide.
        """
        memory = torch.zeros(1, self.policy.memory_dim, device=self.device)
        with torch.random.fork_rng(devices=[self.device]):
            for _ in range(3):
                with (
                    torch.inference_mode(),
                    torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16),
                ):
                    self.policy.actor(
                        memory,
                        noise=torch.zeros(1, self.policy.actor.noise_dim, device=self.device),
                        deterministic=self.deterministic,
                    )
        torch.cuda.synchronize()

    def act(self, rgb, timestamp, precomputed=None):
        device = self.device
        # The worker downscales on the capture side when it can, which keeps a 33 MB
        # frame off the wire and the resize out of this loop entirely. Fall back to
        # resizing here, on the GPU, when it handed back a full frame instead.
        global_view, tiles = precomputed if precomputed is not None else views(rgb, device=device)
        # The worker hands back numpy; the local fallback hands back device tensors.
        global_view = torch.as_tensor(global_view, device=device)
        tiles = torch.as_tensor(tiles, device=device)
        self.history.append((timestamp, global_view))
        times = np.array([t for t, _ in self.history])
        desired = timestamp - np.arange(15, -1, -1) / 7.5
        ids = np.searchsorted(times, desired, side="right") - 1
        clip = torch.stack([self.history[max(0, int(i))][1] for i in ids])
        before = (
            np.zeros(self.policy.memory_dim, np.float32)
            if self.hidden is None
            else self.hidden[0].float().cpu().numpy()
        )
        with (
            torch.inference_mode(),
            torch.autocast(torch.device(device).type, dtype=torch.bfloat16),
        ):
            self.hidden, value, _ = self.policy(
                normalize(clip).permute(3, 0, 1, 2)[None],
                normalize(tiles).permute(0, 3, 1, 2)[None],
                torch.from_numpy(self.previous)[None].to(device),
                self.hidden,
            )
            noise = act_noise(
                self.config["objective"],
                self.policy.actor.noise_dim,
                self.deterministic,
                device,
            )
            action, logp, _ = self.policy.actor(
                self.hidden, noise=noise, deterministic=self.deterministic
            )
        action = action[0].cpu().numpy()
        sample = {
            "clip": clip.cpu().numpy(),
            "tiles": tiles.cpu().numpy(),
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
    # Salt with the pair id so each match is independently seeded yet reproducible.
    seed = seed_everything(config.get("seed", 42), salt=config["pair_id"])
    deterministic = bool(config.get("deterministic", False))
    # Native-resolution audit video costs a 33 MB frame per tick per side. Off by
    # default: the worker then sends only the policy views and template crops, and the
    # audit video records the global view the policy actually saw.
    record_full = bool(config.get("record_full", False))
    actors = [
        Actor(left_checkpoint, config.get("model_path"), deterministic=deterministic),
        Actor(right_checkpoint, config.get("model_path"), deterministic=deterministic),
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
        "seed": seed,
        "config_seed": config.get("seed", 42),
        "deterministic": deterministic,
        "record_full": record_full,
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
                    downscale=config.get("downscale", True),
                )
            )
        for env in environments:
            env.record_full = record_full
        pair = ArenaPair(*environments)
        pair.reset()
        for index, env in enumerate(environments):
            recorder = Recorder(
                root / f"player-{index}" / "recording", env.last, source="policy", hz=5
            )
            recorder.append(env.last)
            recorders.append(recorder)
            env.recorder = recorder
        while True:
            actions = []
            samples = []
            # Read the Frame rather than the returned observation: it carries the views
            # the worker already downscaled, so no full frame is needed or resized here.
            for actor, env in zip(actors, environments, strict=True):
                action, sample = actor.act(
                    env.last.rgb, env.last.meta["t_ns"] / 1e9, precomputed=env.last.views
                )
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
            if count % 25 == 0:
                misses = sum(bool(r[4].get("deadline_miss")) for r in results)
                log.info(
                    "step %d: elapsed=%.1f s deadline_misses_this_step=%d",
                    count,
                    max(r[4].get("elapsed_seconds", 0.0) for r in results),
                    misses,
                )
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
        for index, desktop in enumerate(desktops):
            try:
                lines = desktop.worker_log()
                if lines:
                    (root / f"worker-{index}.log").write_text("\n".join(lines) + "\n")
            except Exception as error:  # noqa: BLE001 - diagnostics must never mask cleanup.
                log.warning("could not write worker-%d.log: %s", index, error)
            try:
                desktop.close()
                # close() records a failed release instead of raising out of __exit__;
                # an unreleased worker still invalidates the run.
                if desktop.close_error:
                    reason = reason or f"Worker release failed: {desktop.close_error}"
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
        # collect_pair re-raises, so the caller never sees the return value. Say where the
        # evidence landed rather than leaving only a traceback.
        log.info("manifest written: %s (valid=%s)", root / "manifest.json", manifest["valid"])
    # Same contract as record: the evidence is written first, then the run fails loudly.
    # A permanently miscalibrated rules file otherwise invalidates every episode in an
    # unattended batch while every invocation still exits zero.
    if not manifest["valid"]:
        raise RuntimeError(
            f"Match invalid ({manifest.get('error') or manifest.get('outcomes')}); "
            f"see {root / 'manifest.json'}"
        )
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


def train_ppo(
    rollouts, checkpoint, output, epochs=3, sequence=8, burn_in=2, model_path=None, seed=42
):
    from .learning import gae, ppo_loss, save_checkpoint

    seed = seed_everything(seed)
    policy, config, digest = load_policy(checkpoint, model_path)
    policy.train()
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-5)
    episodes = []
    for manifest_path in sorted(Path(rollouts).glob("*/manifest.json")):
        meta = json.loads(manifest_path.read_text())
        excluded = ppo_exclusion(meta)
        if excluded:
            log.info("excluding rollout %s: %s", manifest_path.parent, excluded)
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
            "seed": seed,
            "epochs": epochs,
            "sequence": sequence,
            "burn_in": burn_in,
            "gameplay_verified": False,
        },
    )
    return {
        "episodes": len(episodes),
        "updates": len(losses),
        "mean_loss": float(np.mean(losses)),
        "seed": seed,
        "selection_requires_held_out_games": True,
    }

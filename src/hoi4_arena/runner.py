"""Direct recurrent-policy collection. Match weights are immutable until collection ends."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .actions import GRID, SLOTS
from .dataset import CLIP_FRAMES, clip_frame_ids, normalize, recorded_speed, views
from .desktop import Desktop
from .environment import ArenaEnv, ArenaPair
from .learning import approximate_kl, file_hash, value_estimate
from .models import CELL_DIM, Policy, build_encoder, halve_frozen
from .recording import Recorder
from .remote import RemoteDesktop
from .vision import ScreenRules

log = logging.getLogger(__name__)

# Which views a rollout stored. 1: global view and five 224 px tiles. 2: global view, four
# 448 px quadrants and a 224 px fovea, and the game speed. 3: the same views at 16:9, a
# 448x256 global view and 576x320 quadrants.
OBSERVATION = 3


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

    A rollout collected under a different clip length is excluded for the same reason,
    and it is the quieter failure of the two. The encoder positions tokens with RoPE, so
    a sixteen-frame clip replays through an eight-frame policy without complaint -- the
    shapes fit, nothing raises, and the ratio is simply computed against a likelihood
    from an observation the current policy never sees. Rollouts predating the field were
    all sixteen.
    """
    if not meta.get("complete"):
        return "incomplete"
    if not meta.get("valid"):
        return "invalid"
    if meta.get("deterministic"):
        return "deterministic evaluation rollout, not on-policy"
    frames = meta.get("clip_frames", 16)
    if frames != CLIP_FRAMES:
        return f"collected against a {frames}-frame clip; the policy now reads {CLIP_FRAMES}"
    layout = meta.get("observation", 1)
    if layout != OBSERVATION:
        return f"observation layout {layout}; the policy now reads layout {OBSERVATION}"
    return None


def load_policy(checkpoint, model_path=None, device="cuda"):
    path = Path(checkpoint)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if file_hash(path) != metadata["sha256"]:
        raise ValueError("Checkpoint does not match its immutable manifest")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = saved["config"]
    encoder = build_encoder(model_path or config["model_path"], config["variant"])
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
        self,
        checkpoint,
        model_path=None,
        deterministic=False,
        device="cuda",
        compile_head=True,
        *,
        game_speed,
    ):
        self.policy, self.config, self.digest = load_policy(checkpoint, model_path, device)
        # The policy is told the speed the match runs at: the same clip is a different
        # amount of game time at each one.
        self.speed = recorded_speed(game_speed)["game_speed"]
        self.policy.eval().requires_grad_(False)
        # Evaluation runs take the argmax so paired_evaluation's bound is not inflated by
        # sampling noise the analysis does not model. Self-play collection must sample.
        self.deterministic = deterministic
        self.device = device
        self.hidden = None
        self.previous = np.zeros((SLOTS, 3), dtype=np.int64)
        self.history = deque(maxlen=64)
        self.compiled = compile_head and device == "cuda" and self._compile_head()

    def reset_episode(self):
        """Drop the GRU state and the clip. A second match on this actor must not see the first."""
        self.hidden = None
        self.previous = np.zeros((SLOTS, 3), dtype=np.int64)
        self.history.clear()

    def _compile_head(self):
        """Capture the action head into a CUDA graph, and prove it before trusting it.

        The head is eight fixed-length slots of very small tensors, so nearly all of its
        cost is launching them: 7.72 ms of work that measures 0.741 ms once replayed from
        a graph. The encoder is deliberately left alone. Compiling it is worth about
        24 ms, but it moves a stored old_logp by 4.6e-4 and there is no configuration
        that removes the gap -- inductor's no-grad graph and its grad graph differ by
        3.4e-4 from each other, so compiling PPO's side too does not fix it. Eager
        collection and eager update agree exactly today, and that property is worth more
        than the milliseconds. The head is a different case: compiled, it measured
        bit-for-bit identical to the eager head on this GPU in both modes, so it costs
        nothing at all. Stated with the machine attached deliberately -- the sibling
        claim about the fused head held here and failed on CI's CPU.

        Returns whether compilation took, because inductor needs Triton and a failure
        here must cost latency rather than the match.
        """
        try:
            self.policy.actor.compile(mode="reduce-overhead", dynamic=False)
            self._warm_head()
        except Exception as error:  # noqa: BLE001 - an uncompiled actor is still correct.
            self._uncompile(f"{type(error).__name__}: {error}")
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
        cells = torch.zeros(1, GRID, CELL_DIM, device=self.device)
        # Inductor keeps its cudagraph tree manager in thread-local storage, so a graph
        # captured here can only be replayed from this thread. `collect_pair` calls act
        # from the main thread and only hands capture and dispatch to a pool, so the
        # invariant holds -- but it is invisible in the code that depends on it, and
        # violating it raises an AssertionError from inside inductor rather than
        # anything a match loop would recognize. Recorded so act can check it.
        self.graph_thread = threading.get_ident()
        with torch.random.fork_rng(devices=[self.device]):
            for _ in range(3):
                with (
                    torch.inference_mode(),
                    torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16),
                ):
                    self.policy.actor(
                        memory,
                        cells,
                        noise=torch.zeros(1, self.policy.actor.noise_dim, device=self.device),
                        deterministic=self.deterministic,
                    )
        torch.cuda.synchronize()

    def _uncompile(self, reason):
        """Drop back to the eager head, keeping the match alive."""
        log.warning("running the action head eagerly: %s", reason)
        self.policy.actor = getattr(self.policy.actor, "_orig_mod", self.policy.actor)
        self.compiled = False

    def act(self, rgb, timestamp_ns, precomputed=None, cursor=None):
        device = self.device
        # A replay from the wrong thread would take the match down with an assertion
        # from inside inductor. Losing the graph costs about seven milliseconds a tick;
        # losing the match costs the match.
        if self.compiled and threading.get_ident() != self.graph_thread:
            self._uncompile("the cuda graph was captured on another thread")
        # The worker downscales on the capture side when it can, which keeps a 33 MB
        # frame off the wire and the resize out of this loop entirely. Fall back to
        # resizing here, on the GPU, when it handed back a full frame instead. The
        # fallback still needs the pointer, because the fovea is centred on it.
        seen = precomputed if precomputed is not None else views(rgb, device=device, cursor=cursor)
        # The worker hands back numpy; the local fallback hands back device tensors.
        global_view, quads, fovea = (torch.as_tensor(v, device=device) for v in seen)
        # Same lookback Sessions uses. Integer nanoseconds: dividing t_ns by 1e9 and
        # stepping in float seconds would not land on the same frames.
        timestamp_ns = int(timestamp_ns)
        self.history.append((timestamp_ns, global_view))
        times = np.array([t for t, _ in self.history], dtype=np.int64)
        ids = clip_frame_ids(times, timestamp_ns)
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
            self.hidden, value, _, cells = self.policy(
                normalize(clip).permute(3, 0, 1, 2)[None],
                normalize(quads).permute(0, 3, 1, 2)[None],
                normalize(fovea).permute(2, 0, 1)[None],
                torch.from_numpy(self.previous)[None].to(device),
                torch.tensor([self.speed], device=device),
                self.hidden,
            )
            noise = act_noise(
                self.config["objective"],
                self.policy.actor.noise_dim,
                self.deterministic,
                device,
            )
            action, logp, entropy = self.policy.actor(
                self.hidden, cells, noise=noise, deterministic=self.deterministic
            )
        # One host transfer for the whole sample. Four separate .cpu()/.item() calls
        # each waited for the GPU, on the same thread that has to start the next capture.
        host = {
            "clip": clip.detach().to("cpu", non_blocking=True),
            "quadrants": quads.detach().to("cpu", non_blocking=True),
            "fovea": fovea.detach().to("cpu", non_blocking=True),
            "action": action[0].detach().to("cpu", non_blocking=True),
            "old_logp": logp.detach().to("cpu", non_blocking=True),
            "old_entropy": entropy.float().detach().to("cpu", non_blocking=True),
            "old_value": value_estimate(value).detach().to("cpu", non_blocking=True),
            "noise": noise[0].detach().to("cpu", non_blocking=True),
        }
        if torch.device(device).type == "cuda":
            torch.cuda.current_stream().synchronize()
        action_np = host["action"].numpy()
        sample = {
            "clip": host["clip"].numpy(),
            "quadrants": host["quadrants"].numpy(),
            "fovea": host["fovea"].numpy(),
            "speed": np.int64(self.speed),
            "hidden": before,
            "previous": self.previous.copy(),
            "action": action_np,
            "old_logp": float(host["old_logp"]),
            # The collecting policy's entropy here: InfoPPO's information density.
            "old_entropy": float(host["old_entropy"]),
            "old_value": float(host["old_value"]),
            "noise": host["noise"].numpy(),
        }
        self.previous = action_np
        return action_np, sample


def commit_transition(held, result):
    """Attach this step's observation to the action that just finished.

    step() captures at the start of the action it was given, after joining the
    previous interval. The reward on that frame was produced by the action already
    in flight, not by the one that is about to start.
    """
    if held is None:
        return None
    _observation, reward, terminated, truncated, info = result
    finished = dict(held)
    finished.update(
        reward=reward,
        terminal=bool(terminated or truncated),
        valid=bool(info["valid"]),
        elapsed=float(info.get("elapsed_seconds", 0.0)),
    )
    return finished


class TransitionWriter:
    """Compress rollout samples off the decision thread.

    np.savez_compressed of an eight-frame clip is the Python work that misses the
    interval. Actor.act builds every array fresh, so nothing is copied here. The queue
    is bounded: a writer that falls behind blocks the next save, and the stall shows
    as a deadline miss instead of growing memory by ~20 MB/s until the match dies.
    """

    def __init__(self, backlog=64):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rollout-save")
        self.slots = threading.BoundedSemaphore(backlog)
        self.pending = []

    def save(self, path, sample):
        self.slots.acquire()
        job = self.pool.submit(np.savez_compressed, path, **sample)
        job.add_done_callback(lambda _job: self.slots.release())
        self.pending.append(job)

    def finish(self):
        errors = []
        for job in self.pending:
            try:
                job.result()
            except Exception as error:  # noqa: BLE001 - reported with the manifest.
                errors.append(error)
        self.pool.shutdown(wait=True)
        if errors:
            raise errors[0]


def collect_pair(config_path, output, left_checkpoint, right_checkpoint):
    """Prototype coordinator: both actors infer here; second PC supplies its own pixels.

    This mode must pass its measured throughput gate before unattended acceptance.
    Model weights are loaded once; neither actor gets its opponent's observations.
    """
    config = json.loads(Path(config_path).read_text())
    # Before the run directory exists. A match with no recorded speed cannot grow one,
    # and the documented default of 2 is not what the long runs used.
    speed = recorded_speed(config.get("game_speed"))
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
        Actor(
            checkpoint,
            config.get("model_path"),
            deterministic=deterministic,
            game_speed=speed["game_speed"],
        )
        for checkpoint in (left_checkpoint, right_checkpoint)
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
        "clip_frames": CLIP_FRAMES,
        "observation": OBSERVATION,
        **speed,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    reason = None
    count = 0
    writer = None
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
                    # West is Blue and east is Red unless the pair says otherwise.
                    # The territory reward is that country's own share.
                    country=spec.get("country", "BLU" if side == "left" else "RED"),
                    # The arena mod reports every match in game.log, so that is what
                    # scores it. A vanilla lobby has no mod and sets "screen".
                    reward=config.get("reward", "log"),
                )
            )
        for env in environments:
            env.record_full = record_full
        for actor in actors:
            actor.reset_episode()
        pair = ArenaPair(*environments)
        pair.reset()
        writer = TransitionWriter()
        held = [None, None]
        for index, env in enumerate(environments):
            recorder = Recorder(
                root / f"player-{index}" / "recording",
                env.last,
                source="policy",
                hz=5,
                game_speed=speed["game_speed"],
            )
            recorder.append(env.last)
            recorders.append(recorder)
            env.recorder = recorder
        tick = 0
        while True:
            actions = []
            samples = []
            # Read the Frame rather than the returned observation: it carries the views
            # the worker already downscaled, so no full frame is needed or resized here.
            for actor, env in zip(actors, environments, strict=True):
                action, sample = actor.act(
                    env.last.rgb,
                    env.last.meta["t_ns"],
                    precomputed=env.last.views,
                    cursor=env.last.meta.get("cursor"),
                )
                actions.append(action)
                samples.append(sample)
            results = pair.step(actions)
            saved = False
            for i, (sample, result) in enumerate(zip(samples, results, strict=True)):
                finished = commit_transition(held[i], result)
                if finished is not None:
                    writer.save(root / f"player-{i}" / f"{count:06d}.npz", finished)
                    saved = True
                # The action just started has not had its effect captured yet. A terminal
                # or invalid step aborts that action, so it is not a training sample.
                ended = bool(result[2] or result[3]) or not result[4]["valid"]
                held[i] = None if ended else sample
            if saved:
                count += 1
            tick += 1
            if tick % 25 == 0:
                misses = sum(bool(r[4].get("deadline_miss")) for r in results)
                log.info(
                    "step %d: elapsed=%.1f s deadline_misses_this_step=%d",
                    tick,
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
        if writer is not None:
            try:
                writer.finish()
            except Exception as error:  # noqa: BLE001 - the manifest has to record it.
                reason = reason or f"Rollout save failed: {error}"
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
    """Stored rollout steps as one batch, normalized the way Actor.act normalized them."""
    rows = [dict(np.load(path)) for path in files]
    missing = {"quadrants", "fovea", "speed"} - rows[0].keys()
    if missing:
        raise ValueError(f"rollout step predates observation layout {OBSERVATION}: {missing}")
    batch = {
        name: torch.as_tensor(np.stack([r[name] for r in rows]), device=device) for name in rows[0]
    }
    batch["clips"] = normalize(batch.pop("clip")).permute(0, 4, 1, 2, 3)
    batch["quadrants"] = normalize(batch["quadrants"]).permute(0, 1, 4, 2, 3)
    batch["fovea"] = normalize(batch["fovea"]).permute(0, 3, 1, 2)
    return batch


def _replay(policy, batch, burned, device, *, policy_grad=True):
    """Run stored steps through the policy, warming the memory on the first `burned`.

    Returns the scored steps' memories, value logits, log-likelihoods of the stored
    actions and entropies. `policy_grad` False keeps the whole pass out of the graph.
    """
    hidden = batch["hidden"][0:1].float()
    hiddens, values, logps, entropies = [], [], [], []
    for t in range(batch["clips"].shape[0]):
        burn = t < burned
        with (
            torch.set_grad_enabled(policy_grad and not burn),
            torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"),
        ):
            step = slice(t, t + 1)
            hidden, value, _, cells = policy(
                batch["clips"][step],
                batch["quadrants"][step],
                batch["fovea"][step],
                batch["previous"][step],
                batch["speed"][step],
                hidden,
            )
            if burn:
                hidden = hidden.detach()
                continue
            _, logp, entropy = policy.actor(
                hidden, cells, batch["action"][step], noise=batch["noise"][step]
            )
            hiddens.append(hidden)
            values.append(value)
            logps.append(logp)
            entropies.append(entropy)
    return torch.cat(hiddens), torch.cat(values), torch.cat(logps), torch.cat(entropies)


def _windows(episodes, sequence, burn_in, rng):
    """Every (episode, start, begin, end) window of every episode, in shuffled order."""
    for index in rng.permutation(len(episodes)):
        files = episodes[int(index)][0]
        starts = list(range(0, len(files), sequence))
        rng.shuffle(starts)
        for start in starts:
            yield int(index), start, max(0, start - burn_in), min(start + sequence, len(files))


def train_ppo(
    rollouts,
    checkpoint,
    output,
    epochs=3,
    sequence=8,
    burn_in=4,
    model_path=None,
    seed=42,
    kl_limit=0.02,
    gae_lambda=None,
    critic="pact",
    critic_epochs=1,
    ratio_range=(0.0, 6.0),
    clock="ticks",
    clip="fixed",
    info_clip=None,
):
    """Recurrent PPO. Windows are shuffled, and training stops once the KL leaves the trust region.

    `burn_in` refreshes the GRU from the behavior policy's stored state. Two steps were
    not enough once the encoder's last blocks had moved; four is the default. The first
    window whose approximate KL exceeds `kl_limit` ends every remaining epoch, not just
    the current one.

    `critic` chooses how the value head learns:

    - "pact" (Fu et al., 2026): the actor is updated first, without a value term. Then
      each stored step is replayed under the updated policy, and the value head alone is
      trained toward the return weighted by that step's likelihood ratio between the
      updated and the collecting policy, so it estimates the value of the policy just
      trained rather than of the one before it. Ratios outside `ratio_range` are left out.
      Only the head moves in that phase: the memory it reads is the actor's.
    - "joint": the usual PPO, a value term in the same loss, critic one update behind.

    Both train the critic with binary cross-entropy on the scaled return (learning).

    InfoPPO (Zeng et al., 2026) measures each step by the collecting policy's entropy
    there, its information density. `clock="information"` discounts over that instead of
    ticks, so the many ticks a policy confidently spends waiting cost no horizon, and
    `clip="adaptive"` lets the ratio move further where the policy was unsure and less
    where it was sure (`info_clip`, default learning.INFO_CLIP). Their gains were on
    language models; both are off by default until self-play can compare them.
    """
    from .learning import (
        GAE_LAMBDA,
        INFO_CLIP,
        adaptive_clip,
        critic_loss,
        gae,
        information_density,
        normalize_advantages,
        ppo_loss,
        save_checkpoint,
    )

    if critic not in {"pact", "joint"}:
        raise ValueError("critic must be 'pact' or 'joint'")
    if clock not in {"ticks", "information"} or clip not in {"fixed", "adaptive"}:
        raise ValueError("clock is 'ticks' or 'information', clip 'fixed' or 'adaptive'")
    info_clip = tuple(INFO_CLIP if info_clip is None else info_clip)
    informed = clock == "information" or clip == "adaptive"
    keys = ["reward", "old_value", "terminal", "valid", "elapsed"]
    keys += ["old_entropy"] if informed else []
    gae_lambda = GAE_LAMBDA if gae_lambda is None else gae_lambda
    output = Path(output)
    if output.exists():
        raise FileExistsError("Checkpoints are immutable")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = seed_everything(seed)
    policy, config, digest = load_policy(checkpoint, model_path, device)
    policy.train()
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-5)
    episodes, pending = [], []
    speeds = []
    for manifest_path in sorted(Path(rollouts).glob("*/manifest.json")):
        meta = json.loads(manifest_path.read_text())
        excluded = ppo_exclusion(meta)
        if excluded:
            log.info("excluding rollout %s: %s", manifest_path.parent, excluded)
            continue
        on_policy = [player for player in range(2) if meta["checkpoint_sha256"][player] == digest]
        if not on_policy:
            continue  # Historical opponents are not on-policy training data.
        # A rollout that would train with no recorded speed is a data error, not a skip.
        # Each step also stores the speed the policy was told, so matches at different
        # speeds can train together.
        speed = recorded_speed(meta.get("game_speed"))["game_speed"]
        for player in on_policy:
            files = sorted((manifest_path.parent / f"player-{player}").glob("*.npz"))
            scalars = []
            for path in files:
                with np.load(path) as archive:
                    if informed and "old_entropy" not in archive:
                        raise ValueError(
                            f"{path} was collected before steps stored the policy's entropy,"
                            " which the information clock and adaptive clip need"
                        )
                    scalars.append({key: archive[key] for key in keys})
            if not scalars or not scalars[-1]["terminal"]:
                raise ValueError("Incomplete episode cannot enter PPO")
            columns = {k: torch.as_tensor(np.array([r[k] for r in scalars])) for k in keys}
            pending.append((files, columns))
            speeds.append(speed)
    if not pending:
        raise ValueError("No completed valid on-policy episodes; collection must precede PPO")
    # The density is normalized over the whole update, so it waits for every episode.
    densities = (
        information_density([columns["old_entropy"] for _, columns in pending])
        if informed
        else [None] * len(pending)
    )
    for (files, columns), density in zip(pending, densities, strict=True):
        advantages, returns = gae(
            columns["reward"],
            columns["old_value"],
            torch.tensor(0.0),
            columns["terminal"],
            columns["valid"],
            columns["elapsed"],
            lam=gae_lambda,
            density=density if clock == "information" else None,
        )
        bounds = adaptive_clip(density, info_clip) if clip == "adaptive" else None
        episodes.append((files, normalize_advantages(advantages), returns.float(), bounds))
    game_speed = sorted(set(speeds))
    losses, critic_losses = [], []
    kept = total = 0
    rng = np.random.default_rng(seed)
    early_stop = False
    for _ in range(epochs):
        for index, start, begin, end in _windows(episodes, sequence, burn_in, rng):
            files, advantages, returns, bounds = episodes[index]
            if bounds is not None:
                bounds = tuple(b[start:end].to(device) for b in bounds)
            batch = replay_batch(files[begin:end], device)
            optimizer.zero_grad(set_to_none=True)
            _, values, new_logp, entropies = _replay(policy, batch, start - begin, device)
            old_logp = batch["old_logp"][start - begin :].float()
            kl = approximate_kl(new_logp.detach(), old_logp)
            if float(kl) > kl_limit:
                log.info("stopping PPO, approximate KL %.4f", float(kl))
                early_stop = True
                break
            loss = ppo_loss(
                new_logp,
                old_logp,
                values if critic == "joint" else None,
                returns[start:end].to(device),
                advantages[start:end].to(device),
                entropies,
                bounds=bounds,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        if early_stop:
            break
    if critic == "pact":
        head = torch.optim.AdamW(policy.value.parameters(), lr=1e-4)
        for _ in range(critic_epochs):
            for index, start, begin, end in _windows(episodes, sequence, burn_in, rng):
                files, _, returns, _ = episodes[index]
                batch = replay_batch(files[begin:end], device)
                with torch.no_grad():
                    hiddens, _, new_logp, _ = _replay(
                        policy, batch, start - begin, device, policy_grad=False
                    )
                # The current step's ratio only, detached: the whole continuation's
                # product is exact but its variance grows with every step of a long match.
                ratio = (new_logp.float() - batch["old_logp"][start - begin :].float()).exp()
                inside = (ratio >= ratio_range[0]) & (ratio <= ratio_range[1])
                kept += int(inside.sum())
                total += len(ratio)
                if not inside.any():
                    continue
                head.zero_grad(set_to_none=True)
                logits = policy.value(hiddens.float()).squeeze(-1)
                per_step = critic_loss(logits, returns[start:end].to(device), weight=ratio)
                loss = (per_step * inside).sum() / inside.sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite critic objective")
                loss.backward()
                head.step()
                critic_losses.append(loss.item())
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
            "game_speeds": game_speed,
            "epochs": epochs,
            "sequence": sequence,
            "burn_in": burn_in,
            "kl_limit": kl_limit,
            "gae_lambda": gae_lambda,
            "critic": critic,
            "clock": clock,
            "clip": clip,
            "info_clip": list(info_clip) if clip == "adaptive" else None,
            "gameplay_verified": False,
        },
    )
    return {
        "episodes": len(episodes),
        "updates": len(losses),
        "mean_loss": float(np.mean(losses)) if losses else None,
        "critic": critic,
        "clock": clock,
        "clip": clip,
        "critic_updates": len(critic_losses),
        "mean_critic_loss": float(np.mean(critic_losses)) if critic_losses else None,
        "critic_ratio_kept": kept / total if total else None,
        "early_stop": early_stop,
        "seed": seed,
        "selection_requires_held_out_games": True,
    }

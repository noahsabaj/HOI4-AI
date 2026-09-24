from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Discount per 200 ms of wall time.
GAMMA = 0.9999
# GAE's lambda. One, not the customary 0.95. A match is thousands of decisions scored
# mostly at its end, so the true credit of most decisions is near zero (Fu et al., 2026,
# "PACT", Theorem 4: the expected number of credits larger than e is at most 1/(4e^2) for
# a reward in [0, 1], however long the episode). With lambda below one every intermediate
# value error is carried into the advantage and can swamp that credit; at one they cancel
# and only the value error at the step itself remains. Their PPO at 0.95 collapsed where
# 1.0 did not.
GAE_LAMBDA = 1.0
# Every return this project can produce lies within plus or minus this: a win or a loss
# is 1, and the log's shaping sums to the change in a potential that stays within 2 either
# side (arena_log: surrender progress difference within 1, half a state share within 1).
RETURN_BOUND = 5.0
# InfoPPO's adaptive clip (Zeng et al., 2026, Appendix B): the ratio may move within
# [1 / (1 + log(1 + low * rho)), 1 + log(1 + high * rho)], rho the step's information
# density in [0, 1].
INFO_CLIP = (10.0, 20.0)


def scale_return(returns):
    """A return mapped into [0, 1], the range the critic predicts in."""
    return (returns + RETURN_BOUND) / (2 * RETURN_BOUND)


def value_estimate(value_logit):
    """The critic's output, a logit, as a return."""
    return torch.sigmoid(value_logit.float()) * (2 * RETURN_BOUND) - RETURN_BOUND


def critic_loss(value_logit, target, weight=None):
    """Binary cross-entropy against a target in return units, per step.

    The critic predicts the scaled return through a sigmoid and is trained with BCE, not
    squared error. For a target in [0, 1] both have the same minimizer, the conditional
    mean, but in PACT's controlled comparison the BCE critic converged faster and told
    winning from losing trajectories apart better. `weight` multiplies the target, for
    the importance-corrected targets of the critic phase; the conditional mean it
    recovers is still that of the weighted target, which is the point.
    """
    scaled = scale_return(target.float())
    if weight is not None:
        scaled = scaled * weight
    return F.binary_cross_entropy_with_logits(value_logit.float(), scaled, reduction="none")


def information_density(entropies):
    """Each step's entropy under the collecting policy over the largest in the update.

    InfoPPO's clock (Zeng et al., 2026): a step advances time by how uncertain the policy
    was there, in [0, 1], rather than by one. A slot's entropy counts where to point only
    in proportion to the chance of moving at all, so a tick the policy confidently spends
    waiting is close to zero and a real decision is not. Normalized over the whole update,
    the paper's "batch level", which was steadier than per episode. A list of per-episode
    tensors in, a list out.
    """
    largest = max((float(e.max()) for e in entropies if e.numel()), default=0.0)
    if largest <= 0:
        return [torch.zeros_like(e, dtype=torch.float32) for e in entropies]
    return [(e.float() / largest).clamp(0, 1) for e in entropies]


def adaptive_clip(density, info_clip=INFO_CLIP):
    """InfoPPO's per-step ratio bounds: wide where the policy was unsure, tight where not.

    A softmax policy's total-variation move from one gradient step is bounded by its
    entropy (their Proposition 4.3), so an unsure state can move further for the same
    guarantee. Logarithmic in the density, their conservative choice over linear.
    """
    low, high = info_clip
    return 1 / (1 + torch.log1p(low * density)), 1 + torch.log1p(high * density)


def gae(
    rewards,
    values,
    bootstrap,
    terminated,
    valid,
    elapsed,
    gamma=GAMMA,
    lam=GAE_LAMBDA,
    density=None,
):
    """Discount actual wall time in units of 200ms. Never learn from invalid episodes.

    `density` (information_density) makes time information time instead: each step's
    discount and trace decay are raised to its density, so waiting costs no horizon and
    a match is discounted over its decisions (InfoPPO, their Eq. 23).
    """
    if not valid.all():
        raise ValueError("Invalid episodes must be quarantined, not assigned zero reward")
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(bootstrap)
    next_value = bootstrap
    for t in reversed(range(rewards.shape[0])):
        continuation = (~terminated[t]).float()
        clock = 1.0 if density is None else density[t]
        discount = gamma ** (elapsed[t] / 0.2 * clock)
        delta = rewards[t] + discount * next_value * continuation - values[t]
        carry = delta + discount * lam**clock * continuation * carry
        advantages[t] = carry
        next_value = values[t]
    return advantages, advantages + values


def normalize_advantages(advantages):
    """One scale for the whole episode.

    An 8-step window of a flat reward has a tiny standard deviation. Dividing by that
    window turns the noise into a unit-scale advantage, and the ratio then trains on it.
    A constant episode has no relative advantage; leave it at zero rather than dividing
    by a floor. A single transition is already the whole episode: centering it would
    erase the only reward, including a one-step win. The divisor is floored at 1e-2, so
    a spread too small to be a terminal return stays proportionally small and flicker
    cannot grow to 1, and the scale is continuous across the floor.
    """
    advantages = advantages.float()
    if advantages.numel() < 2:
        return advantages.clone()
    scale = advantages.std(unbiased=False)
    if not torch.isfinite(scale) or float(scale) < 1e-6:
        return torch.zeros_like(advantages)
    centered = advantages - advantages.mean()
    return centered / max(float(scale), 1e-2)


def approximate_kl(logp, old_logp):
    """Schulman's non-negative approximation of the mean KL from the behavior policy."""
    log_ratio = logp - old_logp
    return (log_ratio.exp() - 1 - log_ratio).mean()


def ppo_loss(logp, old_logp, value_logits, returns, advantages, entropy, clip=0.2, bounds=None):
    """Score one window. Advantages are already scaled over the episode; do not rescale.

    `value_logits` None leaves the critic out: the actor-then-critic mode trains it in a
    phase of its own, after the actor. `bounds`, per-step (low, high) from adaptive_clip,
    replace the fixed [1 - clip, 1 + clip].
    """
    ratio = (logp - old_logp).exp()
    if bounds is None:
        clipped = ratio.clamp(1 - clip, 1 + clip)
    else:
        clipped = torch.minimum(torch.maximum(ratio, bounds[0]), bounds[1])
    policy = -torch.minimum(ratio * advantages, clipped * advantages).mean()
    loss = policy - 0.001 * entropy.mean()
    if value_logits is not None:
        loss = loss + 0.5 * critic_loss(value_logits, returns).mean()
    return loss


def save_checkpoint(path, policy, config, *, auxiliary=None, optimizer=None, provenance=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Checkpoints are immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"policy": policy.state_dict(), "config": config, "provenance": provenance or {}}
    if auxiliary is not None:
        payload["auxiliary"] = auxiliary.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)
    digest = file_hash(path)
    path.with_suffix(".json").write_text(
        json.dumps({"sha256": digest, "config": config, "provenance": provenance}, indent=2)
    )
    return digest


def replace_patiently(source, target, tries=10, wait=1.0):
    """Rename `source` over `target`, retrying while Windows reports the target in use.

    A scanner opening the freshly written file (or another reader) makes the rename fail
    with WinError 32 for a moment; on 2026-09-24 that ended a training run at a routine
    save. A few retries a second apart ride it out; a lasting lock still raises.
    """
    for attempt in range(tries):
        try:
            Path(source).replace(target)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(wait)


class RunLock:
    """A folder's lock for one run at a time: a second copy of a run refuses to start.

    `run.lock` holds the owner's process id. A lock whose process has gone (a killed run)
    is taken over. On 2026-09-24 a double launch ran two trainings into one folder, and one
    died when the other held its progress file.
    """

    def __init__(self, folder, name="run.lock"):
        self.path = Path(folder) / name

    def __enter__(self):
        import os

        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    owner = int(self.path.read_text().strip() or 0)
                except (OSError, ValueError):
                    owner = 0
                if owner and _alive(owner):
                    raise RuntimeError(
                        f"another run (process {owner}) holds {self.path}; one run per folder"
                    ) from None
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(handle, "w") as out:
                out.write(str(os.getpid()))
            return self
        raise RuntimeError(f"could not take {self.path}")

    def __exit__(self, *_):
        self.path.unlink(missing_ok=True)


def _alive(pid):
    """Whether a process with this id is running (Windows and elsewhere)."""
    import os

    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class Progress:
    """A training run's position and state, saved as it goes so an interruption costs minutes.

    On 2026-09-23 a session restart killed a behaviour-cloning run at step 2141 of about
    2300, 72 minutes in, and nothing was kept: checkpoints were written only at the end of
    an epoch. Now `progress.pt` in the run's folder holds the model, the optimizer, the
    random state and where the run was, rewritten every `every` seconds of training and
    after every epoch, and removed when the run finishes. Resuming loads it and skips the
    batches already trained. The data comes back in the same order, because the windows
    are shuffled by seed and epoch alone (VideoSessions), so the skipped batches are
    exactly the ones trained before. They are still decoded to be skipped, which is the
    cost of resuming: a few minutes, not the run.

    The config must match the saved one, so a resume cannot quietly continue a different
    run. `every` 0 saves only after each epoch.
    """

    def __init__(self, output, config, *, every=600.0, resume=False, clock=time.monotonic):
        self.path = Path(output) / "progress.pt"
        self.config, self.every, self.resume, self.clock = config, every, resume, clock
        self.last = clock()
        if not resume and self.path.exists():
            raise FileExistsError(
                f"A run in progress is saved in {self.path.parent}: resume it (--resume) "
                "or train into another folder"
            )
        if resume and not self.path.exists():
            raise FileNotFoundError(f"No run in progress to resume in {self.path.parent}")

    def start(self, modules, optimizer):
        """Where to begin, (epoch, batches to skip), with the saved state loaded if resuming."""
        if not self.resume:
            return 0, 0
        saved = torch.load(self.path, map_location="cpu", weights_only=True)
        if saved["config"] != self.config:
            raise ValueError("The run saved here was made with other settings; it cannot resume")
        for name, module in modules.items():
            module.load_state_dict(saved["modules"][name])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"]["torch"])
        if torch.cuda.is_available() and saved["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        return saved["epoch"], saved["step"]

    def save(self, epoch, step, modules, optimizer):
        """Write the state now: `step` batches of `epoch` are done."""
        payload = {
            "config": self.config,
            "epoch": epoch,
            "step": step,
            "modules": {name: module.state_dict() for name, module in modules.items()},
            "optimizer": optimizer.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        # Written whole and renamed, so an interruption mid-write leaves the last one.
        temp = self.path.with_suffix(".tmp")
        torch.save(payload, temp)
        replace_patiently(temp, self.path)
        self.last = self.clock()

    def tick(self, epoch, step, modules, optimizer):
        """After each batch: save if `every` seconds have passed since the last save."""
        if self.every and self.clock() - self.last >= self.every:
            self.save(epoch, step, modules, optimizer)

    def finish(self):
        self.path.unlink(missing_ok=True)


class League:
    def __init__(self, path, seed=42):
        self.path = Path(path)
        self.rng = np.random.default_rng(seed)
        self.entries = json.loads(self.path.read_text()) if self.path.exists() else []

    def add(self, checkpoint):
        checkpoint = Path(checkpoint).resolve()
        digest = file_hash(checkpoint)
        if not any(e["sha256"] == digest for e in self.entries):
            self.entries.append({"path": str(checkpoint), "sha256": digest})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps(self.entries, indent=2))
            temp.replace(self.path)

    def sample(self):
        if not self.entries:
            raise ValueError("League needs a trained imitation checkpoint")
        entry = (
            self.entries[-1]
            if self.rng.random() < 0.5
            else self.entries[int(self.rng.integers(len(self.entries)))]
        )
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("League checkpoint changed after registration")
        return entry


def paired_evaluation(rows, seed=42):
    """Bootstrap whole side-swapped pairs; draws score half, invalid pairs excluded."""
    groups = {}
    for row in rows:
        groups.setdefault(row["pair_id"], []).append(row)
    scores = []
    win_scores = []
    counts = {"win": 0, "draw": 0, "loss": 0}
    excluded = 0
    for pair in groups.values():
        if (
            len(pair) != 2
            or {r["side"] for r in pair} != {"left", "right"}
            or not all(r["valid"] for r in pair)
        ):
            excluded += 1
            continue
        if len({r["scenario"] for r in pair}) != 1:
            raise ValueError("A pair must share its scenario")
        scores.append(np.mean([{"win": 1.0, "draw": 0.5, "loss": 0.0}[r["outcome"]] for r in pair]))
        win_scores.append(np.mean([r["outcome"] == "win" for r in pair]))
        for row in pair:
            counts[row["outcome"]] += 1
    if not scores:
        return {"pairs": 0, "excluded_pairs": excluded, "score": None, "ci95": None}
    x = np.array(scores)
    samples = np.random.default_rng(seed).choice(x, (10000, len(x)), replace=True).mean(1)
    radius = float(np.sqrt(np.log(40) / (2 * len(x))))

    def bounded(mean):
        return [max(0.0, float(mean) - radius), min(1.0, float(mean) + radius)]

    return {
        "pairs": len(x),
        "excluded_pairs": excluded,
        "score": float(x.mean()),
        "ci95": bounded(x.mean()),
        "ci_method": "Hoeffding bound over independent side-swapped pairs",
        "bootstrap_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "win_rate": float(np.mean(win_scores)),
        "win_rate_ci95": bounded(np.mean(win_scores)),
        "outcomes": counts,
        "acceptance_sample_complete": len(x) >= 50,
    }

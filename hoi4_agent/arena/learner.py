"""Recurrent PPO primitives for completed episodes; live collection is separately gated."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .actions import choice_index
from .contracts import STANCES, ArenaError, BuildFingerprint, PlayerObservation, fraction
from .fingerprint import verify_fingerprint
from .policy import RecurrentPolicy
from .returns import advantages, transition_reward
from .trajectory import Trajectory


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    # Per game hour, not per step: a 2160-hour match keeps 0.9995**2160 ~= 0.34 of
    # its terminal reward at the first decision (0.997 left 0.0015).
    gamma_per_hour: float = 0.9995
    lambda_per_hour: float = 0.998
    shaping: float = 0.0
    clip: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    epochs: int = 4
    sequence_length: int = 32

    def __post_init__(self) -> None:
        from .contracts import finite
        fraction(self.gamma_per_hour, "gamma_per_hour")
        fraction(self.lambda_per_hour, "lambda_per_hour")
        finite(self.learning_rate, "learning_rate", 1e-10)
        for name in ("shaping", "entropy_coefficient", "value_coefficient", "clip"):
            finite(getattr(self, name), name)
        if self.epochs <= 0 or self.sequence_length <= 0:
            raise ArenaError("PPO epochs and sequence length must be positive")


class PPOLearner:
    def __init__(self, policy: RecurrentPolicy, config: PPOConfig) -> None:
        self.policy, self.config = policy, config
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)
        self.updates = 0

    def update(self, episodes: list[Trajectory], *, unit_test_fixture: bool = False) -> dict[str, float]:
        """No mixing rollout policies; rejected orders retain their sampled probability.

        Segment-start memories are recomputed once per episode per epoch in a
        single no-grad pass (linear in episode length), so they are at most one
        episode of optimizer steps stale, never stale across epochs. Episode
        boundaries always initialize GRU memory to zero.
        """
        if not episodes or not all(episode.complete for episode in episodes):
            raise ArenaError("PPO requires completed episodes")
        if not unit_test_fixture:
            for episode in episodes:
                episode.require_live_training_data()
        if len({episode.provenance.policy_id for episode in episodes}) != 1:
            raise ArenaError("on-policy update cannot mix rollout policy versions")
        cfg, device = self.config, self.policy.device
        prepared = []
        for episode in episodes:
            transitions = episode.transitions
            rewards = [transition_reward(t.observation, t.next_observation, cfg.gamma_per_hour, cfg.shaping)
                       for t in transitions]
            adv, targets = advantages(rewards, [t.value for t in transitions],
                [t.elapsed_hours for t in transitions], [t.next_observation.terminal for t in transitions],
                gamma_per_hour=cfg.gamma_per_hour, lambda_per_hour=cfg.lambda_per_hour)
            prepared.append((episode, torch.tensor(adv, device=device, dtype=torch.float32),
                             torch.tensor(targets, device=device, dtype=torch.float32)))
        all_advantages = torch.cat([item[1] for item in prepared])
        mean, std = all_advantages.mean(), all_advantages.std(unbiased=False)
        losses, entropies = [], []
        for _ in range(cfg.epochs):
            for episode, adv_tensor, targets_tensor in prepared:
                starts = self.segment_memories(episode)
                for start in range(0, len(episode.transitions), cfg.sequence_length):
                    memory = starts[start // cfg.sequence_length]
                    terms: list[Tensor] = []
                    for i in range(start, min(start + cfg.sequence_length, len(episode.transitions))):
                        transition = episode.transitions[i]
                        output = self.policy(transition.observation, memory, transition.intent)
                        memory = output.memory
                        action = torch.tensor(choice_index(transition.observation, transition.order), device=device)
                        ratio = (output.distribution.log_prob(action) - transition.log_probability).exp()
                        advantage = ((adv_tensor[i] - mean) / std.clamp_min(1e-8)
                                     if len(all_advantages) > 1 and std > 1e-8 else adv_tensor[i])
                        policy_loss = -torch.minimum(ratio * advantage,
                            ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantage)
                        entropy = output.distribution.entropy()
                        terms.append(policy_loss + cfg.value_coefficient * (output.value - targets_tensor[i]).square()
                                     - cfg.entropy_coefficient * entropy)
                        entropies.append(float(entropy.detach()))
                    loss = torch.stack(terms).mean()
                    if not torch.isfinite(loss):
                        raise ArenaError("non-finite PPO loss")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5, error_if_nonfinite=True)
                    self.optimizer.step()
                    losses.append(float(loss.detach()))
        self.updates += 1
        return {"loss": sum(losses) / len(losses), "entropy": sum(entropies) / len(entropies),
                "transitions": float(len(all_advantages)), "updates": float(self.updates)}

    @torch.no_grad()
    def segment_memories(self, episode: Trajectory) -> list[Tensor]:
        memory, result = self.policy.initial_memory(), []
        for index, transition in enumerate(episode.transitions):
            if index % self.config.sequence_length == 0:
                result.append(memory)
            memory = self.policy(transition.observation, memory, transition.intent).memory
        return result

    def distill_intents(self, samples: list[tuple[PlayerObservation, dict[str, dict[str, float]]]],
                        epochs: int = 1) -> float:
        """Teach the intent head a teacher's per-sector stance probabilities (e.g. Jev).

        Soft-label cross-entropy on single observations with fresh memory; the
        action and value heads are untouched except through the shared encoder.
        """
        if not samples or epochs <= 0:
            raise ArenaError("intent distillation requires samples and positive epochs")
        losses = []
        for _ in range(epochs):
            for observation, teacher in samples:
                output = self.policy(observation, None)
                terms = []
                for sector, probabilities in teacher.items():
                    if sector not in output.intent_logits or set(probabilities) != set(STANCES):
                        raise ArenaError("teacher intent must cover every stance of a known sector")
                    target = output.value.new_tensor([probabilities[stance] for stance in STANCES])
                    if not torch.isfinite(target).all() or (target < 0).any() or abs(float(target.sum()) - 1) > 1e-3:
                        raise ArenaError("teacher intent probabilities must form a distribution")
                    terms.append(-(target * output.intent_logits[sector].log_softmax(-1)).sum())
                if not terms:
                    continue
                loss = torch.stack(terms).mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5, error_if_nonfinite=True)
                self.optimizer.step()
                losses.append(float(loss.detach()))
        if not losses:
            raise ArenaError("no sample carried a sector the observation defines")
        return sum(losses) / len(losses)

    def clone_demonstrations(self, episodes: list[Trajectory], epochs: int = 1) -> float:
        if not episodes or epochs <= 0:
            raise ArenaError("behavior cloning requires demonstrations and positive epochs")
        for episode in episodes:
            episode.require_live_training_data()
            if episode.provenance.demonstration_source is None or any(not t.receipt.accepted for t in episode.transitions):
                raise ArenaError("behavior cloning requires accepted demonstration commands")
        losses = []
        for _ in range(epochs):
            for episode in episodes:
                starts = self.segment_memories(episode)
                for start in range(0, len(episode.transitions), self.config.sequence_length):
                    memory = starts[start // self.config.sequence_length]
                    terms = []
                    for transition in episode.transitions[start:start + self.config.sequence_length]:
                        output = self.policy(transition.observation, memory, transition.intent)
                        memory = output.memory
                        action = torch.tensor(choice_index(transition.observation, transition.order), device=self.policy.device)
                        terms.append(-output.distribution.log_prob(action))
                    loss = torch.stack(terms).mean()
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5, error_if_nonfinite=True)
                    self.optimizer.step()
                    losses.append(float(loss.detach()))
        return sum(losses) / len(losses)

    def save(self, path: Path, fingerprint: BuildFingerprint, league: dict[str, Any],
             provenance: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "width": self.policy.width, "config": asdict(self.config),
            "policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict(), "updates": self.updates,
            "fingerprint": asdict(fingerprint), "league": league, "provenance": provenance,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def resume(cls, path: Path, expected: BuildFingerprint, device: str = "cpu") -> tuple[PPOLearner, dict[str, Any]]:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["schema_version"] != 1:
            raise ArenaError("unsupported checkpoint version")
        verify_fingerprint(expected, BuildFingerprint(**payload["fingerprint"]))
        learner = cls(RecurrentPolicy(payload["width"]).to(device), PPOConfig(**payload["config"]))
        learner.policy.load_state_dict(payload["policy"])
        learner.optimizer.load_state_dict(payload["optimizer"])
        learner.updates = payload["updates"]
        torch.set_rng_state(payload["torch_rng"])
        if device.startswith("cuda") and payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        return learner, {"league": payload["league"], "provenance": payload["provenance"]}

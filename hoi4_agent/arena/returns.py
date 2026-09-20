"""Game-time returns and potential shaping; independent of wall-clock speed."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .contracts import ArenaError, PlayerObservation, finite, fraction


def objective_potential(observation: PlayerObservation) -> float:
    if observation.terminal:
        return 0.0
    total = sum(province.victory_points for province in observation.provinces)
    if not total:
        return 0.0
    return sum(province.victory_points * (1 if province.controller == observation.country else
               -1 if province.controller == observation.country.opponent else 0)
               for province in observation.provinces) / total


def transition_reward(before: PlayerObservation, after: PlayerObservation,
                      gamma_per_hour: float = 0.9995, shaping: float = 0.0) -> float:
    fraction(gamma_per_hour, "gamma_per_hour")
    finite(shaping, "shaping")
    if before.episode_id != after.episode_id or before.country != after.country or before.terminal:
        raise ArenaError("reward transition crosses an episode/player boundary")
    elapsed = after.game_hour - before.game_hour
    if elapsed < 0:
        raise ArenaError("game time moved backwards inside a transition")
    terminal = 0.0
    if after.terminal and after.winner is not None:
        terminal = 1.0 if after.winner == after.country else -1.0
    return terminal + shaping * (gamma_per_hour ** elapsed * objective_potential(after)
                                 - objective_potential(before))


def advantages(rewards: list[float], values: list[float], elapsed_hours: list[int],
               terminals: list[bool], bootstrap_value: float = 0.0,
               gamma_per_hour: float = 0.9995, lambda_per_hour: float = 0.998,
               ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    fraction(gamma_per_hour, "gamma_per_hour")
    fraction(lambda_per_hour, "lambda_per_hour")
    length = len(rewards)
    if not length or not all(len(items) == length for items in (values, elapsed_hours, terminals)):
        raise ArenaError("GAE arrays must be nonempty and aligned")
    if any(hours < 0 for hours in elapsed_hours):
        raise ArenaError("elapsed game hours must not be negative")
    if not np.isfinite([*rewards, *values, bootstrap_value]).all():
        raise ArenaError("non-finite rewards or values")
    result = np.zeros(length, dtype=np.float64)
    accumulator, next_value = 0.0, bootstrap_value
    for i in reversed(range(length)):
        discount = gamma_per_hour ** elapsed_hours[i]
        trace = lambda_per_hour ** elapsed_hours[i]
        continuing = 0.0 if terminals[i] else 1.0
        delta = rewards[i] + discount * continuing * next_value - values[i]
        accumulator = delta + discount * trace * continuing * accumulator
        result[i] = accumulator
        next_value = values[i]
    return result, result + np.asarray(values)

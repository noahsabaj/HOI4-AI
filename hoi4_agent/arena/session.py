"""Live session boundary. A diagnostic/fixture bridge cannot start a match."""

from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any

from .contracts import (
    PROTOCOL_VERSION, ArenaError, ArenaSpec, BuildFingerprint, CapabilityError,
    Country, Order, OrderReceipt, PlayerObservation,
)
from .diagnostics import REQUIRED_CAPABILITIES
from .fingerprint import verify_fingerprint
from .protocol import Transport


class BridgeSession:
    def __init__(self, transport: Transport, expected: BuildFingerprint) -> None:
        self.transport = transport
        self.expected = expected
        self._counter = 0
        self.episode_id: str | None = None
        self._observations: dict[Country, PlayerObservation] = {}
        # Receipt time on this process's clock. captured_monotonic_ns belongs to
        # the bridge's clock (different epoch/resolution) and is provenance only.
        self._received_ns: dict[Country, int] = {}
        self._orders: set[str] = set()
        self._countries: tuple[Country, ...] = ()
        self._spec: ArenaSpec | None = None
        try:
            self.hello = self.request("hello", {})
            self.assert_ready()
        except (ArenaError, KeyError, TypeError, ValueError):
            self.close()
            raise

    def request(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._counter += 1
        response = self.transport.exchange({"version": PROTOCOL_VERSION, "id": self._counter,
                                            "method": method, "payload": payload})
        if response.get("version") != PROTOCOL_VERSION or response.get("id") != self._counter:
            self.transport.close()
            raise ArenaError("bridge returned a mismatched protocol/request ID")
        if response.get("ok") is not True:
            raise CapabilityError(str(response.get("error", "bridge rejected request")))
        result = response.get("result")
        if not isinstance(result, dict):
            raise ArenaError("bridge result must be an object")
        return result

    def assert_ready(self) -> None:
        if self.hello.get("backend") != "hoi4_native":
            raise CapabilityError("live sessions require a verified HOI4 adapter, not a diagnostic/fixture host")
        missing = [cap for cap in REQUIRED_CAPABILITIES if self.hello.get("capabilities", {}).get(cap) is not True]
        if missing:
            raise CapabilityError("unverified engine capabilities: " + ", ".join(missing))
        verify_fingerprint(self.expected, BuildFingerprint(**self.hello["fingerprint"]))

    def reset(self, spec: ArenaSpec, model_countries: tuple[Country, ...]) -> str:
        verify_fingerprint(self.expected, spec.fingerprint)
        if (not model_countries or len(set(model_countries)) != len(model_countries) or
            not all(isinstance(country, Country) for country in model_countries)):
            raise ArenaError("choose one or two unique model countries")
        previous_episode = self.episode_id
        self.episode_id = None
        self._observations.clear()
        self._received_ns.clear()
        self._orders.clear()
        self._countries = ()
        self._spec = None
        result = self.request("reset", {"spec": asdict(spec), "model_countries": model_countries})
        episode = result.get("episode_id")
        if not isinstance(episode, str) or not episode or episode == previous_episode:
            raise ArenaError("reset did not establish a fresh episode")
        if not result.get("engine_state_fingerprint"):
            raise ArenaError("reset did not report its engine state fingerprint")
        self.episode_id = episode
        self._observations.clear()
        self._received_ns.clear()
        self._orders.clear()
        self._countries = model_countries
        self._spec = spec
        return episode

    def observe(self, country: Country) -> PlayerObservation:
        if self.episode_id is None:
            raise ArenaError("reset the session before observing")
        requested_ns = time.monotonic_ns()  # conservative: age includes the bridge round trip
        observation = PlayerObservation.from_dict(self.request("observe", {"country": country}))
        if observation.episode_id != self.episode_id or observation.country != country:
            raise ArenaError("bridge crossed episode/player observation boundaries")
        previous = self._observations.get(country)
        if previous and (observation.sequence < previous.sequence or observation.game_hour < previous.game_hour):
            raise ArenaError("observation sequence or game time moved backwards")
        self._observations[country] = observation
        self._received_ns[country] = requested_ns
        return observation

    def submit(self, order: Order) -> OrderReceipt:
        observation = self._observations.get(order.country)
        if order.episode_id != self.episode_id or order.country not in self._countries:
            raise ArenaError("order is outside this session's country/episode")
        if observation is None or observation.sequence != order.observation_sequence or observation.terminal:
            raise ArenaError("order requires the latest nonterminal player observation")
        assert self._spec is not None
        age_ms = (time.monotonic_ns() - self._received_ns[order.country]) / 1_000_000
        if not 0 <= age_ms <= self._spec.max_observation_age_ms:
            raise ArenaError("observation deadline exceeded; existing orders must continue")
        owned = {unit.id for unit in observation.units if unit.country == order.country}
        if not set(order.unit_ids) <= owned:
            raise ArenaError("order references a destroyed, unobserved or unowned unit")
        if order.target_province_id is not None and order.target_province_id not in {p.id for p in observation.provinces}:
            raise ArenaError("order target is outside the player-visible map")
        if order.id in self._orders:
            raise ArenaError("duplicate order ID; query its receipt instead of replaying it")
        self._orders.add(order.id)  # includes uncertain outcomes after disconnect
        receipt = OrderReceipt(**self.request("submit", asdict(order)))
        if receipt.order_id != order.id or receipt.episode_id != self.episode_id:
            raise ArenaError("receipt does not match submitted order")
        return receipt

    def close(self) -> None:
        self.transport.close()
        self.episode_id = None
        self._observations.clear()
        self._received_ns.clear()
        self._countries = ()
        self._spec = None

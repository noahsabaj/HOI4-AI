"""Small graph/entity encoder, GRU memory, and masked normal-order policy.

Tier 1 of the hierarchy: runs locally every tick. No full-state input, embedded
tactics, pretrained language model, or game import. Tier-2 guidance arrives as an
optional Intent (goal conditioning); the intent head predicts that guidance so
it can be distilled and eventually taken out of the loop.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from .actions import Choice, choices
from .contracts import STANCES, UNIT_KINDS, ArenaError, Intent, Order, PlayerObservation, Verb

TERRAINS = ("plains", "forest", "hills", "mountain", "desert", "marsh", "urban", "jungle", "unknown")
VERBS = tuple(Verb)
PROVINCE_FEATURES = len(TERRAINS) + 8 + len(STANCES) + 1
UNIT_FEATURES = 15 + len(UNIT_KINDS)
GLOBAL_FEATURES = 1 + 5 + 1 + 2  # hour, speed one-hot, paused, intent present/age
INTENT_STALE_HOURS = 72.0


def uncertain(value: float | None) -> tuple[float, float]:
    return (0.0, 0.0) if value is None else (float(value), 1.0)


@dataclass
class PolicyOutput:
    distribution: Categorical
    value: Tensor
    memory: Tensor
    actions: tuple[Choice, ...]
    intent_logits: dict[str, Tensor]  # sector -> logits over STANCES


class RecurrentPolicy(nn.Module):
    def __init__(self, width: int = 128) -> None:
        super().__init__()
        if not 16 <= width <= 256:
            raise ArenaError("policy width must be between 16 and 256")
        self.width = width
        self.province_encoder = nn.Sequential(nn.Linear(PROVINCE_FEATURES, width), nn.Tanh())
        self.graph_layers = nn.ModuleList(nn.Linear(width * 2, width) for _ in range(2))
        self.entity_encoder = nn.Sequential(nn.Linear(UNIT_FEATURES + width * 2, width), nn.Tanh())
        self.recurrent = nn.GRUCell(width * 3 + GLOBAL_FEATURES, width)
        self.verb_embedding = nn.Embedding(len(VERBS), width // 4)
        self.action_head = nn.Sequential(nn.Linear(width * 3 + width // 4 + 2, width), nn.Tanh(), nn.Linear(width, 1))
        self.value_head = nn.Linear(width, 1)
        self.intent_head = nn.Sequential(nn.Linear(width * 2, width), nn.Tanh(), nn.Linear(width, len(STANCES)))
        if sum(parameter.numel() for parameter in self.parameters()) >= 10_000_000:
            raise ArenaError("policy exceeds local parameter budget")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def initial_memory(self) -> Tensor:
        return torch.zeros(1, self.width, device=self.device)

    def forward(self, observation: PlayerObservation, memory: Tensor | None = None,
                intent: Intent | None = None) -> PolicyOutput:
        provinces = sorted(observation.provinces, key=lambda p: p.id)
        units = sorted(observation.units, key=lambda u: u.id)
        index = {province.id: i for i, province in enumerate(provinces)}
        features = []
        for province in provinces:
            terrain = province.terrain if province.terrain in TERRAINS else "unknown"
            stance = intent.stance(province.sector) if intent is not None and province.sector else None
            features.append([float(terrain == name) for name in TERRAINS] +
                            [province.x, province.y, float(province.controller == observation.country),
                             float(province.controller == observation.country.opponent),
                             float(province.controller is not None), province.victory_points / 10,
                             *uncertain(province.supply),
                             *(float(stance == name) for name in STANCES), float(stance is None)])
        nodes = self.province_encoder(torch.tensor(features, dtype=torch.float32, device=self.device))
        adjacency = torch.eye(len(provinces), device=self.device)
        for row, province in enumerate(provinces):
            for neighbor in province.neighbors:
                adjacency[row, index[neighbor]] = 1
        adjacency = adjacency / adjacency.sum(dim=1, keepdim=True)
        for layer in self.graph_layers:
            nodes = torch.tanh(layer(torch.cat((nodes, adjacency @ nodes), dim=-1)))
        zero = nodes.new_zeros(self.width)
        entities: dict[int, Tensor] = {}
        own: list[Tensor] = []
        enemy: list[Tensor] = []
        for unit in units:
            scalars = nodes.new_tensor([float(unit.country == observation.country),
                float(unit.country != observation.country), *uncertain(unit.organization),
                *uncertain(unit.strength), *uncertain(unit.supply),
                *uncertain(None if unit.in_combat is None else float(unit.in_combat)),
                *uncertain(unit.entrenchment), float(unit.order_target_province_id is not None),
                min(unit.count, 24) / 24, unit.confidence,
                *(float(unit.kind == kind) for kind in UNIT_KINDS)])
            target = unit.order_target_province_id
            encoded = self.entity_encoder(torch.cat((nodes[index[unit.province_id]],
                                                     nodes[index[target]] if target is not None else zero, scalars)))
            entities[unit.id] = encoded
            (own if unit.country == observation.country else enemy).append(encoded)
        own_pool = torch.stack(own).mean(0) if own else zero
        enemy_pool = torch.stack(enemy).mean(0) if enemy else zero
        age = 1.0 if intent is None else min(intent.age_hours(observation.game_hour) / INTENT_STALE_HOURS, 1.0)
        pooled = torch.cat((nodes.mean(0), own_pool, enemy_pool, nodes.new_tensor(
            [observation.game_hour / 2160, *(float(observation.game_speed == speed) for speed in range(1, 6)),
             float(observation.paused), float(intent is not None), age]))).unsqueeze(0)
        memory = self.recurrent(pooled, self.initial_memory() if memory is None else memory)
        candidates = choices(observation)
        location = {unit.id: unit.province_id for unit in units}
        rivers = {province.id: province.river_neighbors for province in provinces}

        def extras(action: Choice) -> list[float]:
            crossing = action.unit_id is not None and action.target_id in rivers[location[action.unit_id]]
            return [float(crossing), (action.speed or 0) / 5]

        candidate_features = torch.stack([
            torch.cat((memory[0], entities[action.unit_id] if action.unit_id is not None else zero,
                       nodes[index[action.target_id]] if action.target_id is not None else zero,
                       self.verb_embedding.weight[VERBS.index(action.verb)],
                       nodes.new_tensor(extras(action)))) for action in candidates
        ])
        logits = self.action_head(candidate_features).squeeze(-1)
        sectors = sorted({province.sector for province in provinces if province.sector})
        intent_logits = {
            sector: self.intent_head(torch.cat((memory[0], torch.stack(
                [nodes[index[p.id]] for p in provinces if p.sector == sector]).mean(0))))
            for sector in sectors
        }
        return PolicyOutput(Categorical(logits=logits), self.value_head(memory).squeeze(), memory, candidates,
                            intent_logits)

    @torch.no_grad()
    def act(self, observation: PlayerObservation, memory: Tensor | None,
            intent: Intent | None = None) -> tuple[Order, Tensor]:
        output = self(observation, memory, intent)
        action = output.actions[int(output.distribution.sample().item())]
        return action.order(observation, uuid.uuid4().hex), output.memory

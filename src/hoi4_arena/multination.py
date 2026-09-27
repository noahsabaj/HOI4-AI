"""Arenas for three and four nations: who they are, the designs, and the wars they fight.

A multi-nation arena (mapgen_multi.py) is one nation's share of the map turned N times
about the map's centre, so every nation holds the same ground, the same borders and the
same distances to every other: N = 3 turns a hexagonal lattice a third of a turn, N = 4
turns a square one a quarter turn. The nations are numbered round the centre in the turn's
direction, so nation k borders nations k - 1 and k + 1 (and, with three, both others).

The wars are a parameter (`Wars`, `wars()`): the nations are split into sides, a side of
more than one is a faction, and every pair of sides is at war from the first day, through
the game's own mechanics (a faction created in the leader's history, `declare_war_on`,
`add_to_war`). Nations on no side stay neutral. A setup is fair when the turn that maps the
map onto itself can carry any side onto any other; 3v1 and 2v1 cannot be, and say so.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .arenas import Patch

# The nations in the order they stand round the centre. Blue and Red are the two-country
# arena's; the others are chosen to be told apart on screen. The map draws a country's
# colour faintly over the ground, and what survives is its tint (vision.country_pixels:
# Blue's land about (120, 134, 145), Red's (168, 145, 131) at 1080p). Fitted to those two,
# the tint is about 0.18 of the colour's chroma plus a slight olive from the ground, which
# predicts yellow's tint within 35 degrees of Red's; violet and lime come out 64 to 180
# degrees from every other. Violet also still reads as Blue's kind and lime as Red's to
# the two-country tint test (b - r > 10, r - b > 15), and the four alternate round the
# centre, so each border of the four-nation arena is still a blue-red border to it.
NATIONS = ("BLU", "RED", "PUR", "GRN")
NAMES = {"BLU": "Blue", "RED": "Red", "PUR": "Purple", "GRN": "Green"}
COLOUR = {"BLU": (40, 100, 220), "RED": (220, 60, 60), "PUR": (130, 40, 240), "GRN": (120, 200, 40)}
COLOUR_UI = {
    "BLU": (70, 130, 255),
    "RED": (255, 90, 90),
    "PUR": (170, 100, 255),
    "GRN": (160, 230, 90),
}
# Character names, as the two-country arena's: every country needs a list.
NAME_LISTS = {
    "PUR": (
        "Albert Anselm Bruno Conrad Emil Ernst Felix Florian Gustav Hugo Julius Karl Konrad "
        "Leopold Lorenz Ludwig Moritz Oskar Otto Paul Rudolf Siegfried Theodor Ulrich Viktor",
        "Adele Agnes Berta Clara Dora Elsa Emma Frieda Gerda Hedwig Helene Ida Ilse Johanna "
        "Klara Lena Lotte Marie Martha Olga Paula Rosa Ursula Wilma Zita",
        "Adler Bauer Brandt Eckert Falk Graf Hahn Hartmann Keller Kraus Lang Lorenz Maier "
        "Neumann Pohl Roth Sauer Schenk Seidel Stark Vogel Wagner Weber Winter Zeller",
    ),
    "GRN": (
        "Alvar Anders Arvid Bengt Birger Einar Erik Gunnar Halvar Harald Ingvar Johan Knut "
        "Lars Leif Magnus Nils Olof Per Rune Sigurd Sten Sven Torsten Ulf Vidar",
        "Agnes Astrid Birgit Dagny Ebba Elin Freja Gerd Greta Gudrun Hanna Hilda Ingrid "
        "Karin Kerstin Linnea Maja Märta Ragnhild Sigrid Siv Solveig Tora Tyra Ylva",
        "Alm Berg Dahl Ek Falk Frost Hag Holm Kvist Lind Lund Mark Nord Rask Ros Sand Sjö "
        "Stav Strand Ström Sund Vall Varg Vik Wall",
    ),
}


# How a country's colour shows on the map: this share of its chroma (the colour less its
# grey) over the ground's own slight tint, fitted to Blue's and Red's land at 1080p.
SCREEN_TINT = 0.18
GROUND_TINT = (1.1, 4.5, -5.6)


def arena_tags(mod):
    """An arena's countries in their order round it: a multi-nation arena's from its
    generation.json, and Blue and Red for every other."""
    try:
        found = json.loads((Path(mod) / "generation.json").read_text()).get("tags")
    except (OSError, ValueError):
        found = None
    return tuple(found) if found else ("BLU", "RED")


def classify(pixels, tags, land=False, least=6.0, agree=0.6):
    """Which of `tags` each pixel's colour is, by the direction of its tint: an index into
    `tags`, or -1 where the tint is fainter than `least` or no country's lies within
    `agree` (a cosine) of it. With `land`, the pixels are the map's, where a country shows
    faintly over the ground (SCREEN_TINT); without, they are the colours themselves, as a
    flag shows them. Uncalibrated for any country but Blue and Red: a prediction."""
    pixels = np.asarray(pixels, dtype=np.float32)
    references = []
    for tag in tags:
        colour = np.array(COLOUR[tag], np.float32)
        chroma = colour - colour.mean()
        if land:
            chroma = SCREEN_TINT * chroma + np.array(GROUND_TINT, np.float32)
        references.append(chroma / np.linalg.norm(chroma))
    tint = pixels - pixels.mean(axis=-1, keepdims=True)
    size = np.linalg.norm(tint, axis=-1)
    cosine = tint @ np.array(references).T / np.maximum(size, 1e-6)[..., None]
    best = cosine.argmax(axis=-1)
    return np.where((size >= least) & (cosine.max(axis=-1) >= agree), best, -1)


# ---------------------------------------------------------------------------------------
# Designs.


@dataclass(frozen=True)
class NationPreset:
    """A named multi-nation arena. One nation's share is drawn and turned N times.

    Patches (arenas.Patch) are drawn in the nation frame, in provinces from the map's
    centre: +x runs out through the middle of nation 0 and +y a quarter turn clockwise
    from it, so nation 0's borders run out along the angles +-180/N degrees. As on the
    two-country arenas, whatever is drawn anywhere is drawn for every nation.
    """

    title: str
    summary: str
    nations: int
    seed: int
    patches: tuple = ()
    # Extra lakes, in the nation frame, one per point and each turned N times.
    lakes: tuple = ()
    base: str = "plains"
    # How far province borders wander (pixels) and seeds stray (share of a province): the
    # two-country presets' warp; the seeds stray a little less, since a hexagonal lattice
    # strayed too far makes near-ties that can cost three nations their exact graph.
    warp: float = 26.0
    jitter: float = 0.18
    states: int = 8
    wars: str = ""


# The three-nation arena's borders run out from the centre, where the three meet, at
# +-60 degrees in the nation frame.
_R60 = (0.5, 0.8660254)


def _ray(direction, start, end):
    """A line from `start` to `end` provinces out along a unit direction."""
    return ((direction[0] * start, direction[1] * start), (direction[0] * end, direction[1] * end))


def _quad(t):
    """A point on nation 0's border with nation 3 on the four-nation arena, `t` provinces
    out. The four stand in a pinwheel round the centre cell, so a border runs half a cell
    beside the line through the centre: between the cells right of and on the axis below
    the centre (x = 0.5 on the map, y = t), which the nation frame turns by 45 degrees."""
    return ((0.5 + t) / 2**0.5, (t - 0.5) / 2**0.5)


NATION_PRESETS = {
    "tri-plains": NationPreset(
        title="Three Plains",
        summary="Three nations round a meeting point, each touching both others: open "
        "farmland, a few woods and low hills. Every nation holds the same ground, turned.",
        nations=3,
        seed=1301,
        patches=(
            Patch("forest", "blob", (5.2, -1.8), 1.5),
            Patch("hills", "blob", (7.4, 1.8), 1.5),
            Patch("forest", "blob", (3.0, 1.6), 1.0),
            Patch("marsh", "blob", (8.6, -1.0), 0.8),
        ),
        wars="1v1v1",
    ),
    "tri-ridges": NationPreset(
        title="Three Ridges",
        summary="Three nations, each border a ridge of hills with mountains along its "
        "crest, crossed by two open passes: every war is fought for the passes.",
        nations=3,
        seed=1302,
        patches=(
            Patch("hills", "line", _ray(_R60, 1.2, 11.0), 1.6, 0.15),
            Patch("mountain", "line", _ray(_R60, 1.8, 11.0), 0.75, 0.1),
            # Two passes through each ridge.
            Patch("plains", "blob", (_R60[0] * 3.6, _R60[1] * 3.6), 1.1, 0.1),
            Patch("plains", "blob", (_R60[0] * 6.8, _R60[1] * 6.8), 1.1, 0.1),
            Patch("forest", "blob", (5.0, 0.0), 1.2),
            Patch("hills", "blob", (8.0, 0.0), 0.8),
        ),
        wars="1v1v1",
    ),
    "quad-plains": NationPreset(
        title="Four Plains",
        summary="Four nations round a central lake, each touching two others: open "
        "farmland, a few woods and low hills. Every nation holds the same ground, turned.",
        nations=4,
        seed=1401,
        patches=(
            Patch("forest", "blob", (5.6, -2.0), 1.5),
            Patch("hills", "blob", (8.8, 1.8), 1.5),
            Patch("forest", "blob", (3.4, 2.0), 1.0),
            Patch("marsh", "blob", (10.6, -1.0), 0.8),
        ),
        wars="1v1v1v1",
    ),
    "quad-ridges": NationPreset(
        title="Four Ridges",
        summary="Four nations round a central lake, each border a ridge of hills with "
        "mountains along its crest, crossed by two open passes.",
        nations=4,
        seed=1402,
        patches=(
            Patch("hills", "line", (_quad(1.0), _quad(13.0)), 1.6, 0.15),
            Patch("mountain", "line", (_quad(1.5), _quad(13.0)), 0.75, 0.1),
            Patch("plains", "blob", _quad(3.8), 1.1, 0.1),
            Patch("plains", "blob", _quad(7.4), 1.1, 0.1),
            Patch("forest", "blob", (6.2, 0.0), 1.3),
            Patch("hills", "blob", (10.0, 0.0), 0.8),
        ),
        wars="1v1v1v1",
    ),
}


# ---------------------------------------------------------------------------------------
# Wars.


@dataclass(frozen=True)
class Wars:
    """Which nations fight which: `sides` are tuples of tags, the first of each its leader
    (and its faction's, when it has allies); every pair of sides is at war. `tags` are all
    the arena's nations in their order round the centre; those on no side are neutral."""

    name: str
    tags: tuple
    sides: tuple

    @property
    def neutral(self):
        placed = {tag for side in self.sides for tag in side}
        return tuple(tag for tag in self.tags if tag not in placed)

    @property
    def spec(self):
        """The sides as logged: "BLU+RED:PUR+GRN"."""
        return ":".join("+".join(side) for side in self.sides)

    @property
    def fair(self):
        """True if some turn of the map carries any side onto any other, neutrals onto
        neutrals: then every side holds the same position, ground and enemies included."""
        n = len(self.tags)
        index = {tag: k for k, tag in enumerate(self.tags)}
        sides = [frozenset(index[t] for t in side) for side in self.sides]
        neutral = frozenset(index[t] for t in self.neutral)
        reached = {sides[0]}
        for turn in range(n):
            moved = [frozenset((k + turn) % n for k in side) for side in sides]
            if set(moved) == set(sides) and frozenset((k + turn) % n for k in neutral) == neutral:
                reached.add(moved[0])
        return len(reached) == len(sides)

    @property
    def pairs(self):
        """Each war as a pair of side indexes."""
        return [(i, j) for i in range(len(self.sides)) for j in range(i + 1, len(self.sides))]

    def enemies(self, tag):
        mine = next((side for side in self.sides if tag in side), None)
        if mine is None:
            return ()
        return tuple(t for side in self.sides if side is not mine for t in side)

    def underdogs(self):
        """The nations on sides smaller than the largest: those a bonus would help."""
        largest = max(len(side) for side in self.sides)
        return tuple(t for side in self.sides if len(side) < largest for t in side)

    def note(self):
        if self.fair:
            return "fair: every side holds the same position, turned"
        sizes = "v".join(str(len(side)) for side in self.sides)
        if len({len(side) for side in self.sides}) > 1:
            return f"unfair by design: {sizes}, sides of different sizes"
        return (
            "not exactly fair: the sides are the same size, but no turn of the map carries "
            "one onto the other (the neutrals stand to one side of them)"
        )


# Named setups for each number of nations. "AvB..." splits the nations into sides of those
# sizes in their order round the centre, so a side's nations are neighbours; "+K" leaves the
# last K neutral; a trailing "x" deals the nations to the sides in turn instead, so allies
# stand apart (2v2x: Blue and Purple against Red and Green).
SETUPS = {
    3: ("1v1v1", "2v1", "1v1+1"),
    4: ("1v1v1v1", "2v2", "2v2x", "3v1", "2v1v1", "1v1v1+1", "2v1+1", "1v1+2"),
}
_PATTERN = re.compile(r"^(?P<sizes>\d+(?:v\d+)+)(?:\+(?P<neutral>\d+))?(?P<dealt>x?)$")


def wars(spec, tags):
    """The wars `spec` names for nations `tags` (in their order round the centre).

    `spec` is a named setup ("2v2", "1v1+1", "2v2x") or the sides themselves, tags joined
    by "+" within a side and ":" between sides ("BLU+PUR:RED:GRN"); tags left out are
    neutral.
    """
    tags = tuple(tags)
    n = len(tags)
    found = _PATTERN.match(spec)
    if found:
        sizes = [int(s) for s in found["sizes"].split("v")]
        neutral = int(found["neutral"] or 0)
        if any(size < 1 for size in sizes) or sum(sizes) + neutral != n:
            raise ValueError(f"{spec} does not split {n} nations")
        if found["dealt"]:
            if neutral or len(set(sizes)) != 1:
                raise ValueError(f"{spec}: x deals equal sides with no neutrals")
            count = len(sizes)
            sides = [tuple(tags[k] for k in range(i, n, count)) for i in range(count)]
        else:
            sides, start = [], 0
            for size in sizes:
                sides.append(tags[start : start + size])
                start += size
    else:
        sides = [tuple(part.split("+")) for part in spec.split(":")]
        named = [tag for side in sides for tag in side]
        unknown = sorted(set(named) - set(tags))
        if unknown:
            raise ValueError(f"{spec} names {', '.join(unknown)}, not in {'+'.join(tags)}")
        if len(named) != len(set(named)):
            raise ValueError(f"{spec} names a nation twice")
    if len(sides) < 2 or any(not side for side in sides):
        raise ValueError(f"{spec} needs at least two sides to be at war")
    return Wars(spec, tags, tuple(tuple(side) for side in sides))


# ---------------------------------------------------------------------------------------
# The script that sets the wars up.

FACTION_TEMPLATE = "arena_faction"


def faction_template():
    """common/factions/templates: a faction for script only, with no rule that could
    dismiss a member or refuse one. The generic manifest is the stock one."""
    return (
        f"{FACTION_TEMPLATE} = {{\n"
        "\tname = ARENA_FACTION\n"
        "\tmanifest = faction_manifest_strength_in_unity\n"
        "\tcan_leader_join_other_factions = no\n"
        "\ticon = GFX_faction_logo_generic\n"
        "\tvisible = { always = no }\n"
        "\tgoals = { }\n"
        "\tdefault_rules = { }\n"
        "}\n"
    )


def faction_history(plan, tag):
    """The lines of `tag`'s history file that make its faction, if it leads one."""
    for number, side in enumerate(plan.sides, 1):
        if len(side) > 1 and side[0] == tag:
            return (
                f"create_faction_from_template = {{ template = {FACTION_TEMPLATE}"
                f" name = ARENA_FACTION_{number} }}\n"
                + "".join(f"add_to_faction = {member}\n" for member in side[1:])
            )
    return ""


def declarer(plan, first, pair):
    """Which side of war `pair` declares it, when nation `first` has the first say: the
    side whose nation comes first counting round from `first`."""
    order = plan.tags[first:] + plan.tags[:first]
    i, j = pair
    rank = {tag: k for k, tag in enumerate(order)}
    return i if min(rank[t] for t in plan.sides[i]) < min(rank[t] for t in plan.sides[j]) else j


def setup_effect(plan, first):
    """The effect that starts every war, with nation `first` having the first say.

    Each war is declared by one side's leader on the other's, with an annex-everything war
    goal, and every ally not yet in it is added on its leader's side: a faction member
    attacked joins by itself, but an attacker's allies are only asked to, so they are
    added rather than left to accept.
    """
    parts = [" set_global_flag = arena_declared"]
    for pair in plan.pairs:
        attack = declarer(plan, first, pair)
        defend = pair[1] if attack == pair[0] else pair[0]
        leader, target = plan.sides[attack][0], plan.sides[defend][0]
        parts.append(
            f" {leader} = {{ declare_war_on = {{ target = {target} type = annex_everything }} }}"
            f' log = "ARENA declare {leader} {target}"'
        )
        for side, ally_of, enemy in ((attack, leader, target), (defend, target, leader)):
            for ally in plan.sides[side][1:]:
                parts.append(
                    f" {ally} = {{ if = {{ limit = {{ NOT = {{ has_war_with = {enemy} }} }}"
                    f" add_to_war = {{ targeted_alliance = {ally_of} enemy = {enemy}"
                    " hostility_reason = asked_to_join } } }"
                )
    return "".join(parts)


def events(plan):
    """events/arena.txt: arena.k starts the wars with nation k - 1 having the first say, as
    the two-country arena's arena.1 has Blue declare and arena.2 Red."""
    return "add_namespace = arena\n" + "".join(
        f"country_event = {{ id = arena.{k + 1} hidden = yes is_triggered_only = yes"
        f" immediate = {{{setup_effect(plan, k)} }} }}\n"
        for k in range(len(plan.tags))
    )


def fallback(plan):
    """What the first nation's daily on_action runs until the wars have started: a draw of
    who has the first say, for games where nobody fires an event from the console."""
    draws = "".join(f" 1 = {{{setup_effect(plan, k)} }}" for k in range(len(plan.tags)))
    return (
        " if = { limit = { NOT = { has_global_flag = arena_declared } }"
        f" random_list = {{{draws} }} }}"
    )


def startup_log(plan):
    """The on_startup line that states the wars: sides, neutrals and whether it is fair."""
    neutral = f" neutral {'+'.join(plan.neutral)}" if plan.neutral else ""
    return f' log = "ARENA wars {plan.spec}{neutral} fair {"yes" if plan.fair else "no"}"'


def strategies(plan):
    """common/ai_strategy: each nation drives its fronts against each other nation it is
    at war with, as the two-country arena's front_control does against its one enemy."""
    return "".join(
        f"{tag}_arena_offensive_{enemy} = {{\n"
        f"\tallowed = {{ original_tag = {tag} }}\n"
        f"\tenable = {{ has_war_with = {enemy} }}\n"
        f"\tabort = {{ always = no }}\n\n"
        f"\tai_strategy = {{\n\t\ttype = front_control\n\t\ttag = {enemy}\n"
        f"\t\tratio = 0.1\n\t\tpriority = 100\n\t\tordertype = front\n"
        f"\t\texecution_type = rush\n\t\texecute_order = yes\n\t\tmanual_attack = yes\n\t}}\n"
        f"}}\n"
        for tag in plan.tags
        for enemy in plan.tags
        if enemy != tag
    )


def underdog_idea(bonus):
    """The idea a lone side can be given to even an unfair setup: its attack and defence
    raised by `bonus` (0.25 is +25%)."""
    return (
        "\t\tarena_underdog = {\n"
        "\t\t\tallowed = { always = no }\n\t\t\tremoval_cost = -1\n"
        f"\t\t\tmodifier = {{ army_attack_factor = {bonus:g} army_defence_factor = {bonus:g} }}\n"
        "\t\t}\n"
    )

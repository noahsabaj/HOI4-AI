"""The state channel: the arena's whole game state, logged by its mod every game day.

Training may read the game's state; the player that plays never does (the goal binds
pixels only at play time). Arenas since v3 already log a `day` line per side (arena_log).
This adds what a teacher that acts on state needs and the screen shows only in part:

    provinces 1 489 490 491 492 513 514 515 516 537 538 539 540
    state  23:00, 1 January, 1936 RED day 706640 held 65280 mask 0 ... 4095 law 1 economy 2
        trade 1 pp 1.2 command 0.6 stability 0.95 war 1 manpower 0 max 1001.248 queue 0
        stock 50000 orders 1 battalions 48
    army  23:00, 2 January, 1936 RED units 8 plans 1 orders 1 planning 0.05128 ready 0
        combat 8 attack 8 defend 0 progress 0.49955 entrench 0 rifles 4800 needed 4800
        group 0 at 1=0 ... 15=4 16=4 name Red Marshal

(as logged in a live game here on 2026-09-26, with the "ARENA " prefix removed).

- `provinces`, once at startup: each state's land provinces, in the order of its mask bits.
- `state`, each day for each side:
  - `day`: the game's own day count (global.num_days, from year 1);
  - `held`: a bit per state (state s is bit s-1) that the side controls. A game variable
    holds about 21 bits, so an arena of more than 16 states (the multi-nation arenas) logs
    it as one number per 16 states, lowest first;
  - `mask`: per state, a bit per province (in `provinces` order) that it controls; the two
    sides' masks for a state add up to all its provinces;
  - the conscription, economy and trade laws as indexes into CONSCRIPTION, ECONOMY and
    TRADE (-1 for none of them);
  - political and command power, stability and war support (0 to 1), manpower and its
    ceiling (both in thousands), manpower in the deployment queue, rifles in the
    stockpile;
  - `orders`: its orders groups (armies and army groups), and `battalions`.
- `army`, each day for each leader in command: its divisions, battle plans (fronts and
  offensives drawn for it), planning bonus and share of divisions ready for the plan,
  divisions fighting (attacking, defending) and the average progress of their battles,
  entrenchment, rifles held and needed, its divisions per state, and `group`, the game's
  is_leading_army_group. The scripted player forms one army under one general, where
  this is exact. The game's AI splits its divisions over the marshal and all three
  generals, and in the live check two of the four read `group` 1 and their divisions
  summed to 11 of 8, so for the AI, trust `units` per leader rather than their sum.

Every line starts "ARENA " in game.log, like the rest of the mod's report, and arena_log's
reader skips kinds it does not know, so older readers are not disturbed. The script only
reads and logs: it sets temporary variables and changes no rule, no random draw and
nothing a player sees.

`upgrade` rewrites an existing arena's on_actions with it, so a map and its start saves
stay exactly as they were. `Channel` folds the lines into the latest state of each side,
and `snapshot` returns that as plain numbers.
"""

from __future__ import annotations

import re
from pathlib import Path

from .arena_log import DATE

# Law ideas in the order a player climbs them; the logged number indexes these.
CONSCRIPTION = (
    "disarmed_nation", "volunteer_only", "limited_conscription", "extensive_conscription",
    "service_by_requirement", "all_adults_serve", "scraping_the_barrel",
)  # fmt: skip
ECONOMY = (
    "undisturbed_isolation", "isolation", "civilian_economy", "low_economic_mobilisation",
    "partial_economic_mobilisation", "war_economy", "tot_economic_mobilisation",
)  # fmt: skip
TRADE = ("free_trade", "export_focus", "limited_exports", "closed_economy")
# States per `held` number: a variable is fixed point, and 2^21 is about its largest.
HELD_BITS = 16

# Number variables of the country, as logged: (field, script value).
COUNTRY = (
    ("pp", "political_power"),
    ("command", "command_power"),
    ("stability", "stability"),
    ("war", "has_war_support"),
    ("manpower", "manpower_k"),
    ("max", "max_manpower_k"),
    ("queue", "amount_manpower_in_deployment_queue"),
    ("stock", "num_equipment@infantry_equipment"),
    ("orders", "num_orders_groups"),
    ("battalions", "num_battalions"),
)
# Number variables of an army's leader, as logged.
ARMY = (
    ("units", "num_units"),
    ("plans", "num_battle_plans"),
    ("orders", "has_orders_group"),
    ("planning", "avg_unit_planning_ratio"),
    ("ready", "unit_ratio_ready_for_plan"),
    ("combat", "num_units_in_combat"),
    ("attack", "num_units_offensive_combats"),
    ("defend", "num_units_defensive_combats"),
    ("progress", "avg_combat_status"),
    ("entrench", "avg_unit_entrenchment_ratio"),
    ("rifles", "num_equipment@infantry_equipment"),
    ("needed", "num_target_equipment@infantry_equipment"),
)


def _laws(variable, ideas):
    """Sets `variable` to the index of the idea in `ideas` the country has, else -1."""
    chain = "".join(
        f" if = {{ limit = {{ has_idea = {idea} }} set_temp_variable = {{ {variable} = {i} }} }}"
        for i, idea in enumerate(ideas)
    )
    return f" set_temp_variable = {{ {variable} = -1 }}{chain}"


def startup_effect(states):
    """Logs each state's provinces once, in the order of the daily masks' bits."""
    return "".join(
        f' log = "ARENA provinces {state} {" ".join(map(str, provinces))}"'
        for state, provinces in sorted(states.items())
    )


def daily_effect(states):
    """The effect run by each side every day: its `state` line, then an `army` line for
    each of its leaders in command. `states` maps state id to its land provinces."""
    ids = sorted(states)
    chunks = (max(ids, default=1) - 1) // HELD_BITS + 1
    held = ["arena_held"] + [f"arena_held{c + 1}" for c in range(1, chunks)]
    parts = [f" set_temp_variable = {{ {name} = 0 }}" for name in held]
    for state in ids:
        parts.append(
            f" if = {{ limit = {{ {state} = {{ is_controlled_by = ROOT }} }}"
            f" add_to_temp_variable = {{ {held[(state - 1) // HELD_BITS]} ="
            f" {2 ** ((state - 1) % HELD_BITS)} }} }}"
        )
        parts.append(f" set_temp_variable = {{ arena_m{state} = 0 }}")
        for bit, province in enumerate(states[state]):
            parts.append(
                f" if = {{ limit = {{ controls_province = {province} }}"
                f" add_to_temp_variable = {{ arena_m{state} = {2**bit} }} }}"
            )
    parts.append(_laws("arena_law", CONSCRIPTION))
    parts.append(_laws("arena_economy", ECONOMY))
    parts.append(_laws("arena_trade", TRADE))
    parts.extend(f" set_temp_variable = {{ arena_{name} = {value} }}" for name, value in COUNTRY)
    parts.append(" set_temp_variable = { arena_days = global.num_days }")
    parts.append(
        ' log = "ARENA state [GetDateText] [ROOT.GetTag] day [?arena_days] held'
        + "".join(f" [?{name}]" for name in held)
        + " mask"
        + "".join(f" [?arena_m{state}]" for state in ids)
        + " law [?arena_law] economy [?arena_economy] trade [?arena_trade]"
        + "".join(f" {name} [?arena_{name}]" for name, _ in COUNTRY)
        + '"'
    )
    army = (
        "".join(f" set_temp_variable = {{ arena_a_{name} = {value} }}" for name, value in ARMY)
        + " set_temp_variable = { arena_a_group = 0 }"
        " if = { limit = { is_leading_army_group = yes } set_temp_variable = { arena_a_group = 1 } }"
        + "".join(
            f" set_temp_variable = {{ arena_a{state} = num_units_in_state@{state} }}"
            for state in ids
        )
        + ' log = "ARENA army [GetDateText] [ROOT.GetTag]'
        + "".join(f" {name} [?arena_a_{name}]" for name, _ in ARMY)
        + " group [?arena_a_group] at"
        + "".join(f" {state}=[?arena_a{state}]" for state in ids)
        + ' name [THIS.GetName]"'
    )
    parts.append(f" every_army_leader = {{ limit = {{ is_assigned = yes }}{army} }}")
    return "".join(parts)


def mod_states(root):
    """A generated arena's states and their provinces, from its history/states files."""
    states = {}
    for path in sorted((Path(root) / "history" / "states").glob("*.txt")):
        text = path.read_text()
        state = int(re.search(r"\bid\s*=\s*(\d+)", text).group(1))
        listed = re.search(r"provinces\s*=\s*\{([^}]*)\}", text).group(1)
        states[state] = [int(p) for p in listed.split()]
    return states


MARKER = "ARENA state "


def upgrade(root):
    """Adds the state channel to an arena generated before it, in place, touching only
    common/on_actions/arena.txt: the startup provinces and each side's daily report go at
    the end of its on_startup and on_daily_TAG effects. True if it changed the file; an
    arena that has the channel already is left alone."""
    path = Path(root) / "common" / "on_actions" / "arena.txt"
    text = path.read_text()
    if MARKER in text:
        return False
    states = mod_states(root)
    daily = daily_effect(states)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("on_daily_", "on_startup")):
            if not stripped.endswith("} }"):
                raise ValueError(f"unexpected on_actions line in {path}: {stripped[:60]}")
            extra = startup_effect(states) if stripped.startswith("on_startup") else daily
            cut = line.rstrip().rfind("} }")
            line = f"{line[:cut].rstrip()} {extra.strip()} }} }}"
        lines.append(line)
    path.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""))
    return True


# The lines, with the "ARENA " prefix the worker removes.
NUMBER = r"-?[\d.]+"
PATTERNS = {
    "provinces": re.compile(r"^provinces (?P<state>\d+)(?P<list>(?: \d+)+)$"),
    "state": re.compile(
        rf"^state\s+{DATE} (?P<tag>[A-Z]{{3}}) day (?P<day>{NUMBER})"
        rf" held (?P<held>{NUMBER}(?: {NUMBER})*)"
        rf" mask(?P<mask>(?: {NUMBER})+) law (?P<law>{NUMBER}) economy (?P<economy>{NUMBER})"
        rf" trade (?P<trade>{NUMBER})(?P<rest>.*)$"
    ),
    "army": re.compile(
        rf"^army\s+{DATE} (?P<tag>[A-Z]{{3}})(?P<rest>.*?) at(?P<at>(?: \d+={NUMBER})*)"
        r" name (?P<name>.*)$"
    ),
}


def _fields(text):
    """ "a 1 b 2.5" as {"a": 1.0, "b": 2.5}."""
    words = text.split()
    return {words[i]: float(words[i + 1]) for i in range(0, len(words) - 1, 2)}


def parse(line):
    """A state-channel line as a dict with its `kind`, or None for any other line."""
    line = line.strip()
    for kind, pattern in PATTERNS.items():
        match = pattern.match(line)
        if not match:
            continue
        found = match.groupdict()
        if kind == "provinces":
            return {"kind": kind, "state": int(found["state"]), "provinces": [
                int(p) for p in found["list"].split()
            ]}  # fmt: skip
        event = {"kind": kind, "date": found["date"], "tag": found["tag"]}
        event.update(_fields(found["rest"]))
        if kind == "state":
            event["day"] = int(float(found["day"]))
            event["held"] = sum(
                int(float(part)) << (HELD_BITS * c) for c, part in enumerate(found["held"].split())
            )
            event["mask"] = [int(float(m)) for m in found["mask"].split()]
            for key in ("law", "economy", "trade"):
                event[key] = int(float(found[key]))
        else:
            pairs = (item.split("=") for item in found["at"].split())
            event["at"] = {int(s): round(float(n)) for s, n in pairs}
            event["name"] = found["name"].strip()
        return event
    return None


class Channel:
    """The latest state of each side, folded from the channel's lines as they are read.

    `feed` takes a line (with or without the "ARENA " prefix) and returns its event, or
    None for a line of another kind. Armies are reported one line each after their side's
    `state` line, so a side's armies are replaced whenever its next `state` line comes.
    """

    def __init__(self):
        self.provinces = {}
        self.states = {}
        self.armies = {}
        self._pending = {}

    def feed(self, line):
        event = parse(line.removeprefix("ARENA "))
        if event is None:
            return None
        if event["kind"] == "provinces":
            self.provinces[event["state"]] = event["provinces"]
        elif event["kind"] == "state":
            self.states[event["tag"]] = event
            self.armies[event["tag"]] = self._pending[event["tag"]] = []
        else:
            self._pending.setdefault(event["tag"], []).append(event)
            self.armies.setdefault(event["tag"], self._pending[event["tag"]])
        return event

    def controllers(self, tag):
        """{province: True if `tag` controls it}, from its latest mask."""
        state = self.states.get(tag)
        if state is None:
            return {}
        control = {}
        for mask, (_, listed) in zip(state["mask"], sorted(self.provinces.items())):
            for bit, province in enumerate(listed):
                control[province] = bool(mask >> bit & 1)
        return control

    def snapshot(self, tag):
        """`tag`'s latest state as plain numbers: None before its first report."""
        state = self.states.get(tag)
        if state is None:
            return None
        leaders = self.armies.get(tag, [])
        # An army group's leader counts its armies' divisions again.
        armies = [a for a in leaders if not a.get("group")]
        return {
            "date": state["date"],
            "day": state["day"],
            "held": [bool(state["held"] >> s & 1) for s in range(len(state["mask"]))],
            "provinces": sum(bin(m).count("1") for m in state["mask"]),
            "law": state["law"],
            "economy": state["economy"],
            "trade": state["trade"],
            **{name: state.get(name) for name, _ in COUNTRY},
            "armies": len(armies),
            "commanded": sum(a.get("units", 0) for a in armies),
            "groups": len(leaders) - len(armies),
            "plans": sum(a.get("plans", 0) for a in leaders),
            "fighting": sum(a.get("combat", 0) for a in armies),
        }

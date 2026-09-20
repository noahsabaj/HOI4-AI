"""Live benchmark of TypeSafe Jev on HOI4 decisions (22 single-Choice cases + one fan-out call).

Costs ~23 real API calls. Needs TYPESAFE_API_KEY in the environment or the repo-root .env.
The cases are hand-written situations, not game captures: this measures Jev's judgment and
latency, it is not evidence about playing strength.

    ./.venv/Scripts/python.exe scripts/bench_jev.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hoi4_agent.brain.jev import JevClient, choice, noul  # noqa: E402
from hoi4_agent.errors import AgentError  # noqa: E402

P = "You are an expert Hearts of Iron IV player. "
UNITS = [{"id": 1, "org": 35}, {"id": 2, "org": 91}, {"id": 3, "org": 58}, {"id": 4, "org": 12}]
FRONT = {"north": {"ours": 2, "enemy": 6}, "center": {"ours": 5, "enemy": 5}, "south": {"ours": 5, "enemy": 1}}
IDS = {"1": None, "2": None, "3": None, "4": None}
SECTORS = {"north": None, "center": None, "south": None}
CASES = [
    ("semantic", {"country": "Germany", "date": "1936-03", "event": "Remilitarization of the Rhineland: France has not reacted to our troops entering the Rhineland."},
     P + "Which event option is best?", {"a": "Excellent. (gain political power)", "b": "Withdraw the troops and apologize."}, {"a"}),
    ("semantic", {"country": "Germany", "date": "1936-01", "goal": "maximize long-run industrial capacity", "free_civilian_factories": 15},
     P + "What should the free civilian factories build first?", {"civ": "Civilian factories", "mil": "Military factories", "fort": "Land forts on the French border", "aa": "Anti-air"}, {"civ"}),
    ("semantic", {"country": "Germany", "date": "1936-01", "research_slots_free": 1, "goal": "economy first"},
     P + "Which technology should be researched first?", {"construction_1": "Construction I (+10% construction speed)", "naval_bomber_1": "Naval Bomber I", "heavy_cruiser_2": "Heavy cruiser 1936 hull", "rocket_engines": "Rocket engines"}, {"construction_1"}),
    ("semantic", {"country": "Germany", "date": "1936-01"},
     P + "Which national focus is the standard strong opener?", {"rhineland": "Rhineland", "four_year": "Four Year Plan", "naval": "Naval Rearmament", "oppose": "Oppose Hitler"}, {"rhineland", "four_year"}),
    ("semantic", {"screen_text": "PAUSED. Menu: Resume, Save Game, Options, Exit to main menu"},
     "The automated player wants to return to the map and continue playing. Which button?", {"resume": "Resume", "save": "Save Game", "options": "Options", "exit": "Exit to main menu"}, {"resume"}),
    ("semantic", {"popup": "Our justification on Poland has completed. We can now declare war.", "army_ready": False, "note": "Army is mid-reorganization and understrength; allies not yet called."},
     P + "What to do now?", {"declare": "Declare war immediately", "wait": "Close the popup and wait until the army is ready"}, {"wait"}),
    ("semantic", {"event": "A delegation from Hungary requests to join our faction.", "country": "Germany", "at_war_with": ["Poland", "France", "United Kingdom"]},
     P + "Accept?", {"accept": "Accept them into the faction", "refuse": "Refuse"}, {"accept"}),
    ("semantic", {"trade": "We lack rubber (-24) and oil (-10). Production of trucks and fighters is penalized.", "free_civilian_factories": 6},
     P + "Best immediate fix?", {"import": "Trade civilian factories for rubber and oil imports", "ignore": "Ignore the shortage", "forts": "Build forts", "navy": "Build battleships"}, {"import"}),
    ("semantic", {"division_template": "7 infantry battalions + 2 artillery, engineers, recon support", "enemy": "entrenched infantry in mountains behind a river"},
     P + "How should this attack be approached?", {"frontal": "Attack immediately across the river with everything", "flank": "Look for a flank or non-river tile, use planning bonus and air support first", "retreat": "Retreat from the whole front"}, {"flank"}),
    ("semantic", {"alert": "12 divisions are out of supply in Libya; attrition is high and organization is dropping."},
     P + "Best response?", {"more": "Send 12 more divisions to Libya", "fewer": "Pull some divisions out and improve the supply hub/ports", "attack": "Order an all-out offensive"}, {"fewer"}),
    ("semantic", {"situation": "Three of our divisions have broken through and are 1 province from closing a pocket around 9 enemy divisions. The only exit of the pocket is that province."},
     P + "Best order?", {"close": "Move into the exit province to close the encirclement", "halt": "Halt and dig in", "back": "Pull the spearhead back to the start line"}, {"close"}),
    ("semantic", {"situation": "Our 4 divisions hold a river line in forest. 10 fresh enemy divisions are massing opposite. No reinforcements are available for 30 days."},
     P + "Best order?", {"hold": "Stay entrenched behind the river and defend", "attack": "Attack across the river now", "spread": "Spread out to attack everywhere at once"}, {"hold"}),
    ("numeric", {"our_division": {"org": 22, "strength": 61}, "enemy_division": {"org": 78, "strength": 96}, "terrain": "plains"},
     P + "Organization and strength are percentages. Should our division attack this enemy alone?", {"attack": "Attack", "dont": "Do not attack"}, {"dont"}),
    ("numeric", {"provinces": {"A": {"enemy_divisions": 5}, "B": {"enemy_divisions": 1}, "C": {"enemy_divisions": 3}}, "ours_adjacent_to_all": 4},
     P + "Which province is the weakest point to attack?", {"A": None, "B": None, "C": None}, {"B"}),
    ("numeric", {"battle": {"our_org_avg": 64, "enemy_org_avg": 9, "progress_bar": "87 in our favour"}},
     P + "Is this battle being won; keep attacking or cancel?", {"continue": "Continue the attack", "cancel": "Cancel the attack"}, {"continue"}),
    ("numeric", {"battle": {"our_org_avg": 11, "enemy_org_avg": 70, "progress_bar": "14 in our favour"}},
     P + "Keep attacking or cancel?", {"continue": "Continue the attack", "cancel": "Cancel the attack"}, {"cancel"}),
    ("numeric", {"units": UNITS}, "Which unit has the highest organization?", IDS, {"2"}),
    ("numeric", {"units": UNITS}, "Which unit most needs to be pulled off the line to recover?", IDS, {"4"}),
    ("numeric", {"supply": {"hub_capacity": 20, "divisions_drawing": 34}},
     P + "Is this area oversupplied or undersupplied?", {"under": "Undersupplied: too many divisions for the hub", "over": "Fine: capacity exceeds demand"}, {"under"}),
    ("spatial", {"adjacency": {"P1": ["P2"], "P2": ["P1", "P3", "P4"], "P3": ["P2", "P5"], "P4": ["P2", "P5"], "P5": ["P3", "P4", "CAPITAL"], "CAPITAL": ["P5"]}, "our_unit_in": "P1", "enemy_units_in": ["P3"]},
     "Our unit wants to reach CAPITAL while avoiding enemy units. After P2, which province should it enter?", {"P3": None, "P4": None, "P1": None}, {"P4"}),
    ("spatial", {"front": FRONT}, P + "Where should we launch the main offensive?", SECTORS, {"south"}),
    ("spatial", {"front": FRONT}, P + "Which sector most urgently needs reinforcements to avoid a breakthrough against us?", SECTORS, {"north"}),
]


def main() -> int:
    client = JevClient()
    rows: list[tuple[str, bool, float, float]] = []  # category, correct, confidence, latency
    tokens = [0, 0]
    for category, state, instructions, options, wanted in CASES:
        try:
            result = client.ask(state, {"q": choice(instructions, options)}, timeout=30)
        except AgentError as exc:
            print(f"ERROR {type(exc).__name__}: {exc}")
            return 1
        answer = result.choice("q")
        ok = answer.choice in wanted
        rows.append((category, ok, answer.confidence, result.latency_s))
        tokens = [tokens[0] + result.input_tokens, tokens[1] + result.output_tokens]
        short = instructions.removeprefix(P)
        print(f"{category:9s} {'OK ' if ok else 'BAD'} got={answer.choice:15s} conf={answer.confidence:.2f} "
              f"{result.latency_s * 1000:5.0f} ms | {short[:62]}")

    print("\ncategory    correct   wrong-answer confidences")
    for category in ("semantic", "numeric", "spatial", "all"):
        sub = [row for row in rows if category in (row[0], "all")]
        wrong = ", ".join(f"{row[2]:.2f}" for row in sub if not row[1]) or "-"
        print(f"{category:9s}   {sum(row[1] for row in sub):2d} / {len(sub):2d}   {wrong}")
    latencies = sorted(row[3] for row in rows)
    print(f"latency ms: median {statistics.median(latencies) * 1000:.0f}  "
          f"p90 {latencies[int(len(latencies) * 0.9) - 1] * 1000:.0f}  max {latencies[-1] * 1000:.0f}")
    print(f"tokens: input {tokens[0]}  output {tokens[1]}  over {len(rows)} calls")

    questions = {"offensive": choice(CASES[20][2], SECTORS), "reinforce": choice(CASES[21][2], SECTORS)}
    for name in SECTORS:
        questions["outnumbered_" + name] = noul(f"Are we outnumbered in the {name} sector?")
    fan = client.ask({"front": FRONT}, questions, timeout=30)
    print(f"\nfan-out, {len(questions)} questions in one request: {fan.latency_s * 1000:.0f} ms "
          f"(in={fan.input_tokens} out={fan.output_tokens})")
    print(f"  offensive={fan.choice('offensive').choice} ({fan.choice('offensive').confidence:.2f})  "
          f"reinforce={fan.choice('reinforce').choice} ({fan.choice('reinforce').confidence:.2f})  " +
          "  ".join(f"outnumbered_{name}={fan.noul('outnumbered_' + name).probability:.2f}" for name in SECTORS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

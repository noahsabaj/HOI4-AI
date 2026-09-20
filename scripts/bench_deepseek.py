"""Live benchmark of DeepSeek V4.1 Flash on HOI4 screenshot tasks (19 calls: 18 scored, 1 unscored).

Images are repo files only: ``templates/*.png`` (real HOI4 UI crops) and the two samples under
``tests/data/arena``. Needs DEEPSEEK_API_KEY in the environment or the repo-root .env.
Pass ``think`` to run with thinking enabled (slower, several times the output tokens).

    ./.venv/Scripts/python.exe scripts/bench_deepseek.py [think]
"""
from __future__ import annotations

import json
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hoi4_agent.brain.deepseek import DeepSeekClient  # noqa: E402
from hoi4_agent.errors import AgentError  # noqa: E402

SYSTEM = "You read Hearts of Iron IV screenshots. Output JSON only."
MAP = ROOT / "tests/data/arena/sample_map_600x400.png"
CROP = ROOT / "tests/data/arena/sample_counter_crop.png"
MENU = ROOT / "templates/pause_menu.png"
LOC = ' Reply {"x": X, "y": Y}: the CENTER of the target, integers normalized 0-1000 on this image (0,0 top-left).'
MENU_LABELS = ["load game", "save game", "game options", "playthrough overview", "hearts of iron wiki",
               "exit to menu", "exit game", "close"]
# The indicator has five segments. Lit segments 1-4 are bright green; the fifth lights up RED.
SPEED = ('This is the game-speed indicator: a row of 5 segments. A lit segment is brightly coloured: bright '
         'green for segments 1 to 4, bright red for segment 5. An unlit segment is dark. How many segments are '
         'lit? Reply {"value": N}.')

Check = Callable[[dict[str, Any]], bool]


def near(got: dict[str, Any], want: tuple[int, int], tolerance: float = 40) -> bool:
    try:
        return ((got["x"] - want[0]) ** 2 + (got["y"] - want[1]) ** 2) ** 0.5 <= tolerance
    except (KeyError, TypeError):
        return False


def lit(k: int) -> Check:
    return lambda got: got.get("value") == k


TASKS: list[tuple[str, Path, int, str, Check | None]] = [
    (f"count  speed_{k}", ROOT / f"templates/speed_{k}.png", 4, SPEED, lit(k)) for k in range(1, 6)]
TASKS += [
    ("ocr    menu buttons", MENU, 1, 'List every button label top to bottom. Reply {"labels": [..]}.',
     lambda g: [str(s).lower() for s in g.get("labels", [])] == MENU_LABELS),
    ("locate menu Close", MENU, 1, "Locate the 'Close' button." + LOC, lambda g: near(g, (505, 763))),
    ("locate menu Save Game", MENU, 1, "Locate the 'Save Game' button." + LOC, lambda g: near(g, (505, 160))),
    ("count  unit counters", MAP, 1, "How many unit counters (small rectangular boxes with a flag, an icon and a "
     'number) are on the map? Reply {"value": N}.', lambda g: g.get("value") == 2),
    ("ocr    counter numbers", MAP, 1, "Read the number shown on each unit counter, left to right. "
     'Reply {"numbers": [..]}.', lambda g: g.get("numbers") == [5, 7]),
    ("ocr    city", MAP, 1, 'Which city is labelled on the map? Reply {"city": "..."}.',
     lambda g: str(g.get("city", "")).lower() == "warsaw"),
    ("ocr    arrow label", MAP, 1, 'What does the text along the blue battle-plan arrow say? Reply {"text": "..."}.',
     lambda g: "12 divisions" in str(g.get("text", "")).lower() and "army 1" in str(g.get("text", "")).lower()),
    ("locate counter 5", MAP, 1, "Locate the unit counter showing the number 5." + LOC, lambda g: near(g, (265, 580))),
    ("locate counter 7", MAP, 1, "Locate the unit counter showing the number 7." + LOC, lambda g: near(g, (425, 757))),
    ("locate Warsaw star", MAP, 1, "Locate the star marking Warsaw." + LOC, lambda g: near(g, (858, 243))),
    ("reason closer counter", MAP, 1, "Which unit counter is closer to Warsaw, the one showing 5 or the one "
     'showing 7? Reply {"value": N}.', lambda g: g.get("value") == 7),
    ("reason plan target", MAP, 1, 'The blue arrow is an offensive plan. Which city does it target? '
     'Reply {"city": "..."}.', lambda g: str(g.get("city", "")).lower() == "warsaw"),
    ("ocr    crop number", CROP, 4, 'This is one unit counter. What number does it show? Reply {"value": N}.',
     lambda g: g.get("value") == 5),
    ("bars   crop (unscored)", CROP, 4, "This unit counter has two horizontal bars under the icon: the upper one "
     'is green, the lower one is orange/brown. Estimate how full each is, 0-100. Reply {"green": N, "orange": N}.',
     None),
]


def load(path: Path, scale: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image if scale == 1 else image.resize((image.width * scale, image.height * scale),
                                                 Image.Resampling.LANCZOS)


def main() -> int:
    thinking = len(sys.argv) > 1 and sys.argv[1] == "think"
    client = DeepSeekClient()
    groups: dict[str, list[bool]] = {}
    latencies: list[float] = []
    image_tokens: list[int] = []
    output_tokens = 0
    print("mode:", "thinking" if thinking else "non-thinking")
    for name, path, scale, prompt, check in TASKS:
        try:
            reply = client.chat(system=SYSTEM, user=prompt, images=[load(path, scale)], thinking=thinking,
                                max_tokens=4000 if thinking else 200, timeout=120)
            got = reply.json()
        except AgentError as exc:
            print(f"ERR {name:24s} {type(exc).__name__}: {str(exc)[:120]}")
            if check is not None:
                groups.setdefault(name.split()[0], []).append(False)
            continue
        latencies.append(reply.latency_s)
        image_tokens.append(reply.prompt_tokens)
        output_tokens += reply.completion_tokens
        mark = "   "
        if check is not None:
            ok = bool(check(got))
            groups.setdefault(name.split()[0], []).append(ok)
            mark = "OK " if ok else "BAD"
        print(f"{mark} {name:24s} {reply.latency_s * 1000:6.0f} ms in={reply.prompt_tokens} "
              f"out={reply.completion_tokens} -> {json.dumps(got)[:110]}")
    if not latencies:
        return 1
    print("\nskill      correct")
    for skill, results in (*groups.items(), ("all", [ok for results in groups.values() for ok in results])):
        print(f"{skill:9s}  {sum(results):2d} / {len(results):2d}")
    ordered = sorted(latencies)
    print(f"latency ms: median {statistics.median(ordered) * 1000:.0f}  "
          f"p90 {ordered[int(len(ordered) * 0.9) - 1] * 1000:.0f}  max {ordered[-1] * 1000:.0f}")
    print(f"prompt tokens per image call: min {min(image_tokens)}  max {max(image_tokens)}; "
          f"output tokens total {output_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

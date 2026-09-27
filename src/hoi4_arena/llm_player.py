"""A vision-language model plays one side of an arena game against the game's AI, from pixels.

The bitter-lesson test: no training, no script, no reading of the game's numbers. The
model sees the screen as a player does, one picture per turn, and answers with mouse and
keyboard actions, which the second PC's worker applies. The arena log is read here only
to referee the game, as for every player.

The model thinks for seconds, so the game is played in turns. It starts paused; each turn
the model sees the paused screen and gives its actions (orders can be given while paused,
as a player can), then the harness runs the game for the seconds the model asked for, at
speed 5, and pauses it again. The pause and the speed are the harness's; the model may
not press space or the speed keys, nor the console key or F12.

The model is reached through an OpenAI-style chat API (DeepSeek's by default). Its key is
read from DEEPSEEK_API_KEY (the process's environment, then the user's saved environment
on Windows) or ~/.deepseek/api_key, and is never written or logged. Every turn is kept:
the picture it saw, its reasoning, its answer, what was applied and what it cost.
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from .ai_games import SPEED_UP, Station, act, focus, on_screen, recentre, start_game, tap
from .arena_log import ArenaLog
from .desktop import DesktopError

log = logging.getLogger(__name__)

API = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
# The picture the model sees: the 1920x1080 screen at 70%, under the model's own
# ~1300 px limit, with a ruler every 100 px along the edges to read coordinates by.
VIEW = (1344, 756)
RULER = 100
# USD per million tokens (DeepSeek's price list, 2026-09-27): input from the cache, input
# not from the cache, output. Peak hours cost double.
PRICES = {"hit": 0.003, "miss": 0.15, "out": 0.6}
PEAK_UTC = ((1, 4), (6, 10))
SPACE, ESCAPE = 0x20, 0x1B
SPEED_CLICKS = 4
# Keys the model may press, by name. Not space (pause), + and - (speed), the console key
# or F-keys (F12 is the worker's emergency stop): the harness keeps those.
KEYS = {
    # Virtual-key codes of letters are the capitals' (0x41-0x5A); the small letters'
    # codes are the numpad's and F1-F11's, which the worker refuses.
    **{chr(c).lower(): c for c in range(ord("A"), ord("Z") + 1)},
    **{str(d): 0x30 + d for d in range(10)},
    "esc": ESCAPE, "enter": 0x0D, "tab": 0x09, "backspace": 0x08,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
}  # fmt: skip
# The worker refuses Alt and Delete (valid_event in the worker): not offered.
MODIFIERS = {"shift": 0x10, "ctrl": 0x11}
BUTTONS = {"left": 0, "right": 1, "middle": 2}
# Actions a turn may carry, and the game seconds it may ask for.
MAX_ACTIONS, MAX_RUN = 30, 5.0
# The share of the screen that must change for a turn's actions to count as doing
# something: the model is told when its actions left the screen as it was.
CHANGED = 0.01
# Turns in a row without game time before the harness runs the game anyway, against a
# model stuck on one screen. A player may stay paused as long as they like, and with 5 the
# model's setup (about 50 turns) left the AI weeks of free time: 40 (2026-09-27).
MAX_IDLE, IDLE_RUN = 40, 2.0

BRIEF = """You are playing Hearts of Iron IV, the grand-strategy game, as a human would: you see \
the screen and use the mouse and keyboard. This is a small custom arena map with two \
countries at war: BLU (blue) and RED (red). You play {country}; the game's own AI plays \
the other. You win when the enemy capitulates (you take enough of its land, above all its \
capital and victory points); you lose if you capitulate. Everything is decided on the \
battlefield, so use every tool the game gives you: armies, generals, battle plans (front \
lines and offensive arrows), mobilisation laws, recruiting and deploying divisions, and \
production if it helps.

How turns work: the game is paused while you decide. Each turn you get one screenshot \
({width}x{height} pixels, a ruler every 100 px on the top and left edges). You answer with \
actions, applied in order while the game stays paused (orders given while paused take \
effect), then the game runs at full speed for `run_seconds` real seconds (about 2-3 game \
days per second) and pauses again for your next turn. You cannot unpause, pause or change \
speed yourself; the space bar, + and - are not yours to press. Popups (events, messages) \
must be closed by you when they block the screen.

Coordinates are pixels in the screenshot you see, x from the left, y from the top. Read every position afresh from the current screenshot; never reuse coordinates from your notes, which may be wrong. Each turn you are told whether your last actions changed the screen at all: if they did not, they missed or did nothing, so try something different.

The camera: the game starts with the whole arena in view, zoomed fully out (scrolling out further does nothing). Scroll in (positive clicks) over a place to look closer; pan with the arrow keys or w/a/s/d. The map is small: two countries side by side. Your divisions show as unit counters on the map; the army cards at the bottom of the screen select whole armies. If {max_idle} turns in a row ask for no game time, the game runs {idle_run:.0f} s anyway: time only matters when it passes, so let it pass once your orders are in.

Answer with one JSON object and nothing else:
{{"notes": "what you see, what you intend, and what to remember next turn (short)",
  "actions": [ ... up to {max_actions} ... ],
  "run_seconds": 0-{max_run}}}

Actions:
{{"type": "click", "x": 100, "y": 200, "button": "left"|"right", "mods": ["shift"|"ctrl"]}}
{{"type": "double_click", "x": 100, "y": 200}}
{{"type": "drag", "x1": 100, "y1": 200, "x2": 300, "y2": 250, "button": "left"|"right", "mods": [...]}}
{{"type": "key", "key": "a"-"z"|"0"-"9"|"esc"|"enter"|"tab"|"backspace"|"left"|"up"|"right"|"down", "mods": [...]}}
{{"type": "scroll", "x": 100, "y": 200, "clicks": -3..3}}   (positive zooms in, negative out)
{{"type": "move", "x": 100, "y": 200}}   (hover, e.g. to read a tooltip next turn)
"button", "mods" are optional (left, none). run_seconds 0 means: no game time, just show me \
the screen again after these actions (use it to check a menu or finish a setup)."""


# The controls, as the game's tutorial or a friend would tell a new player (from what the
# scripted player was calibrated on, scripted.py). Given with --manual: without it the
# model lost its first two games at the controls, never forming an army (2026-09-27).
MANUAL = """

Controls you will need (Hearts of Iron IV):
- Esc closes the open window or panel. A right-click on another country's land opens diplomacy with it; it does not order anything.
- Your divisions start unassigned: the top bar shows a red "unassigned divisions" alert (red fists). Shift+click it to select all of them.
- With divisions selected, the green + in the army bar at the bottom of the screen creates an army from them, and selects it. The cards in that bar are armies, not units; a click on a card selects that army and shows the Battle Plans bar above it.
- A new army has no commander: click the portrait on its panel and pick a general.
- With the army selected: press z (Front Line tool) and click the border with the enemy to put the army on a front along it. Then press x (Offensive Line tool) and right-drag from the front into enemy land to draw an offensive. Then click the green arrow button above the army card to execute the plan (it activates the attack).
- The bin at the right end of the Battle Plans bar deletes the army's orders (right-click it, then confirm).
- q opens the political screen; its first law slot is conscription. A click lists the laws; picking a higher one and OK raises your manpower, for political power.
- u opens Recruit & Deploy: train more divisions and set where they deploy; new ones arrive unassigned (shift+click the alert, then right-click the army card to add them).
- The game's AI does all of the above within days: be quick about your setup, while the game is paused."""


def api_key():
    """DEEPSEEK_API_KEY from this process, the user's saved environment (setx, which a
    process started earlier does not see), or ~/.deepseek/api_key."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key and os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as env:
                key = winreg.QueryValueEx(env, "DEEPSEEK_API_KEY")[0]
        except OSError:
            key = None
    if not key:
        path = Path.home() / ".deepseek" / "api_key"
        key = path.read_text().strip() if path.exists() else None
    if not key:
        raise RuntimeError("no DEEPSEEK_API_KEY (environment or ~/.deepseek/api_key)")
    return key.strip()


def peak(now=None):
    now = now or datetime.now(timezone.utc)
    return now.weekday() < 5 and any(a <= now.hour < b for a, b in PEAK_UTC)


def cost(usage, prices=PRICES, at_peak=False):
    """USD for one call's `usage`, as DeepSeek reports it."""
    hit = usage.get("prompt_cache_hit_tokens", 0)
    miss = usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0) - hit)
    out = usage.get("completion_tokens", 0)
    usd = (hit * prices["hit"] + miss * prices["miss"] + out * prices["out"]) / 1e6
    return usd * (2 if at_peak else 1)


class BudgetSpent(RuntimeError):
    pass


class Brain:
    """The model behind the chat API: a screenshot and the last few turns' notes in, a
    turn's JSON out. Counts what it spends and refuses to go past `budget_usd`."""

    def __init__(self, *, model=MODEL, api=API, budget_usd=5.0, history=8, thinking=True,
                 timeout=180.0, key=None):  # fmt: skip
        self.model, self.api, self.budget, self.history = model, api, budget_usd, history
        self.thinking, self.timeout = thinking, timeout
        self._key = key or api_key()
        self.spent = 0.0
        self.calls = 0
        self.notes = []

    def reset(self):
        self.notes = []

    def decide(self, rgb, brief, status):
        if self.spent >= self.budget:
            raise BudgetSpent(f"spent ${self.spent:.3f} of ${self.budget:.2f}")
        picture = view(rgb)
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=85)
        data = base64.b64encode(buffer.getvalue()).decode()
        past = "\n".join(f"turn {t}: {n}" for t, n in self.notes[-self.history :])
        text = f"{status}\n\nYour notes from recent turns:\n{past or '(none yet)'}"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": brief},
                {"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
                ]},
            ],
            "max_tokens": 16000,
            # JSON mode: without it the model now and then answered in its own tool-call
            # markup instead (turn 1 of the first game, 2026-09-27).
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled" if self.thinking else "disabled"},
        }  # fmt: skip
        began = time.monotonic()
        reply = self._post(body)
        message = reply["choices"][0]["message"]
        usage = reply.get("usage") or {}
        usd = cost(usage, at_peak=peak())
        self.spent += usd
        self.calls += 1
        turn = parse_turn(message.get("content") or "")
        self.notes.append((self.calls, turn.get("notes", "")[:600]))
        return {
            "turn": turn,
            "content": message.get("content"),
            "reasoning": message.get("reasoning_content"),
            "usage": usage,
            "usd": round(usd, 5),
            "seconds": round(time.monotonic() - began, 1),
            "picture": picture,
        }

    def _post(self, body, tries=6):
        request = urllib.request.Request(
            self.api,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"},
        )
        for attempt in range(tries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")[:300]
                if error.code in (429, 500, 502, 503) and attempt + 1 < tries:
                    log.warning("the API refused (%s): %s; again in a moment", error.code, detail)
                    time.sleep(5 * (attempt + 1))
                    continue
                raise RuntimeError(f"the API refused ({error.code}): {detail}") from None
            except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
                # Dropped connections too: a reply cut short (IncompleteRead) ended the
                # fifth game, 2026-09-27.
                if attempt + 1 < tries:
                    log.warning("the API did not answer (%s); again in a moment", error)
                    time.sleep(5 * (attempt + 1))
                    continue
                raise RuntimeError(f"the API did not answer: {error}") from error
        raise RuntimeError("unreachable")


def view(rgb):
    """The screen as the model sees it: scaled to VIEW, with rulers on two edges."""
    picture = Image.fromarray(rgb).resize(VIEW, Image.BILINEAR)
    draw = ImageDraw.Draw(picture)
    width, height = VIEW
    # The top ruler sits under the game's top bar (56 px deep here), not over its numbers.
    for x in range(RULER, width, RULER):
        draw.line([(x, 56), (x, 64)], fill=(255, 255, 0), width=2)
        draw.text((x + 2, 62), str(x), fill=(255, 255, 0))
    for y in range(RULER, height, RULER):
        draw.line([(0, y), (8, y)], fill=(255, 255, 0), width=2)
        draw.text((10, y - 5), str(y), fill=(255, 255, 0))
    return picture


def parse_turn(content):
    """The model's JSON object, from an answer that may wrap it in words or a fence."""
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        # No game time for a turn that could not be read: the model looks again at once.
        return {"notes": "(no JSON)", "actions": [], "run_seconds": 0, "error": "no JSON"}
    try:
        turn = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        return {"notes": "(bad JSON)", "actions": [], "run_seconds": 0, "error": str(error)}
    return turn if isinstance(turn, dict) else {"actions": [], "run_seconds": 3}


def to_events(action, width=VIEW[0], height=VIEW[1]):
    """One action of the model's, as worker events (x, y as fractions of the screen).
    Raises ValueError for anything not allowed."""
    kind = action.get("type")

    def at(x, y):
        x, y = float(x) / width, float(y) / height
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"off the screen: {action}")
        return {"kind": "move", "x": x, "y": y}

    def press(button, down):
        return {"kind": "button", "button": BUTTONS[button], "down": down}

    mods = [MODIFIERS[m] for m in action.get("mods") or []]
    before = [{"kind": "key", "vk": vk, "down": True} for vk in mods]
    after = [{"kind": "key", "vk": vk, "down": False} for vk in reversed(mods)]
    button = action.get("button", "left")
    if button not in BUTTONS:
        raise ValueError(f"no button {button!r}")
    if kind == "click":
        body = [at(action["x"], action["y"]), press(button, True), press(button, False)]
    elif kind == "double_click":
        body = [at(action["x"], action["y"])] + [press(button, d) for d in (True, False) * 2]
    elif kind == "drag":
        start, end = at(action["x1"], action["y1"]), at(action["x2"], action["y2"])
        steps = [
            {"kind": "move", "x": start["x"] + (end["x"] - start["x"]) * i / 6,
             "y": start["y"] + (end["y"] - start["y"]) * i / 6}
            for i in range(1, 7)
        ]  # fmt: skip
        body = [start, press(button, True), *steps, press(button, False)]
    elif kind == "key":
        name = str(action.get("key", "")).lower()
        if name not in KEYS:
            raise ValueError(f"key {name!r} is not allowed")
        body = tap(KEYS[name])
    elif kind == "scroll":
        clicks = max(-3, min(3, int(action.get("clicks", 1))))
        body = [at(action["x"], action["y"])]
        body += [{"kind": "wheel", "delta": 120 if clicks > 0 else -120}] * abs(clicks)
    elif kind == "move":
        body = [at(action["x"], action["y"])]
    else:
        raise ValueError(f"no action {kind!r}")
    return before + body + after


def apply_turn(desk, actions):
    """Apply a turn's actions; the ones refused, with why."""
    refused = []
    events = []
    for action in (actions or [])[:MAX_ACTIONS]:
        try:
            events += to_events(action)
        except (ValueError, KeyError, TypeError) as error:
            refused.append({"action": action, "why": str(error)})
    if events:
        try:
            act(desk, events, pause=0.06)
        except DesktopError as error:
            if "invalid_event" not in str(error):
                raise
            refused.append({"action": "the turn's batch", "why": str(error)})
    return refused, len(events)


def changed(before, after):
    """The share of the screen that differs visibly between two frames (on a small grey
    copy of each, so the blinking pause mark and the moving pointer barely count)."""
    import numpy as np

    small = [
        np.asarray(Image.fromarray(rgb).convert("L").resize((320, 180)), dtype=np.int16)
        for rgb in (before, after)
    ]
    return float((np.abs(small[0] - small[1]) > 12).mean())


def is_paused(desk, rules, looks=8):
    """Whether the pause mark shows in any of a few frames: it blinks."""
    for _ in range(looks):
        if rules.matches("paused", on_screen(desk.capture(full=True)).rgb):
            return True
        time.sleep(0.25)
    return False


def pause(desk, rules):
    """Pause the game (space), making sure it took."""
    for _ in range(3):
        if is_paused(desk, rules):
            return
        act(desk, tap(SPACE))
    if not is_paused(desk, rules):
        raise RuntimeError("the game would not pause")


def run_for(desk, rules, seconds):
    """Run the paused game for `seconds`, then pause it again."""
    if seconds <= 0:
        return
    act(desk, [{"kind": "move", "x": 0.5, "y": 0.6}, *tap(SPACE)])
    time.sleep(seconds)
    act(desk, tap(SPACE))
    pause(desk, rules)


def set_speed(desk):
    """Speed 5, by clicks on the speed control's +, the game still paused."""
    press = [{"kind": "button", "button": 0, "down": d} for d in (True, False)]
    act(desk, [{"kind": "move", "x": SPEED_UP[0], "y": SPEED_UP[1]}, *press * SPEED_CLICKS])


def play_llm_game(desk, brain, root, *, rules, country, cap_minutes=60.0, max_turns=400,
                  arena_name=None, after_surrender=3.0, manual=False):  # fmt: skip
    """One game, `brain` playing `country` from the game paused at its start. Ends when
    the arena log names a winner, or at `cap_minutes` or `max_turns` (a draw)."""
    root = Path(root)
    (root / "turns").mkdir(parents=True, exist_ok=True)
    brain.reset()
    arena = ArenaLog(desk)
    arena.poll()
    brief = BRIEF.format(country=country, width=VIEW[0], height=VIEW[1],
                         max_actions=MAX_ACTIONS, max_run=int(MAX_RUN), max_idle=MAX_IDLE,
                         idle_run=IDLE_RUN) + (MANUAL if manual else "")  # fmt: skip
    outcome, reason = "timeout", None
    start = time.monotonic()
    turns = 0
    ran = 0.0
    idle = 0
    effect = None
    with (root / "turns.jsonl").open("a") as book:
        try:
            pause(desk, rules)
            set_speed(desk)
            # The arena in the middle of the screen, zoomed out, as the scripted player
            # starts; the start saves open on the camera wherever it was when saved.
            recentre(desk)
            while time.monotonic() - start < cap_minutes * 60 and turns < max_turns:
                if not focus(desk):
                    raise RuntimeError("could not bring the game window to the front")
                rgb = on_screen(desk.capture(full=True)).rgb
                turns += 1
                status = (
                    f"Turn {turns}. The game is paused. You play {country}. "
                    f"Game time run so far: {ran:.0f} s. Turns in a row without game time: "
                    f"{idle} (at {MAX_IDLE} the game runs {IDLE_RUN:.0f} s anyway)."
                )
                if effect is not None:
                    status += f" Your last turn's actions: {effect}."
                decision = brain.decide(rgb, brief, status)
                turn = decision.pop("turn")
                picture = decision.pop("picture")
                picture.save(root / "turns" / f"{turns:04d}.jpg", quality=80)
                refused, applied = apply_turn(desk, turn.get("actions"))
                change = None
                if applied:
                    time.sleep(0.3)
                    change = changed(rgb, on_screen(desk.capture(full=True)).rgb)
                    effect = (
                        f"changed {change:.0%} of the screen" if change >= CHANGED
                        else "changed NOTHING on the screen (they missed or did nothing)"
                    )  # fmt: skip
                else:
                    effect = "none were given"
                if refused:
                    effect += "; refused, not applied: " + "; ".join(
                        str(r["why"])[:80] for r in refused[:3]
                    )
                try:
                    seconds = max(0.0, min(MAX_RUN, float(turn.get("run_seconds", 3))))
                except (TypeError, ValueError):
                    seconds = 3.0
                idle = idle + 1 if seconds == 0 else 0
                if idle >= MAX_IDLE:
                    seconds, idle = IDLE_RUN, 0
                run_for(desk, rules, seconds)
                ran += seconds
                arena.poll()
                book.write(json.dumps({
                    "turn": turns, "t": round(time.monotonic() - start, 1), "turn_json": turn,
                    "refused": refused, "events": applied, "changed": change, "ran": seconds,
                    "spent": round(brain.spent, 4), "winner": arena.winner, **decision,
                }) + "\n")  # fmt: skip
                book.flush()
                log.info(
                    "turn %d: %d actions (%d refused), ran %.0f s, $%.3f; %s",
                    turns, len(turn.get("actions") or []), len(refused), seconds, brain.spent,
                    str(turn.get("notes", ""))[:160],
                )  # fmt: skip
                if arena.winner:
                    time.sleep(after_surrender)
                    outcome = arena.winner
                    break
        except BudgetSpent as error:
            reason = f"budget: {error}"
        except Exception as error:  # noqa: BLE001 - recorded with the game.
            reason = f"{type(error).__name__}: {error}"
            log.warning("game ended early: %s", reason)
    (root / "arena-log.txt").write_text("\n".join(arena.lines) + "\n")
    manifest = {
        "winner": outcome, "reason": reason, "started_as": country, "arena": arena_name,
        "declarer": arena.declarer, "players": arena.players, "surrendered": arena.surrendered,
        "turns": turns, "game_seconds_run": ran, "seconds": round(time.monotonic() - start),
        "model": brain.model, "thinking": brain.thinking, "manual": manual,
        "spent_usd": round(brain.spent, 4),
        "driver": "vision-language model from pixels, in turns; harness pauses and runs",
    }  # fmt: skip
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return outcome, reason, manifest


def evaluate_llm(output, *, games, peer, mod="arena-plains-v6", saves=None,
                 rules="artifacts/calibration-1080p/rules.json", countries=("BLU", "RED"),
                 budget_usd=5.0, cap_minutes=60.0, max_turns=400, model=MODEL,
                 thinking=True, seed=None, manual=False):  # fmt: skip
    """Up to `games` games of the model against the game's AI on the second PC, sides
    alternating, until the budget is spent. HOI4 is left closed. Results as play-policy's."""
    from .scripted import win_rate
    from .vision import ScreenRules

    out_root = Path(output)
    out_root.mkdir(parents=True, exist_ok=True)
    screen_rules = ScreenRules(rules)
    rng = random.Random(seed)
    brain = Brain(model=model, budget_usd=budget_usd, thinking=thinking)
    station = Station("peer", peer)
    results = []
    try:
        for index in range(games):
            country = countries[index % len(countries)]
            name = time.strftime("llm-peer-%Y%m%d-%H%M%S")
            entry = {"game": name, "station": "peer", "started_as": country, "arena": mod,
                     "model": model, "manual": manual,
                     "plan": {"variant": "llm-manual" if manual else "llm"}}  # fmt: skip
            entry["declare_drawn"] = rng.choice(("BLU", "RED"))
            save = (saves or {}).get(country)
            entry["start_save"] = save
            try:
                station.quit()
                station.launch(mod, save=save)
                if not save:
                    time.sleep(25)
                with station.connect() as desk:
                    if not focus(desk):
                        raise RuntimeError("could not bring the game window to the front")
                    start_game(
                        desk, screen_rules, out_root / f"{name}-start-failed.png", country, 5,
                        observe=False, declarer=entry["declare_drawn"], saved=bool(save),
                    )  # fmt: skip
                    log.info("%s: %s plays %s on %s", name, model, country, mod)
                    outcome, reason, manifest = play_llm_game(
                        desk, brain, out_root / name, rules=screen_rules, country=country,
                        cap_minutes=cap_minutes, max_turns=max_turns, arena_name=mod,
                        manual=manual,
                    )  # fmt: skip
            except (DesktopError, RuntimeError, OSError) as error:
                entry["error"] = f"{type(error).__name__}: {error}"
                log.warning("%s failed: %s", name, entry["error"])
            else:
                entry.update(winner=outcome, reason=reason, **{
                    k: manifest[k] for k in ("seconds", "turns", "declarer", "players",
                                             "spent_usd", "game_seconds_run")
                })  # fmt: skip
                log.info("%s: winner %s (%s)", name, outcome, reason or "played out")
            results.append(entry)
            (out_root / "results-llm.json").write_text(json.dumps(results, indent=2))
            if brain.spent >= brain.budget:
                log.info("budget spent: $%.3f", brain.spent)
                break
            if entry.get("error") or (entry.get("reason") and "budget" not in entry["reason"]):
                log.warning("%s ended early; stopping the test", name)
                break
    finally:
        try:
            station.quit()
        except Exception as error:  # noqa: BLE001 - the games are saved.
            log.warning("quit failed: %s", error)
    return {"results": results, "win_rate": win_rate(results) if results else {},
            "spent_usd": round(brain.spent, 4), "calls": brain.calls}  # fmt: skip

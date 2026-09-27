"""The second harness for a vision-language model playing an arena game: an agent loop.

The first (llm_player.play_llm_game) gave the model one screenshot a turn and took up to
30 actions back, applied blind: one missed click and the rest landed on the wrong screen,
and its setup took some 50 turns where the game's AI takes seconds. It learned nothing it
did not write in eight short notes, and nothing at all from one game to the next.

This harness gives the model what a player has around their eyes and hands, and nothing
about this game in particular:

- Closed-loop hands. Each action is a tool call that returns the screen after it, so the
  model acts, looks and acts again. Hovering is an action too, which shows tooltips.
- A fovea. `look` returns a region of the screen at full resolution (up to 3x), with a
  ruler in screenshot coordinates, to read small icons and text.
- Working memory. `remember` replaces a notebook the model keeps through the game, shown
  at the start of every turn.
- Learning between games. After each game the model rewrites a lessons document from
  its notebook, the outcome and its last turns; the next game starts with it.
- Time as a player has it. The game is paused while the model works; `end_turn` lets
  game time pass (at most MAX_RUN seconds at speed 5) and pauses again.

What stays as before: pixels only while playing (the arena log only referees), the
harness keeps pause and speed, and the keys the worker refuses are not offered.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .ai_games import act, focus, on_screen, recentre
from .arena_log import ArenaLog
from .desktop import DesktopError, EmergencyStop
from .llm_player import (
    CHANGED,
    KEYS,
    MANUAL,
    MODIFIERS,
    VIEW,
    BudgetSpent,
    changed,
    cost,
    pause,
    peak,
    run_for,
    set_speed,
    to_events,
    view,
)

log = logging.getLogger(__name__)
# The rulers' numbers, large enough to read in a scaled screenshot.
FONT = ImageFont.load_default(size=14)

# Game seconds a turn may let pass (about 12 game days at speed 5), and what a turn that
# never ends lets pass when it is ended for the model.
MAX_RUN, FORCED_RUN = 5.0, 2.0
# Tool calls in one turn before the harness ends it: a stuck model still sees time pass.
MAX_STEPS = 40
# Screenshots kept in the conversation of a turn; older ones become a line of text, so a
# long turn costs what a short one does.
KEEP_IMAGES = 3
# How long the screen is given to settle after an action before it is taken.
SETTLE = 0.35
# The notebook's and the lessons' length, in characters.
NOTEBOOK, LESSONS = 4000, 6000
# Times a game may lose its connection to the worker and carry on over a new one: the
# tunnel dropped a frame mid-game on 2026-09-27 ("Truncated frame").
RECONNECTS = 20
# The close-up's longest side, and its greatest magnification.
LOOK_SIZE, LOOK_ZOOM = 1024, 3.0

BRIEF = """You are playing Hearts of Iron IV, the grand-strategy game, as a human would: you see \
the screen and use the mouse and keyboard through tools. This is a small custom arena map \
with two countries at war, BLU (blue) and RED (red). You play {country}; the game's own AI \
plays the other. You win when the enemy capitulates (you take enough of its land, above \
all its capital and victory points); you lose if you capitulate. Everything is decided on \
the battlefield, so use every tool the game gives you.

How play works: the game is paused while you work, as long as you like. Every action \
(click, drag, key, scroll, hover) returns the screen as it is after it, so act, look, and \
act again. Orders given while paused take effect. When your orders are in, call end_turn \
to let game time pass (up to {max_run:.0f} real seconds, about 2-3 game days each); the \
game then pauses again and your next turn begins. You cannot pause, unpause or change the \
speed yourself. After {max_steps} actions in one turn the turn ends for you.

Screenshots are {width}x{height} pixels, with a yellow ruler every 100 px (x under the \
top bar, y along the left edge). All coordinates you give are in these screenshot pixels. \
Read positions from the newest screenshot, not from memory. Small icons and text are hard \
to read at this size: use look on a region to see it at full resolution. Tooltips show \
where the pointer rests: hover to read them.

Your notebook (remember) is shown to you at the start of every turn, and is all you keep \
from one turn to the next besides the screen: write down what you have learned (where \
things are, what worked, your plan). Lessons from your earlier games, if any, follow."""

REFLECT = """You just finished a game of Hearts of Iron IV on the arena map, playing {country} \
against the game's AI through the screen, mouse and keyboard. Outcome: {outcome}. It \
lasted {turns} turns and {seconds:.0f} seconds of game time.

Your notebook at the end:
{notebook}

Your notes from the last turns:
{last}

The lessons you started the game with:
{lessons}

Rewrite the lessons document for your next game on this map, so that you play faster and \
win. Keep what is still true, correct what was wrong, add what you learned: how the \
interface works (where things are, which keys and clicks do what, what failed), and what \
wins against this AI. Be concrete and brief, at most {limit} characters. Answer with the \
document only."""


def _tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_XY = {"x": {"type": "number"}, "y": {"type": "number"}}
_MODS = {"type": "array", "items": {"type": "string", "enum": sorted(MODIFIERS)}}
_BUTTON = {"type": "string", "enum": ["left", "right", "middle"]}
TOOLS = [
    _tool(
        "click",
        "Click at a point of the screenshot (optionally double, with a button and modifier "
        "keys held). Returns the screen after it.",
        {**_XY, "button": _BUTTON, "mods": _MODS, "double": {"type": "boolean"}},
        ["x", "y"],
    ),
    _tool(
        "drag",
        "Press a button at one point, move to another and let go (a right-drag, for "
        "example). Returns the screen after it.",
        {
            "x1": {"type": "number"},
            "y1": {"type": "number"},
            "x2": {"type": "number"},
            "y2": {"type": "number"},
            "button": _BUTTON,
            "mods": _MODS,
        },
        ["x1", "y1", "x2", "y2"],
    ),  # fmt: skip
    _tool(
        "key",
        "Press and release a key, optionally with modifiers held. Returns the screen after it.",
        {"key": {"type": "string", "enum": sorted(KEYS)}, "mods": _MODS},
        ["key"],
    ),
    _tool(
        "scroll",
        "Turn the mouse wheel over a point: positive clicks zoom the map in, negative out. "
        "Returns the screen after it.",
        {**_XY, "clicks": {"type": "integer", "minimum": -5, "maximum": 5}},
        ["x", "y", "clicks"],
    ),
    _tool(
        "hover",
        "Move the pointer to a point and rest there, to read its tooltip. Returns the "
        "screen after it.",
        _XY,
        ["x", "y"],
    ),
    _tool(
        "look",
        "A close look: the region of the current screen at x, y (its top-left corner, in "
        "screenshot pixels), w wide and h high, at full resolution with a ruler in "
        "screenshot pixels. Changes nothing in the game.",
        {**_XY, "w": {"type": "number"}, "h": {"type": "number"}},
        ["x", "y", "w", "h"],
    ),
    _tool(
        "remember",
        "Replace your notebook with this text (it is shown to you every turn).",
        {"text": {"type": "string"}},
        ["text"],
    ),
    _tool(
        "end_turn",
        "Let game time pass: the game runs this many real seconds, then pauses for your next turn.",
        {
            "seconds": {"type": "number", "minimum": 1, "maximum": MAX_RUN},
            "note": {"type": "string", "description": "What you did and expect, briefly"},
        },
        ["seconds"],
    ),  # fmt: skip
]


REMEMBER = next(t for t in TOOLS if t["function"]["name"] == "remember")
CONSOLIDATE = (
    "The turn is ending, and your notebook is all you will keep of it. Call remember now "
    "with your whole notebook: what you learned this turn (where things are, what each "
    "control did, what failed), what you have done so far in the game, and your plan. "
    "Keep what is still true from before."
)


def encode(picture):
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def close_up(rgb, x, y, w, h):
    """The region (x, y, w, h), given in view pixels, of full-resolution `rgb`, enlarged
    to at most LOOK_SIZE (and LOOK_ZOOM times), with a ruler in view pixels."""
    height, width = rgb.shape[:2]
    sx, sy = width / VIEW[0], height / VIEW[1]
    w, h = max(8.0, float(w)), max(8.0, float(h))
    x0 = int(max(0, min(width - 1, float(x) * sx)))
    y0 = int(max(0, min(height - 1, float(y) * sy)))
    x1 = int(max(x0 + 1, min(width, round((float(x) + w) * sx))))
    y1 = int(max(y0 + 1, min(height, round((float(y) + h) * sy))))
    crop = Image.fromarray(rgb[y0:y1, x0:x1])
    zoom = min(LOOK_ZOOM, LOOK_SIZE / max(crop.size))
    crop = crop.resize((max(1, round(crop.width * zoom)), max(1, round(crop.height * zoom))))
    draw = ImageDraw.Draw(crop)
    # A tick every 10, 20, 50 or 100 view pixels, whichever gives about eight.
    span = max(w, h)
    step = next((s for s in (10, 20, 50, 100) if span / s <= 10), 200)
    per_view_x = crop.width / ((x1 - x0) / sx)
    per_view_y = crop.height / ((y1 - y0) / sy)
    left, top = x0 / sx, y0 / sy
    for tick in range(int(left // step + 1) * step, int((x1 / sx) + 1), step):
        px = (tick - left) * per_view_x
        draw.line([(px, 0), (px, 10)], fill=(255, 255, 0), width=2)
        draw.text((px + 2, 10), str(tick), fill=(255, 255, 0), font=FONT)
    for tick in range(int(top // step + 1) * step, int((y1 / sy) + 1), step):
        py = (tick - top) * per_view_y
        draw.line([(0, py), (10, py)], fill=(255, 255, 0), width=2)
        draw.text((12, py - 5), str(tick), fill=(255, 255, 0), font=FONT)
    region = f"x {left:.0f}-{x1 / sx:.0f}, y {top:.0f}-{y1 / sy:.0f}"
    return crop, f"close-up of {region} (screenshot pixels), magnified {per_view_x:.1f}x"


def action_of(name, args):
    """A tool call's arguments as one of llm_player's actions (to_events reads them)."""
    if name == "click":
        kind = "double_click" if args.get("double") else "click"
        return {"type": kind, **{k: args[k] for k in ("x", "y") if k in args},
                "button": args.get("button", "left"), "mods": args.get("mods") or []}  # fmt: skip
    if name == "drag":
        return {"type": "drag", **args}
    if name == "key":
        return {"type": "key", "key": args.get("key"), "mods": args.get("mods") or []}
    if name == "scroll":
        return {"type": "scroll", "x": args["x"], "y": args["y"], "clicks": args.get("clicks", 1)}
    if name == "hover":
        return {"type": "move", "x": args["x"], "y": args["y"]}
    raise ValueError(f"no action {name!r}")


class Agent:
    """The model behind the chat API, in a loop of tool calls. Counts what it spends."""

    def __init__(self, brain, *, lessons_path=None, manual=False):
        self.brain = brain
        self.lessons_path = Path(lessons_path) if lessons_path else None
        self.lessons = ""
        if self.lessons_path and self.lessons_path.exists():
            self.lessons = self.lessons_path.read_text(encoding="utf-8")
        self.manual = manual
        self.notebook = ""
        self.notes = []

    def call(self, messages, tools=True):
        brain = self.brain
        if brain.spent >= brain.budget:
            raise BudgetSpent(f"spent ${brain.spent:.3f} of ${brain.budget:.2f}")
        body = {
            "model": brain.model,
            "messages": messages,
            "max_tokens": 16000,
            "thinking": {"type": "enabled" if brain.thinking else "disabled"},
        }
        if tools:
            body["tools"] = TOOLS if tools is True else tools
        began = time.monotonic()
        reply = brain._post(body)
        usage = reply.get("usage") or {}
        usd = cost(usage, at_peak=peak())
        brain.spent += usd
        brain.calls += 1
        return reply["choices"][0]["message"], usage, usd, time.monotonic() - began

    def system(self, country):
        text = BRIEF.format(country=country, width=VIEW[0], height=VIEW[1], max_run=MAX_RUN,
                            max_steps=MAX_STEPS)  # fmt: skip
        if self.manual:
            text += MANUAL
        text += "\n\nLessons from your earlier games:\n" + (self.lessons or "(none yet)")
        return text

    def reflect(self, root, *, country, outcome, turns, seconds):
        """Rewrite the lessons from this game; kept with the game and at lessons_path."""
        last = "\n".join(f"turn {t}: {n}" for t, n in self.notes[-12:]) or "(none)"
        prompt = REFLECT.format(country=country, outcome=outcome, turns=turns, seconds=seconds,
                                notebook=self.notebook or "(empty)", last=last,
                                lessons=self.lessons or "(none)", limit=LESSONS)  # fmt: skip
        message, _usage, usd, _ = self.call([{"role": "user", "content": prompt}], tools=False)
        lessons = (message.get("content") or "").strip()[:LESSONS]
        if lessons:
            self.lessons = lessons
            (Path(root) / "lessons.md").write_text(lessons, encoding="utf-8")
            if self.lessons_path:
                self.lessons_path.parent.mkdir(parents=True, exist_ok=True)
                self.lessons_path.write_text(lessons, encoding="utf-8")
        return lessons, usd


def prune(messages, keep=KEEP_IMAGES):
    """Drop all but the last `keep` screenshots from a turn's conversation, in place."""
    seen = 0
    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for index in range(len(content) - 1, -1, -1):
            if content[index].get("type") == "image_url":
                seen += 1
                if seen > keep:
                    content[index] = {"type": "text", "text": "[an earlier screenshot]"}


def screenshot_message(text, picture):
    return {"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": encode(picture)}},
    ]}  # fmt: skip


def play_agent_game(desk, agent, root, *, rules, country, cap_minutes=90.0, max_turns=400,
                    arena_name=None, after_surrender=3.0, recentred=True,
                    reconnect=None):  # fmt: skip
    """One game, `agent` playing `country` from the game paused at its start. Ends when
    the arena log names a winner, or at `cap_minutes` or `max_turns` (a draw); then the
    agent rewrites its lessons. `reconnect`, if given, opens a new connection to the
    worker: a game whose connection drops carries on over it (RECONNECTS times), the turn
    in progress played again from the paused screen."""
    root = Path(root)
    (root / "steps").mkdir(parents=True, exist_ok=True)
    agent.notebook, agent.notes = "", []
    arena = ArenaLog(desk)
    arena.poll()
    system = {"role": "system", "content": agent.system(country)}
    outcome, reason = "timeout", None
    start = time.monotonic()
    turns = 0
    ran = 0.0
    book = (root / "steps.jsonl").open("a", encoding="utf-8")
    try:
        pause(desk, rules)
        set_speed(desk)
        if recentred:
            recentre(desk)
        reconnects = 0
        while time.monotonic() - start < cap_minutes * 60 and turns < max_turns:
            turns += 1
            try:
                seconds, note = play_turn(desk, agent, root, book, system, country, turns, ran)
                run_for(desk, rules, seconds)
            except EmergencyStop:
                raise
            except DesktopError as error:
                if reconnect is None or reconnects >= RECONNECTS or "invalid_event" in str(error):
                    raise
                reconnects += 1
                log.warning("lost the worker (%s); connecting again (%d of %d)", error,
                            reconnects, RECONNECTS)  # fmt: skip
                # The worker takes one connection at a time: the dead one goes first.
                try:
                    desk.close()
                except Exception:  # noqa: BLE001 - it is already broken.
                    pass
                time.sleep(5)
                desk = reconnect()
                arena.desktop = desk
                if not focus(desk):
                    raise RuntimeError("could not bring the game window to the front") from error
                pause(desk, rules)
                turns -= 1
                continue
            ran += seconds
            agent.notes.append((turns, note))
            arena.poll()
            log.info("turn %d: ran %.0f s, $%.3f; %s", turns, seconds, agent.brain.spent,
                     note[:160])  # fmt: skip
            if arena.winner:
                time.sleep(after_surrender)
                outcome = arena.winner
                break
    except BudgetSpent as error:
        reason = f"budget: {error}"
    except Exception as error:  # noqa: BLE001 - recorded with the game.
        reason = f"{type(error).__name__}: {error}"
        log.warning("game ended early: %s", reason)
    finally:
        book.close()
    (root / "arena-log.txt").write_text("\n".join(arena.lines) + "\n")
    won = "won" if outcome == country else "lost" if outcome in ("BLU", "RED") else "no result"
    lessons_usd = 0.0
    try:
        lessons, lessons_usd = agent.reflect(root, country=country, outcome=f"you {won}",
                                             turns=turns, seconds=ran)  # fmt: skip
        log.info("lessons rewritten (%d characters)", len(lessons))
    except Exception as error:  # noqa: BLE001 - the game is kept either way.
        log.warning("could not rewrite the lessons: %s", error)
    manifest = {
        "winner": outcome, "reason": reason, "started_as": country, "arena": arena_name,
        "declarer": arena.declarer, "players": arena.players, "surrendered": arena.surrendered,
        "turns": turns, "game_seconds_run": ran, "seconds": round(time.monotonic() - start),
        "model": agent.brain.model, "thinking": agent.brain.thinking, "manual": agent.manual,
        "harness": "agent", "spent_usd": round(agent.brain.spent, 4),
        "lessons_usd": round(lessons_usd, 5), "notebook": agent.notebook,
        "driver": "vision-language model from pixels, tool calls while paused",
    }  # fmt: skip
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return outcome, reason, manifest


def play_turn(desk, agent, root, book, system, country, turn, ran):
    """One turn: tool calls until end_turn (or MAX_STEPS). The game seconds to run next,
    and the model's note on the turn."""
    if not focus(desk):
        raise RuntimeError("could not bring the game window to the front")
    rgb = on_screen(desk.capture(full=True)).rgb
    status = (
        f"Turn {turn}. The game is paused. You play {country}. Game time run so far: "
        f"{ran:.0f} s.\n\nYour notebook:\n{agent.notebook or '(empty)'}\n\nThe screen now:"
    )
    picture = view(rgb)
    notebook = agent.notebook
    messages = [system, screenshot_message(status, picture)]
    picture.save(root / "steps" / f"{turn:04d}-00.jpg", quality=75)
    for step in range(1, MAX_STEPS + 1):
        message, usage, usd, seconds = agent.call(messages)
        calls = message.get("tool_calls") or []
        kept = {"role": "assistant", "content": message.get("content") or ""}
        if message.get("reasoning_content"):
            kept["reasoning_content"] = message["reasoning_content"]
        if calls:
            kept["tool_calls"] = calls
        messages.append(kept)
        record = {"turn": turn, "step": step, "content": message.get("content"),
                  "reasoning": message.get("reasoning_content"), "calls": calls,
                  "usage": usage, "usd": round(usd, 5), "seconds": round(seconds, 1),
                  "spent": round(agent.brain.spent, 4)}  # fmt: skip
        if not calls:
            messages.append({"role": "user", "content": "Use the tools: act, or end_turn to "
                             "let game time pass."})  # fmt: skip
            book.write(json.dumps(record) + "\n")
            continue
        results, newest, end = [], None, None
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError as error:
                args, text = None, f"refused: arguments are not JSON ({error})"
            if args is not None:
                text, shot, rgb, ending = do(desk, agent, name, args, rgb)
                if shot is not None:
                    newest = shot
                if ending is not None:
                    end = ending
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
            results.append({"tool": name, "args": args, "result": text})
        record["results"] = results
        book.write(json.dumps(record) + "\n")
        book.flush()
        if end is not None:
            consolidate(agent, messages, notebook, book, turn)
            return end
        if newest is not None:
            picture, label = newest
            picture.save(root / "steps" / f"{turn:04d}-{step:02d}.jpg", quality=75)
            messages.append(screenshot_message(label, picture))
            prune(messages)
    consolidate(agent, messages, notebook, book, turn)
    return FORCED_RUN, "(the turn ran out of actions)"


def consolidate(agent, messages, notebook, book, turn):
    """At a turn's end, if the model left its notebook as it was, ask it once to write
    down what the turn taught it: without this it explored for a whole turn, kept
    nothing, and began the next turn exploring again (2026-09-27)."""
    if agent.notebook != notebook:
        return
    messages.append({"role": "user", "content": CONSOLIDATE})
    message, usage, usd, seconds = agent.call(messages, tools=[REMEMBER])
    text = None
    for call in message.get("tool_calls") or []:
        if call["function"]["name"] == "remember":
            try:
                text = json.loads(call["function"].get("arguments") or "{}").get("text")
            except json.JSONDecodeError:
                text = None
    text = text or (message.get("content") or "").strip()
    if text:
        agent.notebook = str(text)[:NOTEBOOK]
    book.write(json.dumps({"turn": turn, "step": "notebook", "text": agent.notebook,
                           "usd": round(usd, 5), "usage": usage}) + "\n")  # fmt: skip


def do(desk, agent, name, args, rgb):
    """Carry out one tool call. Its text result, a picture to show (picture, label) or
    None, the newest full screen, and (seconds, note) if it ends the turn."""
    if name == "end_turn":
        try:
            seconds = max(1.0, min(MAX_RUN, float(args.get("seconds", MAX_RUN))))
        except (TypeError, ValueError):
            seconds = MAX_RUN
        return "the game runs", None, rgb, (seconds, str(args.get("note", ""))[:600])
    if name == "remember":
        agent.notebook = str(args.get("text", ""))[:NOTEBOOK]
        return "noted", None, rgb, None
    if name == "look":
        try:
            crop, label = close_up(rgb, args["x"], args["y"], args["w"], args["h"])
        except (KeyError, TypeError, ValueError) as error:
            return f"refused: {error}", None, rgb, None
        return label, (crop, label), rgb, None
    try:
        events = to_events(action_of(name, args))
    except (KeyError, TypeError, ValueError) as error:
        return f"refused: {error}", None, rgb, None
    try:
        act(desk, events, pause=0.06)
    except DesktopError as error:
        if "invalid_event" not in str(error):
            raise
        return "refused by the input filter", None, rgb, None
    time.sleep(SETTLE)
    after = on_screen(desk.capture(full=True)).rgb
    change = changed(rgb, after)
    said = (
        f"done; the screen changed {change:.0%}" if change >= CHANGED
        else "done; the screen did NOT change (it missed or did nothing)"
    )  # fmt: skip
    return said, (view(after), f"The screen after {name}:"), after, None

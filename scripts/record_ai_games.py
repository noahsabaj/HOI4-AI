"""Record AI-vs-AI games on an arena map, one after another, until a time budget runs out.

Each game launches HOI4 with the arena, starts as Blue, hands both countries to the AI with
the `observe` console command, sets speed 4, and records native frames while a second worker
connection moves the camera the way a player would. A game ends when the capitulation popup
("<country> equipment seized") appears, or at the cap. The popup names the country that
capitulated with its flag on the left, so the winner is the other one.

The recordings carry no actions, so they cannot teach clicks. They are for the encoder, for
predicting who wins, and for measuring how often a match ends inside the time limit.

    python scripts/record_ai_games.py artifacts/ai-games --minutes 90

It takes over this PC's screen. Anything else that takes focus stops input to the game.
"""

import argparse
import json
import random
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from hoi4_arena.desktop import Desktop, DesktopError
from hoi4_arena.recording import Recorder
from hoi4_arena.vision import ScreenRules

SCRIPTS = Path(__file__).resolve().parent
# The game runs in a 1920x1080 window (Test-ArenaLoad -Window), and these are fractions of
# it, measured on 2026-09-22.
WINDOW = "1920x1080"
# The capitulation popup at 1920x1080: the tank artwork at the left of its title bar, which
# is the same whichever side lost, and the flag of the country that capitulated.
POPUP_ART = (744, 390, 86, 65)
LOSER_FLAG = (764, 475, 24, 12)
SINGLE_PLAYER, NEW_GAME = (0.5, 290 / 1080), (0.5, 420 / 1080)
SELECT_COUNTRY, START = (1043 / 1920, 875 / 1080), (1777 / 1920, 1038 / 1080)
SPEED_UP = (1789 / 1920, 20 / 1080)


def say(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def pwsh(*args, timeout=300):
    command = ["pwsh", "-NoProfile", *args]
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def focus():
    pwsh("-Command", "(New-Object -ComObject WScript.Shell).AppActivate((Get-Process hoi4).Id)")
    time.sleep(0.5)


def act(events, pause=0.15):
    focus()
    with Desktop() as desk:
        desk.arm(setup=True)
        try:
            for event in events:
                desk.apply([event])
                time.sleep(pause)
        finally:
            desk.release()


def click(x, y):
    press = [{"kind": "button", "button": 0, "down": d} for d in (True, False)]
    act([{"kind": "move", "x": x, "y": y}, *press])


def key(vk):
    act([{"kind": "key", "vk": vk, "down": d} for d in (True, False)])


def shot():
    with Desktop() as desk:
        return desk.capture(full=True).rgb


def crop(rgb, rect):
    x, y, w, h = rect
    return rgb[y : y + h, x : x + w].astype(np.float32)


def capitulation_winner(rgb, template):
    """BLU or RED once the capitulation popup is up, otherwise None."""
    if float(np.abs(crop(rgb, POPUP_ART) - template).mean()) > 12:
        return None
    r, g, b = crop(rgb, LOSER_FLAG).mean(axis=(0, 1))
    if r > b + 60:
        return "BLU"  # Red capitulated.
    if b > r + 60:
        return "RED"
    return "unknown"


def kill_game():
    pwsh(
        "-Command",
        "Get-Process hoi4 -ErrorAction SilentlyContinue | Stop-Process -Force; "
        "while (Get-Process hoi4 -ErrorAction SilentlyContinue) { Start-Sleep 1 }",
    )
    # Let the previous launch's watcher put the player's display settings back first.
    time.sleep(5)


def start_game(mod, rules, failure_shot):
    kill_game()
    out = pwsh("-File", str(SCRIPTS / "Test-ArenaLoad.ps1"), "-Mod", str(mod), "-Window", WINDOW)
    say("launch:", out.stdout.strip().replace("\n", " | "), out.stderr.strip()[-200:])
    time.sleep(25)
    click(*SINGLE_PLAYER)
    time.sleep(8)
    click(*NEW_GAME)
    time.sleep(40)
    click(*SELECT_COUNTRY)  # Blue is preselected.
    time.sleep(40)
    click(*START)
    time.sleep(20)
    rgb = shot()
    if not rules.matches("healthy", rgb):
        Image.fromarray(rgb).resize((960, 540)).save(failure_shot)
        raise RuntimeError("game did not reach the map")
    pwsh("-File", str(SCRIPTS / "Send-HoiConsole.ps1"), "observe")
    act([{"kind": "move", "x": 0.5, "y": 0.5}] + [{"kind": "wheel", "delta": -120}] * 14, 0.1)
    # The framing the hand-centred runs used: right for 0.5 s, then left for 0.28 s.
    act([{"kind": "key", "vk": 0x27, "down": d} for d in (True, False)], pause=0.5)
    act([{"kind": "key", "vk": 0x25, "down": d} for d in (True, False)], pause=0.28)
    for _ in range(3):
        click(*SPEED_UP)  # The on-screen + button: speed 1 to 4.
    act([{"kind": "move", "x": 0.5, "y": 0.75}])
    key(0x20)  # Unpause
    time.sleep(2)
    rgb = shot()
    if not rules.matches("speed", rgb) or rules.matches("paused", rgb):
        Image.fromarray(rgb).resize((960, 540)).save(failure_shot)
        raise RuntimeError("game is not running at speed 4")


def camera(stop):
    """Look around like a player, always coming back to the start view.

    Every pan is undone by the opposite pan of the same length, and every zoom-in by a
    zoom-out at the same pointer position. A random walk drifted off the arena onto open
    sea within minutes, because pan speed changes with zoom.
    """
    rng = random.Random()
    opposite = {0x25: 0x27, 0x27: 0x25, 0x26: 0x28, 0x28: 0x26}
    with Desktop() as desk:

        def do(events, pause=0.08):
            # The worker disarms after 750 ms without input, so arm for each burst.
            desk.arm(setup=True)
            for event in events:
                desk.apply([event])
                time.sleep(pause)

        def hold(vk, seconds):
            desk.arm(setup=True)
            desk.apply([{"kind": "key", "vk": vk, "down": True}])
            try:
                time.sleep(seconds)
            finally:
                desk.apply([{"kind": "key", "vk": vk, "down": False}])

        def linger():
            # Point around while away, like a player reading the map.
            for _ in range(rng.randint(1, 3)):
                if stop.wait(rng.uniform(0.6, 1.8)):
                    return
                x, y = rng.uniform(0.1, 0.9), rng.uniform(0.15, 0.85)
                do([{"kind": "move", "x": x, "y": y}])

        try:
            while not stop.wait(rng.uniform(0.8, 2.5)):
                try:
                    roll = rng.random()
                    if roll < 0.35:
                        vk = rng.choice(list(opposite))
                        seconds = rng.uniform(0.08, 0.25)
                        hold(vk, seconds)
                        linger()
                        hold(opposite[vk], seconds)
                    elif roll < 0.65:
                        x, y = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
                        notches = rng.randint(1, 4)
                        at = {"kind": "move", "x": x, "y": y}
                        do([at] + [{"kind": "wheel", "delta": 120}] * notches)
                        linger()
                        do([at] + [{"kind": "wheel", "delta": -120}] * notches)
                    else:
                        linger()
                except DesktopError as error:
                    say("  camera:", error)
                    focus()
        finally:
            desk.release()


def play(root, template, args):
    stop = threading.Event()
    mover = threading.Thread(target=camera, args=(stop,), daemon=True)
    outcome, reason, seen = "timeout", None, 0
    with Desktop() as desk:
        first = desk.capture()
        rec = Recorder(root, first, game_speed=4, source="ai", hz=args.hz, codec=args.codec)
        start = deadline = time.monotonic()
        late = 0
        try:
            rec.append(first)
            mover.start()
            while time.monotonic() - start < args.cap_minutes * 60:
                deadline += 1 / args.hz
                time.sleep(max(0, deadline - time.monotonic()))
                frame = desk.capture()
                if not frame.meta.get("foreground"):
                    focus()
                    continue
                rec.append(frame)
                if time.monotonic() - deadline > 1:
                    late += 1
                    deadline = time.monotonic()
                winner = capitulation_winner(frame.rgb, template)
                seen = seen + 1 if winner else 0
                if seen >= 2:
                    outcome = winner
                    Image.fromarray(frame.rgb).save(Path(root) / "capitulation.png")
                    break
                if rec.manifest["frames"] % (60 * int(args.hz)) == 0:
                    say(f"  {rec.manifest['frames'] // int(args.hz) // 60} min recorded")
        except Exception as error:  # noqa: BLE001 - recorded in the manifest.
            reason = f"{type(error).__name__}: {error}"
        finally:
            stop.set()
            mover.join(timeout=10)
            rec.manifest.update(
                winner=outcome,
                seconds=round(time.monotonic() - start),
                late_ticks=late,
                arena=Path(args.mod).name,
                driver="observe + scripted camera",
            )
            rec.close(complete=reason is None, reason=reason)
    return outcome, reason, rec.manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output")
    parser.add_argument("--minutes", type=float, required=True, help="Total time budget.")
    parser.add_argument("--mod", default="artifacts/mods/small-arena-v1")
    parser.add_argument("--rules", default="artifacts/calibration-1080p/rules.json")
    parser.add_argument(
        "--popup",
        default="artifacts/screens-1080p/capitulation-popup.png",
        help="A 1920x1080 capture of the capitulation popup, to cut its artwork from.",
    )
    parser.add_argument("--hz", type=float, default=5)
    parser.add_argument("--codec", choices=["ffv1", "x264"], default="x264")
    parser.add_argument("--cap-minutes", type=float, default=32)
    args = parser.parse_args()
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    rules = ScreenRules(args.rules)
    template = crop(np.asarray(Image.open(args.popup).convert("RGB")), POPUP_ART)
    end = time.monotonic() + args.minutes * 60
    results = []
    # A game needs about 3 minutes to launch and most end within 10; do not start one
    # that cannot plausibly finish.
    while time.monotonic() + 12 * 60 < end:
        name = time.strftime("ai-%Y%m%d-%H%M%S")
        try:
            start_game(args.mod, rules, out_root / f"{name}-start-failed.png")
        except Exception as error:  # noqa: BLE001 - reported, then the next game is tried.
            say("start failed:", error)
            results.append({"game": name, "error": str(error)})
            continue
        say("recording", name)
        outcome, reason, manifest = play(out_root / name, template, args)
        say("finished", name, "winner", outcome, "after", manifest["seconds"], "s", reason or "")
        results.append(
            {
                "game": name,
                "winner": outcome,
                "seconds": manifest["seconds"],
                "frames": manifest["frames"],
                "complete": manifest["complete"],
                "reason": reason,
            }
        )
        (out_root / f"results-{time.strftime('%Y%m%d')}.json").write_text(
            json.dumps(results, indent=2)
        )
    kill_game()
    say("done", json.dumps(results))


if __name__ == "__main__":
    main()

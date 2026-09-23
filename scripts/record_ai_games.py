"""Record AI-vs-AI games on an arena map, one after another, until a time budget runs out.

Each game launches HOI4 with the arena, starts as Blue, hands both countries to the AI with
the `observe` console command, sets speed 4, and records native frames while the camera
moves the way a player's would. A game ends when the capitulation popup ("<country>
equipment seized") appears, or at the cap. The popup names the country that capitulated
with its flag on the left, so the winner is the other one.

With --peer, the second PC records its own games at the same time, driven over its worker
connection; its frames are recorded here. The second PC's worker must be running and the
arena deployed to it (Deploy-Peer.ps1 -Mod), which this script does itself.

The recordings carry no actions, so they cannot teach clicks. They are for the encoder, for
predicting who wins, and for measuring how often a match ends inside the time limit.

    python scripts/record_ai_games.py artifacts/ai-games-1080p --minutes 90
    python scripts/record_ai_games.py artifacts/ai-games-1080p --minutes 90 --peer artifacts/pairing/peer.json

It takes over the screen of each PC it uses. Anything else that takes focus stops input to
the game until the recorder brings it back.
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
from hoi4_arena.remote import RemoteDesktop
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
GRAVE, ENTER = 0xC0, 0x0D


def say(station, *parts):
    print(time.strftime("%H:%M:%S"), f"[{station}]", *parts, flush=True)


def pwsh(*args, timeout=300):
    command = ["pwsh", "-NoProfile", *args]
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


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


class Station:
    """This PC, or the second PC through its worker bridge and shared folder."""

    def __init__(self, name, peer=None):
        self.name, self.peer = name, peer
        if peer:
            host = json.loads(Path(peer).read_text())["host"]
            self.share = Path(rf"\\{host}\HOI4Worker")

    def connect(self):
        return RemoteDesktop(self.peer) if self.peer else Desktop()

    def _request(self, line, timeout=240):
        # The second PC's idle bridge notices launch.txt, runs it, and writes the outcome.
        # It only does so between connections, so none may be open while this waits.
        result = self.share / "launch-result.txt"
        result.unlink(missing_ok=True)
        (self.share / "launch.txt").write_text(line)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(2)
            try:
                text = result.read_text(errors="replace").strip()
            except OSError:
                continue
            if text.endswith("end of request"):
                return text.removesuffix("end of request").strip()
        raise RuntimeError(f"second PC did not answer '{line}' within {timeout} s")

    def quit(self):
        if self.peer:
            self._request("quit")
        else:
            # Asked to close first, as on the second PC: a game killed outright left the
            # next one there hanging at startup.
            pwsh(
                "-Command",
                "Get-Process hoi4 -ErrorAction SilentlyContinue | ForEach-Object { [void]$_.CloseMainWindow() }; "
                "$end = (Get-Date).AddSeconds(30); "
                "while ((Get-Process hoi4 -ErrorAction SilentlyContinue) -and (Get-Date) -lt $end) { Start-Sleep 1 }; "
                "Get-Process hoi4 -ErrorAction SilentlyContinue | Stop-Process -Force; "
                "while (Get-Process hoi4 -ErrorAction SilentlyContinue) { Start-Sleep 1 }",
            )  # fmt: skip
        # Let the previous launch's watcher put the player's display settings back first.
        time.sleep(5)

    def launch(self, mod):
        if self.peer:
            out = self._request(f"{Path(mod).name} {WINDOW}")
            if "Timed out waiting for the game log" in out:
                # A game that never gets far enough to log has hung at startup; a stuck
                # Discord overlay did that once. Restart Discord and try once more.
                say(self.name, "launch hung; restarting Discord and retrying")
                self._request("restart-discord")
                self.quit()
                out = self._request(f"{Path(mod).name} {WINDOW}")
            if "Arena load test PID" not in out or "Timed out" in out:
                raise RuntimeError(f"second PC did not launch: {out}")
        else:
            run = pwsh(
                "-File", str(SCRIPTS / "Test-ArenaLoad.ps1"), "-Mod", str(mod), "-Window", WINDOW
            )
            out = run.stdout.strip() + run.stderr.strip()[-200:]
        say(self.name, "launch:", out.replace("\n", " | "))


def focus(desk, tries=5):
    for _ in range(tries):
        try:
            if desk.focus():
                return True
        except DesktopError:
            pass  # Refused while the camera has input armed; it disarms within a second.
        time.sleep(1)
    return False


def act(desk, events, pause=0.15):
    focus(desk)
    desk.arm(setup=True)
    try:
        for event in events:
            desk.apply([event])
            time.sleep(pause)
    finally:
        desk.release()


def click(desk, x, y):
    press = [{"kind": "button", "button": 0, "down": d} for d in (True, False)]
    act(desk, [{"kind": "move", "x": x, "y": y}, *press])


def tap(vk):
    return [{"kind": "key", "vk": vk, "down": d} for d in (True, False)]


def console(desk, command):
    """Type a console command through the worker, which allows the console key in setup."""
    act(desk, tap(GRAVE))
    time.sleep(0.6)
    keys = [e for c in command.upper() for e in tap(0x20 if c == " " else ord(c))]
    act(desk, keys + tap(ENTER), pause=0.05)
    time.sleep(0.4)
    act(desk, tap(GRAVE))


def on_screen(frame):
    """The frame, unless the worker had to fall back from desktop duplication.

    A monitor switched off disconnects on DisplayPort: Windows shrinks the desktop to
    1024x768, the game window fits on no screen, and the fallback frames are wrong.
    """
    if frame.meta.get("backend") == "gdi_bgra":
        raise RuntimeError("the game window is on no screen; is the monitor switched off?")
    return frame


def screen(desk, tries=5):
    """A full frame. The first capture on a new connection is sometimes all black."""
    for _ in range(tries):
        rgb = on_screen(desk.capture(full=True)).rgb
        if rgb.max() > 0:
            break
        time.sleep(0.5)
    return rgb


def start_game(desk, rules, failure_shot):
    """From the main menu to an AI-vs-AI game running at speed 4."""
    click(desk, *SINGLE_PLAYER)
    time.sleep(8)
    click(desk, *NEW_GAME)
    time.sleep(40)
    click(desk, *SELECT_COUNTRY)  # Blue is preselected.
    time.sleep(40)
    click(desk, *START)
    time.sleep(20)
    rgb = screen(desk)
    # A new game starts paused. The pause mark is the same on both PCs; the alert row that
    # "healthy" reads is not, as each PC shows different alerts.
    if not rules.matches("paused", rgb):
        Image.fromarray(rgb).resize((960, 540)).save(failure_shot)
        raise RuntimeError("game did not reach the map")
    console(desk, "observe")
    act(desk, [{"kind": "move", "x": 0.5, "y": 0.5}] + [{"kind": "wheel", "delta": -120}] * 14, 0.1)
    # The framing the hand-centred runs used: right for 0.5 s, then left for 0.28 s.
    act(desk, tap(0x27), pause=0.5)
    act(desk, tap(0x25), pause=0.28)
    for _ in range(3):
        click(desk, *SPEED_UP)  # The on-screen + button: speed 1 to 4.
    act(desk, [{"kind": "move", "x": 0.5, "y": 0.75}])
    act(desk, tap(0x20))  # Unpause
    time.sleep(2)
    rgb = screen(desk)
    if not rules.matches("speed", rgb) or rules.matches("paused", rgb):
        Image.fromarray(rgb).resize((960, 540)).save(failure_shot)
        raise RuntimeError("game is not running at speed 4")


def camera(desk, stop, station):
    """Look around like a player, always coming back to the start view.

    Every pan is undone by the opposite pan of the same length, and every zoom-in by a
    zoom-out at the same pointer position. A random walk drifted off the arena onto open
    sea within minutes, because pan speed changes with zoom. It shares the recorder's
    connection: the second PC's bridge accepts only one.
    """
    rng = random.Random()
    opposite = {0x25: 0x27, 0x27: 0x25, 0x26: 0x28, 0x28: 0x26}

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
                say(station, "camera:", error)
                try:
                    desk.release()
                except DesktopError:
                    pass
                focus(desk, tries=1)
    finally:
        try:
            desk.release()
        except DesktopError:
            pass


def play(desk, root, template, args, station):
    stop = threading.Event()
    mover = threading.Thread(target=camera, args=(desk, stop, station), daemon=True)
    outcome, reason = "timeout", None
    first = desk.capture()
    rec = Recorder(root, first, game_speed=4, source="ai", hz=args.hz, codec=args.codec)
    start = deadline = time.monotonic()
    late = seen = 0
    try:
        rec.append(first)
        mover.start()
        while time.monotonic() - start < args.cap_minutes * 60:
            deadline += 1 / args.hz
            time.sleep(max(0, deadline - time.monotonic()))
            frame = on_screen(desk.capture())
            if not frame.meta.get("foreground"):
                focus(desk, tries=1)
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
                say(station, f"{rec.manifest['frames'] // int(args.hz) // 60} min recorded")
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
            station=station,
        )
        rec.close(complete=reason is None, reason=reason)
    return outcome, reason, rec.manifest


def run_station(station, out_root, rules, template, args, end):
    results = []
    # A game needs about 3 minutes to launch and most end within 10; do not start one
    # that cannot plausibly finish.
    while time.monotonic() + 12 * 60 < end:
        name = time.strftime(f"ai-{station.name}-%Y%m%d-%H%M%S")
        entry = {"game": name, "station": station.name}
        try:
            station.quit()
            station.launch(args.mod)
            time.sleep(25)
            with station.connect() as desk:
                if not focus(desk):
                    raise RuntimeError("could not bring the game window to the front")
                start_game(desk, rules, out_root / f"{name}-start-failed.png")
                say(station.name, "recording", name)
                outcome, reason, manifest = play(
                    desk, out_root / name, template, args, station.name
                )
        except Exception as error:  # noqa: BLE001 - reported, then the next game is tried.
            say(station.name, "start failed:", error)
            entry["error"] = f"{type(error).__name__}: {error}"
        else:
            say(
                station.name,
                "finished",
                name,
                "winner",
                outcome,
                "after",
                manifest["seconds"],
                "s",
                reason or "",
            )
            entry.update(
                winner=outcome,
                seconds=manifest["seconds"],
                frames=manifest["frames"],
                complete=manifest["complete"],
                reason=reason,
            )
        results.append(entry)
        path = out_root / f"results-{station.name}-{time.strftime('%Y%m%d')}.json"
        path.write_text(json.dumps(results, indent=2))
    try:
        station.quit()
    except Exception as error:  # noqa: BLE001 - the games are already saved.
        say(station.name, "quit failed:", error)
    say(station.name, "done", json.dumps(results))
    return results


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
    parser.add_argument("--peer", help="The second PC's pairing file, to record there too.")
    parser.add_argument("--peer-only", action="store_true", help="Leave this PC free.")
    args = parser.parse_args()
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    rules = ScreenRules(args.rules)
    template = crop(np.asarray(Image.open(args.popup).convert("RGB")), POPUP_ART)
    stations = [] if args.peer_only else [Station("here")]
    if args.peer:
        deploy = pwsh(
            "-File", str(SCRIPTS / "Deploy-Peer.ps1"), "-SkipBuild", "-PeerConfig", args.peer,
            "-Mod", args.mod,
        )  # fmt: skip
        if deploy.returncode:
            raise SystemExit(f"Deploy-Peer failed: {deploy.stdout}{deploy.stderr}")
        stations.append(Station("peer", args.peer))
    end = time.monotonic() + args.minutes * 60
    threads = [
        threading.Thread(target=run_station, args=(s, out_root, rules, template, args, end))
        for s in stations
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()

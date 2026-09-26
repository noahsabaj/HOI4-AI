from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path


def local_time(text):
    """A local time written YYYYMMDD-HHMMSS, as unix seconds."""
    return time.mktime(time.strptime(text, "%Y%m%d-%H%M%S"))


def add_sampling(parser):
    """How a policy playing live samples its actions (runner.resolve_temperatures)."""
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sample the policy at this temperature: 1 as trained, below 1 sharper (each "
        "slot's likeliest input and spot gain), 0 always the likeliest. The pointer follows "
        "it unless --pointer-temperature is given",
    )
    parser.add_argument(
        "--pointer-temperature",
        type=float,
        help="Where a move goes (its cell and the spot inside) at this temperature instead",
    )
    parser.add_argument(
        "--point",
        action="store_true",
        help="Place each move on its likeliest spot: --pointer-temperature 0",
    )


# Commands that never touch a model, so they run without importing torch.
NO_TORCH = {
    "record",
    "record-ai",
    "control",
    "telemetry",
    "job",
    # It only sends a session to the second PC: this PC's GPU stays with training.
    "on-peer",
    "probe-peer",
    "capture",
    "win-rate",
    "salvage",
    # Its Gaussian process imports torch itself, on the CPU; the GPU is left alone.
    "tune",
    "live",
    "live-say",
}


def build_parser():
    """Every command's arguments (on-peer reads a session's own with it too)."""
    parser = argparse.ArgumentParser(description="Screen-only HOI4 research prototype")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Progress and diagnostics go to stderr; JSON results go to stdout.",
    )
    parser.add_argument("--traceback", action="store_true", help="Re-raise instead of exiting 1")
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Allow TF32 float32 matmuls. Measured worth nothing here and not free; "
        "see models.configure_precision.",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.9,
        help="Fail with out-of-memory past this fraction of the GPU, rather than let Windows "
        "spill into system memory and run many times slower (models.limit_gpu_memory). "
        "The rest is left for the game and the desktop.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--model", default="models/levjepa-large")
    bench.add_argument("--output", default="artifacts/benchmark")
    bench.add_argument("--iterations", type=int, default=30)
    bench.add_argument("--offline", action="store_true")
    record = sub.add_parser("record")
    record.add_argument("output")
    record.add_argument("--seconds", type=float, default=1200)
    record.add_argument("--hz", type=float, default=10)
    record.add_argument(
        "--peer",
        help="Record the second PC instead (its pairing file): you play there, its worker "
        "captures the screen and your inputs, and the video is written here. Use --hz 5.",
    )
    record.add_argument("--split", choices=["train", "validation", "test"])
    record.add_argument(
        "--codec",
        choices=["ffv1", "x264", "nvenc", "nvenc-hevc", "x264-source", "ffv1-source"],
        default="ffv1",
        help="ffv1 is lossless. x264 is visually lossless (CRF 18, 4:4:4) at about a "
        "hundredth of the size. nvenc (H.264 4:4:4 at least as faithful as x264 CRF 18) and "
        "the -source codecs are encoded by the worker on the PC that captures, on its own "
        "clock; only the video crosses the network (falls back to x264 here if it cannot).",
    )
    record.add_argument(
        "--game-speed",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5],
        help="Speed the game is set to for the whole session. Written into the manifest. "
        "There is no default: a 1.6 s clip is 3.2 in-game hours at speed 2 and 16 at "
        "speed 4, and the speed bars are not read back.",
    )
    ai = sub.add_parser(
        "record-ai",
        help="Record AI-vs-AI arena games on this PC, the second PC, or both, until the "
        "time budget runs out",
    )
    ai.add_argument("output")
    ai.add_argument("--minutes", type=float, required=True, help="Total time budget.")
    ai.add_argument(
        "--mod",
        nargs="+",
        default=["artifacts/mods/arena-12x8-v2"],
        help="Arena mod folders, played in turn, each as both countries.",
    )
    ai.add_argument("--rules", default="artifacts/calibration-1080p/rules.json")
    ai.add_argument(
        "--ok-button",
        nargs="+",
        default=["artifacts/screens-1080p/ok-button.png", "artifacts/screens-1080p/event-ok.png"],
        help="1920x1080 crops of popup Ok buttons, clicked wherever they appear.",
    )
    ai.add_argument("--hz", type=float, default=5)
    ai.add_argument(
        "--codec",
        choices=["ffv1", "x264", "nvenc", "nvenc-hevc", "x264-source", "ffv1-source"],
        default="nvenc",
        help="As for record. nvenc (the default) is recorded on the worker's clock and "
        "encoded where the game runs; x264 is the old way, a request per frame, encoded here.",
    )
    ai.add_argument("--cap-minutes", type=float, default=45)
    ai.add_argument(
        "--speeds",
        type=int,
        nargs="+",
        choices=[4, 5],
        default=[4, 5],
        help="Game speeds taken in turn, each for both countries. Speed 5 finishes more "
        "games an hour; speed 4 looks like normal play.",
    )
    ai.add_argument("--peer", help="The second PC's pairing file, to record there too.")
    ai.add_argument("--peer-only", action="store_true", help="Leave this PC free.")
    ai.add_argument(
        "--player",
        choices=["observe", "scripted"],
        default="observe",
        help="observe: the game's AI plays both countries. scripted: the scripted player "
        "fights the recorder's country through the interface, with a random strategy each "
        "game, against the AI (needs an arena v3 or later for its daily state reports).",
    )
    ai.add_argument(
        "--arena-queue",
        help='A folder of arena test requests (<name>.json with {"mod": <folder>}). A scripted '
        "station plays each once, first, and answers in the sibling results/ folder; arenas "
        "that pass join the turn (accepted.json).",
    )
    ai.add_argument(
        "--queue-stations",
        nargs="+",
        choices=["here", "peer"],
        help="The stations that serve the arena queue (default: every one).",
    )
    ai.add_argument(
        "--start-save",
        dest="start_saves",
        nargs="+",
        help="ARENA:COUNTRY:SAVE, a save made paused at the start of a new game on that arena "
        "as that country: games launch straight into it, skipping the menus.",
    )
    ai.add_argument(
        "--opening",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Seconds, drawn per game, that a game runs at speed 1 before the war begins, so "
        "games that start alike do not play alike.",
    )
    ai.add_argument(
        "--main-only",
        action="store_true",
        help="Play only the given arenas; accepted test arenas do not join the turn (each "
        "arena switch is a launch of HOI4).",
    )
    ai.add_argument(
        "--eval-dir",
        help="Lend the second PC between games to live evaluations that reserve it: "
        '<dir>/queue/<name>.json ({"minutes": N}) is answered by <dir>/granted/<name>.json '
        "with HOI4 closed, and play resumes at <dir>/done/<name>.json or after N+15 minutes.",
    )
    ai.add_argument(
        "--camera-kicks",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="Every LOW to HIGH seconds, knock the camera off an edge of the map or right in "
        "on a random spot, unrecorded as input, so the recording shows it finding the front "
        "again: recovery for a learned camera to imitate.",
    )
    ai.add_argument(
        "--tune",
        help="A tuning study's SQLite file (tuning.py): the scripted player's exploring games "
        "play the best plan with the settings its Gaussian process asks for, and report "
        "their scores to it.",
    )
    ai.add_argument(
        "--tune-skip",
        nargs="+",
        help="Arenas (folder names) whose exploring games are not tuned, such as one every "
        "plan loses on.",
    )
    tower = sub.add_parser(
        "cache-tower",
        help="Run a checkpoint's frozen vision tower once over every frame of the "
        "recordings and keep what it read, for train-bc --tower-cache.",
    )
    tower.add_argument("data")
    tower.add_argument("checkpoint")
    tower.add_argument("output")
    tower.add_argument("--model", dest="model_path")
    tower.add_argument("--sources", nargs="+")
    tower.add_argument(
        "--spill",
        help="A second folder, on another drive, for the recordings the output's drive "
        "cannot hold while keeping --keep-free. Without it a build that does not fit is "
        "refused before it starts.",
    )
    tower.add_argument(
        "--keep-free", type=float, default=30.0, help="GB to leave free on each drive"
    )
    tower.add_argument(
        "--bf16", dest="int8", action="store_false",
        help="Keep the grid in bfloat16, twice the space of the default int8 (a scale per "
        "frame and channel)",
    )  # fmt: skip
    tower.add_argument("--int8", action="store_true", default=True, help=argparse.SUPPRESS)
    tower.add_argument(
        "--dry-run", action="store_true",
        help="Work out the build's size and where it would go, and write nothing",
    )  # fmt: skip
    drill = sub.add_parser(
        "practice",
        help="A trained policy practises the setup on the second PC: short episodes from the "
        "start save, each scored on the army, its general, its front and the game running, "
        "with the scripted player taking late steps over (practice.py).",
    )
    drill.add_argument("checkpoint")
    drill.add_argument("output", help="A folder for the episodes and practice-peer.json")
    drill.add_argument("--peer", required=True, help="The second PC's pairing file")
    drill.add_argument("--episodes", type=int, default=20)
    drill.add_argument("--minutes", type=float, required=True, help="Time budget for all")
    drill.add_argument("--seconds", type=float, default=90.0, help="Each episode's length")
    drill.add_argument("--countries", nargs="+", choices=["BLU", "RED"], default=["BLU", "RED"])
    drill.add_argument(
        "--no-coach", dest="coach", action="store_false",
        help="Only watch and score; never take a step over",
    )  # fmt: skip
    drill.add_argument("--reservation", help="Reserve the second PC first, as play-policy")
    drill.add_argument("--held-previous", action="store_true", help="As play-policy's")
    add_sampling(drill)
    drill.add_argument("--rules", default="artifacts/calibration-1080p/rules.json")
    drill.add_argument("--model", dest="model_path")
    drill.add_argument("--seed", type=int)
    drill = sub.add_parser(
        "drills",
        help="The scripted player's setup alone, over and over, on the second PC: recorded "
        "demonstrations of the part of a game the learned player gets wrong, many from a "
        "scrambled start (practice.drills).",
    )
    drill.add_argument("output", help="A folder for the drills and drills-peer.json")
    drill.add_argument("--peer", required=True, help="The second PC's pairing file")
    drill.add_argument("--episodes", type=int, default=40)
    drill.add_argument("--minutes", type=float, required=True, help="Time budget for all")
    drill.add_argument("--arenas", nargs="+", default=["arena-12x8-v4"])
    drill.add_argument("--countries", nargs="+", choices=["BLU", "RED"], default=["BLU", "RED"])
    drill.add_argument(
        "--scrambled", type=float, default=0.7, help="The share of drills from a scramble"
    )
    drill.add_argument("--block", type=int, default=4, help="Drills in a row on an arena")
    drill.add_argument("--after", type=float, default=8.0, help="Seconds run after the setup")
    drill.add_argument("--reservation", help="Reserve the second PC first, as play-policy")
    drill.add_argument("--rules", default="artifacts/calibration-1080p/rules.json")
    drill.add_argument("--seed", type=int)
    live = sub.add_parser(
        "play-policy",
        help="A trained policy plays arena games against the game's AI on the second PC, "
        "from pixels, recorded; prints its record with 95%% intervals.",
    )
    live.add_argument("checkpoint")
    live.add_argument("output", help="A folder for the recorded games and results-peer.json")
    live.add_argument("--peer", required=True, help="The second PC's pairing file")
    live.add_argument("--games", type=int, default=2)
    live.add_argument("--minutes", type=float, required=True, help="Time budget for all games")
    live.add_argument("--mod", default="arena-12x8-v4", help="The arena, as deployed there")
    live.add_argument("--rules", default="artifacts/calibration-1080p/rules.json")
    live.add_argument(
        "--reservation",
        help="Reserve the second PC from the scripted player's agent under this name first "
        "(artifacts/eval/queue), and hand it back after (artifacts/eval/done).",
    )
    live.add_argument("--countries", nargs="+", choices=["BLU", "RED"], default=["BLU", "RED"])
    live.add_argument("--cap-minutes", type=float, default=15.0)
    live.add_argument(
        "--setup-seconds",
        type=float,
        default=90.0,
        help="Start the paused game after this long if the policy has not clicked + by then",
    )
    live.add_argument(
        "--memory-window",
        type=int,
        help="Run the memory afresh over the last N decisions at each one, as in training",
    )
    add_sampling(live)
    live.add_argument(
        "--held-previous",
        action="store_true",
        help="Show the policy what it still holds even if its training did not "
        "(a checkpoint trained with --held-previous always does)",
    )
    live.add_argument(
        "--start-save",
        nargs="+",
        metavar="COUNTRY:SAVE",
        help="Launch games as COUNTRY straight into SAVE, a save made paused at the start of a "
        "new game (as record-ai --start-save), skipping the menus",
    )
    live.add_argument("--model", dest="model_path")
    live.add_argument("--seed", type=int)
    heat = sub.add_parser(
        "heatmap",
        help="Draw where a policy wants to point, as a heat map over a recording's frames, "
        "with the demonstrated move marked",
    )
    heat.add_argument("checkpoint")
    heat.add_argument("recording")
    heat.add_argument("output", help="A folder for the images and heatmaps.json")
    heat.add_argument("--decisions", type=int, nargs="+", help="Decision indices to draw")
    heat.add_argument("--count", type=int, default=24, help="Else this many, spread over moves")
    heat.add_argument(
        "--targets",
        choices=["moves", "clicks"],
        default="moves",
        help="Spread over every move, or only the aimed ones: moves onto something pressed",
    )
    heat.add_argument("--model")
    aim = sub.add_parser(
        "setup-pointing",
        help="How near a policy points to the setup's clicks (the unassigned-divisions alert, "
        "the create-army +, the commander portrait, the first front line) in recordings, its "
        "memory carried from each one's start: miss in pixels and the chance of a hit",
    )
    aim.add_argument("checkpoint")
    aim.add_argument("data", help="A recording, or a folder of them (with splits.json)")
    aim.add_argument("--split", default="validation", help="Which of a folder's recordings")
    aim.add_argument("--radius", type=float, default=30.0, help="A hit's distance in pixels")
    aim.add_argument("--output", help="Also write the report to this JSON file")
    aim.add_argument("--model")
    rate = sub.add_parser(
        "win-rate",
        help="The scripted player's record against the game's AI, from record-ai results",
    )
    rate.add_argument("results", nargs="+", help="results-*.json files written by record-ai")
    tune = sub.add_parser(
        "tune",
        help="The study behind record-ai --tune: ask it for a plan's settings, seed it with "
        "the best plan's earlier games, or show it",
    )
    tune.add_argument("study", help="The study's SQLite file")
    tune.add_argument("action", choices=["ask", "seed", "show"])
    tune.add_argument("roots", nargs="*", help="seed: run folders whose games to add")
    tune.add_argument("--skip", nargs="+", default=[], help="seed: arenas to leave out")
    live = sub.add_parser(
        "live",
        help="Watch the games from a phone: each PC's game live, what is going on, a feed "
        "like a stream's chat, replays and the record, served on 127.0.0.1 (publish it with "
        "tailscale serve), until stopped",
    )
    live.add_argument(
        "--runs",
        nargs="+",
        default=["artifacts/*", "artifacts/learned/*"],
        help="Globs of run folders whose games to follow (the newest being recorded): "
        "record-ai's runs and play-policy's live tests by default.",
    )
    live.add_argument("--out", help="Where the stream is written (default: temp/hoi4-live).")
    live.add_argument("--port", type=int, default=8765)
    live.add_argument(
        "--peer",
        help="The second PC's pairing file: show its own view of the game window (smooth, "
        "menus included) instead of following the recording at 5 frames a second.",
    )
    live.add_argument("--hz", type=int, default=60, help="The view's frames a second.")
    live.add_argument(
        "--label",
        nargs="+",
        default=[],
        metavar="STATION=NAME",
        help='What the page calls a PC, e.g. peer="Lent PC" (default: Second PC, This PC)',
    )
    live.add_argument("--feed", default="artifacts/live", help="Where the chat and flags are kept")
    say = sub.add_parser("live-say", help="Post a message to the live view's chat, as Claude")
    say.add_argument("text")
    say.add_argument("--who", default="Claude")
    say.add_argument("--feed", default="artifacts/live")
    check = sub.add_parser(
        "check-session",
        help="Check that a recording can train, and count its decisions. Training reads "
        "recordings directly; nothing is prepared.",
    )
    check.add_argument("source")
    check.add_argument("--sources", nargs="+", default=["human", "ai"])
    check.add_argument("--lead-in", type=int, help="Decision intervals before the first one")
    check.add_argument(
        "--drop-keys",
        nargs="+",
        default=[],
        type=lambda text: int(text, 0),
        help="Key codes left out of the labels (0x20: space)",
    )
    check.add_argument(
        "--drop-parking", action="store_true", help="Leave out the pointer's parking moves"
    )
    check.add_argument("--look-before-click", action="store_true")
    rescue = sub.add_parser(
        "salvage",
        help="Finish recordings whose recorder was killed before it closed them, so training "
        "takes them: rows past the video's end go, and the manifest says complete and "
        "salvaged. What the recorder left is kept beside them; --undo puts it back. Skips "
        "any recording that may still be recording.",
    )
    rescue.add_argument("paths", nargs="+", help="Recordings, or folders to search for them")
    rescue.add_argument("--undo", action="store_true", help="Put salvaged recordings back")
    rescue.add_argument("--dry-run", action="store_true", help="Say what would be done")
    train = sub.add_parser("train-bc")
    train.add_argument("data", help="A folder of recordings, each with its own manifest.")
    train.add_argument(
        "--sources",
        nargs="+",
        choices=["human", "ai", "scripted", "policy", "idm"],
        default=["human"],
        help="Whose inputs are demonstrations: the player's, the AI games' scripted "
        "camera and popup clicks, the scripted player's, a learned policy's own games "
        "(play-policy), and inputs the inverse dynamics model labelled.",
    )
    train.add_argument("output")
    train.add_argument("--model", default="models/qwen3-vit-88m")
    train.add_argument("--variant", choices=["large", "tiny", "screen"], default="screen")
    train.add_argument("--student")
    train.add_argument("--auxiliary", choices=["none", "dense", "sparse"], default="none")
    train.add_argument(
        "--sparsity-shift",
        type=float,
        default=0.0,
        help="LpWM's target mean for --auxiliary sparse: below 0, sparser codes (it swept "
        "0, -1, -2).",
    )
    train.add_argument(
        "--temporal-jaccard",
        type=float,
        default=0.0,
        help="Weight of LpWM's temporal Jaccard loss on the sparse codes, so their support "
        "changes with the game rather than the camera (LpWM used 0.005 to 0.1).",
    )
    train.add_argument(
        "--projections", type=int, default=256, help="RDMReg's random projections (LpWM: 1024+)."
    )
    train.add_argument("--objective", choices=["bc", "xm"], default="bc")
    train.add_argument(
        "--xm-candidates",
        type=int,
        default=5,
        help="Latents explored per decision for --objective xm (the paper sweeps 1, 2, 3, "
        "5, 8, 12).",
    )
    train.add_argument(
        "--xm-form",
        choices=["hard", "smooth"],
        default="hard",
        help="hard trains the best candidate only; smooth, -log of the candidates' mean "
        "likelihood, trains them all.",
    )
    train.add_argument(
        "--xm-latents",
        type=int,
        default=0,
        help="Learn this many latents to explore instead of drawing Gaussian noise "
        "(the paper's discrete XM); 0 keeps the noise.",
    )
    train.add_argument(
        "--idm-min-logp",
        type=float,
        help="Drop inverse-dynamics labels whose log-likelihood (summed over the slots, 0 "
        "is certain) is below this; a window holding one is not trained on.",
    )
    train.add_argument(
        "--idm-weight",
        type=float,
        default=1.0,
        help="Loss weight of an inverse-dynamics label against a recorded one, in (0, 1]: "
        "inferred labels are noisier, and hurt precise control most (D2E, 2510.05684).",
    )
    train.add_argument(
        "--advantage",
        action="store_true",
        help="Count each decision of a weighed player's recording by its advantage "
        "(hoi4-arena advantage): advantage-weighted imitation, offline RL",
    )
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument(
        "--pointer-sigma",
        type=float,
        default=0.0,
        help="Score each demonstrated pointer position against a Gaussian blob this wide, in "
        "lattice units (1 is about 1.9 px across and 1.1 px down at 1920x1080), instead of "
        "its one exact point: a click anywhere on a button is right. 0 keeps exact targets",
    )
    train.add_argument(
        "--look-before-click",
        action="store_true",
        help="The policy may press a button only where its pointer already was when the "
        "decision began, so it has seen what it clicks; decisions that press after a move "
        "are left out of training",
    )
    train.add_argument(
        "--save-every",
        type=float,
        default=600.0,
        help="Seconds of training between saves of the run in progress (progress.pt), so an "
        "interruption costs minutes; 0 saves after each epoch only",
    )
    train.add_argument(
        "--resume",
        action="store_true",
        help="Continue the run saved in the output folder, same settings, from where it stopped",
    )
    train.add_argument("--sequence", type=int, default=8)
    train.add_argument("--burn-in", type=int, default=2)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--no-checkpoint",
        dest="recompute",
        action="store_false",
        help="Keep perception's trainable activations instead of recomputing them: faster, but a "
        "batch of 2 no longer fits 8 GB.",
    )
    train.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Background processes that decode the video and cut training windows while "
        "the GPU trains; 0 does it on the training thread.",
    )
    train.add_argument(
        "--chunk",
        type=int,
        default=4,
        help="Frames the vision tower reads at once. Larger is faster until the backward "
        "pass's recomputation no longer fits the card.",
    )
    train.add_argument(
        "--lead-in",
        type=int,
        help="Decision intervals of video before a recording's first decision (default: a "
        "clip and one more, 9). 0 suits an encoder that reads no clip: the scripted player "
        "forms its army in the first 2.5 s.",
    )
    train.add_argument(
        "--drop-keys",
        type=lambda v: int(v, 0),
        nargs="+",
        default=[],
        help="Key codes left out of the labels, such as 0x20: space unpauses the game, which "
        "the harness does when a learned policy plays.",
    )
    train.add_argument(
        "--loser-weight",
        type=float,
        default=1.0,
        help="Loss weight of every decision of a game the recording's player did not win.",
    )
    train.add_argument(
        "--state-weight",
        type=float,
        default=0.0,
        help="Weight of the privileged-state loss: the memory predicts the arena's true state "
        "(from its log) at each decision. Training only; 0 is off.",
    )
    train.add_argument(
        "--order-weight",
        type=float,
        default=0.0,
        help="Weight of the next-order loss: the memory predicts the scripted player's next "
        "order and the time until it, from the manifest's orders. Training only; 0 is off.",
    )
    train.add_argument(
        "--train-last",
        type=int,
        help="How many of the vision tower's last blocks train (default 2); 0 freezes it, "
        "which makes a step about a third cheaper.",
    )
    train.add_argument(
        "--tower-cache",
        help="Read the frozen tower's output from this cache (cache-tower) instead of "
        "running it; needs --train-last 0.",
    )
    train.add_argument(
        "--carry",
        action="store_true",
        help="Carry the memory through each game, window after window in order "
        "(truncated backpropagation through time), instead of from empty per window.",
    )
    train.add_argument(
        "--reinit",
        nargs="+",
        default=[],
        help="Policy layers to start afresh after --init, such as fusion memory",
    )
    train.add_argument(
        "--press-weight",
        type=float,
        default=1.0,
        help="Loss weight of decisions that press a key or button, or move onto what is "
        "pressed next (dataset.acting), against 1 for waiting and the camera",
    )
    train.add_argument(
        "--drop-parking",
        action="store_true",
        help="Leave out the moves that only park the pointer so the scripted player can read "
        "the screen (dataset.parking_moves)",
    )
    train.add_argument(
        "--gpu-views",
        action="store_true",
        help="Loader workers only decode: each frame's quadrants are cut on the GPU "
        "(dataset.quadrant_views), the same pixels, without the CPU's ~20 ms a frame",
    )
    train.add_argument(
        "--balance",
        action="store_true",
        help="Weigh every (arena, side) group of recordings the same in the loss "
        "(dataset.balance_weights)",
    )
    train.add_argument(
        "--held-previous",
        action="store_true",
        help="Show the policy, in its previous action, every key and button it still holds "
        "(actions.with_held), so it knows to let them go",
    )
    train.add_argument(
        "--setup-weight",
        type=float,
        default=1.0,
        help="Loss weight of the setup's decisions, before the scripted player's run order "
        "(dataset.setup_end), on top of --press-weight",
    )
    train.add_argument(
        "--camera-since",
        type=local_time,
        help="Leave the arrow-key pans out of scripted and AI recordings made before this "
        "local time, YYYYMMDD-HHMMSS (until #90, 20260924-131000, the recorder's camera "
        "panned at random: dataset.camera_keys_dropped)",
    )
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--init", help="Start from this checkpoint's policy weights.")
    weigh = sub.add_parser(
        "advantage",
        help="Offline RL: value every decision of a player's recording with a trained "
        "critic and write how much each input improved the position, for train-bc "
        "--advantage (offline.py)",
    )
    weigh.add_argument(
        "checkpoint",
        nargs="?",
        help="A policy whose critic was trained (train-critic); not needed with --state-value",
    )
    weigh.add_argument("recording")
    weigh.add_argument("--model")
    weigh.add_argument("--n-step", type=int, default=25, help="Decisions looked ahead (5 s)")
    weigh.add_argument("--beta", type=float, default=0.05, help="Weight temperature")
    weigh.add_argument("--max-weight", type=float, default=20.0)
    weigh.add_argument(
        "--state-value",
        help="A win predictor over the arena's logged state (train-state-value), used "
        "instead of the checkpoint's screen critic",
    )
    fit = sub.add_parser(
        "train-state-value",
        help="Fit a win predictor on the arena's daily state reports (v3 arenas), for "
        "advantage --state-value; CPU, seconds",
    )
    fit.add_argument("output", help="Where to write the model, e.g. artifacts/state-value.pt")
    fit.add_argument("recordings", nargs="+", help="Recorded games; others are skipped")
    fit.add_argument("--epochs", type=int, default=300)
    idm = sub.add_parser(
        "train-idm",
        help="Train the inverse dynamics model: inputs inferred from video, trained on "
        "recordings whose inputs are known",
    )
    idm.add_argument("data")
    idm.add_argument("output")
    idm.add_argument("--model", default="models/levjepa-large")
    idm.add_argument("--variant", choices=["large", "tiny", "screen"], default="large")
    idm.add_argument("--epochs", type=int, default=1)
    idm.add_argument(
        "--save-every",
        type=float,
        default=600.0,
        help="Seconds of training between saves of the run in progress (progress.pt), so an "
        "interruption costs minutes; 0 saves after each epoch only",
    )
    idm.add_argument(
        "--resume",
        action="store_true",
        help="Continue the run saved in the output folder, same settings, from where it stopped",
    )
    idm.add_argument("--batch-size", type=int, default=2)
    idm.add_argument(
        "--sequence",
        type=int,
        default=16,
        help="Decisions per window: 16, 32 or 64. At speed 5 an input's effect can show "
        "late, so a longer window lets a label read it from its neighbours' frames.",
    )
    idm.add_argument(
        "--context",
        choices=["gru", "transformer"],
        default="gru",
        help="What runs over the window: a two-way GRU, or full two-way attention (as "
        "VPT's IDM, 2206.11795).",
    )
    idm.add_argument(
        "--context-layers", type=int, default=2, help="Layers of the transformer context."
    )
    idm.add_argument("--seed", type=int, default=42)
    idm.add_argument(
        "--no-checkpoint",
        dest="recompute",
        action="store_false",
        help="Keep perception's trainable activations instead of recomputing them: faster, but a "
        "batch of 2 no longer fits 8 GB.",
    )
    idm.add_argument(
        "--sources", nargs="+", choices=["human", "ai", "scripted"], default=["human", "ai"]
    )
    idm.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Background processes that decode the video and cut training windows while "
        "the GPU trains; 0 does it on the training thread.",
    )
    idm.add_argument(
        "--chunk",
        type=int,
        default=4,
        help="Frames the vision tower reads at once. Larger is faster until the backward "
        "pass's recomputation no longer fits the card.",
    )
    cache = sub.add_parser(
        "cache-features",
        help="Cache what a behaviour-cloned policy's frozen perception reads at every "
        "decision, so its memory can train on long windows",
    )
    cache.add_argument("data")
    cache.add_argument("checkpoint")
    cache.add_argument("output")
    cache.add_argument("--model")
    cache.add_argument("--sources", nargs="+", choices=["human", "ai", "scripted"], default=["ai"])
    memory = sub.add_parser(
        "train-memory",
        help="Train the memory and action head on cached features, and score it on "
        "held-out games (imitation loss, probes of what the memory holds)",
    )
    memory.add_argument("cache")
    memory.add_argument("output")
    memory.add_argument("--memory", choices=["gru", "gdn2", "mamba3", "none"], default="gru")
    memory.add_argument("--window", type=int, default=256)
    memory.add_argument(
        "--no-carry",
        dest="carry",
        action="store_false",
        help="Start each window from an empty memory after --burn-in steps, as train-bc "
        "does, instead of running through whole games",
    )
    memory.add_argument("--burn-in", type=int, default=0)
    memory.add_argument("--decisions", type=int, default=1024, help="Decisions per update")
    memory.add_argument("--epochs", type=int, default=4)
    memory.add_argument("--lr", type=float, default=1e-4)
    memory.add_argument("--seed", type=int, default=0)
    memory.add_argument("--probe-games", type=int)
    pointer = sub.add_parser(
        "pointer",
        help="Save the pointer image Windows is showing in the game, to find it in video",
    )
    pointer.add_argument("output", help="A .png; its hotspot is written beside it as .json.")
    pointer.add_argument("--peer")
    imported = sub.add_parser(
        "import-video",
        help="Make a recording, with no inputs, from plain video of the game, for `label`",
    )
    imported.add_argument("video")
    imported.add_argument("output")
    imported.add_argument("--pointers", nargs="+", required=True, help="Saved pointer images.")
    imported.add_argument("--game-speed", type=int, required=True, choices=[1, 2, 3, 4, 5])
    imported.add_argument("--split", choices=["train", "validation", "test"])
    label = sub.add_parser(
        "label", help="Label a recording's inputs with a trained inverse dynamics model"
    )
    label.add_argument("checkpoint")
    label.add_argument("recordings", nargs="+")
    label.add_argument("--model-path")
    distill = sub.add_parser("distill")
    distill.add_argument("data")
    distill.add_argument("output")
    distill.add_argument("--model", default="models/levjepa-large")
    distill.add_argument("--epochs", type=int, default=1)
    pairing = sub.add_parser("bundle-peer")
    pairing.add_argument("output")
    pairing.add_argument("--host", required=True)
    pairing.add_argument("--coordinator", required=True)
    pairing.add_argument(
        "--port",
        type=int,
        required=True,
        help="Worker port. Required rather than defaulted, because a published "
        "default is a detail of somebody's actual network.",
    )
    probe = sub.add_parser("probe-peer")
    probe.add_argument("config")
    capture = sub.add_parser("capture")
    capture.add_argument("output")
    capture.add_argument("--peer")
    control = sub.add_parser(
        "control",
        help="Launch, close or inspect HOI4 through the worker, here or on the second PC",
    )
    control.add_argument("action", choices=["launch", "quit", "report", "saves", "restart-discord"])
    control.add_argument("--peer", help="The second PC's peer.json; this PC if omitted")
    control.add_argument("--mod", help="Arena mod folder name to launch, as deployed")
    control.add_argument(
        "--window", default="1920x1080", help="Client size of the windowed game to launch"
    )
    control.add_argument(
        "--save",
        help="Load this save game at launch, skipping the main menu: its name in the save "
        "games folder, without .hoi4 (letters, digits and _). `control saves` lists them.",
    )
    tele = sub.add_parser(
        "telemetry",
        help="What a PC is doing: CPU, RAM, GPU (busy, video encoder, VRAM), disks, network, "
        "per process (the game, the worker, the encoder), the game's window and capture "
        "timing. With --peer it uses a read-only connection, so it works during a recording.",
    )
    tele.add_argument("--peer", help="The second PC's peer.json; this PC if omitted")
    tele.add_argument("--watch", type=float, help="Repeat every this many seconds")
    tele.add_argument("--count", type=int, help="With --watch, stop after this many")
    tele.add_argument("--json", dest="as_json", action="store_true", help="Print the raw reply")
    job = sub.add_parser(
        "job",
        help="Run compute on the second PC's GPU: set up its Python environment, run a "
        "training command there, stop one, or list them (scripts/Run-Job.ps1)",
    )
    job.add_argument("action", choices=["start", "stop", "status"])
    job.add_argument("--peer", required=True, help="The second PC's peer.json")
    job.add_argument("--id", dest="job_id", help="A name for the job: letters, digits, _ and -")
    job.add_argument("--kind", choices=["setup", "run", "script"], default="run")
    job.add_argument(
        "args",
        nargs="*",
        help="For run, the hoi4-arena command and its arguments; for script, the script "
        "and its arguments. Paths are inside the second PC's compute folder. Put -- first.",
    )
    away = sub.add_parser(
        "on-peer",
        help="Run a practice, drills or play-policy session on the second PC itself, as a "
        "job: its own Python plays its own game with the policy on its own GPU, and the "
        "session's folder comes back here when it ends (on_peer.py)",
    )
    away.add_argument("--peer", required=True, help="The second PC's pairing file")
    away.add_argument(
        "--reservation",
        help="Reserve the second PC under this name here first (artifacts/eval, as "
        "play-policy's), and hand it back when the session ends",
    )
    away.add_argument("--id", dest="job_id", help="The job's name (default: command and time)")
    away.add_argument(
        "--no-deploy", dest="deploy", action="store_false",
        help="Send no code or data first: what is there already",
    )  # fmt: skip
    away.add_argument(
        "--keep-there", action="store_true",
        help="Leave the session's folder on the second PC too (by default it is moved here)",
    )  # fmt: skip
    away.add_argument(
        "session",
        nargs=argparse.REMAINDER,
        help="The session's command as it would run here, with its --peer (the same "
        "pairing file, whose copy there reaches its own bridge). Put -- first.",
    )
    template = sub.add_parser("template")
    template.add_argument("screenshot")
    template.add_argument("rules")
    template.add_argument("name")
    template.add_argument("--rect", nargs=4, type=int, required=True, metavar=("X", "Y", "W", "H"))
    template.add_argument(
        "--max-mae",
        type=int,
        default=5,
        help="How far this crop may drift and still match. The default of 5 is tighter "
        "than a live HUD holds still: measure the drift before trusting it.",
    )
    clock = sub.add_parser("clock", help="Calibrate the changing-clock ROI collection requires")
    clock.add_argument("screenshot")
    clock.add_argument("rules")
    clock.add_argument("--rect", nargs=4, type=int, required=True, metavar=("X", "Y", "W", "H"))
    minimap = sub.add_parser(
        "minimap", help="Calibrate the political-minimap crop the territory reward reads"
    )
    minimap.add_argument("screenshot")
    minimap.add_argument("rules")
    minimap.add_argument("--rect", nargs=4, type=int, required=True, metavar=("X", "Y", "W", "H"))
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("results")
    league_add = sub.add_parser("league-add", help="Register an immutable checkpoint in a league")
    league_add.add_argument("league")
    league_add.add_argument("checkpoint")
    league_sample = sub.add_parser(
        "league-sample", help="Sample one registered checkpoint. Does not start a match."
    )
    league_sample.add_argument("league")
    league_sample.add_argument("--seed", type=int, default=42)
    generation = sub.add_parser("generate-map")
    generation.add_argument("output")
    generation.add_argument("--game", required=True)
    generation.add_argument(
        "--preset",
        help="A named arena design (arenas.PRESETS: plains, river, passes, marsh, bay): "
        "terrain, rivers, lakes and cities on the 12x8 grid. Without one, the plain arena.",
    )
    generation.add_argument(
        "--seed",
        type=int,
        help="Redraw a preset's noise, province shapes and river courses: the same design, "
        "another map. Recorded in generation.json.",
    )
    generation.add_argument(
        "--undefended",
        choices=["BLU", "RED"],
        help="Field no divisions for this country. A diagnostic, not a playable arena: "
        "an empty front an AI never enters says it is not attacking at all.",
    )
    generation.add_argument("--columns-per-half", type=int)
    generation.add_argument("--rows", type=int)
    generation.add_argument("--state-columns", type=int)
    generation.add_argument("--state-rows", type=int)
    generation.add_argument(
        "--land-columns",
        type=int,
        help="Shrink each country to this many columns of the full grid, touching the "
        "seam. The rest is sea at the same province size.",
    )
    generation.add_argument(
        "--land-rows", type=int, help="Rows of land per country, centered vertically."
    )
    generation.add_argument(
        "--victory-points-on-border",
        action="store_true",
        help="Put every victory point on the border column. This does not capitulate a "
        "country: surrender is territorial. Kept as the measurement that showed that.",
    )
    inspection = sub.add_parser(
        "audit-map", help="Check a generated arena for references the engine cannot resolve"
    )
    inspection.add_argument("mod")
    picture = sub.add_parser(
        "preview-map", help="Draw a generated arena: terrain, relief, rivers, states, cities"
    )
    picture.add_argument("mod")
    picture.add_argument("output")
    collect = sub.add_parser("collect-pair")
    collect.add_argument("config")
    collect.add_argument("output")
    collect.add_argument("left_checkpoint")
    collect.add_argument("right_checkpoint")
    ppo = sub.add_parser("train-ppo")
    ppo.add_argument("rollouts")
    ppo.add_argument("checkpoint")
    ppo.add_argument("output")
    ppo.add_argument("--epochs", type=int, default=3)
    ppo.add_argument("--model-path")
    ppo.add_argument("--seed", type=int, default=42)
    ppo.add_argument("--burn-in", type=int, default=4)
    ppo.add_argument("--kl-limit", type=float, default=0.02)
    ppo.add_argument(
        "--gae-lambda",
        type=float,
        help="GAE's lambda. Default 1.0: with a reward that comes mostly at the end, "
        "anything lower carries every intermediate value error into the advantage.",
    )
    ppo.add_argument(
        "--critic",
        choices=["pact", "joint"],
        default="pact",
        help="pact: update the actor, then the value head alone on importance-corrected "
        "returns under the updated policy. joint: the usual single PPO loss.",
    )
    ppo.add_argument("--critic-epochs", type=int, default=1)
    ppo.add_argument(
        "--clock",
        choices=["ticks", "information"],
        default="ticks",
        help="information: discount over the collecting policy's entropy per step "
        "(InfoPPO), so confident waiting costs no horizon. ticks: over wall time.",
    )
    ppo.add_argument(
        "--clip",
        choices=["fixed", "adaptive"],
        default="fixed",
        help="adaptive: InfoPPO's per-step ratio bounds, wider where the policy was unsure.",
    )
    ppo.add_argument(
        "--info-clip",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="The adaptive clip's epsilons. Default 10 20, the paper's.",
    )
    critic = sub.add_parser(
        "train-critic",
        help="Pre-train a policy checkpoint's critic on recorded AI games, whose winners are known",
    )
    critic.add_argument("data")
    critic.add_argument("checkpoint")
    critic.add_argument("output")
    critic.add_argument("--model-path")
    critic.add_argument("--epochs", type=int, default=1)
    critic.add_argument("--batch-size", type=int, default=2)
    critic.add_argument("--seed", type=int, default=42)
    critic.add_argument(
        "--no-checkpoint",
        dest="recompute",
        action="store_false",
        help="Keep perception's trainable activations instead of recomputing them: faster, but a "
        "batch of 2 no longer fits 8 GB.",
    )
    critic.add_argument(
        "--trunk", action="store_true", help="Also train the shared trunk, not only the head."
    )
    return parser


def main():
    args = vars(build_parser().parse_args())
    command = args.pop("command")
    logging.basicConfig(
        level=getattr(logging, args.pop("log_level").upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    traceback = args.pop("traceback")
    tf32 = args.pop("tf32")
    gpu_memory = args.pop("gpu_memory")
    try:
        # Recording, control and telemetry need no torch; importing it costs them 4 s and
        # about 1.5 GB for the whole of a recording.
        if command not in NO_TORCH:
            from .models import configure_precision, limit_gpu_memory

            configure_precision(tf32)
            limit_gpu_memory(gpu_memory)
        result = _dispatch(command, args)
    except KeyboardInterrupt:
        logging.getLogger(__name__).error("interrupted")
        raise SystemExit(130) from None
    except Exception as error:
        if traceback:
            raise
        logging.getLogger(__name__).error("%s: %s", type(error).__name__, error)
        raise SystemExit(1) from None
    if result is not None:
        print(json.dumps(result, indent=2))


def _report(problems):
    """One line per unresolved reference, then a short failure the exit code follows."""
    for problem in problems:
        logging.getLogger(__name__).error("arena: %s", problem)
    if problems:
        raise ValueError(f"{len(problems)} references the game cannot resolve")


def control(action, peer=None, mod=None, window="1920x1080", save=None):
    """One control operation through a worker that is not attached to any game.

    On the second PC the worker finds its script and mods beside itself; here they are
    this repo's scripts/ and artifacts/mods/.
    """
    from .desktop import Desktop, local_control_args
    from .remote import RemoteDesktop

    if action == "launch" and not mod:
        raise ValueError("launch needs --mod, the arena's folder name")
    if peer and action in ("report", "saves"):
        # Read-only: through an observer connection, so it works while a recording holds
        # the game there.
        from .telemetry import open_observer

        with open_observer(peer) as desktop:
            return getattr(desktop, action)()
    with (
        RemoteDesktop(peer, attach=False)
        if peer
        else Desktop(worker_args=local_control_args(), attach=False)
    ) as desktop:
        if action == "launch":
            return desktop.launch(mod, window=window, save=save)
        return getattr(desktop, action.replace("-", "_"))()


def _dispatch(command, args):
    result = None
    if command == "benchmark":
        from .benchmark import benchmark

        benchmark(args.pop("model"), **args)
    elif command == "record":
        from .recording import record

        record(args.pop("output"), **args)
    elif command == "record-ai":
        from .ai_games import record_ai_games

        result = record_ai_games(args.pop("output"), **args)
    elif command == "cache-tower":
        from .tower_cache import cache_tower

        result = cache_tower(
            args["data"],
            args["checkpoint"],
            args["output"],
            model_path=args["model_path"],
            sources=args["sources"],
            spill=args["spill"],
            keep_free_gb=args["keep_free"],
            int8=args["int8"],
            dry_run=args["dry_run"],
        )
    elif command == "drills":
        from .practice import drills

        args["arenas"] = tuple(args["arenas"])
        args["countries"] = tuple(args["countries"])
        result = drills(args.pop("output"), **args)
    elif command == "practice":
        from .practice import practice

        args["countries"] = tuple(args["countries"])
        result = practice(args.pop("checkpoint"), args.pop("output"), **args)
    elif command == "play-policy":
        from .play import evaluate_policy

        args["countries"] = tuple(args["countries"])
        args["saves"] = dict(item.split(":", 1) for item in args.pop("start_save") or [])
        result = evaluate_policy(args.pop("checkpoint"), args.pop("output"), **args)
    elif command == "heatmap":
        from .heatmap import draw_heatmaps

        result = draw_heatmaps(
            args["checkpoint"],
            args["recording"],
            args["output"],
            decisions=args["decisions"],
            count=args["count"],
            model_path=args["model"],
            targets=args["targets"],
        )
    elif command == "setup-pointing":
        from .dataset import recording_splits
        from .heatmap import setup_pointing

        data = Path(args["data"])
        if (data / "manifest.json").exists():
            chosen = [data]
        else:
            splits = recording_splits(data)
            chosen = [
                path.parent
                for path in sorted(data.glob("*/manifest.json"))
                if splits.get(path.parent.name, json.loads(path.read_text()).get("split"))
                == args["split"]
            ]
        result = setup_pointing(
            args["checkpoint"], chosen, model_path=args["model"], radius=args["radius"]
        )
        if args["output"]:
            Path(args["output"]).write_text(json.dumps(result, indent=2))
    elif command == "win-rate":
        from .scripted import win_rate

        games = [g for path in args["results"] for g in json.loads(Path(path).read_text())]
        result = win_rate(games)
    elif command == "tune":
        from . import tuning

        if args["action"] == "ask":
            result = tuning.ask(args["study"])
        elif args["action"] == "seed":
            result = {"added": tuning.seed(args["study"], args["roots"], set(args["skip"]))}
        else:
            result = tuning.show(args["study"])
    elif command == "live":
        from .live import watch

        labels = dict(item.split("=", 1) for item in args["label"])
        result = watch(
            args["runs"], out=args["out"], port=args["port"], peer=args["peer"], hz=args["hz"],
            labels=labels, feed=args["feed"],
        )  # fmt: skip
    elif command == "live-say":
        from .live import say

        result = say(Path(args["feed"]) / "chat.jsonl", args["text"], who=args["who"])
    elif command == "salvage":
        from .recording import recordings, salvage, unsalvage

        found = recordings(args["paths"])
        if args["undo"]:
            result = [unsalvage(root) for root in found]
        else:
            result = [salvage(root, dry_run=args["dry_run"]) for root in found]
    elif command == "check-session":
        from .dataset import sequence_starts, session_labels, setup_end

        labels = session_labels(
            args["source"],
            sources=tuple(args["sources"]),
            lead_in=args["lead_in"],
            drop_keys=tuple(args["drop_keys"]),
            drop_parking=args["drop_parking"],
            look_before_click=args["look_before_click"],
        )
        kinds = labels["actions"][..., 0]
        valid = labels["valid"]
        setup = labels["decisions"] < setup_end(labels["manifest"], labels["times"])
        result = {
            "decisions": len(labels["decisions"]),
            "excluded": len(labels["excluded"]),
            "windows": len(sequence_starts(labels["valid"], 8, 2)),
            "split": labels["manifest"]["split"],
            "source": labels["label_source"],
            "game_speed": labels["speed"],
            "parking_dropped": labels["parking"],
            # What the labels hold, over the valid decisions: moves, and the moves to
            # the screen's centre point, in the whole game and in its setup.
            "moves": int((kinds[valid] == 1).sum()),
            "moves_to_centre": int(
                (
                    (kinds == 1)
                    & (labels["actions"][..., 1] == 512)
                    & (labels["actions"][..., 2] == 512)
                )[valid].sum()
            ),
            "setup_decisions": int((setup & valid).sum()),
            "setup_moves": int((kinds[setup & valid] == 1).sum()),
        }
    elif command == "train-bc":
        from .train import train_bc

        args["sources"] = tuple(args["sources"])
        result = train_bc(args.pop("data"), args.pop("model"), args.pop("output"), **args)
    elif command == "advantage":
        from .offline import advantage_labels

        result = advantage_labels(
            args["checkpoint"],
            args["recording"],
            model_path=args["model"],
            n_step=args["n_step"],
            beta=args["beta"],
            max_weight=args["max_weight"],
            state_value=args["state_value"],
        )
    elif command == "train-state-value":
        from .state_value import train_state_value

        result = train_state_value(args["recordings"], args["output"], epochs=args["epochs"])
    elif command == "train-idm":
        from .idm import train_idm

        args["sources"] = tuple(args["sources"])
        result = train_idm(args.pop("data"), args.pop("model"), args.pop("output"), **args)
    elif command == "cache-features":
        from .features import cache_features

        result = cache_features(
            args["data"],
            args["checkpoint"],
            args["output"],
            model_path=args["model"],
            sources=tuple(args["sources"]),
        )
    elif command == "train-memory":
        from .features import train_memory

        result = train_memory(args.pop("cache"), args.pop("output"), **args)
    elif command == "pointer":
        from .desktop import Desktop
        from .remote import RemoteDesktop
        from .video_import import save_pointer

        with RemoteDesktop(args["peer"]) if args["peer"] else Desktop() as desktop:
            result = save_pointer(desktop, args["output"])
    elif command == "import-video":
        from .video_import import import_video

        result = import_video(args.pop("video"), args.pop("output"), **args)
    elif command == "label":
        from .idm import label_recording

        result = {
            path: label_recording(args["checkpoint"], path, model_path=args["model_path"])
            for path in args["recordings"]
        }
    elif command == "distill":
        from .train import distill

        distill(args.pop("data"), args.pop("model"), args.pop("output"), **args)
    elif command == "bundle-peer":
        from .remote import bundle

        result = bundle(**args)
    elif command in {"probe-peer", "capture"}:
        from .desktop import Desktop
        from .remote import RemoteDesktop

        config = args.get("peer", args.get("config"))
        with RemoteDesktop(config) if config else Desktop() as desktop:
            if command == "capture":
                from PIL import Image

                frame = desktop.capture()
                path = Path(args["output"])
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(frame.rgb).save(path)
                result = {k: v for k, v in frame.meta.items() if k != "events"}
            else:
                result = {k: v for k, v in desktop.attached.items() if k != "payload"}
    elif command == "control":
        print(control(**args))
    elif command == "telemetry":
        from .telemetry import watch

        watch(args["peer"], args["watch"], args["as_json"], args["count"])
    elif command == "job":
        from .remote import RemoteDesktop

        # An observer: the full connection may be held by a game, there or here.
        with RemoteDesktop(args["peer"], attach=False, observer=True) as desktop:
            print(desktop.job(args["action"], args["job_id"], args["kind"], args["args"]))
    elif command == "on-peer":
        from .on_peer import run_on_peer

        session = args["session"][1:] if args["session"][:1] == ["--"] else args["session"]
        result = run_on_peer(
            session, args["peer"], reservation=args["reservation"], job_id=args["job_id"],
            deploy_first=args["deploy"], keep_there=args["keep_there"],
        )  # fmt: skip
        from .on_peer import played_nothing

        if played_nothing(result):
            print(json.dumps(result, indent=2, default=str))
            raise SystemExit(f"on-peer: {result.get('job')} played nothing")
    elif command == "template":
        from .vision import add_template

        result = add_template(**args)
    elif command == "clock":
        from .vision import set_clock_rect

        result = set_clock_rect(**args)
    elif command == "minimap":
        from .vision import set_minimap_rect

        result = set_minimap_rect(**args)
    elif command == "evaluate":
        from .learning import paired_evaluation

        result = paired_evaluation(
            [
                json.loads(line)
                for line in Path(args["results"]).read_text().splitlines()
                if line.strip()
            ]
        )
    elif command == "league-add":
        from .learning import League

        league = League(args["league"])
        league.add(args["checkpoint"])
        result = {"entries": len(league.entries)}
    elif command == "league-sample":
        from .learning import League

        result = League(args["league"], seed=args["seed"]).sample()
    elif command == "generate-map":
        from .mapgen import audit, generate

        grid = {
            "columns_per_half": args["columns_per_half"],
            "rows": args["rows"],
            "state_columns": args["state_columns"],
            "state_rows": args["state_rows"],
            "land_columns": args["land_columns"],
            "land_rows": args["land_rows"],
        }
        result = generate(
            args["game"],
            args["output"],
            preset=args["preset"],
            seed=args["seed"],
            undefended=args["undefended"],
            victory_points_on_border=args["victory_points_on_border"],
            **{key: value for key, value in grid.items() if value is not None},
        )
        result["audit"] = audit(args["output"])
        # The map is on disk either way; a bad one must not exit zero, because the next
        # thing that reads it is the game, which crashes instead of reporting.
        _report(result["audit"]["problems"])
    elif command == "audit-map":
        from .mapgen import audit

        result = audit(args["mod"])
        _report(result["problems"])
    elif command == "preview-map":
        from .mapgen import preview

        result = preview(args["mod"], args["output"])
    elif command == "collect-pair":
        from .runner import collect_pair

        result = collect_pair(
            args["config"], args["output"], args["left_checkpoint"], args["right_checkpoint"]
        )
    elif command == "train-ppo":
        from .runner import train_ppo

        result = train_ppo(**args)
    elif command == "train-critic":
        from .train import train_critic

        result = train_critic(args.pop("data"), args.pop("checkpoint"), args.pop("output"), **args)
    return result


if __name__ == "__main__":
    main()

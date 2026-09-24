from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main():
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
        choices=["ffv1", "x264"],
        default="ffv1",
        help="ffv1 is lossless. x264 is visually lossless (CRF 18, 4:4:4) at about a "
        "hundredth of the size.",
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
    ai.add_argument("--mod", default="artifacts/mods/arena-12x8-v2")
    ai.add_argument("--rules", default="artifacts/calibration-1080p/rules.json")
    ai.add_argument(
        "--ok-button",
        nargs="+",
        default=["artifacts/screens-1080p/ok-button.png", "artifacts/screens-1080p/event-ok.png"],
        help="1920x1080 crops of popup Ok buttons, clicked wherever they appear.",
    )
    ai.add_argument("--hz", type=float, default=5)
    ai.add_argument("--codec", choices=["ffv1", "x264"], default="x264")
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
    rate = sub.add_parser(
        "win-rate",
        help="The scripted player's record against the game's AI, from record-ai results",
    )
    rate.add_argument("results", nargs="+", help="results-*.json files written by record-ai")
    check = sub.add_parser(
        "check-session",
        help="Check that a recording can train, and count its decisions. Training reads "
        "recordings directly; nothing is prepared.",
    )
    check.add_argument("source")
    check.add_argument("--sources", nargs="+", default=["human", "ai"])
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
    args = vars(parser.parse_args())
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
    elif command == "play-policy":
        from .play import evaluate_policy

        args["countries"] = tuple(args["countries"])
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
    elif command == "win-rate":
        from .scripted import win_rate

        games = [g for path in args["results"] for g in json.loads(Path(path).read_text())]
        result = win_rate(games)
    elif command == "check-session":
        from .dataset import sequence_starts, session_labels

        labels = session_labels(args["source"], sources=tuple(args["sources"]))
        result = {
            "decisions": len(labels["decisions"]),
            "excluded": len(labels["excluded"]),
            "windows": len(sequence_starts(labels["valid"], 8, 2)),
            "split": labels["manifest"]["split"],
            "source": labels["label_source"],
            "game_speed": labels["speed"],
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
    elif command == "job":
        from .remote import RemoteDesktop

        with RemoteDesktop(args["peer"], attach=False) as desktop:
            print(desktop.job(args["action"], args["job_id"], args["kind"], args["args"]))
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

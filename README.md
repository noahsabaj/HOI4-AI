# HOI4 visual self-play prototype

Project code is dual-licensed under [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE), at your option. Third-party models and dependencies retain their own terms; see [NOTICE.md](NOTICE.md).

One direct policy: screenshots → video encoder and detail crops → GRU memory → raw mouse/keyboard events. The deployed actor has no world-state API or planner. Python supplies recording, training and match coordination; a Rust Windows worker captures pixels and applies input.

The worker downscales before transport. A `capture` can ask for the six policy views — one global frame, four quadrants, and a native crop centered on the pointer — plus the calibrated template crops, instead of the native frame. That is what takes the payload from 33.2 MB down to about a megabyte and removes the resize from the decision loop. The worker's resampler is an exact integer-binned area average that reproduces the training resize bit for bit; both implementations assert the same golden vectors, because a filter mismatch would not fail loudly, it would quietly feed the policy different pixels than it trained on. Apply is handled while that capture runs, so the eight 25 ms slots are not stuck behind the blit.

Pointer positions are quantized onto a square 1024×1024 lattice of the client rectangle. On a 3840×2160 screen that is 3.75 px horizontally and 2.11 px vertically, so controls narrower than about four pixels cannot be addressed exactly and recorded human motion is re-quantized before it becomes a training label.

**Research prototype; no trained combat agent yet.** See [STATUS.md](STATUS.md) for measured results and outstanding acceptance gates. The generated arena now starts a match and runs; combat, supply over time and two-machine self-play remain unverified.

## Setup

Requires Windows, PowerShell 7.5+ (`pwsh`), HOI4, Rust, uv and ffmpeg on PATH. From the repository root:

```powershell
.\scripts\Setup.ps1
.\scripts\Download-Encoder.ps1
.venv\Scripts\hoi4-arena.exe --help
```

Dependencies are locked in `uv.lock` and `Cargo.lock`. The downloaded LeVJEPA custom model code is pinned and checked against `configs/model-source-review.json` before execution. Model weights, recordings, generated maps and pairing credentials are ignored by Git.

The worker only attaches to `hoi4.exe`. It requires foreground focus to apply input, releases held keys/buttons on focus loss, connection loss or its 750 ms watchdog, and provides a physical **F12 emergency stop**. Match actions cannot change game speed. Setup recipes have a separate, limited input mode. Do not treat unit tests as certification of every live release path.

## Second PC

`bundle-peer` writes the pairing to the path you pass it: `peer.json` for this PC and a private `second-pc` folder for the other one. No Python is needed on the second PC, only PowerShell 7.5 or later (`pwsh`; Windows PowerShell 5.1 cannot load the bridge).

The second PC runs from one shared folder that this PC deploys into. Set it up once, on the second PC, in an elevated PowerShell 7:

```powershell
New-Item -ItemType Directory "$HOME\HOI4Worker"
New-SmbShare -Name HOI4Worker -Path "$HOME\HOI4Worker" -ChangeAccess ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -EncryptData $true
```

If that account has no usable password (a Microsoft account signed in by PIN), create a local account for the share instead and grant it the share and the folder (`New-LocalUser`, `Grant-SmbShareAccess -AccessRight Change`, `icacls /grant <name>:(OI)(CI)M`).

On this PC, save that account's sign-in once (`cmdkey` asks for the password), then deploy. `Deploy-Peer.ps1` builds the worker and copies it, its scripts and the pairing files into the share, skipping anything unchanged.

```powershell
cmdkey /add:<second-pc-ip> /user:<second-pc-account> /pass
.\scripts\Deploy-Peer.ps1
```

Then on the second PC, once, in PowerShell 7: `& "$HOME\HOI4Worker\Start-Worker.ps1" -Install`. That starts the worker now and at every logon, hidden, so there is no window to close by accident. It writes `worker.log` in that folder, which this PC can read through the share (`HOI4 worker ready` means it is listening). `-Stop` stops it; undo the install with `-Stop` and by deleting `HOI4 Worker` from `shell:startup`. After that, deploys need nothing on the second PC: a new worker is swapped in before the next connection, and a changed script or pairing restarts the bridge once it is idle, never during a match. Keep the folder private: it contains pairing credentials.

For a two-player match, `Deploy-Peer.ps1 -Mod artifacts\mods\<arena>` copies the arena there, and `control launch` asks the worker to start HOI4 with it and prints the outcome. It is refused if HOI4 is already running there. `control quit` closes HOI4, `control restart-discord` restarts Discord, and `control report` prints its windows, busiest processes and log ends. Without `--peer` they act on this PC. The worker runs these through `Game-Control.ps1` and refuses them while input is armed.

```powershell
.venv\Scripts\hoi4-arena.exe control launch --mod <arena> --peer artifacts/pairing/peer.json
.venv\Scripts\hoi4-arena.exe probe-peer artifacts/pairing/peer.json
.venv\Scripts\hoi4-arena.exe capture artifacts/peer.png --peer artifacts/pairing/peer.json
```

The connection uses a pinned TLS certificate, a random token and the coordinator's source IP. It exposes worker operations, not a remote shell. Both addresses and the port are supplied to `bundle-peer` and stored in the generated config, which is ignored by Git; a DHCP change means regenerating it. No firewall rules are changed automatically. Actual second-PC screenshots, menu mouse/keyboard input and watchdog release have passed. Full screenshot round-trip p95 was 411 ms over 20 menu captures; the transport still needs optimization before the 5 Hz runtime gate.

## Demonstrations and learning

Record 1–4 hours of human play, in complete sessions, with HOI4 in front. Reserve entire sessions for validation and test. Real input timestamps, the pointer position and frame capture times are stored beside the video; the video's frame rate is not the timing source. `--game-speed` is the speed the game is set to for the whole session. It is required and written into the manifest, and the policy is told it, so sessions at different speeds can train together.

```powershell
.venv\Scripts\hoi4-arena.exe record data/raw/session-001 --seconds 1200 --hz 10 --split train --game-speed 4 --codec x264
.venv\Scripts\hoi4-arena.exe check-session data/raw/session-001
.venv\Scripts\hoi4-arena.exe train-bc data/raw artifacts/bc-none --auxiliary none
```

Training reads the recordings directly: each video is decoded front to back and cut into training windows as it plays, several at once through a shuffle buffer, with the views computed on the GPU. Nothing is prepared ahead, so the view sizes can change without redoing any data. `check-session` runs the same checks before any decoding. Bursts of more than eight inputs in one 200 ms interval are excluded rather than relabelled.

What the policy sees of each decision: the last eight global views (224 px, the video encoder's clip), the four screen quadrants at 448 px (so 10 px text survives at about 4.5 px), and a 224 px native fovea centred on the pointer. It places the pointer by picking one of 32x32 screen cells, scored against what each cell shows, then a position inside it. `views` resizes with an exact area average in float32, and the worker does the same in Rust, bit for bit.

Other demonstrations:

- `--sources ai` also trains on the AI games' scripted camera and popup clicks, which `record-ai` stores with each game.
- The inverse dynamics model (`train-idm`) learns to read inputs from video, looking 0.8 s past each decision, on recordings whose inputs are known. `label` then writes its labels into recordings that have none, and `--sources idm` trains the policy on them (the Video PreTraining recipe).

```powershell
.venv\Scripts\hoi4-arena.exe train-idm data/raw artifacts/idm
.venv\Scripts\hoi4-arena.exe label artifacts/idm/epoch-0000.pt data/unlabelled/game-001
.venv\Scripts\hoi4-arena.exe train-bc data/raw artifacts/bc-idm --sources human idm
```

Video from elsewhere (a friend's recording, a published video) has no inputs and no pointer position. `pointer` saves the pointer image the game is showing (repeat it for the game's other pointers), and `import-video` turns a video into a recording: times from its frame rate, the pointer found in each frame by matching those images. `label` then gives it inputs. The worker draws the pointer into every frame it captures, so recordings made here show it the way such videos do.

```powershell
.venv\Scripts\hoi4-arena.exe pointer artifacts/screens-1080p/pointer-menu.png
.venv\Scripts\hoi4-arena.exe import-video clip.mp4 data/unlabelled/clip-001 --pointers artifacts/screens-1080p/pointer-*.png --game-speed 3
```
`--variant screen` swaps the LeVJEPA video encoder for SigLIP 2 (`google/siglip2-base-patch16-naflex`, kept in a local folder passed as `--model`), which reads the four quadrants as one 896 px screen. `distill` trains the compact LeVJEPA student from any recordings.
Repeat BC with `--auxiliary dense` and `--auxiliary sparse`, holding seed, demonstrations, encoder initialization and other settings fixed. Compare `--objective xm --auxiliary none` separately. This is a discrete, noise-conditioned best-of-five **XM-inspired adaptation**, not a faithful reproduction of a continuous-action XM method.

Dense and sparse predictive objectives use separate projection modules. Sparse training uses RepReLU and a rectified-Laplace RDM regularizer inspired by LpWM; 256 projections are a hardware adaptation. Neither auxiliary runs at deployment. These mechanisms have gradient tests, not demonstrated HOI4 learning gains.

## Arena and self-play

```powershell
.venv\Scripts\hoi4-arena.exe generate-map artifacts/mods/infantry-arena --game 'C:\Program Files (x86)\Steam\steamapps\common\Hearts of Iron IV'
.venv\Scripts\hoi4-arena.exe audit-map artifacts/mods/infantry-arena
.\scripts\Test-ArenaLoad.ps1 -Mod artifacts/mods/infantry-arena
```

Generation requires a new output directory. The disposable launch script temporarily selects the mod and restores the prior mod-selection file. Its current 20-second startup assumption requires local verification. Normal later launches use the restored selection. The map is an original rotationally mirrored island with equal infantry forces and ordinary supply; playable match startup remains unverified.

`generate-map` audits what it wrote and exits non-zero if anything is wrong, and `audit-map` re-checks a mod on disk. The audit exists because the engine does not report bad map data: `CProvinceProvider::GetProvince` returns null for any id below 1, and the match-start callers dereference the result without checking, so an unset province id ends the process with an access violation and no log line. It checks every province id the generated files ask the engine to resolve, that both sides of a coast agree, that every province carries the unit-counter anchors and building placements the stock database supplies for it, and that each strategic region has all twelve weather periods.

`template` creates screenshot ROI templates. `clock` calibrates the changing-clock ROI and `speed` is the selected-speed indicator; collection refuses to start without them. A running frame that stops matching `speed` ends the episode, because a click on the speed control would otherwise falsify the manifest. `configs/pair.example.json` shows the two-player configuration, including its `seed` and `deterministic` keys. Real ready/healthy/paused/win/loss/disconnect/desync templates, a changing-clock ROI and observed lobby/reset recipes must be calibrated before collection. Missing evidence fails closed. There are no fabricated default victory templates.

```powershell
.venv\Scripts\hoi4-arena.exe template screen.png artifacts/calibration-left/rules.json healthy --rect 100 40 220 60
.venv\Scripts\hoi4-arena.exe clock screen.png artifacts/calibration-left/rules.json --rect 3420 60 180 34
```

A screen that stops matching `healthy` gets two bounded budgets: a short one while it matches no template at all, and a longer one once some terminal template is in flight, since the outcome debounce cannot start until the panel renders. A terminal template that never converges exhausts the longer budget and invalidates the episode, so it cannot suppress the pause and clock-liveness gates. Any fault inside a step — including a template or resolution mismatch — ends the episode as invalid rather than aborting the coordinator.

```powershell
.venv\Scripts\hoi4-arena.exe collect-pair configs/pair.json artifacts/rollouts/match-001 artifacts/bc-none/epoch-0000.pt artifacts/bc-none/epoch-0000.pt
.venv\Scripts\hoi4-arena.exe train-ppo artifacts/rollouts artifacts/bc-none/epoch-0000.pt artifacts/ppo
```

Set `"downscale": false` in the pair config to make the worker send native frames, and `"record_full": true` to keep native-resolution audit video; by default the audit video records the global view the policy actually saw, and the manifest's `video_source` says which. Checkpoints are hashed and frozen during collection. Both commands seed torch, CUDA and NumPy and record the seed in the run manifest or checkpoint provenance; collection salts the seed with `pair_id` so matches stay reproducible without replaying one RNG stream across a league. Set `"deterministic": true` in the pair config for evaluation matches: the actor then takes the argmax *and* pins its latent, which an xm checkpoint needs to be greedy at all. Greedy rollouts are recorded in the manifest and excluded from PPO, since their likelihoods are not samples from the behavior policy. Leave it false for self-play. Progress and worker diagnostics go to stderr (`--log-level`), JSON results to stdout, and each run writes the worker's captured stderr beside its manifest. Both long-running commands write their evidence and then exit non-zero on failure. Recurrent PPO excludes invalid episodes and historical-opponent data, accounts for elapsed wall time, and normalizes advantages once over the episode. A rollout with no recorded game speed does not train; each step stores the speed the policy was told, so rollouts at different speeds can train together. A rollout stored before the quadrants and fovea (observation layout 1) is excluded. The pair config's `game_speed` is that record; `configs/pair.example.json` shows it. The league class samples current/historical checkpoints, but an unattended league scheduler is not yet wired to the CLI. Collection currently runs both policy actors on the coordinator GPU. A tick dispatches its eight event slots on their own thread, so capture and inference overlap the interval rather than following it; the next tick blocks on that dispatch finishing, which is the cadence barrier. The loop therefore holds 5 Hz for any policy whose capture and inference fit inside the interval, and records `late_seconds` and `deadline_miss` when they do not. Locally, with a simulated 150 ms policy and a 20 ms capture, the tick held 203 ms where the previous serial loop would have taken 370 ms. This has not been measured end to end against a live game with two real actors.

`evaluate results.jsonl` analyzes complete side-swapped pairs. Each row contains `pair_id`, candidate `side` (`left` or `right`), `scenario`, `valid`, and candidate `outcome` (`win`, `draw`, `loss`). Use a frozen imitation baseline and 50 predeclared pairs. The report includes pair-aware uncertainty, invalid exclusions and a sample-completion flag. Training metrics do not select the winner.

`configs/arena.toml` documents intended defaults; it is not currently a CLI configuration loader. Inspect command help for implemented arguments.

## Checks

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check src tests scripts
cargo test --locked
cargo clippy --locked -- -D warnings
```

Model attribution and usage terms are in [NOTICE.md](NOTICE.md).

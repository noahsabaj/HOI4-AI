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

For a two-player match, `Deploy-Peer.ps1 -Mod artifacts\mods\<arena>` copies the arena there, and `control launch` asks the worker to start HOI4 with it and prints the outcome. It is refused if HOI4 is already running there. `control quit` closes HOI4, `control restart-discord` restarts Discord, and `control report` prints its windows, busiest processes and log ends. `control launch --save <name>` loads that save game at startup, skipping the main menu (the game's `-start_save`), so a match can start mid-game; `control saves` lists the save games there. Without `--peer` they act on this PC. The worker runs these through `Game-Control.ps1` and refuses them while input is armed.

```powershell
.venv\Scripts\hoi4-arena.exe control launch --mod <arena> --peer artifacts/pairing/peer.json
.venv\Scripts\hoi4-arena.exe probe-peer artifacts/pairing/peer.json
.venv\Scripts\hoi4-arena.exe capture artifacts/peer.png --peer artifacts/pairing/peer.json
```

The second PC's GPU can train too. `Deploy-Peer.ps1 -Compute` copies the package, its lock file, the study scripts, uv and ffmpeg into the share's `compute` folder (`-Data <folder>` mirrors data such as a feature cache to the same path there). `hoi4-arena job` then runs compute there through the worker's `job` operation and `Run-Job.ps1`. It can build the Python environment (`--kind setup`), run one of a fixed list of `hoi4-arena` training commands (`--kind run`), or run a study script (`--kind script`); `job stop` and `job status` (jobs, GPU, free disk) complete it. Jobs run hidden and detached, with their output in the share's `jobs` folder. Arguments may be flags, values or paths inside `compute`, nothing else, and the worker and the script both check them.

```powershell
.venv\Scripts\hoi4-arena.exe job start --peer artifacts/pairing/peer.json --id setup-1 --kind setup
.venv\Scripts\hoi4-arena.exe job start --peer artifacts/pairing/peer.json --id study --kind script -- memory_study.py artifacts/features artifacts/memory-study --seeds 3 4
.venv\Scripts\hoi4-arena.exe job status --peer artifacts/pairing/peer.json
```

The connection uses a pinned TLS certificate, a random token and the coordinator's source IP. It exposes worker operations, not a remote shell: the compute jobs, approved on 2026-09-23 so the second GPU can work, start only the fixed commands above. Both addresses and the port are supplied to `bundle-peer` and stored in the generated config, which is ignored by Git; a DHCP change means regenerating it. No firewall rules are changed automatically. Actual second-PC screenshots, menu mouse/keyboard input and watchdog release have passed. Full screenshot round-trip p95 was 411 ms over 20 menu captures; the transport still needs optimization before the 5 Hz runtime gate.

## Demonstrations and learning

Record 1–4 hours of human play, in complete sessions, with HOI4 in front. Reserve entire sessions for validation and test. Real input timestamps, the pointer position and frame capture times are stored beside the video; the video's frame rate is not the timing source. `--game-speed` is the speed the game is set to for the whole session. It is required and written into the manifest, and the policy is told it, so sessions at different speeds can train together.

To play on the second PC instead, add `--peer artifacts/pairing/peer.json` and `--hz 5`: its worker captures the screen and your inputs there, and the video is written here. On an arena game, `record` also reads the mod's log, so the manifest names who declared, which country you played and who won, and the win predictor can learn from your games as it does from the AI's. Start recording before you start the game. It stops by itself 15 s after the log names a winner, or when you press F12 or Ctrl+C, and nothing recorded is lost to a mistake: while the game is out of focus (a click outside it, an alt-tab) recording pauses and resumes, and anything that ends it early keeps what it has, with the reason in the manifest's `ended`.

```powershell
.venv\Scripts\hoi4-arena.exe record data/raw/session-001 --seconds 1200 --hz 10 --split train --game-speed 4 --codec x264
.venv\Scripts\hoi4-arena.exe record data/raw/session-002 --seconds 1800 --hz 5 --game-speed 5 --codec x264 --peer artifacts/pairing/peer.json
.venv\Scripts\hoi4-arena.exe check-session data/raw/session-001
.venv\Scripts\hoi4-arena.exe train-bc data/raw artifacts/bc-none --auxiliary none
```

Training reads the recordings directly: each video is decoded front to back and cut into training windows as it plays, several at once through a shuffle buffer. Nothing is prepared ahead, so the view sizes can change without redoing any data. `--workers` (2 by default) background processes do the decoding and compute the views on the CPU, byte for byte what the GPU computes, while the GPU trains; each plays its own share of the recordings, so every window is still read once. `--workers 0` does it on the training thread with the views on the GPU, in the order training always used. The global clips are built only for an encoder that reads them: the default Qwen3.5 tower reads the quadrants alone. `check-session` runs the same checks before any decoding. Bursts of more than eight inputs in one 200 ms interval are excluded rather than relabelled. A run in progress saves itself every 10 minutes (`--save-every`, seconds) to `progress.pt` in its output folder, and after each epoch; `--resume` continues it from there with the same settings, so an interrupted run costs minutes, not the run.

Each training step reads its whole window's frames before running the memory over them, since what the screen shows does not depend on the memory. The encoder's frozen blocks run once, without a graph; only the blocks that train, and the detail and fovea readers, are recomputed in the backward pass, `--chunk` frames (4) at a time. Before, each decision ran on its own and was recomputed whole, frozen blocks included. The results are the same to float rounding, and a test holds the new path to the old one's outputs and gradients.

What the policy sees of each decision: the last eight global views (448x256, the video encoder's clip), the four screen quadrants at 576x320 (so 10 px text survives at about 6 px), and a 224 px native fovea centred on the pointer. The first two are 16:9, like the screen; view sizes are (height, width) in Python and [width, height] on the worker's wire. It places the pointer by picking one of 32x32 screen cells, scored against what each cell shows, then a position inside it. `views` resizes with an exact area average in float32, and the worker does the same in Rust, bit for bit.

Where the policy wants to point is a heat map: hot on what it wants to click, cold everywhere else. `heatmap` draws it over a recording's frames, with the demonstrated move marked, and reports how far the hottest point lies from the demonstrated one and how often that point falls in the hottest 1% of the screen. Two training options follow GUI-Actor (arXiv 2506.03143), which points with an attention map over the screen instead of writing coordinates:

- `--pointer-sigma S` scores each demonstrated move against a Gaussian blob S lattice units wide (1 unit is about 1.9 px across and 1.1 px down at 1920x1080), over the cells it covers and the positions inside each, instead of its one exact point. A click anywhere on a button is right, so one 3 px off should not be scored as wrong as one across the screen.
- `--look-before-click` lets the policy press a button only where its pointer already was when the decision began, so it clicks only what its fovea has seen. Decisions that press after a move are left out of training. The recorders now wait 0.25 to 0.45 s between moving onto something and pressing it, so their games follow the rule.

Both are off by default until measured against exact targets.

```powershell
.venv\Scripts\hoi4-arena.exe heatmap artifacts/bc/epoch-0000.pt data/raw/session-001 artifacts/heatmaps --count 24
.venv\Scripts\hoi4-arena.exe train-bc data/raw artifacts/bc-soft --pointer-sigma 4 --look-before-click
```

Other demonstrations:

- `--sources ai` also trains on the AI games' scripted camera and popup clicks, which `record-ai` stores with each game.
- The inverse dynamics model (`train-idm`) learns to read inputs from video, looking 0.8 s past each decision, on recordings whose inputs are known. It reads with LeVJEPA (`--model models/levjepa-large --variant large`, the default), four 448x256 frames in sequence, which read the camera's inputs best. Over the window it runs a two-way GRU, or with `--context transformer` full two-way attention, as VPT's model did; `--sequence 32` or `64` lets a label read further around it, since at speed 5 an input's effect can show late. `label` then writes its labels into recordings that have none, each with its likelihood, and `--sources idm` trains the policy on them (the Video PreTraining recipe). Inferred labels are noisier than recorded ones: `--idm-min-logp` drops the ones the model was least sure of, and `--idm-weight` makes the rest count less.

```powershell
.venv\Scripts\hoi4-arena.exe train-idm data/raw artifacts/idm
.venv\Scripts\hoi4-arena.exe label artifacts/idm/epoch-0000.pt data/unlabelled/game-001
.venv\Scripts\hoi4-arena.exe train-bc data/raw artifacts/bc-idm --sources human idm --idm-weight 0.5
```

Offline reinforcement learning, before any live self-play: `advantage` has a trained critic (`train-critic`) value every decision of one of your recordings, with the memory carried from the game's start, and writes how much each input improved the position over the next 25 decisions (5 s), from your side. `train-bc --advantage` then counts each decision in proportion to exp(advantage / beta): advantage-weighted imitation (AWR, 1910.00177), the simplest form of learning from outcomes offline (AlphaStar Unplugged, 2308.03526; RECAP, 2511.14759). AI games are refused: their recorded inputs are the camera's, and the game's AI decided who won.

```powershell
.venv\Scripts\hoi4-arena.exe advantage artifacts/critic.pt data/human/session-001
.venv\Scripts\hoi4-arena.exe train-bc data/human artifacts/bc-awr --advantage
```

A scripted player makes games whose recorded inputs do decide who wins, without anyone at the keyboard. `record-ai --player scripted` has it fight the recorder's country against the game's AI through the real interface. While the game is still paused it forms the divisions into an army (shift+click on the "Unassigned divisions" alert, then the green + in the army bar). It draws a front line on the border (Z, then a click) and an offensive into enemy land (X, then a right-drag). Then it runs the game and activates the plan (the arrow above the army card). Each game draws its strategy at random: a near or deep offensive, or none; a wait of 0 to 60 s before activating; and sometimes a new offensive every 40 to 120 s. Its inputs are stored as labels, like the camera's, and the manifest lists every order with the frame it was given at and the enemy state it aimed at. `win-rate` reads the results files and reports its record against the AI, overall, by side and by strategy, with 95% intervals. It is the first baseline a learned agent must beat. Its games train with `--sources scripted`.

Since v4 (`arena-12x8-v4`) the recorder decides who declares the war with a fair coin, fired from the console, because the game's own flip at startup came out Red in 36 of 44 games. Arenas since v3 also report each side's true state every day in game.log: divisions in each state, the game's estimate of its army's strength against the enemy's, casualties, manpower, and rifles held against rifles needed. Recordings keep every mod line with the frame it was read at (`arena-log.jsonl`). `train-state-value` fits a small win predictor on that state, on the CPU in seconds. `advantage --state-value` then values each decision from the state rather than from the screen. Training may read the state; the agent never does, and a vanilla lobby has no mod.

```powershell
.venv\Scripts\hoi4-arena.exe record-ai artifacts/scripted-games --minutes 240 --player scripted --mod artifacts/mods/arena-12x8-v4 --speeds 5 --peer artifacts/pairing/peer.json --peer-only
.venv\Scripts\hoi4-arena.exe win-rate artifacts/scripted-games/results-peer-20260923.json
.venv\Scripts\hoi4-arena.exe train-state-value artifacts/state-value.pt artifacts/scripted-games/scripted-peer-20260923-185544
.venv\Scripts\hoi4-arena.exe advantage --state-value artifacts/state-value.pt artifacts/scripted-games/scripted-peer-20260923-185544
```

A learned policy imitates the scripted player's games and then plays them itself. `--lead-in 0` starts a recording's decisions at its first frame (the Qwen tower reads no clip, and the scripted player forms its army in the first 2.5 s), and `--drop-keys 0x20` leaves the space bar out of the labels, since the harness presses it. `--state-weight` and `--order-weight` add training-only losses: the memory predicts the arena's true state from its log and the scripted player's next order. A `splits.json` in the data folder chooses the held-out games. `play-policy` then has a checkpoint play on the second PC against the game's AI, from the screen, recorded; it reserves that PC from the scripted player's recorder first (`artifacts/eval`), and `--point` places each move on its likeliest spot.

```powershell
.venv\Scripts\hoi4-arena.exe train-bc data/scripted artifacts/bc-scripted --sources scripted --lead-in 0 --drop-keys 0x20 --look-before-click --state-weight 0.5 --order-weight 0.2
.venv\Scripts\hoi4-arena.exe play-policy artifacts/bc-scripted/epoch-0000.pt artifacts/live --peer artifacts/pairing/peer.json --games 2 --minutes 40 --reservation first-look --point
```

Video from elsewhere (a friend's recording, a published video) has no inputs and no pointer position. `pointer` saves the pointer image the game is showing (repeat it for the game's other pointers), and `import-video` turns a video into a recording: times from its frame rate, the pointer found in each frame by matching those images. `label` then gives it inputs. The worker draws the pointer into every frame it captures, so recordings made here show it the way such videos do.

```powershell
.venv\Scripts\hoi4-arena.exe pointer artifacts/screens-1080p/pointer-menu.png
.venv\Scripts\hoi4-arena.exe import-video clip.mp4 data/unlabelled/clip-001 --pointers artifacts/screens-1080p/pointer-*.png --game-speed 3
```
The default image encoder (`--variant screen`) is the vision tower of Qwen3.5-0.8B, which reads the four quadrants as one 1152x640 screen. It was chosen from the 2026 encoders by probing each on our own frames (STATUS.md). Its weights (timm's `qwen3_vit_88m_enc.qwen3_5_0_8b`, Apache-2.0, 400 MB) go in `models/qwen3-vit-88m`, which is `--model`'s default:

```powershell
.venv\Scripts\hf.exe download timm/qwen3_vit_88m_enc.qwen3_5_0_8b model.safetensors --local-dir models/qwen3-vit-88m
```

Memory on long windows: `train-bc` trains on 16 decisions (3.2 s) at a time, because the vision tower runs at every step. `cache-features` freezes a trained policy's perception and stores what it reads at each decision (about 0.5 MB a decision), and `train-memory` then trains the memory and action head on that cache over whole games, with the memory carried from each game's start. It compares memory cells at the same budget: the GRU, Gated DeltaNet-2, Mamba-3 (both in plain PyTorch, checked against their reference implementations) and none. It scores each on held-out games, by imitation loss and by linear probes of what the memory holds that the screen does not show.

```powershell
.venv\Scripts\hoi4-arena.exe cache-features data/ai artifacts/bc/epoch-0000.pt artifacts/features
.venv\Scripts\hoi4-arena.exe train-memory artifacts/features artifacts/memory/gdn2-256 --memory gdn2 --window 256
.venv\Scripts\hoi4-arena.exe train-memory artifacts/features artifacts/memory/gru-16 --window 16 --no-carry --burn-in 2
```

`--variant large` keeps the LeVJEPA video encoder (`--model models/levjepa-large`), and `distill` trains its compact student. `scripts/probe_encoders.py` reruns the comparison on any two recordings.
Repeat BC with `--auxiliary dense` and `--auxiliary sparse`, holding seed, demonstrations, encoder initialization and other settings fixed. Compare `--objective xm --auxiliary none` separately. It is Explorative Modeling (arXiv 2607.27372) applied to the action head: the head is conditioned on a latent, K latents are explored per decision, and the one that best explains the demonstration trains. `--xm-candidates` sets K (5 by default; the paper sweeps 1, 2, 3, 5, 8, 12), `--xm-form smooth` trains every candidate on the mixture's likelihood instead of only the best, and `--xm-latents N` learns N latents to choose among instead of Gaussian noise, as the paper's discrete models do. The paper found autoregressive models like this head the hardest to improve, so treat it as an experiment.

Dense and sparse predictive objectives use separate projection modules. Sparse training uses RepReLU and a rectified-Laplace RDM regularizer inspired by LpWM; 256 projections are a hardware adaptation (`--projections`). LpWM's paper (arXiv 2608.22764) adds two options, both off by default: `--sparsity-shift` moves the target's mean below zero for sparser codes, and `--temporal-jaccard` penalizes codes whose support changes between steps, which in LpWM made the support follow contact rather than the arm's motion (here, the game rather than the camera). Neither auxiliary runs at deployment. These mechanisms have gradient tests, not demonstrated HOI4 learning gains.

## Arena and self-play

```powershell
.venv\Scripts\hoi4-arena.exe generate-map artifacts/mods/infantry-arena --game 'C:\Program Files (x86)\Steam\steamapps\common\Hearts of Iron IV'
.venv\Scripts\hoi4-arena.exe audit-map artifacts/mods/infantry-arena
.\scripts\Test-ArenaLoad.ps1 -Mod artifacts/mods/infantry-arena
```

`--preset` picks a named arena instead of the plain one, and `preview-map` draws it from its files. There are six, all 12x8 provinces a side in 8 states, each an exact half-turn mirror, painted with stock terrain, relief, rivers on province borders, trees, cities holding the victory points, a trunk railway, and borders that wander: `plains` (open farmland), `river` (a large river behind each border), `passes` (a mountain range on the border crossed by two valleys), `marsh` (a marsh and lake in the middle, forests on the wings), `bay` (the sea cuts in at the border, leaving an isthmus) and `salient` (the border bends round a bulge on each side). `--seed` draws another map of the same design. STATUS.md says what each changes.

```powershell
.venv\Scripts\hoi4-arena.exe generate-map artifacts/mods/arena-river-v3 --preset river --game 'C:\Program Files (x86)\Steam\steamapps\common\Hearts of Iron IV'
.venv\Scripts\hoi4-arena.exe preview-map artifacts/mods/arena-river-v3 artifacts/arenas/previews/arena-river-v3.png
```

Generation requires a new output directory. The disposable launch script temporarily selects the mod and restores the prior mod-selection file. Its current 20-second startup assumption requires local verification. Normal later launches use the restored selection. The map is an original rotationally mirrored island with equal infantry forces and ordinary supply; playable match startup remains unverified.

`generate-map` audits what it wrote and exits non-zero if anything is wrong, and `audit-map` re-checks a mod on disk. The audit exists because the engine does not report bad map data: `CProvinceProvider::GetProvince` returns null for any id below 1, and the match-start callers dereference the result without checking, so an unset province id ends the process with an access violation and no log line. It checks every province id the generated files ask the engine to resolve, that both sides of a coast agree, that every province carries the unit-counter anchors and building placements the stock database supplies for it, and that each strategic region has all twelve weather periods. It also checks the bitmaps: each province is one piece with no four-way pixel corners, its painted terrain matches `definition.csv`, land is above sea level, rivers can be traced (a source at a free end, one pixel wide, joins marked) and lie on province borders where they are crossed, and a preset is an exact half-turn mirror with no seam down the middle.

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

Set `"downscale": false` in the pair config to make the worker send native frames, and `"record_full": true` to keep native-resolution audit video; by default the audit video records the global view the policy actually saw, and the manifest's `video_source` says which. Checkpoints are hashed and frozen during collection. Both commands seed torch, CUDA and NumPy and record the seed in the run manifest or checkpoint provenance; collection salts the seed with `pair_id` so matches stay reproducible without replaying one RNG stream across a league. Set `"deterministic": true` in the pair config for evaluation matches: the actor then takes the argmax *and* pins its latent, which an xm checkpoint needs to be greedy at all. Greedy rollouts are recorded in the manifest and excluded from PPO, since their likelihoods are not samples from the behavior policy. Leave it false for self-play. Progress and worker diagnostics go to stderr (`--log-level`), JSON results to stdout, and each run writes the worker's captured stderr beside its manifest. Both long-running commands write their evidence and then exit non-zero on failure. Recurrent PPO excludes invalid episodes and historical-opponent data, accounts for elapsed wall time, and normalizes advantages once over the episode. A rollout with no recorded game speed does not train; each step stores the speed the policy was told, so rollouts at different speeds can train together. A rollout stored in an older view layout (before the quadrants and fovea, or before the 16:9 views) is excluded. The pair config's `game_speed` is that record; `configs/pair.example.json` shows it. The league class samples current/historical checkpoints, but an unattended league scheduler is not yet wired to the CLI. Collection currently runs both policy actors on the coordinator GPU. A tick dispatches its eight event slots on their own thread, so capture and inference overlap the interval rather than following it; the next tick blocks on that dispatch finishing, which is the cadence barrier. The loop therefore holds 5 Hz for any policy whose capture and inference fit inside the interval, and records `late_seconds` and `deadline_miss` when they do not. Locally, with a simulated 150 ms policy and a 20 ms capture, the tick held 203 ms where the previous serial loop would have taken 370 ms. This has not been measured end to end against a live game with two real actors.

The critic follows PACT (Fu et al., 2026, arXiv:2609.26355). GAE uses lambda = 1 (`--gae-lambda`): a match is thousands of decisions scored mostly at its end, and below 1 every intermediate value error leaks into the advantages. The value head predicts the return scaled into [0, 1] and trains with binary cross-entropy. By default (`--critic pact`) PPO updates the actor first, then trains the value head alone on the stored steps replayed under the updated policy, each return weighted by that step's likelihood ratio (ratios outside [0, 6] left out), so the critic estimates the policy just trained rather than the one before it. `--critic joint` is the usual single loss. Before self-play, `train-critic` pre-trains the value head of a behaviour-cloned checkpoint on recorded AI games, whose winners are known, and reports how often it names the winner.

Two options follow InfoPPO (Zeng et al., 2026, arXiv:2609.24380), which measures each step by the collecting policy's entropy there. Every stored step keeps that entropy. `--clock information` discounts over it instead of over ticks, so the many ticks a policy confidently spends waiting cost no horizon and a match is discounted over its decisions. `--clip adaptive` replaces PPO's fixed [0.8, 1.2] with per-step bounds that widen with the entropy, logarithmically (`--info-clip`, default the paper's 10 and 20), so a confident step barely moves and an unsure one may move further. Both are off by default: the paper measured them on language models.

```powershell
.venv\Scripts\hoi4-arena.exe train-critic artifacts/ai-games-1080p artifacts/bc-none/epoch-0000.pt artifacts/bc-critic.pt
```
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

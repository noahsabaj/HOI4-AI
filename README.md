# HOI4 visual self-play prototype

Project code is dual-licensed under [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE), at your option. Third-party models and dependencies retain their own terms; see [NOTICE.md](NOTICE.md).

One direct policy: screenshots → video encoder and detail crops → GRU memory → raw mouse/keyboard events. The deployed actor has no world-state API or planner. Python supplies recording, training and match coordination; a Rust Windows worker captures pixels and applies input.

Pointer positions are quantized onto a square 1024×1024 lattice of the client rectangle. On a 3840×2160 screen that is 3.75 px horizontally and 2.11 px vertically, so controls narrower than about four pixels cannot be addressed exactly and recorded human motion is re-quantized before it becomes a training label.

**Research prototype; no trained combat agent yet.** See [STATUS.md](STATUS.md) for measured results and outstanding acceptance gates. The generated arena currently reaches country selection but crashes when starting a match on HOI4 1.19.3.

## Setup

Requires Windows, HOI4, Rust, uv and ffmpeg on PATH. From the repository root:

```powershell
.\scripts\Setup.ps1
.\scripts\Download-Encoder.ps1
.venv\Scripts\hoi4-arena.exe --help
```

Dependencies are locked in `uv.lock` and `Cargo.lock`. The downloaded LeVJEPA custom model code is pinned and checked against `configs/model-source-review.json` before execution. Model weights, recordings, generated maps and pairing credentials are ignored by Git.

The worker only attaches to `hoi4.exe`. It requires foreground focus to apply input, releases held keys/buttons on focus loss, connection loss or its 750 ms watchdog, and provides a physical **F12 emergency stop**. Match actions cannot change game speed. Setup recipes have a separate, limited input mode. Do not treat unit tests as certification of every live release path.

## Second PC

The prepared private bundle is `artifacts/pairing/second-pc.zip`. Extract it on the peer machine, open HOI4 and run `Start-Worker.ps1`. It prints `HOI4 worker ready`. No Python is needed there. Keep the ZIP private: it contains pairing credentials.

```powershell
.venv\Scripts\hoi4-arena.exe probe-peer artifacts/pairing/peer.json
.venv\Scripts\hoi4-arena.exe capture artifacts/peer.png --peer artifacts/pairing/peer.json
```

The connection uses a pinned TLS certificate, a random token and the coordinator's source IP. It exposes worker operations, not a remote shell. The prepared addresses are coordinator `<coordinator-address>`, peer `<peer-address>`, TCP port `<worker-port>`; DHCP changes require updating configuration. No firewall rules are changed automatically. Actual second-PC screenshots, menu mouse/keyboard input and watchdog release have passed. Full screenshot round-trip p95 was 411 ms over 20 menu captures; the transport still needs optimization before the 5 Hz runtime gate.

## Demonstrations and learning

Record 2–4 hours of human combat, in complete sessions, with HOI4 foreground. Reserve entire sessions for validation and test. Real input timestamps and frame capture intervals are stored alongside native-resolution lossless FFV1 video; video frame rate alone is not the timing source.

```powershell
.venv\Scripts\hoi4-arena.exe record data/raw/session-001 --seconds 1200 --hz 10 --split train
.venv\Scripts\hoi4-arena.exe prepare data/raw/session-001 data/prepared/session-001
.venv\Scripts\hoi4-arena.exe distill data/prepared models/student.pt
.venv\Scripts\hoi4-arena.exe train-bc data/prepared artifacts/bc-none --variant tiny --student models/student.pt --auxiliary none
```

Collect separate `--split validation` and `--split test` sessions before training. The default automatic split hashes whole sessions; explicit splits are useful with few sessions. Data preparation rejects unsupported or excessive action bursts rather than silently changing their labels. Distillation and training have not run on expert data yet.

Repeat BC with `--auxiliary dense` and `--auxiliary sparse`, holding seed, demonstrations, encoder initialization and other settings fixed. Compare `--objective xm --auxiliary none` separately. This is a discrete, noise-conditioned best-of-five **XM-inspired adaptation**, not a faithful reproduction of a continuous-action XM method.

Dense and sparse predictive objectives use separate projection modules. Sparse training uses RepReLU and a rectified-Laplace RDM regularizer inspired by LpWM; 256 projections are a hardware adaptation. Neither auxiliary runs at deployment. These mechanisms have gradient tests, not demonstrated HOI4 learning gains.

## Arena and self-play

```powershell
.venv\Scripts\hoi4-arena.exe generate-map artifacts/mods/infantry-arena --game 'C:\Program Files (x86)\Steam\steamapps\common\Hearts of Iron IV'
.\scripts\Test-ArenaLoad.ps1 -Mod artifacts/mods/infantry-arena
```

Generation requires a new output directory. The disposable launch script temporarily selects the mod and restores the prior mod-selection file. Its current 20-second startup assumption requires local verification. Normal later launches use the restored selection. The map is an original rotationally mirrored island with equal infantry forces and ordinary supply; playable match startup remains unresolved.

`template` creates screenshot ROI templates and `clock` calibrates the changing-clock ROI; collection refuses to start without both. `configs/pair.example.json` shows the two-player configuration, including its `seed` and `deterministic` keys. Real ready/healthy/speed-two/win/loss/disconnect/desync templates, a changing-clock ROI and observed lobby/reset recipes must be calibrated before collection. Missing evidence fails closed. There are no fabricated default victory templates.

```powershell
.venv\Scripts\hoi4-arena.exe template screen.png artifacts/calibration-left/rules.json healthy --rect 100 40 220 60
.venv\Scripts\hoi4-arena.exe clock screen.png artifacts/calibration-left/rules.json --rect 3420 60 180 34
```

A screen that stops matching `healthy` gets two bounded budgets: a short one while it matches no template at all, and a longer one once some terminal template is in flight, since the outcome debounce cannot start until the panel renders. A terminal template that never converges exhausts the longer budget and invalidates the episode, so it cannot suppress the speed-two and clock-liveness gates. Any fault inside a step — including a template or resolution mismatch — ends the episode as invalid rather than aborting the coordinator.

```powershell
.venv\Scripts\hoi4-arena.exe collect-pair configs/pair.json artifacts/rollouts/match-001 artifacts/bc-none/epoch-0000.pt artifacts/bc-none/epoch-0000.pt
.venv\Scripts\hoi4-arena.exe train-ppo artifacts/rollouts artifacts/bc-none/epoch-0000.pt artifacts/ppo
```

Checkpoints are hashed and frozen during collection. Both commands seed torch, CUDA and NumPy and record the seed in the run manifest or checkpoint provenance; collection salts the seed with `pair_id` so matches stay reproducible without replaying one RNG stream across a league. Set `"deterministic": true` in the pair config for evaluation matches: the actor then takes the argmax *and* pins its latent, which an xm checkpoint needs to be greedy at all. Greedy rollouts are recorded in the manifest and excluded from PPO, since their likelihoods are not samples from the behavior policy. Leave it false for self-play. Progress and worker diagnostics go to stderr (`--log-level`), JSON results to stdout, and each run writes the worker's captured stderr beside its manifest. Both long-running commands write their evidence and then exit non-zero on failure. Recurrent PPO excludes invalid episodes and historical-opponent data, and accounts for elapsed wall time. The league class samples current/historical checkpoints, but an unattended league scheduler is not yet wired to the CLI. Collection currently runs both policy actors on the coordinator GPU. It records deadline misses; five decisions per second has not been achieved end to end.

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

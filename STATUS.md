# Status — 2026-09-23

The pipeline runs end to end in pieces, but there is no trained agent yet and no match
between two agents has been played. This file lists what has actually been measured.
Wrong claims are fixed in place; `git log` keeps the history.

## What works

| Check | Evidence | Result |
|---|---|---|
| Hardware | Consumer desktop, 8 GB VRAM, 32 GB RAM; the second PC is the same | All timings below are on this machine |
| Screen capture at 3840×2160 | `artifacts/worker-menu-v3-*` | Follows real menu changes. Desktop Duplication is the fast path, GDI blit the fallback |
| Mouse and keyboard input | `artifacts/worker-smoke/` | Clicks advance menus. A held key is released by the watchdog |
| Second PC | `artifacts/pairing/integration/report.json`; `probe-peer` and a 4K capture, 2026-09-22 | TLS connection, remote capture, clicks, Escape and the watchdog all work |
| LAN capture per tick | `artifacts/pairing-roundtrip-downscaled.json` | Views plus screen checks: 83.9 ms p50, 99.5 ms p95, 0.87 MB |
| Arena loads and runs | `infantry-arena-v13`, `logs/game.log` | 1537 provinces, no map errors, ran to April 1936 at speed 4 |
| Combat | `infantry-arena-v12` | Orders, battles and results all work. Two Blue divisions attacked one Red defender and lost |
| Long run | `capitulation-harness-v2` | 1936 to May 1940 at speed 4–5 with no crash |
| **A capitulation on demand** | `capitulation-harness-v5`, `artifacts/capitulation-harness-run-2026-09-22/` | **Works on a small map.** Blue (AI) beat an unarmed Red and signed a peace taking 2 of Red's 4 states by 29 Jan 1936 |
| **An armed match ends in time** | `small-arena-v1`, `artifacts/small-arena-armed-run-2026-09-22/`, `…-armed-run2-…` | Both sides armed, both AI (observer mode), speed 4. Run 1: Blue surrendered in early 1937, about 17–18 minutes of real time. Run 2: Red surrendered on 1 Jul 1936, about 7.5 minutes. Two of two ended in time, with a different winner each time |

Automated checks: 155 Python tests and 24 Rust tests (2 need a live desktop and are
skipped in CI), plus Ruff and Clippy. CI runs all of them on Windows.

**Not yet shown:** a match between two agents. A two-player match across the two PCs
was played to a surrender on 2026-09-22, driven from this PC.

## How a country surrenders

- **Surrender depends on occupied territory, not victory points.** In
  `capitulation-harness-v2`, Blue held all of Red's victory points from 13 Jan 1936 and
  Red still had not surrendered by May 1940. The tooltip value that matters grows as
  land is occupied.
- **On the full map, the AI stops early.** With Red fielding no army at all, Blue's AI
  took about 47% of Red's 600 provinces in 1936 and then did nothing for 43 months. Why
  is unknown: supply, a front it considers held, or an objective it thinks it reached.
- **On a small map it finishes.** `capitulation-harness-v5` gives each side 24 land
  provinces in 4 states (6 columns by 4 rows) inside the normal-size grid. Blue declared
  war on 1 Jan 1936, and by 29 Jan Red had surrendered and lost 2 states in the peace
  deal. At speed 5 that took seconds of real time.
- **With both sides armed it still finishes.** On `small-arena-v1` (same map, both
  armies) both runs ended in a surrender, after about 14 months of game time in one and
  6 months in the other. The winner took land in the peace deal but did not annex the
  loser, and the game kept running at peace. So a match must end at the surrender, not when a country disappears.
- Observer mode: the `observe` console command (debug mode) hands both countries to the
  AI. The worker allows the console key in setup mode only, never in a match, so the
  recorder can type it on either PC.
- Useful game constants: `BASE_SURRENDER_LEVEL = 1.0` (`NDiplomacy`) is the surrender
  threshold. `BASE_SURRENDER_LIMIT = 0.8` is an occupation fraction, not the threshold.
- Game speed in wall-clock seconds per in-game hour: `{2.0, 0.5, 0.2, 0.1, 0.0}` for
  speeds 1 to 5. Speed 1 is 48 s per in-game day. Speed 5 does not wait at all.

## Match-end screens

HOI4 has no "game over" screen. What each side actually sees when the other surrenders
(single player, 2026-09-22, frames in `artifacts/match-end-screens-2026-09-22/`):

- **The winning player** gets the full-screen peace conference ("Make your Demands",
  "Confirm and Exit") with a "Red has capitulated" equipment popup on top. The game
  waits there until the player confirms. `healthy` stops matching on this screen.
- **The losing player, with an AI winner,** sees no surrender popup and no conference.
  The first sign is the peace summary popup, "Treaty of East Capital: Blue took 2
  states", over the map. The treaty is named after the loser's capital. Observer mode
  sees the same popup.
- **In a two-player match both players get the peace conference.** Tested 2026-09-22
  with this PC hosting Blue and the second PC as Red, joined by Server ID. When Red
  surrendered, Blue got "Make your Demands" with the capitulation popup, and Red got
  the same window titled **Defeated** ("You have been defeated. The victors are
  currently making demands"). Both games wait there.
- Switching sides mid-game: the `tag BLU` console command. That is how the winner's view
  was captured: play Red, then take Blue just before Red surrenders.

## Screen calibration

**The arena now runs in a 1920×1080 window** (`Test-ArenaLoad.ps1 -Window 1920x1080`),
so the game renders at 1080p and captures and recordings are that size too. HOI4 is not
DPI aware: on this 150%-scaled 4K monitor Windows stretched a 1920×1080 window to
2880×1620, so the launch marks only that process DPI aware. The script also switches
the game to windowed and puts the player's own display settings back after the game
exits (HOI4 writes them back on exit, so restoring them sooner is undone).

Calibrated at 1920×1080 in `artifacts/calibration-1080p/rules.json`: `ready`, `paused`,
`healthy`, `speed`, `clock_rect`, the territory crop, and, from a two-player match on
2026-09-22 with both PCs at 1080p, `win` ("Make your Demands", `[889, 93, 138, 18]`),
`loss` ("Defeated", `[120, 111, 92, 18]`) and `disconnect` ("Server Lost!",
`[862, 431, 192, 26]`). Each matches only its own screen: the nearest other captured
screen is 36 away for `win` and `loss` and 42 for `disconnect`. On a running game
`healthy` and `speed` read 0 and `ready` 35 or more. Only `desync` is left. The layout
differs from 4K, so no 4K rect carries over.

- At 1080p the client showed "Server Lost!" 19 s after the host died.
- A windowed game started by a background process does not take focus, so the worker
  has a setup-only `focus` command (`Desktop.focus()`) that brings the game to the
  front. It is refused while armed. Anything else in front, such as the Claude app,
  stops input and capture until the game has focus again.
- A settings file that never had a windowed size gets one added; without it the second
  PC opened its window at the desktop size.
- In a two-player match the host's lobby panel has its own Start button, and the game
  does not begin until it is pressed.

**Reward from the log.** On the arena a match is now scored from the mod's game.log
lines (`reward: "log"`, the default in the pair config), not from pixels. The surrender
line gives the win or loss. Between surrenders the reward is the change in a potential:
the enemy's surrender progress minus one's own, plus half the difference in states held
as a share of one side's states. That is potential-based shaping, so it adds early
signal without changing which policy is best, and it does not depend on the camera.
Screen scoring stays for a vanilla lobby, which has no mod.

**Territory reward from the screen.** HOI4 has no minimap, so the reward reads the main view, but only
when the camera shows the whole arena at full zoom-out: the arena is then 469 px wide
at 1080p (`minimap_span`), and any frame whose land is not within 15% of that width is
not a reading. The map draws the country colours faintly (Blue's land about
(120, 134, 145), Red's (168, 145, 131)), so land is told apart by tint, not by the
written colours, and only the largest connected patch counts, which drops lit cloud.
At the start Blue reads 52% and Red 48%.

The 4K calibration below (`artifacts/calibration-live/rules.json`) is kept for
reference.

Calibrated at 3840×2160 in `artifacts/calibration-live/rules.json`: `healthy`, `paused`,
`clock_rect`, `speed` (the speed-4 bars at `[3416, 52, 186, 9]`), `win` (the "Make your
Demands" text at `[1810, 140, 215, 30]`) and `loss` (the "Defeated" title at
`[181, 166, 137, 28]`), `disconnect` (the "Server Lost!" title at `[1770, 915, 300, 45]`)
and `ready` (the clock reading "12:00, 1 Jan, 1936" at the start of a game, max 23).
Each matches only its own screen: the nearest other captured screen is 21 away for
`win`, 33 for `loss`, 26 for `disconnect` and 29 for `ready`. **Still needed before a
match can run:** `desync`, which can't be produced on demand.

- The start clock drifts by up to 19 between captures of the same paused frame (the
  pause hatching moves), so `ready` needs a looser threshold than the other rules.
- The client shows "Server Lost!" 25 to 65 seconds after the host dies, not at once.
  The host is healthy the whole time, so it stays in the match; a match needs its own
  timeout on that wait.

What calibration taught us:

- Nothing on a live screen holds perfectly still. The HUD drifts by a mean absolute
  error of about 7 between captures, because the day/night line moves under it.
  Templates take `--max-mae`; `healthy` uses 15.
- The play/pause glyph at `[3392, 20, 20, 20]` is the reliable pause check (`paused`,
  max 18). The speed bars look the same running or paused, so they only show which
  speed is selected.
- The clock check allows a difference of 15 between crops, because even a paused clock
  differs by 6.4 between captures.
- Speed changes with `+` and `-` in setup mode only; the number keys do nothing. Space
  and Escape pause the game, so they are not allowed during a match either.

## The map generator

`hoi4-arena generate-map` writes an arena mod, and `audit-map` checks it for anything
the engine would crash on. The engine does not report bad map data, it crashes, so
every rule below came from a crash dump or the stock files.

- **Small countries must be a land block inside the normal grid** (`--land-columns`,
  `--land-rows`). A smaller grid centred on the map left the surrounding sea to a few
  provinces up to 2245×769 px, and the game crashed while loading.
- **Stock `events` and `common/on_actions` must be replaced.** They refer to states the
  arena doesn't have, and the game crashed on its first daily tick.
- Other crash causes, all fixed: provinces the map never placed; countries with no name
  list or leader; stock `tutorial/tutorial.txt` pointing at state 550; an empty tutorial
  file. The tutorial has to be overridden with exactly one block that names nothing.
- Map facts: the `-1;-1;...` row ends `adjacencies.csv` and is required. Terrain
  palette 0 and 1 are plain plains and forest. A sea strategic region needs
  `naval_terrain`. Map colour comes from `common/countries/colors.txt`. The map must be
  5632×2048 (the stock size) or the camera shows the world repeating.
- Province size is about 88×85 px on a 32×24 grid per side, and armies march 4× faster,
  so one border crossing takes about 1.6 days. The old 8×12 grid took 26 days a crossing.
- Not blockers: missing generals only slow planning, and fronts are made by the engine
  rather than by AI strategy files.
- Diagnostics: `--undefended BLU|RED` gives one side no army. `--victory-points-on-border`
  puts all victory points on one border province; it showed that victory points alone do
  not cause a surrender.

## The model and its data (2026-09-23)

The policy was rebuilt so it can read the screen and point at what it sees:

- **Views.** Eight 224 px global views (the video encoder's clip), the four quadrants at
  448 px instead of 224, and a 224 px native fovea on the pointer. The worker's views of
  a live 1080p frame matched `dataset.views` byte for byte, and a views-only capture took
  21.5 ms (p50, this PC).
- **Reader.** The video encoder's last-frame patch grid is kept, not only its summary
  token. The quadrants go through a convolutional reader that keeps a stride-16 map
  instead of pooling each tile to 2x2. Both are combined into a 32x32 map of the screen.
- **Pointer.** A move picks one of the 32x32 cells, scored against each cell's features,
  then one of 32x32 positions inside it: the same 1024x1024 lattice as before. A test
  trains the head to point at a marked cell placed at random, and it hits it over 90% of
  the time on screens it has not seen.
- **Speed** is an input, so recordings at different speeds train together.
- **Data** is read straight from the recordings' video; nothing is prepared, which at 448
  px would have been about 43 GB per hour. AI games now keep the scripted camera's
  inputs as labels.
- **Inverse dynamics model** (`train-idm`, `label`): the same reader, shown each clip
  shifted 0.8 s past the decision, labels the inputs behind video that has none.
- **The pointer is drawn into every captured frame.** HOI4 uses the Windows cursor,
  which neither capture path includes, so no frame showed the pointer a player sees.
  The worker now draws it (live: the gauntlet pointer, fingertip on the position, views
  still byte-identical to `dataset.views`). `import-video` finds it again in video from
  elsewhere by matching saved pointer images, so that video can be labelled.
- **Screen encoder option** (`--variant screen`, SigLIP 2 base): reads the quadrants as one
  896 px screen.

Measured at batch one in bfloat16 on the 4060 Ti (random weights of the real sizes where
no trained ones exist yet):

| | Result |
|---|---|
| Whole policy step with LeVJEPA: encoder, reader, cells, memory, 8-slot head (eager) | 73.8 ms p50, 761 MiB peak |
| LeVJEPA encoder alone, 8 frames | 63.5 ms |
| Detail reader, four 448 px quadrants | 2.7 ms |
| SigLIP 2 screen encoder at 448 / 672 / 896 px | 8.2 / 16.2 / 31.4 ms |

At 31.4 ms the screen encoder costs half of LeVJEPA, so two actors on one GPU should fit
with it; that is measured once it has weights.

## Performance

All on an RTX 4060 Ti with the game at 3840×2160.

| | Result |
|---|---|
| One actor, whole tick with a live game | 135.0 ms p50, 148.7 ms p95. **Fits the 200 ms tick** |
| Two actors on one GPU (self-play on one PC) | 243.6 ms p50, 279.9 ms p95 of GPU per tick. **Does not fit** |
| Two compact-encoder actors, 8 frames | 85.6 ms p50, 103.6 ms p95. Fits, but the compact encoder is untrained |
| Capture, Desktop Duplication vs GDI blit | 16.2 ms vs 67.1 ms on the same loaded screen |
| Encoder forward, 16 vs 8 frames | 131.5 ms vs 64.9 ms. Fewer than 8 frames gains nothing |
| Worker, five downscaled views | 8.3 ms |

What made the difference: removing hidden GPU syncs in `torch.distributions` argument
checks, capturing the action head as a CUDA graph (26× faster), Desktop Duplication
instead of the blit, and storing frozen weights in bfloat16 (640 MiB instead of 1170).

Tried and not taken: `channels_last` (slower), TF32 (no gain, available as `--tf32`),
resizing on the GPU (would mean sending 33 MB frames), compiling the encoder (changes
stored log-probabilities, which PPO compares against).

Things to know:

- A fresh Desktop Duplication has no image until the screen changes, so the worker
  uses the blit for that one tick instead of giving up on duplication.
- The fused action head is not bit-identical across CPUs; matmul order differs.
- Clips use one frame per decision (200 ms apart, 8 frames = 1.6 s), the same live and
  in training.
- After any change to the Rust worker, rebuild and redeploy it. `Desktop.capture` raises
  if a worker returns the wrong views.

## Second PC

The second PC runs the worker from one shared folder. From this PC,
`scripts/Deploy-Peer.ps1` builds and copies the worker, its scripts and the pairing
files there, skipping unchanged files. `hoi4-arena control` launches, closes and
inspects HOI4 there through the worker. On the second PC, `Start-Worker.ps1
-Install` (PowerShell 7.5+) starts the worker at every logon. After that it applies
updates by itself, but only between connections, never mid-match. See the README.

## Recording AI games

`hoi4-arena record-ai` (`hoi4_arena.ai_games`) plays AI-vs-AI games in observer mode and records them.

- **The arena reports itself in game.log.** The mod logs, without changing any rule,
  `ARENA` lines at the start, every week (states held, divisions, surrender progress per
  country), when a state changes hands, on a surrender (loser and winner) and after the
  peace deal. The worker's `game_log` request returns only these lines, on either PC.
  A game ends on the surrender line, so no pixels are read to call it. Surrender
  progress runs from 0 to 1 (1 is a surrender). `owned` counts states a side both owns
  and controls, so it falls as land is occupied.
- **Popups are clicked, not disabled.** An agent must learn to clear them in a vanilla
  game, so the recorder clicks each popup's Ok button 1 to 4 s after it opens, found
  anywhere on screen with OpenCV template matching. A popup was open in 44% of sampled
  frames of the last game without this, and in 2% of both games with it.
- **The camera comes back to the whole arena** every one to two and a half minutes: it
  zooms fully out and pans until the land is centred, steered by what it sees. Pans
  alone drifted, because pan speed changes with zoom.
- **The default arena is 12x8 provinces a side** (`arena-12x8-v1`, 8 states a side).
  The first two games on it (2026-09-23, one on each PC) both ended in a surrender read
  from the log, after 24.8 and 31.5 minutes at speed 4. Red won both, in July and
  December 1937.
- The pause mark blinks, so the start check looks for it over 10 s rather than in one
  frame.

## Open work

In order:

1. **Record AI-vs-AI games in bulk** with `hoi4-arena record-ai`, on both PCs at once
   with `--peer artifacts/pairing/peer.json`. The second PC's games are launched and
   closed through its worker (`launch`, `quit`) and its frames recorded here: a full
   1080p frame takes about 86 ms over the network, so 5 Hz fits. Both monitors must stay
   switched on (brightness can be zero): a monitor switched off disconnects on
   DisplayPort, Windows shrinks the desktop to 1024x768, and the capture breaks, which
   the recorder reports. On the second PC the Discord overlay is off: after a
   force-closed game it hung every later launch. Games are closed politely first, and a
   hung launch restarts Discord (`restart_discord`) and retries once. `report` lists the
   second PC's windows, busy processes and log ends. The games carry the scripted
   camera's inputs as labels, so they teach camera control and popup clearing, and
   serve the encoder, predicting who wins, and a first opponent. On the 12x8 arena the
   first two games took 24.8 and 31.5 minutes, so a match limit of 1800 s is too short
   there; the recorder's cap is 45 minutes. Recordings are 1080p x264 (CRF 18, 4:4:4).
   `desync` gets calibrated whenever one happens.
2. **Record 1–4 hours of human play** on the arena, with `--game-speed` set to the
   speed used. This is the only source of a player's inputs.
3. **Train the inverse dynamics model** on those and the AI games' inputs, then label
   video that has no inputs (`label`) and train on it (`--sources idm`).
4. **Pick the encoder.** Download the SigLIP 2 weights and compare `--variant screen`
   against LeVJEPA, or distil the compact LeVJEPA student. Either is how two actors fit
   one GPU; the other way is one GPU per side.
5. Train the behaviour-cloning baseline, pre-train its critic on the AI games' winners
   (`train-critic`), then recurrent PPO self-play with a league, scored from the arena
   log, with the PACT critic (lambda = 1, a BCE value head trained after the actor on
   importance-weighted returns). PACT's gains were measured on language models, so its
   pieces are flags (`--gae-lambda`, `--critic`) to compare once self-play runs.
6. Run a two-PC match between agents, and check reset and recovery when something goes
   wrong. (A two-PC match driven by hand, from this PC, works.)
7. Complete 20 unattended matches and 50 side-swapped evaluation pairs.
8. Test the same interface in an unmodified private multiplayer lobby.
# Status — 2026-09-22

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

Automated checks: 125 Python tests and 16 Rust tests (2 need a live desktop and are
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
  AI. The console key is outside the worker's allowed keys, so it is sent separately.
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

Calibrated at 3840×2160 in `artifacts/calibration-live/rules.json`: `healthy`, `paused`,
`clock_rect`, `speed` (the speed-4 bars at `[3416, 52, 186, 9]`), `win` (the "Make your
Demands" text at `[1810, 140, 215, 30]`) and `loss` (the "Defeated" title at
`[181, 166, 137, 28]`), `disconnect` (the "Server Lost!" title at `[1770, 915, 300, 45]`)
and `ready` (the clock reading "12:00, 1 Jan, 1936" at the start of a game, max 23).
Each matches only its own screen: the nearest other captured screen is 21 away for
`win`, 33 for `loss`, 26 for `disconnect` and 29 for `ready`. **Still needed before a
match can run:** `desync`, which can't be produced on demand, and a `minimap_rect` for
the territory reward, which needs a design decision because HOI4 has no minimap.

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
`scripts/Deploy-Peer.ps1` builds and copies the worker, `Start-Worker.ps1` and the
pairing files there, skipping unchanged files. On the second PC, `Start-Worker.ps1
-Install` (PowerShell 7.5+) starts the worker at every logon. After that it applies
updates by itself, but only between connections, never mid-match. See the README.

## Open work

In order:

1. **Finish calibration**: `desync`, and decide how the territory reward reads the map
   (a fixed-camera crop, or an overlay only the reward sees). All other screens are done.
2. **Record AI-vs-AI games** on the arena with `scripts/record_ai_games.py`. They have
   no actions, so they can't teach clicks, but they need no human time and are enough
   for the encoder (step 4), for learning to predict who wins, and as a first opponent.
   Three recorded so far (Red, Red, Blue; 7 to 15 minutes each). Every armed game so far,
   five of five, ended inside 1800 s. At native 4K a game is 7 to 19 GB, so pick a
   smaller storage format before recording many.
3. **Record 1–4 hours of human play** on the arena, with `--game-speed` set to the
   speed actually used. This is the only source of real actions. Recordings need the
   cursor position, which the worker now sends.
4. **Distil the compact encoder** from the AI games and human recordings. It is the only
   measured way to fit two actors in one tick on one GPU; the other is one GPU per side.
5. Train the behaviour-cloning baseline, then recurrent PPO self-play with a league.
6. Run a two-PC match between agents, and check reset and recovery when something goes
   wrong. (A two-PC match driven by hand, from this PC, works.)
7. Complete 20 unattended matches and 50 side-swapped evaluation pairs.
8. Test the same interface in an unmodified private multiplayer lobby.

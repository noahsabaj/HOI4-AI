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

Automated checks: about 220 Python tests and 26 Rust tests (2 need a live desktop and
are skipped in CI), plus Ruff and Clippy. CI runs all of them on Windows.

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
- **Victory-point names go in `localisation/english/replace`.** The stock file names
  thousands of provinces as `VICTORY_POINTS_<id>`, and its names won: Blue's capital
  (province 564) showed as "Kassel".
- **Railways must cross the border.** A supply hub works only while a railway joins it
  to its holder's capital, and nobody in the arena can build one. With no line across
  the border, a captured hub never supplied its captor: on the first terrain arena Blue
  took three states, then stood for four years five provinces from Red's capital,
  against a single Red division. The plain arena still has no crossing lines.
- **Rivers run along province borders.** Only a river on a border is crossed (86% of
  stock river pixels are on one). A river starts at a green source pixel at its free
  end, is one pixel wide and edge-connected, and a tributary ends on a red join pixel.
  Small rivers (indices 3–6) cost 30% of an attack to cross, large ones (7–11) 60%.
- **Lakes are their own class**, as in the stock file: type `lake`, terrain `lakes`,
  never coastal, no unit anchors, and in a land strategic region.
- **Blue's capital must stand mid-country.** The country picker opens centred on it, and
  the recorder picks Red by clicking Red's land there. With the capital near the west
  coast, Red was off the screen and the pick failed.
- **A mirror copied from one half needs symmetric noise.** Copying the western half's
  half turn onto the east is exact, but left a 42-byte cliff down the middle of the
  first mountain arena until the noise itself was symmetric. The map's wrap seam
  (x = 0 meets x = 5631) keeps the plain lattice, because the fix for four-way corners
  never looks there.

## Arena maps (2026-09-24)

`generate-map --preset <name>` writes one of six named arenas (`arenas.PRESETS`), and
`preview-map` draws one from its files. Every one keeps the playable grid (12×8
provinces a side, states 1–8 Blue and 9–16 Red, named "West n" and "East n"), the 35
victory points a side and every rule. Each is an exact half-turn mirror, and `audit()`
checks that, province by province and pixel by pixel.

| Preset | What changes the fight |
|---|---|
| `plains` | Farmland, a few woods and low hills. Rivers run toward the enemy, not across the front |
| `river` | A large river runs coast to coast four provinces behind each border: -60% to attack across |
| `passes` | Mountains two provinces deep on each side of the whole border (-50% attack), crossed by two one-province valleys |
| `marsh` | A marsh round a two-province lake fills the middle of the front; forests on both wings |
| `bay` | The sea cuts in from north and south at the border, leaving a four-province isthmus |
| `salient` | The border itself bends: Blue holds a bulge into Red in the north, Red one into Blue in the south |

What a preset paints, all with stock assets:

- **Terrain in regions:** the stock palette indices each type mostly uses (plains are
  grass with farmland patches, forest dark and light, hills rolling and ridged,
  mountains green slopes and bare rock above byte 150, marsh, urban). Each province's
  painted majority matches its `definition.csv` terrain.
- **Relief:** heights by type, near the stock medians: plains 102, hills 113-116 and
  mountains 164 (90th percentile 190), against the stock's 102, 115 and 129 (172).
  Coasts are ramped over 12 px and rivers lie in shallow valleys. `world_normal.bmp` is
  computed from the heights, so the relief is lit.
- **Cities:** each victory point is an urban province, 60% of it painted as city (stock
  city models and night lights). Forests have stock European trees (85% cover).
- **Borders that wander:** province seeds stray up to 22% of a province, and the Voronoi
  is taken through a displacement that turns with the map, so coasts, state borders and
  the front are no longer ruled lines.

A preset takes 13–17 s to generate. `generation.json` records the preset, its terrain
counts, and a state layout over the land box, from which `scripted.state_at` names the
state under a point on any arena, bulges included.

Live test, `arena-plains-v1` (2026-09-24, second PC, scripted Blue against the AI):
loaded and ran 15 minutes with no errors, and the scripted player found its fronts on
the new ground. It ended as a draw, in the supply stall above; v3 adds the lines across
the border.

## The model and its data (2026-09-23)

The policy was rebuilt so it can read the screen and point at what it sees:

- **Views.** Eight global views at 448x256 (the video encoder's clip), the four
  quadrants at 576x320, and a 224 px native fovea on the pointer. All are 16:9 since
  2026-09-23; before, the screen was squashed into 224 and 448 px squares, which cost
  both encoders what they read (see "LeVJEPA, a second look"). The worker takes view
  sizes as [width, height]. Its views of a live 1080p frame matched `dataset.views` byte
  for byte at the new sizes too. (A views-only capture took 21.5 ms at the old sizes.)
- **Reader.** The video encoder's last-frame patch grid is kept, not only its summary
  token. The quadrants go through a convolutional reader that keeps a stride-16 map
  instead of pooling each tile to 2x2. Both are combined into a 32x32 map of the screen.
- **Pointer.** A move picks one of the 32x32 cells, scored against each cell's features,
  then one of 32x32 positions inside it: the same 1024x1024 lattice as before. A test
  trains the head to point at a marked cell placed at random, and it hits it over 90% of
  the time on screens it has not seen.
  Since 2026-09-23 that distribution can be drawn as a heat map over any recording
  (`heatmap`). Two options, off until measured: `--pointer-sigma` trains against a blob
  around each demonstrated point rather than the point, and `--look-before-click`
  presses only where the pointer already was, after the fovea has seen it (GUI-Actor,
  arXiv 2506.03143, points the same way, with an attention map over the screen).
  The first heat maps (the policy after one epoch on 34 speed-5 AI games, 10 decisions
  of a held-out game) are a near-even wash over the whole screen: a move scored about 2%
  and no cell above 0.5%; the demonstrated point was a median 374 px away and never
  among the hottest 1% of cells. That is right for those games: the scripted camera
  picks where it pans and zooms at random, so nothing on screen says where it will
  point. AI games cannot teach pointing; games whose clicks follow the screen, the
  scripted player's (buttons found by template, fronts drawn on the border), can.
- **Speed** is an input, so recordings at different speeds train together.
- **Data** is read straight from the recordings' video; nothing is prepared, which at 448
  px would have been about 43 GB per hour. AI games now keep the scripted camera's
  inputs as labels.
- **Inverse dynamics model** (`train-idm`, `label`): the same reader, shown each clip
  shifted 0.8 s past the decision, labels the inputs behind video that has none. Its
  encoder is LeVJEPA reading the last four 448x256 frames (`--variant large`), the best
  reader of the camera's inputs probed; it runs offline, so its cost does not matter. One
  epoch on one speed-5 AI game, tested on another: input kind right 93.5% of the time
  (about 3 s a step at batch 1 with the game running beside it).
- **The pointer is drawn into every captured frame.** HOI4 uses the Windows cursor,
  which neither capture path includes, so no frame showed the pointer a player sees.
  The worker now draws it (live: the gauntlet pointer, fingertip on the position, views
  still byte-identical to `dataset.views`). `import-video` finds it again in video from
  elsewhere by matching saved pointer images, so that video can be labelled.
- **The image encoder is now the vision tower of Qwen3.5-0.8B** (`--variant screen`, the
  default), reading the quadrants as one 1152x640 screen; see "Choosing the screen
  encoder". Behaviour cloning on the new views ran: one epoch on one AI game, loss 35 to
  2.5. The whole-step times below are from before the 16:9 views and were not re-timed.

Measured at batch one in bfloat16 on the 4060 Ti (random weights of the real sizes where
no trained ones exist yet):

| | Result |
|---|---|
| Whole policy step with LeVJEPA: encoder, reader, cells, memory, 8-slot head (eager) | 73.8 ms p50, 761 MiB peak |
| LeVJEPA encoder alone, 8 frames | 63.5 ms |
| Detail reader, four 448 px quadrants | 2.7 ms |
| Whole policy step with the Qwen3.5 screen encoder, real weights | 50.4 ms p50, 51.3 ms p95, 353 MiB peak |

Two actors with the screen encoder take about 100 ms of the 200 ms tick, where two with
LeVJEPA did not fit.

**End to end, 2026-09-23.** One `record-ai` game on each PC with all of the above: both
launched and closed through the worker, both ended on the log's surrender (Red both
times, after 19.5 and 30.2 minutes), the pointer was drawn in both PCs' frames, and the
camera's inputs came along as labels (5,825 decisions on this PC, none excluded). Behaviour
cloning on those two games ran with LeVJEPA: the loss on the camera's inputs fell from 37.4
to 20.5 in 20 steps.

**Training memory.** A training step keeps every step's activations for the backward
pass, and at batch 2 (windows of 8 steps after 2 of burn-in) that reached 7 GB of the
card's 8. Windows then quietly moves GPU memory into system memory instead of failing,
and a step took 34.6 s. Two fixes: training recomputes each step in the backward pass
(activation checkpointing, the default for `train-bc`, `train-idm` and `train-critic`;
`--no-checkpoint` turns it off), and every command caps its GPU use at 90% of the card
(`--gpu-memory`), so running out raises out-of-memory instead of spilling.

| | Time per window | Peak |
|---|---|---|
| Batch 1 | 1.0 s | 4.3 GB |
| Batch 1, recomputed | 1.6 s | 2.1 GB |
| Batch 2, recomputed | 1.3 s | 2.5 GB |

## Choosing the screen encoder (2026-09-23)

Four search agents (arXiv, Hugging Face and GitHub, screen and OCR models, small
encoders) and Grok on X listed every image encoder released in 2026, with weight to
July–September. No new screen-specific standalone encoder shipped then; GUI agents reuse
the vision towers of small vision-language models. The candidates that fit an 8 GB card
were probed, frozen, on frames of two recorded AI games (`scripts/probe_encoders.py`):
characters of 10–12 px drawn on the game's own pixels, named by a linear read-out from
the patch they sit in (chance 2.8%), and the pointer found among all patches. Train on one
game, test on the other; times are the whole frame at batch one in bfloat16.

| Encoder (release) | Input | Time | Reads text | Finds pointer |
|---|---|---|---|---|
| **Qwen3.5-0.8B vision tower** (Feb 2026; timm 10 Sep 2026) | 896 square | 40.7 ms | **58.4%** | 93.1% |
| same | 1152x640 | 36.9 ms | 59.6% | 91.9% |
| same | 1280x720 | 49.1 ms | 67.8% | 85.6% |
| same | 768 / 672 square | 28.3 / 21.5 ms | 39.4 / 26.3% | 85.0 / 78.1% |
| Holo-3.1-0.8B's copy, GUI-tuned (weights about 1% apart) | 1152x640 | 36.8 ms | 56.2% | 93.1% |
| MonkeyOCRv2-B (Jul 2026, OCR-trained) | 1008x560 / 1120x616 | 50.8 / 63.5 ms | 46.4 / 48.8% | 98.1 / 96.2% |
| TIPSv2-B/14 (Apr 2026) | 896 square | 47.5 ms | 16.6% | 98.1% |
| EUPE ConvNeXt-S (Mar 2026) | 896 square | 24.5 ms | 12.7% | 100% |
| EUPE ViT-S (Mar 2026) | 896 square | 18.1 ms | 9.6% | 98.8% |
| LingBot-Vision ViT-B / ViT-S (Jul 2026) | 896 square | 39.5 / 17.7 ms | 9.5 / 4.8% | 100 / 83.1% |
| Gemma 4 E4B vision tower (Jul 2026) | 864 square | 88.3 ms | 4.4% | 64.4% |

The Qwen tower reads small text four to six times better than the general-purpose
encoders; it was trained inside a vision-language model on documents, screenshots and
GUIs. It became the default screen encoder (Apache-2.0). The pointer test is noisy with 160
test frames (a rerun of the Qwen row gave 85.6%). Larger towers (C-RADIOv4, the 300M
Qwen3.5-2B tower, Qwen4-Exp, which is also gated) cost 120–160 ms at 896 px. UltraViT,
TuringViT and LiAuto-MindViT have no public weights yet.

## LeVJEPA, a second look (2026-09-23)

LeVJEPA had been set aside on speed alone: it never took the text or pointer probe, it
read 224 px squares (12 px text becomes 2 px), and it re-encoded all 8 frames every tick.
Its paper (Kuhn et al., 2026, arXiv:2608.27395) says why each of those undersold it: it
reads any frame size through 3D rotary positions, it attends block-causally so earlier
frames need no re-encoding, and its frozen features are meant for a nonlinear probe.
So it was probed again (`scripts/probe_video_encoders.py`) on two games recorded with the
new camera: 16:9 frames from 448 to 1152 wide, 1, 4 or 8 frames of context, three depths,
each read out by a linear layer and by a small MLP, against the Qwen tower on the same
frames. A new task, camera motion, asks what the camera did in the last 200 ms, from the
recorder's own inputs (still, pan, zoom in, zoom out; balanced accuracy, chance 25%). An
encoder that reads one frame gets the last four side by side. Best of depth and read-out:

| Encoder | Input | Reads text | Finds pointer | Camera motion |
|---|---|---|---|---|
| Qwen tower | 1152x640 | **58.0%** | 95.0% | 54.4% |
| Qwen tower | 896 square | 56.5% | 96.2% | 50.8% |
| Qwen tower | 896x496 | 33.7% | 98.1% | 50.5% |
| LeVJEPA, 1 frame | 1152x640 | 18.0% | 85.6% | 47.2% |
| LeVJEPA, 1 frame | 896x496 | 13.6% | 92.5% | 46.7% |
| LeVJEPA, 4 frames | 640x352 | 7.3% | 93.1% | 62.3% |
| LeVJEPA, 4 frames | 448x256 | 4.5% | 90.6% | **64.6%** |
| LeVJEPA, 8 frames | 448x256 | 4.6% | 93.1% | 59.1% |
| LeVJEPA, 8 frames (the old setup) | 224 square | 3.4% | 87.5% | not run |

- **Text: the Qwen tower, by far.** At the same resolution it reads three times as many
  characters (58.0% against 18.0% at 1152x640, 33.7% against 13.6% at 896x496).
  LeVJEPA learned from natural video, the tower from documents and screens.
- **Motion: LeVJEPA, with frames in sequence.** Four frames in order beat the tower's four
  side by side, 64.6% against 54.4%. One LeVJEPA frame is no better than the tower, so
  the gain is its video context, as its paper would predict. Four frames did better than
  eight here.
- **Pointer:** about equal.
- **Speed.** Streaming LeVJEPA frame by frame, with each layer's keys and values cached
  (`scripts/stream_levjepa.py`), matches the whole-clip pass (1.7e-5 in float32). Idle
  GPU, bfloat16, one frame per tick:

| | Per tick |
|---|---|
| Qwen tower 1152x640 / 896 square / 896x496 | 32 / 36 / 17 ms |
| LeVJEPA streamed, eager, 448x256 / 640x352 / 896x496 | 91 / 109 / 288 ms |
| LeVJEPA, one frame as a CUDA graph plus the cached frames' attention, same sizes | about 20 / 42 / 115 ms |

  Eager LeVJEPA is dominated by Python overhead (the released code rebuilds its rotary
  tables in every layer), which a CUDA graph removes. The tower gains nothing from one.

**Verdict.** The live policy keeps the Qwen tower: reading the screen's numbers needs
resolution, and LeVJEPA reads text poorly even at 1152 px while costing more there. LeVJEPA
is the better encoder where motion is the task and time is not: the inverse dynamics
model, which runs offline and exists to tell which input caused each change between
frames. At 448x256 with four frames it would also fit beside the tower in a tick, for a
policy that wants both. Two side results: the tower reads 16:9 at 1152x640 better than the
896 square it is fed now, for less time (it is fed quadrants tiled into a square, which
squashes the screen); and the LpWM paper (Kuang et al., 2026, arXiv:2608.22764) is about
sparse latents for learned world models, not an encoder, so it had nothing to probe.

## Performance

Re-measured on 2026-09-23 with the Qwen3.5 tower and the 16:9 views, on the second PC's
idle RTX 4060 Ti (the same card as this PC), without a game running
(`scripts/time_policy.py`, `train-bc` on four fixed games):

| | Result |
|---|---|
| One actor's decision step | 38.6 ms p50 with the compiled action head (63.5 ms eager) |
| Two actors, batched into one pass | 78.5 ms p50 (102.4 ms eager). **Fits the 200 ms tick** |
| Vision tower alone, one frame | 33.5 ms eager, 33.0 ms as a CUDA graph (compute-bound; not worth it) |
| A behaviour-cloning step (batch 2, 10 decisions) | 1.47 s, from 2.0 s before #54 (1.36×) |
| An inverse-dynamics step (LeVJEPA-Large, 4 frames, 32 decisions) | about 9 s, the GPU 81–98% busy |

The compiled action head needs Triton; `triton-windows` 3.6 is locked since #53. Before,
it had been installed by hand on this PC only, and elsewhere the head silently ran
eagerly. The newest Triton, 3.8, was no faster and not bit-exact with the eager head, so
3.6 stays until PyTorch moves.

Earlier, all on an RTX 4060 Ti with the game at 3840×2160 and the LeVJEPA encoder:

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

Since 2026-09-24 the bridge there takes one connection that holds the game (a recording,
a match, a launch) and up to four read-only observers beside it. An observer's worker
hooks no input and refuses input, launches, jobs and recording, so `hoi4-arena telemetry
--peer ...` and `control report --peer ...` work while a game is recorded. Before, the
bridge took one connection at a time, and a report during a game timed out on the TLS
handshake. A second full connection now waits 8 s for the first to finish, then is told
`worker_busy`.

`telemetry` reads, once a second: CPU and memory of the PC and of the processes that
matter (the game, the workers, ffmpeg, the bridge, compute jobs, the busiest others), GPU
use per process (Windows' GPU Engine counters) and for the card (NVML: busy, video
encoder, memory, temperature, power), disks, the network, the game window (responding,
in front, on which screen) and the capture's timing.

## Recording where the game runs (2026-09-24)

Frames used to be pulled one request at a time: every 200 ms the recorder asked the
worker for a full 1080p frame, the frame crossed the network as lz4 (1.2-3.2x on game
frames), and this PC encoded it with x264. Anything else on the connection (the camera's
own screenshots) or on this PC (a busy CPU) made frames late. Over 41 games on the second
PC, recordings reached 3.8-5.0 fps with up to 36 late ticks a game, and even the 5.0 fps
games bunched their frames (10-20% of intervals over 300 ms).

Now the worker keeps the clock. With `--codec nvenc` (`record-ai`'s default, and a
choice for `record`) the worker captures the game 5 times a second on its own timer,
draws the pointer, and hands the frames to ffmpeg on the same PC, which encodes them on
that PC's NVIDIA encoder. Only the video and each frame's row (times, pointer, inputs)
cross the network, and this PC writes them as they come. An older worker, or a PC without
NVENC, falls back to x264 here, and the manifest says which ran.

One AI game each on the second PC, same camera and worker, 2026-09-24:

| | x264 here, one request a frame | NVENC stream |
|---|---|---|
| Frames per second (nominal 5) | 3.34 | 4.99 |
| Late ticks | 59 | 0 |
| Frame interval p50 / p95 / p99 | 199 / 859 / 1039 ms | 200.0 / 200.5 / 205.7 ms |
| Intervals over 300 ms | 25.5% | 0.28% (one stall, since fixed) |
| Out of the second PC | 148 Mbit/s | 56 Mbit/s: 3.4 video, the rest the camera's screenshots |
| Encoder on this PC | 0.49 cores, 554 MB | 0.002 cores, 23 MB (a remux) |
| Recorder process on this PC | 0.68 cores | 0.49 cores, mostly camera and popup image work |
| Worker, encoder, bridge on the second PC | 0.17, none, 0.04 cores (at 3.3 fps) | 0.14, 0.09, 0.03 cores; 3% of the video encoder |

The game itself took 3.4-3.5 cores, 3.2 GB and 30% of the GPU either way. x264 is now
capped at 4 threads: its default here took 80 threads and 2 GB for the same 5 fps.

**Is the video as good?** Against 452 lossless 1080p frames (a clip, and screenshots of
menus, maps and scripted games, each held three frames), each candidate encoded and then
decoded with the training reader's own command (`scripts/codec_fidelity.py`, run as a
compute job on the second PC's GPU):

| | KB/frame | PSNR whole / top bar | Worst pixel, 99.9% | Template scores moved | Decode fps |
|---|---|---|---|---|---|
| x264 CRF 18, 4:4:4 (before) | 86.9 | 44.83 / 43.47 dB | 80, 9 | 0.0115 | 245 |
| **NVENC H.264 4:4:4, p7, QP 14** | 88.7 | **45.47 / 43.97 dB** | **50, 7** | **0.0029** | **260** |
| NVENC QP 16 | 72.1 | 44.21 / 42.59 dB | 64, 9 | 0.0035 | 259 |
| NVENC HEVC 4:4:4, QP 14 | 82.0 | 45.16 / 43.59 dB | 72, 7 | 0.0042 | 239 |
| NVENC lossless | 333.3 | 52.63 / 53.28 dB | 2, 2 | 0.0014 | 224 |

That is 151 sightings of 11 templates from `artifacts/screens-1080p`; none moved in any
candidate, and the screen rules' error changed by 1.80 at most at QP 14 against 3.07 for
x264. So the stream keeps full-resolution colour and keeps what the policy reads better
than x264 did. On real games its files are larger (3.4 against 2.2 Mbit/s) and still
decode faster: 122 against 110 fps through the reader, alternated on this PC.

**Capture is faster.** The worker keeps the desktop image on the GPU, reads back only the
game's window, and no longer waits up to 8 ms for the next present: a 1920x1080 capture
went from 14.9 to 4.9 ms p50 on this PC (16 to 4.8 ms on the second PC), still
pixel-identical to the GDI blit.

Also: the capture thread's own output goes through a queue, so a tick never waits for a
large reply to cross the network (one did, for 618 ms). `apply` can take a decision's
events with their offsets (`at_ms`) and apply them on the worker's clock, one request
instead of eight; offsets of 50 ms came out 50.2 and 50.3 ms apart on the second PC.
Captures travel raw over the local pipe (lz4 cost 16-21 ms a frame). BGRA to RGB takes
2.2 ms instead of 18.4.

## Recording AI games

`hoi4-arena record-ai` (`hoi4_arena.ai_games`) plays AI-vs-AI games in observer mode and records them.

- **The arena reports itself in game.log.** The mod logs, without changing any rule,
  `ARENA` lines at the start (who declared the war, each human's country), every week (states held, divisions, surrender progress per
  country), when a state changes hands, on a surrender (loser and winner) and after the
  peace deal. The worker's `game_log` request returns only these lines, on either PC.
  A game ends on the surrender line, so no pixels are read to call it. Surrender
  progress runs from 0 to 1 (1 is a surrender). `owned` counts states a side both owns
  and controls, so it falls as land is occupied.
- **Popups are clicked, not disabled.** An agent must learn to clear them in a vanilla
  game, so the recorder clicks each popup's Ok button 1 to 4 s after it opens, found
  anywhere on screen with OpenCV template matching. A popup was open in 44% of sampled
  frames of the last game without this, and in 2% of both games with it.
- **The camera watches the front with the unit counters in view.** HOI4 hides the
  counters beyond a camera distance of 900. Measured at 1080p in mouse-wheel notches in
  from fully out: at 0 the arena fills the middle half of the screen, the counters appear
  from 9, about a third of the arena shows at 18, the map turns to terrain past 22, and 26
  is the closest. The first recordings sat almost fully zoomed out, so the counters were
  never on screen. Now the camera stays between 9 and 20 notches, zooming in and out and
  panning, mostly towards where Blue's land meets Red's. Every 20 to 60 seconds it zooms
  fully out for a few seconds, recentres by what it sees (pans alone drifted, because pan
  speed changes with zoom), and closes in on a new point, mostly on the front. Games
  recorded before 2026-09-23 noon are the zoomed-out kind.
- **The default arena is 12x8 provinces a side** (`arena-12x8-v2`, 8 states a side).
  The first two games on it (2026-09-23, one on each PC) both ended in a surrender read
  from the log, after 24.8 and 31.5 minutes at speed 4.
- **Red won all of the first six AI games on it**, where a fair map does that 1.6% of
  the time. The map is a true mirror (terrain, coast, rivers, supply, victory points and
  starting divisions all match under a half turn) and the fronts moved both ways for
  months, but in v1 Blue always declared the war and the recorder always started as
  Blue. In v2 a coin flip picks who declares, logged as `declare`, and the log names
  each human's starting country (`player`); the recorder alternates Blue and Red, the
  two PCs out of step. Each game's manifest records `started_as`, `declarer` and
  `players`, so the next games show which of the two decides it.
- The pause mark blinks, so the start check looks for it over 10 s rather than in one
  frame.
- **Picking Red** clicks Red's land on the picker and checks the selected flag; twice on
  the second PC it had clicked before the picker's map was drawn, and once here it clicked
  the drawn pointer, whose glove reads as red. The log's `player` line is the truth.
- **Games alternate between speed 4 and 5** (`--speeds`), each speed played as both
  countries in turn. Frames come at 5 a second either way, so speed 5 adds games an hour,
  not frames: more winners for the win predictor, in footage that runs faster than normal
  play and at a rate the CPU sets. Each manifest records its speed. Speed 4 proved too
  slow to watch, so the runs since 2026-09-23 12:44 use `--speeds 5`: a game takes about
  three minutes.

## What the 2026 literature changes (2026-09-23)

The plan above followed Video PreTraining (2022) for its labelling step and AlphaStar
(2019) for its reinforcement learning. Both were checked against the work of 2024–2026:
arXiv, GitHub and researchers' posts on X, each claim traced to its primary source.
No published agent plays grand strategy, 4X or RTS well from pixels: every strategy
result reads the game's state through an API or text (Compiled Agency 2609.18996,
CivBench 2609.02459, StarWM 2602.14857). So nothing here can be copied whole, and
each change below is a flag measured against what it replaces.

- **Kept: labelling video with an inverse dynamics model.** It is still the only route
  shown to recover precise mouse and keyboard input, and it now runs at scale:
  Standard Intelligence's FDM-1 (2026) labelled 11M hours of screen recordings with an
  IDM trained on 40k, and D2E's Generalist-IDM-1B (2510.05684) labels PC games'
  keyboard and mouse, including games it never saw. Latent action models (Genie, LAPA
  2410.11758, villa-X 2507.23682) have never been tested on clicks or keys, and they
  absorb change the agent did not cause (2605.20223), which in HOI4 is most of the
  screen: the clock, the AI's units, the map.
- **Changed: memory is trained on long windows.** A decision is 200 ms, so the 16-step
  windows the policy trained on were 3.2 s: nothing longer could be learned. Cutting
  gradients at even 100 steps costs measurably (Memoroids, 2402.09900). The newest
  sequence layers, Mamba-3 (2603.15569) and Gated DeltaNet-2 (2605.22791), have been
  compared only as language models; as an RL agent's memory, GRUs and LSTMs still match
  newer cells (2601.15086, POPGym Arcade 2503.01450). So the window comes first, then a
  measured comparison of the cells, with GDN-2 written from the MIT-licensed
  flash-linear-attention reference or our own (NVlabs' repository is non-commercial).
- **Changed: reinforcement learning uses every game.** On-policy PPO with a league
  assumes millions of cheap games; this arena gives tens a day. Every 2025–26 method
  that improves a policy from minutes to hours of real play is offline or off-policy on
  top of a pretrained one: AlphaStar Unplugged (2308.03526) beat its imitation agent
  90% of the time from replays alone; RECAP (π*0.6, 2511.14759) trains a critic on
  outcomes and the policy on each action's advantage; EXPO-FT (2609.18207) went from
  42% to 97% with 10 minutes of online data. Games can also start from mid-game saves
  (DAGS, 2605.14379). PPO, PACT and InfoPPO stay, as flags to compare.
- **Not now: training inside a world model of the screen.** Dreamer 4 (2509.24527)
  found Minecraft diamonds from offline data alone, but on 256–1024 TPUs, with 9.6 s of
  context; MIRA (2607.05352) needs a B200. On an 8 GB card a pixel model of HOI4 would
  run no faster than the game, and no strategy policy has been trained in one. A small
  model of the game's numbers, stepping once per game-day, is a later pilot.
- **Not now: large vision-language agents** (Lumine 2511.08892, Game-TARS 2510.23691,
  SIMA 2 2512.04797, UI-TARS-2 2509.02544). They are 7–230B parameters, too slow for
  an 8 GB card in real time, and most are closed. What they teach is kept for later:
  emitting short chunks of actions, and thinking only now and then, which a paused
  game allows.

## A scripted player and the true state (2026-09-23)

The AI games teach camera control and clearing popups, but their recorded inputs never
decide who wins: the game's AI fights the war. So learning from outcomes waited on hours
of hand-recorded play. A scripted player now fights one side through the interface
(`record-ai --player scripted`, `scripted.py`), with a strategy drawn at random each
game, so its games are labelled recordings whose inputs do decide the outcome, in any
number. Its win rate against the game's AI is the first baseline a learned agent must
beat (`win-rate`).

- **What it does**, calibrated live at 1080p on the second PC:
  - While the game is paused, a shift+click on the top bar's "Unassigned divisions"
    alert selects all 8 divisions, and the green + in the army bar makes them one army.
  - Z, then a click on the border, draws a front line along the whole border.
  - X, then a right-drag into enemy land, draws an offensive.
  - After a wait of 0 to 60 s, the arrow above the army card activates the plan.
  - Some games redraw the offensive every 40 to 120 s. Some attack not at all.
- **Finding buttons.** Buttons are found by normalised correlation, which scored 1.00
  where a button was (0.88 to 0.92 lit under the pointer) and at most 0.55 anywhere
  else. The first live game failed on the create-army +, which glows while divisions are
  selected: a picture of it taken a minute earlier scored 0.17 by squared difference.
  It is found by its green instead: 201 to 336 green pixels where it showed, none while
  it was grey.
- **The true state.** Arenas since v3 log each side's state every day: divisions in
  every state, the game's estimate of its army's strength against the enemy's,
  casualties, manpower, and rifles held against rifles needed. At the start of a live
  game each side had 4 divisions in each of its two border states, as placed. Recordings
  keep every mod line with the frame it was read at. `train-state-value` fits a win
  predictor on that state, and `advantage --state-value` values a recording's decisions
  from it. Training may read the state; the agent never does.
- **Who declares was not random.** Red declared in 36 of the 44 speed-5 AI games, in
  all 13 started as Red on the second PC, and in all 8 of the first scripted games: the
  game's random draw at startup comes out the same for the same setup. Since arena v4
  the recorder flips the coin itself and fires the result from the console
  (`event arena.1` or `arena.2`), then closes the event window that opens. The game's
  own flip remains only where no one fires either, after the first day. The first v4
  game drew Blue, and Blue declared.
- **First complete game**, 2026-09-23 on the second PC. It played Red (near offensive,
  26 s wait, redraw every 51 s) and lost to the AI in 141 s. The game logged 596 daily
  reports and 14 changes of control. The recording trains: 697 decisions, 84 windows,
  `--sources scripted`.
- **Lost all of the first 10 games**, in 138 to 164 s, on both sides. The daily reports
  showed why: the AI raises its conscription law once it can afford it, and its
  deployed manpower rose from 15k to 16-21k while the script's only fell. The script
  now does the same through the political screen (Q, the law slot, the law, OK), to a
  law drawn per game; a fifth of games keep the starting law, to measure it. In the
  first game with it, as Red, the script moved to Limited in late April 1936 and
  Extensive in late June, its deployed manpower rose from 14k to 32k by January 1937,
  and it was stronger than the AI's army by the game's own estimate (1.6 to 1) from
  September to April. It still lost, in 249 s: the AI took the map's top and bottom
  edges while the script's divisions were split between its front and its offensives.
- **What else was wrong**, each found in the daily reports and a few frames, and fixed:
  - The AI's army has a general (each side has three, skill 3); the script's had none.
    It now takes one from the army panel.
  - Redraws added offensives to the old ones and split the army. A redraw now deletes
    every order first (the Battle Plans bar's bin, right-click, OK).
  - Thin lines on the map were read as land (the blue glow on Red's coast, red arrows
    in Blue), so fronts and offensives started in the wrong places. The land masks are
    now opened by 5 px.
  - After a law change the political screen stayed open for the rest of one game. Every
    click on it is now made only after checking that it shows.
  - A plan redrawn for divisions already in place shows a green check and does not
    execute until clicked; taken for executing, it left one army holding its line for
    five years. The camera also glides on after zooming out, so orders placed from a
    screen taken too early missed. The script now clicks the check too, checks the
    arrow lights up, and waits for the camera to come to rest.
  - Single-state arrows ("near", "deep") lost the flanks or the rear in every game
    with them. A "broad" offensive draws the line across the whole front, a third of
    the way to the enemy's far edge, redrawn every 30 to 90 s; most games use it.
  - Divisions start at 31% strength, and Extensive conscription still left them at
    81% after five years, with 2,000 political power unspent. The script now climbs
    to Service by Requirement or All Adults Serve (150 power a step). Some games hold
    for up to 240 s before attacking, and games stop at 15 minutes as a draw.
- **First win, 2026-09-23**, after 16 losses and 2 draws. As Blue, the script held
  for 142 s with Extensive conscription; its deployed manpower rose to 41k while the
  AI's stayed near 14.5k, and the AI lost 29k men against the held line to Blue's
  10k. Then broad offensives, redrawn every 67 s, took Red's states one by one, and
  Red surrendered in January 1938, after 344 s.
- **4 wins in the first 6 games since the fixes** (67%, 95% interval 30-90%): 2 of 2
  as Blue, 2 of 4 as Red; 3 of 4 with broad offensives, 1 of 2 with a deep arrow. The
  winners' armies filled to 41-48k deployed manpower while the AI's stayed at 8-15k,
  and the AI lost 2 to 4 men for each of the script's. Most plans now hold 90-240 s
  before attacking, from the first win; too few games yet to say which part matters.
- **Recruitment.** Winning games left manpower unused while the AI never had more
  than 8 divisions, so plans now draw 0, 2 or 4 training slots. U opens Recruit &
  Deploy; Train on the army's template adds a deployment line, its "No location set"
  is answered on the map, where the player's own land shows green, and Add Unit adds
  slots. New divisions deploy unassigned: a shift+click on the top bar's alert selects
  them and a right-click on the army's card adds them (8/24 became 16/24 in the
  calibration game), every 20 s. Slots opened at the start took the manpower the
  divisions needed to fill up from 31%: in the first game with them the army's
  deployed manpower fell from 14.7k to 5.5k and Blue surrendered in December 1936. So
  recruiting starts 60 s after the conscription goal is reached. That is late: in the
  games since, no new division deployed before the war ended, so its effect on the win
  rate is not measured yet.
- **The best plan, a challenger, and exploration (2026-09-24).** Every game since the
  fixes that held 90 s or more before attacking had won (6 of 6), and three of four
  that attacked within 60 s had lost. So 40% of games now play the best plan found so
  far (broad offensives after a hold of 120-240 s, All Adults Serve, no recruiting),
  30% play it with one change under test (now: the front line executed alone, no
  offensive line), and 30% draw every choice at random, so the recordings stay
  varied. `win-rate` reports each apart, and by arena.
- **A quiet hold.** The first best-plan loss came before its attack: the plan was
  redrawn every 33 s while the front held, each redraw took 15-30 s, and the
  conscription steps, checked after the redraws, found few turns between them
  (Limited at 67 s, Extensive at 146 s, against 41 and 65 s in the wins). Every redraw
  also deleted the front line under the divisions. The AI broke in while the army was
  still at 17k. Now nothing is redrawn until the attack, conscription comes first, and
  the redraw period counts from the end of the last redraw. Since then the laws have
  come at 38-46, 60-69, 79-94 and 99-117 s, as political power allows.
- **A plan counts as executing only when its arrow is lit.** In the next game the
  attack started 65 s late: selecting the army had left the pointer on its card, the
  general's tooltip covered the execute arrow, neither the idle nor the ready look was
  found, and that was taken for executing. The arrow's green averages 54-65 idle or
  ready and 90-105 executing (the dots or the check before it come and go while it
  executes), so the check now asks for a lit arrow, looked at with the pointer off the
  bar.
- **The attack can swing back.** In 2 of 11 long-hold games the attack let the AI
  into the script's rear: its broad line pulled the army forward while the AI held
  part of the script's border states, and the AI's last divisions walked into the
  empty home half. One of them was lost that way (the AI took the victory points
  first). In the wins the AI held 0-14% of the script's home half when the attack
  began and none after the push; in that loss it grew to 10%, 17%, then 89%. An
  optional guard (`guard` in a plan, off in every plan so far) executes the front line
  alone at a redraw while the enemy holds at least that share.
- **Games an hour (2026-09-24).** Filmed on the second PC, every menu answered within a
  second of its click, while the recorder slept 25, 8, 40, 40 and 20 s through them; the
  map came 3.5-4.5 s after Start. Those waits are now a few seconds. Better, a game can
  launch straight into a start save (`--start-save ARENA:COUNTRY:SAVE`), made with the
  console's `savegame <name>` while paused at the start of a new game: it reaches the
  paused map 7 s after the launch returns, and recording starts 13-22 s after the
  launch, against about 145 s before. A game on another arena that comes through the
  menus saves its own start for the next one there. After a save loads, the arena logs
  no `player` line (on_startup does not fire), so the manifest takes the save's country,
  checked by the flag at the top left.
- **Games that started alike played alike.** With the same side and the same declarer,
  games' daily reports were identical to the hour until the script's first law change,
  about 100 days in: the game's random draws repeat from the same start, so there were
  only four openings. Each game now runs 0-14 s at speed 1 (about half a game hour a
  second) before the war is declared (`--opening`), so its war starts at its own hour.
- **Services beside the games.** `--arena-queue` plays each arena test request once,
  between the station's own games, to the best plan, and answers it in `results/`:
  whether it loaded and started, the outcome, planner errors, the map errors the game
  logged (counted by the worker's report, and once a run on the main arena to compare),
  and screenshots: the start, a full view mid-game, the end, and eight close-ups zoomed
  into the terrain view over both countries. Arenas that pass join the rotation in every
  other pair of games, newest versions only. `--eval-dir` lends the second PC between
  games to a live evaluation that reserves it (queue, granted, done). A claim names its
  process, and a recorder that starts offers the claims of dead ones again. A `DRAIN`
  file in the output folder ends a run between games. Popups are searched on their own
  thread at half size, confirmed at full size: two full-frame searches had taken about
  350 ms on the capture thread and cost frames. Each run writes its own results file.
- **The country picker** closes in on Blue's capital; on an arena with the capitals in
  the rear no Red land showed. It is zoomed out until the country shows.

## A learned player from the scripted games (2026-09-24)

The goal: a policy that reads only the screen and beats the game's AI in at least half
of 20 live games. It learns by imitating the scripted player's recorded games.

- **What there is to imitate.** A whole scripted win has about 35 mouse presses and 10
  key presses that matter: the army, its general, the front (Z and a click), the
  offensive (X and a right-drag), the speed, each law step (Q, the slot, the law, OK),
  each redraw (the army card, the bin, OK, then front and offensive again) and each
  activation. Everything else is the camera. Over 30 games (41,400 decisions), 2.3% of
  decisions press a mouse button.
- **Two labels were being lost.** The scripted player forms its army 0.9 to 2.5 s into
  each game, but training started its decisions after a 1.8 s lead-in that only the
  clip-reading encoder needs; 23 of 31 games never showed that click. `train-bc --lead-in
  0` keeps it. And the space bar that starts the game is outside the policy's inputs, so
  its decision, and every training window around it, was thrown away with the speed
  clicks beside it. `--drop-keys 0x20` leaves only the key out.
- **Only 35 of 934 pressing decisions press after a move in the same decision**, so
  `--look-before-click` costs under 1% of the training windows.
- **Privileged targets.** Two training-only losses shape what the memory keeps: the
  arena's true state from its daily report, 65 numbers from the player's side
  (`--state-weight`; divisions, strength, manpower, casualties, surrender progress, who
  holds each state, divisions in each state, the date), and the scripted player's next
  order and the time until it (`--order-weight`). Neither is seen when the policy plays.
- **Live play** (`play-policy`): the policy plays on the second PC from its screen, at
  5 decisions a second, through the worker in match mode. The harness launches the game,
  picks the country and fires the fair coin for who declares, as record-ai does. Space
  is not an input the policy has, so when it clicks the speed control's + while the game
  is paused (the scripted player's last setup step), or after 90 s, the harness presses
  space and sets speed 5; it does so again if the daily reports stop. Every such step is
  counted in the game's manifest, and the games are recorded as data (source "policy").
  The second PC is reserved from the scripted player's recorder (`--reservation`).

## The memory study (2026-09-24)

Which memory should the policy have, and how should it be trained?
`scripts/memory_study.py` trained six arms, five seeds each, with `train_memory` on one
cache of frozen perception features (`artifacts/bc-v2s5` reading the 44 speed-5 AI
games). Every arm got the same decisions per update and the same passes. Each was scored
on held-out imitation loss with the memory carried from the start of each game, as the
policy plays. The rule for switching was written into the script before any run.

| Arm | Held-out loss | Loss, memory cleared every 18 decisions | Time since zoom-out, R² |
|---|---|---|---|
| GRU, 16-decision windows, memory carried | **3.109 ± 0.003** | 3.108 | 0.010 |
| Mamba-3, 256, carried | 3.120 ± 0.009 | 3.257 | 0.151 |
| GRU, 256, carried | 3.123 ± 0.006 | 3.123 | 0.019 |
| No memory | 3.129 ± 0.004 | 3.129 | 0.032 |
| Gated DeltaNet-2, 256, carried | 3.134 ± 0.016 | 3.134 | 0.051 |
| GRU, 16, from empty (how `train-bc` trains) | 3.169 ± 0.008 | 3.169 | 0.005 |

The verdict, by the rule:
- **Train with the memory carried through whole games.** The 256-decision GRU beats the
  old way, where each window starts from an empty memory, by far more than twice the
  noise. The 16-decision GRU with its memory carried is better still.
- **The GRU stays.** Neither Mamba-3 nor Gated DeltaNet-2 beats the 256-decision GRU by
  twice the noise.

What the numbers say besides:
- **The new cells do hold more.** A linear read-out of Mamba-3's memory recovers how long
  since the camera zoomed out (R² 0.15, against 0.01-0.02 for the GRUs). And clearing
  its memory costs it 0.14, while it costs the GRU nothing. But on these games that
  knowledge does not help predict the next input: the AI games' camera moves at random,
  so there is little for a long memory to find.
- **Carrying matters, though the GRU uses only a few seconds.** Clearing the carried
  GRU's memory every 18 decisions (3.6 s) during evaluation changes nothing. Yet the GRU
  trained from empty windows scores worse than no memory at all (3.169 against 3.129),
  whether its memory is cleared or not. Why training from empty hurts this much is not
  understood yet.
- **Nobody predicts the winner.** Every arm's read-out matches the base rate (0.63).
- **Timings are not comparable.** This PC was shared with other work, and seed 4 ran up
  to 2x slower than seeds 1-3.

Next: train the policy with its memory carried, and repeat the GRU against Mamba-3 on the
scripted player's games, whose plans run for minutes, before closing the question.

## Open work

In order. Since 2026-09-23 the scripted player comes first: it gives a win rate to beat
and games whose inputs decide the outcome, so learning from outcomes no longer waits on
hand-recorded play.

1. **Scripted games in bulk, and their win rate against the AI**, both sides, both PCs,
   unattended (`record-ai --player scripted --mod artifacts/mods/arena-12x8-v3`, then
   `win-rate`). Keep improving the script where it is weak: it is also the first
   opponent.
2. **Learn from outcomes offline.** Fit the win predictor on the true state
   (`train-state-value`), weight the scripted games' decisions by advantage
   (`advantage --state-value`), and train the policy on them (`train-bc --advantage`).
   Judge it by win rate against the AI and against the scripted player, not by loss.
3. **Train the memory carried through whole games.** The study is done: carrying beats
   starting each window empty, and the GRU stays (see "The memory study"). Next, train
   the policy that way on the scripted games, and repeat the GRU against Mamba-3 there.
4. **Record AI-vs-AI games in bulk** with `hoi4-arena record-ai`, on both PCs at once
   with `--peer artifacts/pairing/peer.json`. The second PC's games are launched and
   closed through its worker (`launch`, `quit`) and encoded there on its own clock
   (`--codec nvenc`, "Recording where the game runs"). Both monitors must stay
   switched on (brightness can be zero): a monitor switched off disconnects on
   DisplayPort, Windows shrinks the desktop to 1024x768, and the capture breaks, which
   the recorder reports. On the second PC the Discord overlay is off: after a
   force-closed game it hung every later launch. Games are closed politely first, and a
   hung launch restarts Discord (`restart_discord`) and retries once. `report` lists the
   second PC's windows, busy processes and log ends. The games carry the scripted
   camera's inputs as labels, so they teach camera control and popup clearing, and
   serve the encoder, predicting who wins, and a first opponent. On the 12x8 arena the
   first two games took 24.8 and 31.5 minutes, so a match limit of 1800 s is too short
   there; the recorder's cap is 45 minutes. Recordings are 1080p H.264 4:4:4: NVENC
   at QP 14 since 2026-09-24, x264 at CRF 18 before and as the fallback.
   `desync` gets calibrated whenever one happens.
5. **Record 1–4 hours of human play** on the arena, with `--game-speed` set to the
   speed used. This is the only source of a player's inputs.
6. **The inverse dynamics model**, paused until a playing agent can show that labelled
   video helps. The GRU context scored 95.0% on the input kind on held-out games (44
   speed-5 AI games, 2026-09-23); the transformer run and the Generalist-IDM-1B
   comparison wait. The options are in place (`train-idm --context transformer` with
   `--sequence` 16, 32 or 64, and `train-bc --idm-min-logp` and `--idm-weight`).
7. **Off-policy fine-tuning in live games**, scored from the arena log, started from
   mid-game saves, against a small pool: the game's AI at several difficulties, the
   scripted player and a few frozen snapshots, rather than a full league. Recurrent PPO
   with the PACT critic (`--gae-lambda`, `--critic`) and InfoPPO's clock and clip
   (`--clock`, `--clip`) remain, as the baseline to compare against. Starting from a
   save works: `control launch --save <name>` (#52) went straight to the saved moment on
   the second PC, from a save the console's `savegame` wrote mid-game.
8. **A slow strategist and a fast hand**, and a small model of the game's numbers that
   steps once per game day, fitted to the daily state logs. The strategist could then
   practise in the model for years of game time an hour, and the real game stays the
   judge. Skild AI's robot footballer (2026) learned by self-play in simulation first.
9. Run a two-PC match between agents, and check reset and recovery when something goes
   wrong. (A two-PC match driven by hand, from this PC, works.)
10. Complete 20 unattended matches and 50 side-swapped evaluation pairs.
11. Test the same interface in an unmodified private multiplayer lobby.
12. Later pilots: short action chunks, a planner that thinks while the game is paused,
     and a small world model of the game's numbers.
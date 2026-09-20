# v0.1 build status

Recorded 2026-09-19. Design: [architecture-v0.1.md](architecture-v0.1.md).
**Live status (2026-09-19, v1.19.3):** the vanilla-region arena mod loads and runs; the custom maps crash on load; perception, executor, agents and LAN have not run live. Offline: 346 tests pass, 1 skipped (optional OCR);
ruff and mypy clean. Simulator results are never evidence about real HOI4.

| Slice | Where | Verified | Not verified |
|---|---|---|---|
| Contracts, policy, learner | `arena/contracts.py`, `actions.py`, `policy.py`, `learner.py`, `returns.py`, `trajectory.py`, `layout.py`, `scripted.py` | Unit tests: stacks/count/confidence, speed and pause actions, zero-elapsed decisions, intent conditioning, intent-head distillation | — |
| Simulator and training | `arena/sim/`, `arena/train.py` | 3 independent 25-minute PPO runs all learn (validation panel score 0.44→0.78, 0.46→0.71, 0.46→0.75); run A held-out 0.45→0.81 over 240 side-swapped games; 1.9k–4.6k games/hour on the tiny map | Combat coefficients, speed table and noise rates are placeholders; trained only on the 8-province map with 2 divisions; draw rate vs `hold` is 45% |
| Cloud tiers and agents | `brain/jev.py`, `brain/deepseek.py`, `arena/tiers.py`, `arena/agents.py` | Live round trip: DeepSeek proposes plans (3.7 s), Jev judges (0.55 s); Jev bench 20/22 at 338 ms; DeepSeek bench 16/18 at 1.24 s; tick loop never blocks (tests) | Behaviour in the live game; rate limits; prompt tuning |
| Vision runtime | `arena/vision/`, `config/arena/vision.toml` | Counter reader on two real screenshots: all fully visible counters found, all counts right against hand labels, no false positives, ~77 ms at 1680x1050 | All calibration points (empty on purpose), bar values vs a save, digits 0/6/8/9, selected-counter look, army panel, recorder, hotkeys in the live game |
| LAN coordinator | `arena/lan.py` | 16 loopback tests; multi-process CLI smoke | Two physical PCs; firewall rule for port 47814; pause/speed convention in real multiplayer |
| Vanilla-region mod | `arena/mod/region.py`, `generate.py` → `artifacts/arena_mod/` | Static validation, deterministic hash, 1020 script names checked against the install's documentation | Loading in game; reset effects; armies must be formed by the executor after reset; river crossings look over-detected |
| Custom maps | `arena/mod/mapdesign.py`, `custommap.py`, `bmpio.py`, `preview.py` → `artifacts/arena_custom/<design>/` | Five designs build byte-identically and pass every static validator (BMP headers, palettes, X-crossings, box rule, ids, CRLF, state/region membership, rivers, supply graph, symmetry, routes) | Whether the engine loads a two-country ~400-province world at all; see each design's `FIRST_LAUNCH.md` |

## Commands

`./.venv/Scripts/python.exe -m hoi4_agent.arena.cli --help` lists: `sim-train`, `sim-eval`, `sim-play`,
`agent-info`, `tiers-smoke`, `vision-read`, `vision-observe`, `calibrate-arena`, `vision-audit`, `lan-host`,
`lan-join`, `lan-ping`, `mod-build`, `mod-validate`, `custom-map-build`, `custom-map-validate`,
`custom-map-preview`, `custom-map-scan`, plus the older diagnostic commands.

## Next, in order

1. First live launch of `three_lanes` with `-debug`; iterate on `error.log` until it loads.
2. `calibrate-arena` on the loaded map; verify hotkeys; capture digits 0/6/8/9 and a selected counter.
3. Screenshot + save pairs: audit bar values and settle strength vs supply for the lower bar.
4. Executor: form one army after reset; confirm orders; run the Random and Cloud agents against the AI panel.
5. Two-PC LAN match with the coordinator.
6. Training follow-ups: mask for pause/speed, batched PPO forwards, train on the full map, rasterize the
   custom map's layout into the simulator so both share one graph, Jev distillation, human recorder.

Parked, not deleted: `native/`, `arena/protocol.py`, `arena/session.py`, `arena/probe.py`.

## Live evidence, 2026-09-19

- Launch helper: `arena/launch.py` (isolated profile through `gameDataPath`, settings restored byte for byte).
  `-debug -start_tag=BLU -start_speed=1` starts a single-player game as BLU with no clicks (~45 s).
- Vanilla-region mod (`artifacts/arena_mod/default`, hash `485c0190…`): 0 MAP_ERROR, 28 error.log lines, none from
  the mod. `game.log` carries `ARENA_STARTUP`, `ARENA_SELECT`, `ARENA_RESET`, daily `ARENA_TICK` (with division
  counts) and `ARENA_VP` (holder per VP province); placeholders expand. Divisions spawn for both sides; VPs hold
  at 21:21 over five days. Fixed on the way: states spanning strategic regions, stale state ids in
  `map/buildings.txt`, `on_startup` being global scope, a trailing newline in buildings.txt, missing name lists,
  front VPs left ungarrisoned next to enemy starts.
- Counter reader on the first live arena frame: enemy counter read (count 1, org 0.88, strength 1.0, conf 0.99);
  the overlapped own counter correctly flagged occluded.
- Custom maps: `provinces.bmp`, definitions, regions and supply load with no MAP_ERROR ("Loaded 397 provinces",
  "Calculated 1 land masses"), then an access violation ~1 s later on a worker thread, same address every time.
  Ruled out by live bisection: all 17 replace_path entries, rivers, supply/buildings/unitstacks files, colour maps,
  dynamic tags, launch flags. A horizontal wrap-seam X-crossing bug was found and fixed. Next: compare against a
  known-good total-conversion skeleton (Workshop "Blank Map Template").
- To do before agents can play: camera zoom/centre on the arena after reset, night-time brightness, form one
  army after reset, `calibrate-arena`. Game debug mode opens error.log in Notepad on each launch.

## First live matches, 2026-09-19 (bring-up quality, `source = hoi4_vision`, n = 1 each: not strength evidence)

- Tooling: `scripts/live.py` (window-only capture, real input, foreground check), `scripts/calibrate_live.py`
  (hover a grid, read the debug tooltip's province id with DeepSeek), `scripts/fit_camera.py` (robust homography:
  42/61 samples inside their province, the rest OCR misreads), `scripts/live_setup.py` (relaunch, pause, fixed
  wheel sequence, verify the four starting counters map to their start provinces: passed), `scripts/live_match.py`.
- Orders verified live: left-click counter, right-click calibrated province centre -> the game draws the movement
  arrow; space pauses; numpad + raises speed. A counter hangs ~38 px below its province's unit position.
- Random agent as BLU vs built-in AI, inf2_line, speed 4: lost 18:24 on the day-90 VP timeout (90 days in ~40 s).
- Cloud agent (Jev + DeepSeek intents, scripted executor), speed 2: lost 18:24, 451 decisions, 11 orders sent.
- Found: vanilla news-event popups (e.g. "The Spanish Civil War") cover the map and blind perception for ~30%
  of frames; orders are unverified; stack-level control only; the second own counter can hide under the bottom UI;
  speed 4 runs ~2.4 game days per second, faster than the 1 Hz loop can use.

## Live batch, 2026-09-19 (quieted mod `24838855…`, inf2_line, agent = BLU vs built-in AI, speed 3, n = 3 each)

Results in `artifacts/live_matches/`. Bring-up quality: orders unverified, stack-level control, VP/day/outcome from
the log. Too few games for any strength claim.

| Agent | W-L-D | Notes |
|---|---|---|
| random | 0-3-0 | 13:29, 3:39 (capital lost on day 51), 18:24; 40-45 orders per match |
| scripted:advance | 0-0-3 | 21:21 three times; 84-101 orders sent but no VP ever changed hands: check whether the orders take effect or the attacks simply fail |
| cloud-nobrief (Jev intents) | 0-0-2, 1 unfinished | Jev chose "hold" every time, so 0 orders; one match never ended (12 min cap): the blind space-bar toggle probably paused instead of unpausing |

- Quieting works: blind frames fell from ~30% to 0 in 8 of 9 matches (7 frames in one).
- A match takes ~4 minutes at speed 3 plus ~1 minute to relaunch; `scripts/live_setup.py` now fits a per-session
  camera correction from the four starting counters (rms ~4 px), so calibration survives zoom differences.
- Next: read pause state from the screen instead of toggling blindly; verify orders (arrow check); find out why
  `advance` never takes a VP; give Jev a less passive prompt or a time-pressure signal (a draw is not a win);
  calibrate the reset decision to drop the relaunch; then larger batches and the policy agent.

## Live fixes, 2026-09-19 (later)

Five faults, all found by playing matches rather than by reading code:

| Fault | Fix | Measured |
|---|---|---|
| The space bar was toggled blind, so a stray pause stalled a match | `vision/topbar.py` reads paused/running from the glyph's green-minus-red and counts lit speed segments | one unfinished match before, none since |
| Every tick re-issued the same order, restarting the battle it was fighting | standing orders remembered and fed back as the unit's order target | re-issues 100/match -> 0 |
| Orders were never checked | confirmed by the movement arrow appearing on the unit->target line | 59/59 confirmed |
| A counter whose unit is selected, or whose corner a combat badge covers, was dropped entirely | the cream ring replaces the border rather than surrounding it, so it anchors runs itself; a badged corner re-anchors on the run's right end | frames with no own units 33% -> 1.5% |
| Camera setup relied on a fixed wheel sequence from an unknown start; one run played eight minutes over the British Isles, another ended over the Netherlands | zoom until the arena is the right size on screen, hunt the counters, and ask DeepSeek for a coarse fix when counters merge into a stack icon at wide zoom | camera verified on every run since |

Also: `SetForegroundWindow` fails silently from a background script until the process has input (an ALT
tap fixes it), and Jev held a level race to a draw until the state carried the clock.

Still open: control is per stack; VP/day/outcome come from the log; the reset decision is not calibrated,
so each match costs a relaunch; `in_combat` is still unread (measured, no reliable separation).

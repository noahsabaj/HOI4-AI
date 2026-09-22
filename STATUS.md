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

Automated checks: 125 Python tests and 16 Rust tests (2 need a live desktop and are
skipped in CI), plus Ruff and Clippy. CI runs all of them on Windows.

**Not yet shown:** a match between two agents, a full 1800-second match, and a match
played across two PCs.

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
- Useful game constants: `BASE_SURRENDER_LEVEL = 1.0` (`NDiplomacy`) is the surrender
  threshold. `BASE_SURRENDER_LIMIT = 0.8` is an occupation fraction, not the threshold.
- Game speed in wall-clock seconds per in-game hour: `{2.0, 0.5, 0.2, 0.1, 0.0}` for
  speeds 1 to 5. Speed 1 is 48 s per in-game day. Speed 5 does not wait at all.

## Match-end screens

HOI4 has no "game over" screen. A one-against-one surrender goes:
`surrendered_country_popup` (520×320, centred) → peace conference (full screen) →
"Calculating Effects..." → `peace_summary_popup_window` → back to the map.

- The surrender popup uses the same frame art as the exile popup, so its template must
  be cut from the title text.
- Winner and loser see the same peace conference window. Only the banner art at the top
  differs, so `win` and `loss` templates must be cut from the banner.
- The popup is small. If it doesn't cover the `healthy` rectangle, the match loop keeps
  running and a real surrender ends as a timeout draw.

## Screen calibration

Calibrated at 3840×2160 in `artifacts/calibration-live/rules.json`: `healthy`, `paused`
and `clock_rect`. **Still needed before a match can run:** `ready`, `speed`, `win`,
`loss`, `disconnect`, `desync`, and a `minimap_rect` for the territory reward.

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

1. **Size the playable arena.** The small map produces a surrender quickly with one
   side unarmed. Next, test it with both sides armed and see whether an 1800-second
   match at speed 4 reaches a result. If not, try a slightly larger block or scoring on
   territory.
2. **Calibrate the remaining screens** on the small map: capture the surrender popup,
   the peace conference win and loss banners, `ready`, `speed`, `disconnect`, `desync`,
   and the minimap rectangle. The harness can now produce a surrender whenever needed.
3. **Record 2–4 hours of human play** on the chosen arena, with `--game-speed` set to
   the speed actually used. Recordings need the cursor position, which the worker now
   sends.
4. **Distil the compact encoder** from those recordings. It is the only measured way to
   fit two actors in one tick on one GPU; the other is one GPU per side.
5. Train the behaviour-cloning baseline, then recurrent PPO self-play with a league.
6. Run a two-PC match, and check reset and recovery when something goes wrong.
7. Complete 20 unattended matches and 50 side-swapped evaluation pairs.
8. Test the same interface in an unmodified private multiplayer lobby.

# Verification status — 2026-09-20

The code implements an initial visual learning pipeline. It does not yet deliver reliable self-play or a winning multiplayer agent.

| Check | Evidence | Result |
|---|---|---|
| Local hardware | HP Victus 15L, i7-14700F, 32 GB RAM, RTX 4060 Ti 8 GB | Inspected locally; peer hardware is user-reported identical |
| Released 303M encoder, game menu | `artifacts/benchmark/inference.json` | Encoder p95 148.73 ms, minimum 1553 MiB free |
| Released encoder, loaded vanilla game | `artifacts/benchmark-loaded/inference.json` | Encoder p95 187.72 ms, only 710 MiB free: **fails 1 GB headroom**; game was paused |
| Offline large encoder training capacity | `artifacts/benchmark-loaded/training.json` | Two BF16 Adam updates completed on a synthetic objective, not behavioral training |
| Compact full policy, saved real screenshot | `artifacts/benchmark-policy-tiny.json` | Untrained 5.49M encoder; full-policy p95 51.46 ms; two tiny BC/auxiliary capacity updates completed with game closed |
| Native 3840×2160 capture | `artifacts/worker-menu-v3-new`, `worker-menu-v3-select` | Corrected desktop-DC capture follows two actual menu transitions; prior window-DC capture returned stale/scaled pixels |
| Held input watchdog | `artifacts/worker-smoke/result.json` | Held Shift released after timeout |
| Live worker mouse input | `artifacts/worker-smoke/click-before.png`, `click-after.png` | Menu advanced using a 250 ms button hold |
| TLS peer transport | `artifacts/pairing-kat/integration/report.json` | Actual KATHPINVICTUS authenticated; foreground 3840×2160 capture, menu click, Escape and held-Shift watchdog verified |
| LAN screenshot timing | Same report, 20 menu captures | Status-request p95 1.51 ms; full screenshot round trip p95 **411.15 ms**, exceeds the 200 ms decision budget before inference |
| Arena match start | `artifacts/mods/infantry-arena-v10`, five minidumps under `Documents/Paradox Interactive/Hearts of Iron IV/crashes` | Match starts and runs: war declared, clock advanced 12:00 1 Jan to 20:00 2 Jan 1936, no crash. Combat, supply over time and victory detection **not** yet exercised |

Earlier encoder timing reports used the old window-DC capture. Treat them as capacity measurements, not a validated live visual benchmark. Re-run with the corrected capture backend, a changing game clock and full capture-to-action timing before accepting any latency result. The worker bundle has been rebuilt with the capture fix.

Latest automated checks: **61 Python tests, 7 Rust tests, Ruff lint/format, Cargo format and Clippy pass**. Locked dependency installation and packaged CLI help also pass. Each fix below was mutation-tested: reverting it fails at least one test.

Defects found by an audit of the match loop on 2026-09-20 and fixed, each with a regression test:

- The per-step liveness ladder skipped the speed-two, clock-stall and unrecognized-screen checks for any frame following a terminal template match, because its guard tested `rules.last is None`. Two terminal templates alternating never converge and never reset `last`, so a paused or stalled game could be stepped indefinitely with `valid=True`. Replaced by two explicit budgets: `UNKNOWN_FRAMES` for a screen matching nothing, and the longer `TERMINAL_GRACE_FRAMES` once some terminal template is in flight. A single budget of one debounce depth is wrong in both directions — the first attempt at this fix used one, and it rejected any real victory whose panel took more than one frame to render. Both bounds are now pinned from above and below by separate tests.
- The `deterministic` flag only switched the categorical heads to argmax. `ActionHead` conditions its entire start state on a latent that an xm-objective actor redraws every step, so a "deterministic" xm evaluation stayed stochastic. `act_noise` now pins the latent to the prior mean as well.
- `train_ppo` accepted greedy evaluation rollouts as on-policy data. Their stored `old_logp` is the likelihood of an argmax, so the PPO ratio was meaningless. `ppo_exclusion` now rejects them.
- `record`'s new worker-log write sat unguarded ahead of `recorder.close`, so a failed diagnostic write would have skipped manifest finalization entirely; and its failure gate keyed on `reason`, missing the nonzero-ffmpeg-exit case that `Recorder.close` reports only through `complete`.
- `Desktop.close` swallowing a failed release destroyed the signal `collect_pair` uses to invalidate a run. Both transports now record `close_error` instead, which `collect_pair` reads — raising from `__exit__` would have masked in-flight exceptions.
- Seeding every match from the same constant replayed one RNG stream across a whole league, so collected matches were no longer independent. The per-run seed is now salted with `pair_id`: reproducible, not identical.
- A hand-edited float `clock_rect` passed both validations and then raised `TypeError`, which is not in `FAULTS`, deep inside the match loop. Rejected at load.
- `collect_pair` returned normally on an invalid match, so a permanently miscalibrated rules file invalidated every episode of an unattended batch while every invocation exited zero. It now writes its evidence and then fails, matching `record`.
- `require_match_rules` demanded a `clock_rect` that no command could write, so collection could not be started without hand-editing `rules.json`. Added the `clock` subcommand; the rectangle is bounds-checked at calibration and again at load.
- `ArenaPair.reset` and `.step` disarmed both sides without joining the second future. Because `run_setup` re-arms at every recipe boundary, a side still running its recipe undid the disarm and kept injecting input. Both now join every future before cleanup.
- The step handler caught only `DesktopError`/`TimeoutError`/`OSError`, so the `ValueError` and `KeyError` that `ScreenRules` actually raises escaped through the thread pool and aborted the coordinator instead of invalidating the episode.
- `train_ppo` was unseeded while writing immutable provenance sidecars. Collection and PPO now seed torch, CUDA and NumPy and record the seed.
- `ActionHead`'s `deterministic` path was unreachable, so evaluation would have sampled and inflated variance the paired bound does not model. Wired through `Actor` and the pair config.
- The package had no logging and discarded the worker's stderr, its only diagnostic channel, during 1800-second unattended matches. Worker stderr is now captured, surfaced in error messages and written beside each run's manifest.
- `record` swallowed every failure and exited zero, handing back a session that `prepare_session` will always reject. Both long-running commands now write their evidence and then exit non-zero.

These were latent defects in code paths that have never run against a live match; fixing them does not constitute live verification.

Arena match start, 2026-09-20. **The arena now starts and runs.** A match was started on
`infantry-arena-v10` and left running: the war declaration fired, the clock advanced from
12:00 1 January to 20:00 2 January 1936, province tooltips read `Plains`/`Owner: Blue`/
`Victory Point(s): 20`, and no crash dump was written. The route there was four distinct
faults, each read out of a minidump rather than guessed:

| # | Fault | Evidence | Cause | Fix |
|---|---|---|---|---|
| 1 | `areas.cpp`, walking a `CControllerArea`'s province list | `GetProvince` returns null below id 1; `rdx` held **0** on the null path | a province association the map never set | write the placements and anchors the stock database supplies for every province |
| 2 | `ingameidler.cpp`, `GenerateNonHistoricalAttributes` | log: `character_manager.cpp:261 Failed to generate a name ... for country Blue`, five times | BLU and RED had no `common/names` entry and no country leader, so every generated character was nameless | ship a name list and a `country_leader` for both tags |
| 3 | same frame, null `this` | `rax` held **0x226 = 550**, `rbx` held the ASCII string `state` | the stock `tutorial/tutorial.txt` hard-codes state 550 and provinces 5010/5091/12766, and the hint loader resolves them at every match start | override the file; `replace_path = "tutorial"` does **not** unload that folder |
| 4 | same function, later | `mov rcx,[rax+rcx*8-8]` with `rax = 0` and `rcx = 0` | the loader finishes by marking the **last** entry of the hint list, so an empty tutorial indexes element -1 | ship exactly one block that names no state and no province |

Two of the changes made before any of this was measured were wrong, and are reverted:

- The `-1;-1;;-1;-1;-1;-1;-1;-1` row in `adjacencies.csv` is the engine's end-of-file
  marker, not a stray sentinel. The stock file carries one (line 253, before a trailing
  comment, which is why a `tail` missed it) and the documentation requires it even when
  the file holds no rows.
- Terrain palette 19 is `plains_17`, which sets `perm_snow`, and 13 is `forest_13`, which
  is `type = urban` with `spawn_city`. The original 0 and 1 were right: the stock
  terrain.bmp paints index 0 over 9.8% of the world as plains and index 1 over 5.7% as
  forest, and neither carries a side effect.

Two others were real but were never crash causes, and are kept only because they match the
stock data: since 1.11 the bitmap decides coastal status and `definition.csv` only has to
agree with it, and a single weather period spanning the year is what the documentation's
own example shows.

The rest of the map was audited against the stock database and the map-modding
documentation, which found and fixed: `trees.bmp` at 75/256 of the province bitmap rather
than a quarter; a sea strategic region with no `naval_terrain`, which is where sea
provinces take their provincial terrain from; a heightmap whose every coast was a 50-byte
cliff, steeper than any step on the stock map, now a ramp with a maximum neighbour step of
1; `cities.bmp` filled with index 0, which the stock `cities.txt` maps to a sparse city
group rather than to no cities; ship-in-port anchors missing from provinces that now have
naval bases; all of a country's surrender weight on one province; weather objects anchored
over the wrong region; an X-crossing breaker that never looked at the horizontal wrap seam;
and missing adjective, ideology and victory-point localisation, which is why Red's capital
was labelled "Kargopol" by the stock strings.

**The vanilla Earth over the arena's ocean** was five map-shaped textures the mod never
shipped. Each is a painting of the stock world at the stock map's aspect, so the engine
stretched it over the new one: `colormap_rgb_cityemissivemask_a.dds` (world colour in RGB,
city-light opacity in alpha), `colormap_water_0/1/2.dds` (ocean tint, at provinces/2, /4
and /8), `fow_rgb_waterspec_a.dds` (fog-of-war and water specular), and both minimap
widgets. All are now generated from the arena's own land mask as uncompressed 8.8.8.8 ARGB
with no mip chain, using colours sampled from the stock files.

`hoi4-arena audit-map` checks every one of these without the game, and `generate-map`
audits what it wrote and exits non-zero. It exists because the engine does not report bad
map data: it dereferences it.

**Still unverified:** a full 1800-second match, combat, supply behaviour over time, victory
detection, and anything on two machines. One match ran for a bit over one in-game day.

Capture and preprocessing, measured locally on 2026-09-20:

| Change | Before | After |
|---|---|---|
| Per-tick resize in the decision loop | 60.2 ms p50 (PIL, CPU) | 0 ms; the worker sends views |
| Capture payload | 33.2 MB raw / 3.46 MB lz4 | 1.15 MB raw / 0.77 MB lz4 |
| Worker CPU to produce five views | n/a | 18.0 ms p50 |
| Offline `views` on a 4K frame | 51.3 ms (PIL) | 16.0 ms (GPU, float32) |

The 411 ms screenshot round trip should fall substantially, but it has **not** been re-measured on two PCs.

The step loop no longer serializes the interval against capture and inference. Each tick dispatches its eight slots on a separate thread and blocks the following tick on that dispatch completing, so a tick costs one interval rather than interval plus capture plus inference. Simulated locally against a fake desktop:

| Policy + capture per tick | Serial loop | Pipelined loop |
|---|---|---|
| 51 ms + 15 ms | 266 ms | 203 ms (4.93 Hz) |
| 102 ms + 15 ms | 317 ms | 203 ms (4.93 Hz) |
| 150 ms + 20 ms | 370 ms | 203 ms (4.93 Hz) |

Five Hz is therefore reachable rather than excluded by construction, but it is **not** demonstrated: these are simulated timings, not a live match, and the measured large-encoder p95 of 187.72 ms for a single actor still leaves no room for two on one GPU. The gate needs a real run. The resampler changed from PIL bilinear to an exact area average so the Rust worker can reproduce it bit for bit; this is a deliberate one-time break of any previously prepared dataset, of which there are none.

Open acceptance work:

- Run a full 1800-second match on `infantry-arena-v10` and verify armies, combat, supply, fog, multiple routes and side symmetry in gameplay. Match start itself is now verified.
- Pairing works; complete two-PC lobby/reset calibration and recovery checks. Menu input and watchdog release are verified remotely; match behavior remains untested.
- Verify physical capture/input alignment, live dragging, keyboard effect, focus loss and F12 under load. The worker now releases held input even when stdout fails; live regression remains needed.
- Re-measure the two-PC screenshot round trip with worker-side downscaling enabled.
- Measure end-to-end scheduling against a live game with two real actors. The loop now overlaps dispatch with capture and inference, but rendering, encoding and two actors sharing one GPU are untested at cadence.
- Record 2–4 hours of expert demonstrations. Distill the compact encoder, train the BC baseline and compare held-out gameplay for the auxiliary/XM variants.
- Run actual recurrent PPO self-play. Add an unattended league driver and model-selection schedule; current collector and trainer are separately invoked.
- Complete 20 auditable unattended matches and 50 side-swapped evaluation pairs.
- Test the same screen/input interface in a private unmodified multiplayer lobby.

No gameplay improvement, reliable speed-two operation, trained combat checkpoint, unattended match count or ordinary multiplayer win is claimed.

# Verification status — 2026-09-20

The code implements an initial visual learning pipeline. It does not yet deliver reliable self-play or a winning multiplayer agent.

| Check | Evidence | Result |
|---|---|---|
| Local hardware | consumer desktop, desktop CPU, 32 GB RAM, RTX 4060 Ti 8 GB | Inspected locally; peer hardware is user-reported identical |
| Released 303M encoder, game menu | `artifacts/benchmark/inference.json` | Encoder p95 148.73 ms, minimum 1553 MiB free |
| Released encoder, loaded vanilla game | `artifacts/benchmark-loaded/inference.json` | Encoder p95 187.72 ms, only 710 MiB free: **fails 1 GB headroom**; game was paused |
| Offline large encoder training capacity | `artifacts/benchmark-loaded/training.json` | Two BF16 Adam updates completed on a synthetic objective, not behavioral training |
| Compact full policy, saved real screenshot | `artifacts/benchmark-policy-tiny.json` | Untrained 5.49M encoder; full-policy p95 51.46 ms; two tiny BC/auxiliary capacity updates completed with game closed |
| Native 3840×2160 capture | `artifacts/worker-menu-v3-new`, `worker-menu-v3-select` | Corrected desktop-DC capture follows two actual menu transitions; prior window-DC capture returned stale/scaled pixels |
| Held input watchdog | `artifacts/worker-smoke/result.json` | Held Shift released after timeout |
| Live worker mouse input | `artifacts/worker-smoke/click-before.png`, `click-after.png` | Menu advanced using a 250 ms button hold |
| TLS peer transport | `artifacts/pairing/integration/report.json` | Actual the peer machine authenticated; foreground 3840×2160 capture, menu click, Escape and held-Shift watchdog verified |
| LAN screenshot timing | `artifacts/pairing-roundtrip-downscaled.json`, 25 captures per mode | Live path (five views plus template crops) p50 83.9 ms, **p95 99.5 ms**, 0.87 MB; the full frame it replaces is p95 433.8 ms, 33.2 MB |
| Arena match start | `artifacts/mods/infantry-arena-v10`, five minidumps under `Documents/Paradox Interactive/Hearts of Iron IV/crashes` | Match starts and runs: war declared, clock advanced 12:00 1 Jan to 20:00 2 Jan 1936, no crash |
| Combat resolution | `infantry-arena-v12`, live match, 5 March to mid-May 1936 at game speed four | Battle joined and resolved: 2 Blue divisions attacked 1 Red defender, tooltip gave a running estimate, attacker repulsed, no province changed hands |
| Victory detection | Arithmetic below, from `00_defines.lua` and the live map | **Not reachable at the old province size.** Needed the enemy capital plus two outposts, about five hops and 130 in-game days away, through a front symmetric infantry cannot break |
| Rescaled map starts a match | `infantry-arena-v13`, `logs/game.log`, `logs/error.log`, live screen | `Loaded 1537 provinces`, `[[ Launching SINGLEPLAYER-game ]] Start-date: 1936.1.1.12`, war declared, no crash dump, and it ran to 11 April 1936 at game speed four. Zero `MAP_ERROR`, `TOO LARGE BOX`, `has no continent`, `no pixels` or naval-base lines. State names resolve (`A Plains province in West 13`) |
| Red's AI taking ground | Two runs, `infantry-arena-v13` and `-v14`, debug tooltip at ~100 in-game days each | **None, either time.** `Surrender level: 0.00` with controlled and owned victory points equal. Blue sat still by design. v14 added a field marshal, three corps commanders and a `front_control` strategy with `execute_order = yes` to both sides; all of it loaded with no error and the outcome did not change |

Earlier encoder timing reports used the old window-DC capture. Treat them as capacity measurements, not a validated live visual benchmark. Re-run with the corrected capture backend, a changing game clock and full capture-to-action timing before accepting any latency result. The worker bundle has been rebuilt with the capture fix.

Latest automated checks: **65 Python tests, 7 Rust tests, Ruff lint/format, Cargo format and Clippy pass**. Locked dependency installation and packaged CLI help also pass. Each fix below was mutation-tested: reverting it fails at least one test.

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

One more, found on 2026-09-21 and fixed with a regression test: **a pair had two match
clocks.** Each side started its own `seconds` the moment its own `reset` finished, and two
setups never finish together — the lobby recipe waits on templates and one side is a LAN
round trip away. The side that finished first reached its timeout first and reported a
draw, while the other was still short of its own and had only `PAIR_CONFIRM_SECONDS` to
agree, so any reset skew past three seconds came back `unconfirmed_pair_result` and
invalidated the episode. That is every timeout draw, which is nearly every match while a
decisive result is out of reach. Both sides now take the later start and expire within a
tick of each other.

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

**Soak, 2026-09-20.** A match on `infantry-arena-v10` ran unattended for 1800 seconds of
wall clock **at game speed one**, from 12:00 on 1 January to 10:00 on 7 February 1936: about
37 in-game days. Speed one was not the intention. Speed is changed with `+` and `-`
(`VK_OEM_PLUS` and `VK_OEM_MINUS`, which the worker allows in setup mode and refuses during
a match, exactly as intended); the number keys do nothing, so the attempt to select speed
two silently left it at one. Speed four has since been driven from the `+` control beside the
clock and measured at 97 in-game hours per 10 wall seconds, about 727 days per 1800 s against
speed one's 37; a long soak at that rate has not been run. No crash dump, no line in
`error.log` matching `MAP_ERROR`, `naval base`,
`has no continent` or `no pixels`, and a steady 2.7 GB working set. That is past the window
in which a missing naval-base placement is documented to crash an AI evaluation loop.

Two further map defects, found by looking at the running game rather than at a file:

- **Blue rendered green and Red rendered pale cyan** while their flags were correct. The
  map colour comes from `common/countries/colors.txt`, where the colour space is named:
  `color = rgb { ... }`. A bare `color = { ... }` in a country file is not the map colour,
  so the arena's tags fell back to engine-assigned defaults. The flags were right only
  because they are literal pixels. Both now read one constant.
- **The map was far too small for the camera.** At 2048x1536 the zoom-out limit, which is
  fixed in world units rather than fitted to the map, showed sky above and below the map
  and more than one map width across; since HOI4 wraps horizontally that drew the same two
  countries two and a half times over, only one copy labelled. The arena is now 5632x2048,
  the stock map's exact dimensions, with the same 192 provinces and the same 48 land
  provinces a side. Largest province box 379x213, inside the 704x256 that triggers
  TOO LARGE BOX. Confirmed on screen: at the zoom-out limit the view holds exactly one
  world, Red wrapping to both sides of Blue, with no sky below the map and no unnamed
  repeats. Red appearing twice is the wrap itself and is correct: a world with two
  countries in it looks like that from either one.

**Combat resolution, 2026-09-20: verified.** On `infantry-arena-v12`, playing BLU, two
divisions were ordered across the border into a RED-held province. The battle joined, the
combat tooltip read `Attacker: 2 divisions (Blue) / Defender: 1 divisions (Red) / We are
currently losing! / We estimate that the battle will last for another 74 days`, it ran from
5 March to mid-May 1936, and it concluded with the attacker repulsed and no province
changing hands. Attack order, battle, and resolution to an outcome all work.

The reason the earlier soak saw nothing is **not** that the AI has no strategy plans. That
was wrong twice over: `common/ai_strategy_plans` holds only national-focus, research and
idea picks — it has no key that creates a front, an offensive or a garrison, and 279 of the
game's 364 tags have neither a plan nor a country `ai_strategy` file and still fight. Fronts
are engine-generated, and the arena already has one: the debug tooltip reads
`Front [id=2;idx=0] (RED vs BLU) section IDs: [id=1;idx=0;provs=8]`. The real reason is
**game speed against province scale**:

| | |
|---|---|
| `GAME_SPEED_SECONDS` | `{2.0, 0.5, 0.2, 0.1, 0.0}` wall-seconds per in-game hour |
| Speed 1, measured | 48 s per in-game day — 37.5 days per 1800 s, matching the soak to 1% |
| Speed 4, measured | 97 in-game hours per 10 s wall, about 727 days per 1800 s |
| Arena province | 352x170 px = 59,840 px² |
| Stock land province, mean | 399 px² over 10,154 provinces — the arena's cells are 150x the area, 14.6x the linear size |
| One border crossing | 359 px = 2,556 km at the infantry archetype's 4 km/h = **26.6 in-game days** |

The 1800-second soak at speed one covered about 37 days: barely more than a single province
crossing. Nothing dispersed and nothing failed — the armies had not finished walking.

**Victory detection is not reachable in the arena as generated**, for three compounding
reasons, each measured rather than assumed:

- `BASE_SURRENDER_LIMIT = 0.8` and surrender worth is victory points alone
  (`VICTORY_POINT_WORTH_FACTOR = 10`, and the in-game tooltip in
  `waroverview_l_english.yml:33` states the quantity as the fraction of victory points
  controlled). The arena's 35 VP weight means an attacker needs 28: the 20-point capital
  **plus at least two of the three 5-point outposts**. All three outposts without the
  capital is 15 and capitulates nobody.
- Each capital sits 1,583 px behind its front by construction, about five province hops, so
  roughly 130 in-game days of marching before any fighting.
- Symmetric unsupported infantry does not break through. Six `infantry_equipment_1`
  battalions give 36 soft attack against 132 defence and 18 breakthrough; the live battle
  above is the measurement, not a model — two attacking divisions lost to one defender.

`set_stability = 1` and `set_war_support = 1` do **not** make capitulation harder, which had
been a worry: no stock modifier ties stability to `surrender_limit` at all, and the only
`surrender_limit` entry in `00_static_modifiers.txt` is `-0.3` inside
`war_support_bad_modifier`. Max war support pins the limit at 0.8 rather than raising it.

**There is no bare "game over" screen to template.** What the engine actually shows, and
what a terminal rule would have to anchor on: `surrendered_country_popup` (520x320, centred,
`gfx/interface/capitulation_bg.dds`); `peaceconference_full_window`, which is
`width=100% height=100%` and covers the HUD completely — **the winner and the loser get the
same window**, distinguished only by the top-art sprite
(`GFX_top_art_winning_conference` vs `GFX_top_art_losing_conference`), so win and loss
templates must anchor on that banner and not on the screen as a whole; and the end-game
`playthrough_overview_window` with its CONTINUE/QUIT buttons. `annex_everything` is a peace
conference cost discount, not an auto-annex.

**The match-side vision rules have never been calibrated.** `ScreenRules` ends a match only
on a `win`, `loss`, `disconnect` or `desync` template, and the only two `rules.json` files on
disk hold the four lobby templates and no `clock_rect`, so `require_match_rules()` raises.
An earlier version of this entry said a match ending on an unrecognised screen is silently
recorded as a draw. That was wrong, and reading the code settles it: a screen matching
nothing raises `unrecognized_match_screen`, a terminal candidate that never converges
raises `terminal_screen_never_confirmed`, and a timeout without a healthy HUD raises
`uncertain_timeout`. The gap is narrower and more specific. `surrendered_country_popup` is
520x320 and centred, so it need not cover whichever rectangle `healthy` is calibrated on;
if it does not, the HUD keeps matching, the loop keeps stepping, and the match ends as an
ordinary timeout draw. The requirement that follows is a calibrated template for that
popup, not a louder failure.

Two smaller results from the same pass, both contrary to what was assumed: generals are a
**degrade, not a blocker** (`PLANNING_CAP_NO_HQ_SCALING = 0.8`, and about 60 stock 1936
countries have no commander at all and still fight), so the arena's leader-only character is
worth fixing but is not why nothing moved; and supply reach is about two province hops, so
the single mid-column front hub leaves the ends of an eight-province front column short.

**Still unverified:** victory detection, a two-agent match, and anything on two machines.

Found by the same audit and deliberately left, none of them crash-level:
`common/ai_focuses` is still replaced away, which leaves nine `supports_ai_strategy` tokens
in a file that still loads; the 351 stock country tags still exist with their history files
deleted, so they are present but empty; `map/ambient_object.txt` is emptied rather than
inherited, because the stock world frame is positioned for a 5632x2048 map; and the railway
generator lays a level-1 line on every adjacent land pair, which is far denser than any
stock network. `common/ai_strategy` was removed from the replaced list, because wiping it
also removed `default.txt`, the only country-agnostic AI behaviour file.

Capture and preprocessing, measured locally on 2026-09-20:

| Change | Before | After |
|---|---|---|
| Per-tick resize in the decision loop | 60.2 ms p50 (PIL, CPU) | 0 ms; the worker sends views |
| Capture payload | 33.2 MB raw / 3.46 MB lz4 | 1.15 MB raw / 0.77 MB lz4 |
| Worker CPU to produce five views | n/a | 18.0 ms p50 |
| Offline `views` on a 4K frame | 51.3 ms (PIL) | 16.0 ms (GPU, float32) |

Two-PC round trip, measured 2026-09-20 with the rebuilt worker on both machines, 25
captures per mode:

| Request | p50 | p95 | On the wire |
|---|---|---|---|
| Full frame, the old path | 419.2 ms | 433.8 ms | 33.178 MB |
| Five policy views only | 92.1 ms | 108.8 ms | 0.753 MB |
| **Views plus template crops, what a tick asks for** | **83.9 ms** | **99.5 ms** | 0.868 MB |

That is the 200 ms decision budget met with half of it to spare, against 411 ms p95 before.
It leaves about 100 ms for inference: the compact policy at 51.46 ms p95 fits, the released
303M encoder at 187.72 ms p95 does not.

The first attempt at this measurement returned 424.8 ms p50 for all three modes, because
both workers predated worker-side downscaling: they accept `views` and `regions`, ignore
them, and return the whole frame. The binary in `target/release` was stale the same way, so
the capture table above had been measured against a build that no longer existed on disk.
`Desktop.capture` now raises when a worker accepts `views` and answers without them.

The step loop no longer serializes the interval against capture and inference. Each tick dispatches its eight slots on a separate thread and blocks the following tick on that dispatch completing, so a tick costs one interval rather than interval plus capture plus inference. Simulated locally against a fake desktop:

| Policy + capture per tick | Serial loop | Pipelined loop |
|---|---|---|
| 51 ms + 15 ms | 266 ms | 203 ms (4.93 Hz) |
| 102 ms + 15 ms | 317 ms | 203 ms (4.93 Hz) |
| 150 ms + 20 ms | 370 ms | 203 ms (4.93 Hz) |

Five Hz is therefore reachable rather than excluded by construction, but it is **not** demonstrated: these are simulated timings, not a live match, and the measured large-encoder p95 of 187.72 ms for a single actor still leaves no room for two on one GPU. The gate needs a real run. The resampler changed from PIL bilinear to an exact area average so the Rust worker can reproduce it bit for bit; this is a deliberate one-time break of any previously prepared dataset, of which there are none.

Open acceptance work:

- Decide how the arena reaches a decision at all. Its provinces are 150x the area of a stock land province, so every crossing costs 26 in-game days and each capital is five hops behind its front. The options are to cut the province size (the generator is already parametric in `COLUMNS_PER_HALF` and `ROWS`, but states, buildings, supply and victory points all scale with it), to raise army speed to suit the scale, or to accept that matches are scored on territory rather than capitulation. Nothing else on this list resolves until this does.
- Calibrate the match-side screen rules. `win`, `loss`, `healthy`, `disconnect` and `desync` have never been written for a match, and `require_match_rules()` raises without them. The win and loss templates must anchor on the peace conference's top-art banner, since both sides get the same full-screen window.
- Calibrate a template for `surrendered_country_popup`. It is 520x320 and centred, so it need not cover the `healthy` rectangle; if it does not, a capitulation reads as an ordinary timeout draw.
- Work out why Red's AI takes no ground. Two of the three suspects are now eliminated: province size is no longer the explanation, and giving both sides a field marshal, three corps commanders and a `front_control` strategy with `execute_order = yes` changed nothing over a hundred in-game days. What has **not** been established is whether Red is attacking and failing or not attacking at all — the camera could not be zoomed out to the front during either run, so no battle was ever seen. Settle that first, because a symmetric front that repels every attack is a balance problem and an idle AI is a scripting one. A single unsupported division attacking a dug-in one loses in about two days, so failing attacks would leave the map exactly as still as an idle army.
- Run a full 1800-second match at speed four and verify armies, supply over time, fog, multiple routes and side symmetry in gameplay. Match start and combat resolution are now verified.
- Pairing works; complete two-PC lobby/reset calibration and recovery checks. Menu input and watchdog release are verified remotely; match behavior remains untested.
- Verify physical capture/input alignment, live dragging, keyboard effect, focus loss and F12 under load. The worker now releases held input even when stdout fails; live regression remains needed.
- Re-measure the two-PC screenshot round trip with worker-side downscaling enabled.
- Measure end-to-end scheduling against a live game with two real actors. The loop now overlaps dispatch with capture and inference, but rendering, encoding and two actors sharing one GPU are untested at cadence.
- Record 2–4 hours of expert demonstrations. Distill the compact encoder, train the BC baseline and compare held-out gameplay for the auxiliary/XM variants.
- Run actual recurrent PPO self-play. Add an unattended league driver and model-selection schedule; current collector and trainer are separately invoked.
- Complete 20 auditable unattended matches and 50 side-swapped evaluation pairs.
- Test the same screen/input interface in a private unmodified multiplayer lobby.

No gameplay improvement, reliable speed-two operation, trained combat checkpoint, unattended match count or ordinary multiplayer win is claimed.

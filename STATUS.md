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
| LAN screenshot timing | Same report, 20 menu captures | Status-request p95 1.51 ms; full screenshot round trip p95 **411.15 ms**, exceeds the 200 ms decision budget before inference |
| Arena generation | `artifacts/mods/infantry-arena-v6`, minidumps under `Documents/Paradox Interactive/Hearts of Iron IV/crashes` | v5 crashed on Start; the map defects the dump traces to are fixed and `audit-map` passes, **but no match has been started on the fixed map** |

Earlier encoder timing reports used the old window-DC capture. Treat them as capacity measurements, not a validated live visual benchmark. Re-run with the corrected capture backend, a changing game clock and full capture-to-action timing before accepting any latency result. The worker bundle has been rebuilt with the capture fix.

Latest automated checks: **55 Python tests, 7 Rust tests, Ruff lint/format, Cargo format and Clippy pass**. Locked dependency installation and packaged CLI help also pass. Each fix below was mutation-tested: reverting it fails at least one test.

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

Map diagnosis, 2026-09-20. Revisions 2-5 hit the same access-violation stack at match start, and earlier guesses (adding a few building placements, weather placements, replacing base-map AI/focus definitions, correcting country-history filenames) did not resolve it. The five saved minidumps do. Four of the five fault at one address, in the same second as `frontend.cpp:682: [[ Launching SINGLEPLAYER-game ]]`, with nothing logged after it. Reading the dump against the stock binary's unwind and RTTI tables:

- The faulting frame and its two callers live in `hoi4/source/areas.cpp`. The object being walked is a `CControllerArea`; the list it is walking holds `CProvince*`.
- Per province it calls `CProvinceProvider::GetProvince`, whose prologue is `if (id < 1 || id >= count) return nullptr`, and then reads a member of the result without checking it.
- The id argument survives in `rdx` on the null return path, and it is **0**. So a province association in the generated map was never set, and province 0 is the null province.

Comparing every generated file against the stock database found the unset associations, each of which is now written:

| Generated by revision 5 | Stock database | Now |
|---|---|---|
| No sea province marked coastal, while 36 land provinces are | Coast is a property of both sides; 816 sea provinces are coastal | 42 coastal sea provinces; the flag is derived from the shared bitmap edge, so it cannot disagree |
| One unit-counter anchor, land provinces only | 9 anchors for every land province, 13 for every sea province | All 192 provinces carry the stock anchor set |
| 3 building placements per state | One `supply_node`, `bunker` and `special_project_facility_spawn` per land province; 6 factories, 3 AA and one of each unique building per state; one naval placement per coastal province carrying its adjacent sea province | 514 placements, including 36 ports each naming a real sea province |
| One weather period spanning the year | Twelve, one per month | Twelve |
| `-1;-1;;-1;...` terminator row in `adjacencies.csv` | Ends on a comment; the `-1` sentinel belongs to other Paradox titles | Comment |
| Land painted with terrain palette 0 and 1 | 0 and 1 are the blend terrains `terrain_0`/`terrain_1`; plains is 19 and forest 13 | 19 and 13 |

Which of those supplied the zero is not distinguishable from the dump, so this is not "one fix"; all of them were wrong relative to the stock database and all are now written. `hoi4-arena audit-map` reports 41 problems for `infantry-arena-v5` and none for `infantry-arena-v6`, and every fix is mutation-tested: reverting any one of the seven fails at least one test. The audit runs without the game, which is the point of it: the engine does not log bad map data, it dereferences it.

**This is a static fix against static evidence. The regenerated map has not been loaded, and no match has been started on it.** `artifacts/arena-v5-start-test-focused` records a screen-guarded menu test that correctly stopped when an animating setup screen missed calibration; `artifacts/worker-menu-v5-start` records the subsequent manually reviewed Start input. The original mod selection was restored; no crash reports were submitted.

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

- Load `infantry-arena-v6` and start a match; if it starts, verify armies, war, supply, fog, multiple routes and side symmetry in gameplay. If it still crashes, the new minidump narrows it the same way the last one did.
- Pairing works; complete two-PC lobby/reset calibration and recovery checks. Menu input and watchdog release are verified remotely; match behavior remains untested.
- Verify physical capture/input alignment, live dragging, keyboard effect, focus loss and F12 under load. The worker now releases held input even when stdout fails; live regression remains needed.
- Re-measure the two-PC screenshot round trip with worker-side downscaling enabled.
- Measure end-to-end scheduling against a live game with two real actors. The loop now overlaps dispatch with capture and inference, but rendering, encoding and two actors sharing one GPU are untested at cadence.
- Record 2–4 hours of expert demonstrations. Distill the compact encoder, train the BC baseline and compare held-out gameplay for the auxiliary/XM variants.
- Run actual recurrent PPO self-play. Add an unattended league driver and model-selection schedule; current collector and trainer are separately invoked.
- Complete 20 auditable unattended matches and 50 side-swapped evaluation pairs.
- Test the same screen/input interface in a private unmodified multiplayer lobby.

No gameplay improvement, reliable speed-two operation, trained combat checkpoint, unattended match count or ordinary multiplayer win is claimed.

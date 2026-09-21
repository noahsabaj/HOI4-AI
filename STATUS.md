# Verification status — 2026-09-21

The code implements an initial visual learning pipeline. It does not yet deliver reliable
self-play or a winning multiplayer agent. This file records what has been *measured*.
Where a claim here was later found wrong, it is corrected in place — the history of the
correction is in `git log`, not here.

## Measured

| Check | Evidence | Result |
|---|---|---|
| Local hardware | Consumer desktop: 8 GB VRAM GPU, 32 GB RAM | The figures every benchmark below is read against; the peer is identical |
| Released 303M encoder, game menu | `artifacts/benchmark/inference.json` | p95 148.73 ms, minimum 1553 MiB free |
| Released encoder, loaded vanilla game | `artifacts/benchmark-loaded/inference.json` | p95 187.72 ms, only 710 MiB free: **fails 1 GB headroom**; game was paused |
| Offline large-encoder training capacity | `artifacts/benchmark-loaded/training.json` | Two BF16 Adam updates on a synthetic objective — capacity, not behavioural training |
| Compact full policy | `artifacts/benchmark-policy-tiny.json` | Untrained 5.49M encoder, full-policy p95 51.46 ms, game closed |
| Native 3840×2160 capture | `artifacts/worker-menu-v3-new`, `-select` | Desktop-DC capture follows two real menu transitions; the old window-DC path returned stale/scaled pixels |
| Held input watchdog | `artifacts/worker-smoke/result.json` | Held Shift released after timeout |
| Live worker mouse input | `artifacts/worker-smoke/click-*.png` | Menu advanced on a 250 ms button hold |
| TLS peer transport | `artifacts/pairing/integration/report.json` | the peer machine authenticated; remote 4K capture, menu click, Escape and watchdog all verified |
| LAN screenshot timing | `artifacts/pairing-roundtrip-downscaled.json`, 25 captures per mode | Views plus template crops p50 83.9 ms, **p95 99.5 ms**, 0.87 MB, against p95 433.8 ms / 33.2 MB for the full frame it replaces |
| Arena match start | `infantry-arena-v10` | War declared, clock advanced 1–2 Jan 1936, no crash dump |
| Rescaled map starts a match | `infantry-arena-v13`, `logs/game.log` | `Loaded 1537 provinces`, start-date 1936.1.1.12, ran to 11 Apr 1936 at speed four. Zero `MAP_ERROR`, `TOO LARGE BOX`, `has no continent`, `no pixels` or naval-base lines; state names resolve |
| Combat resolution | `infantry-arena-v12`, 5 Mar – mid-May 1936 | Two Blue divisions attacked one Red defender; tooltip gave a running estimate (`...another 74 days`); attacker repulsed, no province changed hands. Order, battle and resolution all work |
| Victory detection | `capitulation-harness-v2`, 1936-01-01 to 1940-05-05 | **Not reached.** Blue held Red's entire placed victory-point weight from 1936-01-13; Red had not capitulated 52 months later. Victory points are not what decides a surrender — see below |
| An AI against an enemy with no army at all | `capitulation-harness-v2` (Red fielding zero divisions), same run | **It advances, then stops.** Blue was AI-driven throughout: its owned weight rose 950 → 1400 between 1936-01-17 and 1936-10-13 against no opposition, then read exactly 1400 again on 1938-05-03, 1938-07-08 and 1940-05-05. Roughly 47% of an undefended country, then nothing for 43 in-game months |
| Four-year unattended run | Same match, speed four to five | 1936-01-01 to 1940-05-05, no crash dump, no new `error.log` line. The longest the arena has run |
| Live screen stability | Four worker captures at 3840×2160, two running and two paused | **Nothing on screen holds still to within the old hardcoded 5 MAE.** HUD drifts a mean of 6.9 between captures seconds apart while running, 1.6 paused — the day/night terminator sweeping under a translucent HUD |

Earlier encoder timings used the old window-DC capture; treat them as capacity numbers, not
a live visual benchmark. Re-run with the corrected backend, a changing clock and full
capture-to-action timing before accepting any latency result.

Automated checks: **81 Python tests, 7 Rust tests, Ruff lint/format, Cargo fmt and Clippy
pass**, plus locked dependency installation and packaged CLI help. Every fix is
mutation-tested — reverting it fails at least one test.

**Still unverified:** victory detection, a two-agent match, and anything on two machines.

## What a surrender actually costs

**Taking every victory point a country owns does not capitulate it.** `capitulation-harness-v2`
gives Red a single province carrying its whole 35-point weight on the border column, with no
divisions anywhere in the country. Blue took that province on 1936-01-13 — debug tooltip
`Province ID = 1524 ... Owner = Red / Controller = Blue / Local VP: 35` — and Red had not
capitulated by 1940-05-05, far past `DAYS_OF_WAR_BEFORE_SURRENDER = 7`.

The quantity surrender is measured against is territorial. The tooltip reads `Owned VPs: 950`
for a fresh 30-state country whose placed victory points total 35, and `State value: 22.3`
for a state holding none — 30 states at about 31.7 is roughly 950. Blue's own figure rose
from 950 to 1400 as it occupied ground, not as it took victory points.

`BASE_SURRENDER_LIMIT = 0.8` (in `NCountry`) is an occupation fraction and is *not* the
threshold; `BASE_SURRENDER_LEVEL = 1.0` in `NDiplomacy` is the one commented "Surrender when
level reached". `set_stability = 1` and `set_war_support = 1` do not make capitulation
harder: no stock modifier ties stability to `surrender_limit`, and the only entry in
`00_static_modifiers.txt` is `-0.3` inside `war_support_bad_modifier`.

Two constraints on reaching a decision still hold: each capital sits about five province
hops behind its front, and symmetric unsupported infantry does not break through — six
`infantry_equipment_1` battalions give 36 soft attack against 132 defence and 18
breakthrough, and the live battle above is the measurement, not a model.

Province scale was the reason the first soak saw nothing, and is fixed. `GAME_SPEED_SECONDS`
is `{2.0, 0.5, 0.2, 0.1, 0.0}` wall-seconds per in-game hour; speed one measures 48 s per
in-game day (37.5 days per 1800 s, matching the soak to 1%) and speed four about 727 days
per 1800 s. The original 8×12 grid gave provinces 150× the area of a mean stock land
province, so one border crossing cost 26.6 in-game days and the 1800-second soak at speed
one covered barely one crossing — nothing had failed, the armies had not finished walking.
The grid is now 32×24 per half (1536 provinces, ~88×85 px), and an `arena_march_speed`
country spirit multiplies army speed by 4, putting a crossing near 1.6 days. Speed is
changed with `+`/`-` (`VK_OEM_PLUS`/`VK_OEM_MINUS`, allowed in setup and refused during a
match); the number keys do nothing, which is how the first soak silently ran at speed one.

## Terminal screens

There is no bare "game over" screen, and the end-game window is not a fallback. Read out of
the shipped files and adversarially re-checked:

- A one-versus-one capitulation **does** open a conference. The engine's capitulation path
  creates it (`"Creating peace conference between %s and %s"` sits beside the surrender log
  in `country.cpp`), a single-winner conference is a serialized state (`solo_winner`), and
  the peace-conference AI refuses to run for a human country. The only annex-on-war-end path
  in the shipped files is the civil-war one.
- The sequence is `surrendered_country_popup` (520×320) → `peaceconference_full_window`
  (100%×100%) → "Calculating Effects..." → `peace_summary_popup_window` (480×382) → map.
- **`playthrough_overview_window` never fires on conquest.** HOI4 has no victory-condition
  concept anywhere in script or engine; the only end-of-game define is `END_DATE = 1949.1.1.1`.
- `surrendered_country_popup` is `orientation = center`, `position = { -225 -160 }`,
  `size = { 520 320 }`, `moveable = yes`, on a 527×322 `GFX_popup_capitulation_bg`.
  **`exile_country_popup` reuses the same sprite and geometry byte for byte**, so a template
  cut from the frame art cannot tell them apart — it must anchor on the title text box.
- Winner and loser get the same `peaceconference_full_window`, distinguished only by the
  top-art sprite (`GFX_top_art_winning_conference` vs `..._losing_conference`), so win and
  loss templates must anchor on that banner rather than the screen as a whole.
- `annex_everything` is neither an auto-annex nor a peace-conference cost discount: its two
  discount lines are commented out in `common/wargoals/00_invasion.txt:145-146`.

Since it is 520×320 and centred, the popup need not cover whichever rectangle `healthy` is
calibrated on. If it does not, the HUD keeps matching, the loop keeps stepping, and a real
capitulation ends as an ordinary timeout draw. `ScreenRules` is otherwise explicit about
failure: a screen matching nothing raises `unrecognized_match_screen`, a terminal candidate
that never converges raises `terminal_screen_never_confirmed`, and a timeout without a
healthy HUD raises `uncertain_timeout`.

## Calibration

`healthy` and `clock_rect` are cut from a live match and stored at 3840×2160 in
`artifacts/calibration-live/rules.json`. The other five rules are not calibrated, so
`require_match_rules()` still raises. Getting these two exposed three things worth more than
the templates:

| Rectangle | Running | Paused |
|---|---|---|
| HUD icon row, two captures seconds apart | **6.9** | 1.6 |
| Clock crop | **44.8** | 6.4 |
| Speed bars, running versus paused | **0.6** | |
| Play/pause glyph, running versus paused | **26.3**, against 15.6 running-versus-running | |

- **No live screen holds still to within 5.** A `healthy` rectangle cut from one capture
  rejected the next capture of the same screen at 7.3. `template` now takes `--max-mae`
  (bounded 1..96, above which a crop matches anything), and `healthy` is calibrated at 15 on
  `[168, 64, 600, 52]`, verified across four captures spanning four in-game years.
- **The clock stall detector could never fire.** The loop compared successive clock crops
  with `np.array_equal`, and two captures of a *paused* clock differ by 6.4, so every frame
  looked like a fresh tick. `clock_advanced` now compares with a tolerance of 15, between
  the paused 6.4 and the advancing 44.8.
- **`running_speed_two` cannot be built on the speed bars.** They read 0.6 between a running
  and a paused game: the bars show the *selected* speed, not whether time moves. The
  play/pause glyph separates them at 26.3, but against a 15.6 self-drift — a thin margin for
  a gate that invalidates an episode.

## The map generator, and why it is defensive

The engine does not report bad map data; it dereferences it. `hoi4-arena audit-map` checks
for that without the game, and `generate-map` audits what it wrote and exits non-zero. Four
distinct crashes were read out of minidumps rather than guessed:

| Fault | Evidence | Cause | Fix |
|---|---|---|---|
| `areas.cpp`, walking a `CControllerArea` province list | `GetProvince` returns null below id 1; `rdx` held **0** | a province association the map never set | write the placements and anchors the stock database supplies for every province |
| `ingameidler.cpp`, `GenerateNonHistoricalAttributes` | `character_manager.cpp:261 Failed to generate a name ... for country Blue`, ×5 | no `common/names` entry and no country leader, so every generated character was nameless | ship a name list and a `country_leader` for both tags |
| same frame, null `this` | `rax` = **0x226 = 550**, `rbx` = ASCII `state` | stock `tutorial/tutorial.txt` hard-codes state 550 and provinces 5010/5091/12766, resolved at every match start | override the file — `replace_path = "tutorial"` does **not** unload it |
| same function, later | `mov rcx,[rax+rcx*8-8]`, `rax = 0`, `rcx = 0` | the loader marks the **last** hint entry, so an empty tutorial indexes element −1 | ship exactly one block naming no state and no province |

Map facts the generator depends on, each checked against the stock database:

- The `-1;-1;;-1;...` row in `adjacencies.csv` is the engine's end-of-file marker, not a
  stray sentinel, and is required even when the file holds no rows.
- Terrain palette 0 and 1 are the plain plains/forest the stock map paints over 9.8% and
  5.7% of the world. Index 19 is `plains_17` (`perm_snow`) and 13 is `forest_13`
  (`type = urban`, `spawn_city`) — neither can stand in for ordinary ground.
- `trees.bmp` is fixed at 75/256 of the province bitmap, not a quarter.
- A sea strategic region needs `naval_terrain`; it is where sea provinces take their terrain.
- `cities.bmp` index 0 is the stock sparse-city group; index 4 is claimed by no group and is
  how you ask for no cities.
- Since 1.11 the bitmap decides coastal status and `definition.csv` only has to agree.
- Map colour comes from `common/countries/colors.txt` with the space named — `color = rgb { }`.
  A bare `color = { }` in a country file is not the map colour, which is why Blue rendered
  green and Red pale cyan while their flags were right.
- The camera's zoom-out limit is fixed in world units, not fitted to the map, so the arena is
  5632×2048 — the stock map's exact dimensions. Anything smaller shows sky past the edges and
  more than one map width across, which on a horizontally wrapping world draws the same
  countries two and a half times over.
- Also fixed by audit: coastal cliffs replaced by a ramp with a maximum neighbour step of 1;
  ship-in-port anchors for provinces with naval bases; surrender weight spread instead of
  sitting on one province; weather objects anchored over the right region; an X-crossing
  breaker that now looks at the horizontal wrap seam; and adjective, ideology and
  victory-point localisation, absent of which Red's capital was labelled "Kargopol".
- **The vanilla Earth showed over the arena's ocean** because five map-shaped textures were
  never shipped, so the engine stretched the stock world over the new map:
  `colormap_rgb_cityemissivemask_a.dds`, `colormap_water_0/1/2.dds` (at provinces/2, /4, /8),
  `fow_rgb_waterspec_a.dds`, and both minimap widgets. All are now generated from the arena's
  own land mask as uncompressed 8.8.8.8 ARGB with no mip chain.

Two findings that were assumed to be blockers and are not: generals are a **degrade**
(`PLANNING_CAP_NO_HQ_SCALING = 0.8`, and about 60 stock 1936 countries have no commander and
still fight), and `common/ai_strategy_plans` holds only focus, research and idea picks — it
has no key that creates a front, an offensive or a garrison, and 279 of the game's 364 tags
have neither a plan nor a country `ai_strategy` file and still fight. Fronts are
engine-generated and the arena has one. Supply reach is about two province hops.

The generator carries two diagnostic options, recorded in `generation.json`, neither
producing a playable arena: `--undefended BLU|RED` fields no divisions for that side, which
is what separated "the AI attacks and fails" from "the AI does not attack";
`--victory-points-on-border` masses a side's whole weight on one border province. The second
was built to force a capitulation, did not, and is how the victory-point assumption above
came to be refuted.

Known and deliberately left, none crash-level: `common/ai_focuses` is replaced away, leaving
nine `supports_ai_strategy` tokens in a file that still loads; the 351 stock tags exist with
their history files deleted; `map/ambient_object.txt` is emptied rather than inherited, as
the stock world frame is positioned for a 5632×2048 map; and the railway generator lays a
level-1 line on every adjacent land pair, far denser than any stock network.
`common/ai_strategy` was removed from the replaced list because wiping it also removed
`default.txt`, the only country-agnostic AI behaviour file.

## Performance

| Change | Before | After |
|---|---|---|
| Per-tick resize in the decision loop | 60.2 ms p50 (PIL, CPU) | 0 ms; the worker sends views |
| Capture payload | 33.2 MB raw / 3.46 MB lz4 | 1.15 MB raw / 0.77 MB lz4 |
| Worker CPU to produce five views | n/a | 18.0 ms p50 |
| Offline `views` on a 4K frame | 51.3 ms (PIL) | 16.0 ms (GPU, float32) |

Two-PC round trip, rebuilt worker on both machines, 25 captures per mode:

| Request | p50 | p95 | On the wire |
|---|---|---|---|
| Full frame, the old path | 419.2 ms | 433.8 ms | 33.178 MB |
| Five policy views only | 92.1 ms | 108.8 ms | 0.753 MB |
| **Views plus template crops, what a tick asks for** | **83.9 ms** | **99.5 ms** | 0.868 MB |

That meets the 200 ms decision budget with half to spare and leaves about 100 ms for
inference: the compact policy at 51.46 ms p95 fits, the released 303M encoder at 187.72 ms
does not. The first attempt returned 424.8 ms p50 for all three modes because both workers —
and the binary in `target/release` — predated worker-side downscaling and silently returned
the whole frame. `Desktop.capture` now raises when a worker accepts `views` and answers
without them; rebuild and redeploy after touching the Rust crate rather than trusting
`target/release`.

The step loop no longer serializes the interval against capture and inference: each tick
dispatches its eight slots on a separate thread and blocks the next tick on that dispatch.
Simulated against a fake desktop, a tick costs one interval rather than interval plus
capture plus inference — 203 ms (4.93 Hz) at 51+15, 102+15 and 150+20 ms per tick, against
266/317/370 ms serial. Five Hz is therefore not excluded by construction, but it is **not**
demonstrated: these are simulated timings, and 187.72 ms p95 for one large encoder leaves no
room for two actors on one GPU. The resampler changed from PIL bilinear to an exact area
average so the Rust worker can reproduce it bit for bit.

## Open work

- **Decide how the arena reaches a decision at all.** Capitulation is measured against
  territory, not the victory points the generator places, so a decision means occupying most
  of a 600-province country; the harness run took 47% of an undefended Red and stalled. The
  options are a much smaller province grid, a much larger army, or scoring matches on
  territory. Nothing else here resolves until this does.
- **Build a small-map harness so a capitulation can be produced on demand.** This blocks the
  five uncalibrated rules. Make `COLUMNS_PER_HALF`, `ROWS`, `STATE_COLUMNS` and `STATE_ROWS`
  per-call arguments rather than module constants, and generate a grid small enough that an
  undefended country is overrun in days. The conference and the popup are engine UI and look
  the same on any map, so templates cut there transfer to the playable arena.
- Calibrate `win`, `loss`, `ready`, `disconnect`, `desync`, and a `surrendered_country_popup`
  template anchored on its title box.
- Decide what `running_speed_two` should be — the speed bars cannot detect a pause, and the
  name asserts a speed measurement has ruled out as too slow.
- Work out why an AI that *does* advance stops. Province size and missing command are both
  eliminated. What has not been established is what it waits for: a supply limit, a front it
  considers held, or an objective it thinks it has reached.
- Run a full 1800-second match at speed four and verify armies, supply over time, fog,
  multiple routes and side symmetry in gameplay.
- Complete two-PC lobby/reset calibration and recovery checks; match behaviour is untested
  remotely even though menu input and watchdog release are verified.
- Verify physical capture/input alignment, live dragging, keyboard effect, focus loss and
  F12 under load.
- Measure end-to-end scheduling against a live game with two real actors.
- Record 2–4 hours of expert demonstrations, distil the compact encoder, train the BC
  baseline and compare held-out gameplay for the auxiliary/XM variants.
- Run recurrent PPO self-play; add an unattended league driver and model-selection schedule.
- Complete 20 auditable unattended matches and 50 side-swapped evaluation pairs.
- Test the same screen/input interface in a private unmodified multiplayer lobby.

No gameplay improvement, reliable speed-two operation, trained combat checkpoint, unattended
match count or ordinary multiplayer win is claimed.

# Arena implementation and capability report

Measured on 2026-09-13. **The integration gate has not passed. This is not yet a
playable HOI4 combat agent.** The Python learner and diagnostic transport are
implemented and tested; the build-specific engine adapter, playable arena, live
collection loop, and strength experiments remain unfinished.

## Implemented

- A separate `hoi4_agent.arena` package; the existing vision-agent commands and
  calibration assets are preserved.
- Python 3.11 environment bootstrap and exact dependency versions, PyTorch CUDA
  verification, and C++20 builds with MSVC warnings treated as errors.
- Versioned observations, orders, receipts, arena specifications, and trajectory
  provenance. Strict build/DLC/mod hashes, normal-order ownership checks, episode
  isolation, deadlines, and duplicate-command rejection.
- A local Windows named-pipe client with bounded frames, deadlines, cancellation,
  and no automatic command retries. The C++ executable is a diagnostic server.
  Its handshake explicitly reports that gameplay is unavailable. The DLL is an
  experimental bootstrap probe; its in-game loading has **not** been established.
- A 354,690-parameter policy at width 128: two graph-message-passing layers,
  friendly/visible-enemy entity encoders, a GRU, a value head using the same player
  view, and a masked distribution over no-op, cancellation, and adjacent movement
  (which is how HOI4 attacks) or support attack by one division. Unit views carry
  kind, entrenchment, and the current order target; provinces carry river crossings. Larger unit selections are a future vocabulary
  expansion; group demonstrations are rejected explicitly.
- Recurrent PPO updates with episode resets, segment-start memories recomputed
  once per episode per epoch before truncated BPTT, elapsed-hour GAE, terminal outcome rewards, and optional
  potential shaping with zero terminal potential. Behavior-cloning updates accept
  only accepted, aligned engine commands from training scenarios.
- Atomic checkpoints with optimizer/configuration/league/provenance/Torch RNG
  state. The league stores immutable checkpoint copies, freezes opponents per
  episode, preserves its own sampling RNG, and uses the requested 50/25/25 mix.
- Whole-match/scenario demonstration split checks, paired side-balanced score
  reports with confidence intervals, and the four shared experiment
  specifications in `config/arena/experiments.toml`.

These components do not amount to a live trainer. There is no artificial
simulator presented as HOI4 and no fixture-generated match in the capability
report. No substantial training run was started.

## Actual integration evidence

| Check | Observed result |
|---|---|
| Installed game | Operation Postern v1.19.2.0.3eb1 (85f4), installed `open_beta` |
| Executable SHA-256 | `dbc9744158e2e15c4182f04c868e9e697fa917a339072e831c5cde12d4563a76` |
| Static PE inspection | Filesystem/GPU-preference exports and candidate C++ RTTI names; no established player-view/order ABI |
| Isolated profile | Confirmed through `gameDataPath`; both attempted `userdir` forms and `userdir.txt` were ineffective in these launches |
| Probe mod | `Active Mod: HOI4 Arena Binding Probe` confirmed by the isolated game log |
| Legacy `script/autoexec.lua` | No binding dump was produced; its presence is not proof of an active bot API |
| Active defines Lua context | Binding inventory obtained through a deliberate diagnostic error; exposed Lua libraries and `NDefines` configuration tables, no player/unit API in this context |
| Lua file I/O | `io`/`io.open` unavailable in the tested defines context |
| Native DLL | Compiles and loads through Windows' loader in a separate diagnostic Python process |
| Lua native loader | Returned a nil loader and userdata diagnostic; no callable DLL entry point was obtained through this path. The underlying cause remains unresolved |
| Standalone named pipe | Actual Python-to-C++ handshake and gameplay-rejection test passed |
| Live observation/orders | Not implemented or verified |
| Unattended arena matches | 0 / 20 |
| Real-game command reliability sample | 0 / 1,000; accuracy and latency unmeasured |
| Human-versus-model match | Not run |
| Learning/strength milestone | Not run |

Raw evidence in this workspace:

- `artifacts/capabilities.json`: installation hashes and extracted probe evidence.
- `artifacts/lua-probe-02/logs/system.log`: isolated profile and mod load.
- `artifacts/lua-probe-03/logs/error.log`: file-I/O restriction.
- `artifacts/lua-probe-04/lua_bindings.txt`: complete captured globals/types inventory.
- `artifacts/native-probe-02/logs/error.log`: native loader result.

The installed launcher settings were restored byte-for-byte after each completed
probe. The final hash matches the original backup:
`6655308a23e61bc8f1e8c99b1c6677354422295ba3e2523669d8a99c8eb892a6`.
No existing save was loaded or changed. Temporary profiles and their logs remain
under `artifacts/` for inspection.

## Remaining implementation sequence

1. Establish a repeatable native module bootstrap for this exact build. The Lua
   loader probe has not established one. Resolve the engine game-state and
   visibility interfaces, normal-order constructor/validation/dispatch path,
   simulation-thread scheduling point, accepted-command stream, and country AI
   suppression. `native/include/engine_adapter.hpp` is the interface, not an
   implementation. RTTI names must not be treated as addresses or signatures.
2. Implement the original mirrored land mod, its scenario variants and immutable
   train/validation/held-out manifest, starting saves, reset fingerprints, and
   outcome rules. The current diagnostic mod is only a probe, not that arena.
3. Prove both-country control, fog-of-war invariance against changed hidden state,
   movement/attack/cancellation, reset, and uninterrupted human input. Run the
   actual 20-match/1,000-command gate with correct-accounting rate >=99% and p95
   observation-to-submission latency <250 ms. An engine snapshot must be filtered
   before it enters `PlayerObservation`; Python's view-only API does not prove
   that a future native extractor respects fog of war.
4. Connect a continuous real-time collector: game-hour decisions capped at 2/s,
   the same configurable order-rate budget in every mode, discarded late
   predictions, unchanged existing orders after misses, and recorded deadlines.
   No pause command or mouse control belongs in this arena runtime. Recording,
   arena launch, human play, and live training commands are still to be built.
5. Implement several versioned scripted opponents and accepted-command human
   recording; connect the four experiment configurations to the live collector,
   balanced demonstration pretraining, permanent evaluation panel, and league.
   Run three seeds per eligible arm under the 24-hour initial budget. Human arms
   wait for recordings; scratch/scripted arms need not wait for them.
6. Collect at least 100 side-balanced games per candidate per seed. Require the
   >=0.15 score improvement and positive paired interval for each of three seeds,
   plus separate held-out opponent/scenario evidence. The current report helper
   checks pairing and statistics; it does not verify replay authenticity, panel
   permanence, equal compute budgets, or the full acceptance gate by itself.
7. Only after real baseline learning, implement and compare the compact
   structured-state world model under equal budgets. Economy, research,
   construction, air, and navy remain subsequent expansions.

## Local commands available now

Run from the repository root in PowerShell:

```powershell
./scripts/bootstrap_arena.ps1
./.venv/Scripts/python.exe scripts/build_native.py
./.venv/Scripts/python.exe -m hoi4_agent.arena.cli model-info
./.venv/Scripts/python.exe -m hoi4_agent.arena.cli inspect --output artifacts/capabilities.json
./.venv/Scripts/python.exe -m hoi4_agent.arena.cli --help
```

`inspect` returns a nonzero exit status while integration is blocked. Use
`--probe-profile PATH` repeatedly to attach existing probe evidence to its report.

To inspect the standalone pipe, run the executable and client in separate shells:

```powershell
./artifacts/native/Release/hoi4_bridge_probe.exe hoi4-arena-diagnostic
./.venv/Scripts/python.exe -m hoi4_agent.arena.cli bridge --pipe hoi4-arena-diagnostic
```

The response says `backend: diagnostic`; the client exits nonzero because it is
not a gameplay adapter. The probe serves one connection, then exits on disconnect
or inactivity. This is transport evidence only.

```powershell
# Create a fresh diagnostic mod/profile without launching or modifying the installation.
./.venv/Scripts/python.exe -m hoi4_agent.arena.cli lua-probe --output artifacts/my-probe

# Create AND launch a fresh profile; never reuse an existing output directory.
./.venv/Scripts/python.exe -m hoi4_agent.arena.cli lua-probe --output artifacts/my-live-probe --launch
```

The launch command temporarily changes the installed `launcher-settings.json`,
requires write access there, and refuses an already-running HOI4 process. It
restores the exact bytes after startup or failure. Do not launch another game or
edit launcher settings during that interval. The defines inventory deliberately
ends its diagnostic boot via a Lua error when file I/O is unavailable; this is
expected and does not count as a playable launch. If the Python process is
forcibly killed during startup, inspect `hoi4-arena-launch.lock` in the game
installation: it identifies the exact backup to restore before another launch.

`fingerprint --mod MOD_DIR --dlc-load PROFILE/dlc_load.json --output OUTPUT.json`
hashes the executable, installed DLC contents/load configuration, and the exact
enabled mod. `evaluation-report --games RESULTS.json --output REPORT.json`
summarizes records following `EvaluationGame`; it does not play games.

## Verification

```powershell
./.venv/Scripts/python.exe -m pytest tests --basetemp .cache/pytest-final
./.venv/Scripts/python.exe -m ruff check .
./.venv/Scripts/python.exe -m mypy hoi4_agent
```

Observed: 229 passed, 1 skipped; lint and type checks passed. The skipped test is
the pre-existing optional RapidOCR check (the OCR extra is not installed). Both C++
targets build with `/W4 /WX`. CUDA is available on the RTX 4060 Ti. The checkpoint
test verifies that resumption produces exactly the same subsequent weights for
the synthetic update, including optimizer state. These results establish local
software behavior only.

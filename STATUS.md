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
| Arena generation | `artifacts/mods/infantry-arena-v5` | Blue/Red country selection visible; **crashes on Start**, not playable yet |

Earlier encoder timing reports used the old window-DC capture. Treat them as capacity measurements, not a validated live visual benchmark. Re-run with the corrected capture backend, a changing game clock and full capture-to-action timing before accepting any latency result. The worker bundle has been rebuilt with the capture fix.

Latest automated checks: **18 Python tests, 3 Rust tests, Ruff lint/format, Cargo format and Clippy pass**. Locked dependency installation and packaged CLI help also pass.

Map diagnosis: revisions 2–5 hit the same access-violation stack at match start. Adding building placements, small/big weather placements, replacing base-map AI/focus definitions and correcting country-history filenames did not resolve it. `artifacts/arena-v5-start-test-focused` records a screen-guarded menu test that correctly stopped when an animating setup screen missed calibration; `artifacts/worker-menu-v5-start` records the subsequent manually reviewed Start input. The original mod selection was restored; no crash reports were submitted.

Open acceptance work:

- Diagnose arena match-start crash; verify armies, war, supply, fog, multiple routes and side symmetry in gameplay.
- Pairing works; complete two-PC lobby/reset calibration and recovery checks. Menu input and watchdog release are verified remotely; match behavior remains untested.
- Verify physical capture/input alignment, live dragging, keyboard effect, focus loss and F12 under load. The worker now releases held input even when stdout fails; live regression remains needed.
- Fix and measure end-to-end scheduling: the current synchronous environment adds a full action interval after inference, so its nominal 5 Hz is not achieved by construction. Rendering/capture/encoding and two local actors add further cost.
- Record 2–4 hours of expert demonstrations. Distill the compact encoder, train the BC baseline and compare held-out gameplay for the auxiliary/XM variants.
- Run actual recurrent PPO self-play. Add an unattended league driver and model-selection schedule; current collector and trainer are separately invoked.
- Complete 20 auditable unattended matches and 50 side-swapped evaluation pairs.
- Test the same screen/input interface in a private unmodified multiplayer lobby.

No gameplay improvement, reliable speed-two operation, trained combat checkpoint, unattended match count or ordinary multiplayer win is claimed.

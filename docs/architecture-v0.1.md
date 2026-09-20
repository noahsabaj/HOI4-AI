# HOI4-AI architecture v0.1

Drafted 2026-09-19. Supersedes the native-bridge design in the original arena plan.
[arena-status.md](arena-status.md) remains the record of what the native probes found.

## Goal

An agent that plays a compact two-country HOI4 land-combat arena **through the screen,
mouse and keyboard**, improves measurably through learning, and can later grow toward
the full game. Version 1 is a benchmark harness with several agent families on one
leaderboard, not a single agent.

## Decisions taken

| Decision | Outcome |
|---|---|
| Game interface | Vision-first: screenshots in, mouse/keyboard out. No memory reading or DLL injection on the main path. |
| "Human-like" rules | None. Vision-first is the interface, not a handicap. No action cap, pausing allowed, speed is the agent's choice. |
| Game speed | The agent must work at all five speeds. Speed is an observation input and an action. |
| Camera | One fixed screen. The arena map fits without scrolling or zooming. |
| Cloud | Allowed for inference (DeepSeek V4.1 Flash, TypeSafe Jev). Training of the small policy stays local. |
| Hardware | Two or more identical PCs on one Ethernet LAN. One agent per game instance. |
| Decision structure | Hierarchical, three tiers on independent clocks. |
| Unit observation | Enemy: per stack (province, division count, average organization, average strength). Own: same from the map, per-division detail from the army panel. |

## System overview

```
                        ┌──────────────────────── PC 1 ────────────────────────┐
                        │                                                      │
   HOI4 window ──frame──▶  LOCAL PERCEPTION (ms, classical CV)                 │
        ▲               │     calibrated province centres, counter templates,  │
        │               │     glyph digits, bar-pixel fill, speed/pause state  │
        │               │                 │ PlayerObservation (+ confidence)   │
        │               │                 ▼                                    │
        │               │  TIER 1  LEARNED POLICY (ms, every tick, local GPU)  │
        │               │     province graph + entity encoders + GRU memory    │
        │               │     inputs: observation, speed, intent, brief age    │
        │               │     outputs: unit order | set speed | pause | no-op  │
        │               │                 │ Order                  ▲ intent    │
        │               │                 ▼                        │           │
        └──clicks/keys──┤  EXECUTOR  select → right-click → verify │           │
                        │                                          │           │
                        │  TIER 2  JEV (~350 ms, cloud, text only) ┘           │
                        │     one fan-out call: per-sector attack/hold/retreat,│
                        │     which stacks rest, event options, speed change   │
                        │     state = observation JSON + latest brief          │
                        │                 ▲ brief + candidate plans            │
                        │  TIER 3  DEEPSEEK V4.1 FLASH (1–15 s, cloud, async)  │
                        │     reads popups / unknown screens → text            │
                        │     plans with thinking; writes Jev's Choice options │
                        └──────────────────────────────────────────────────────┘
                                   │ LAN multiplayer + TCP coordinator │
                        ┌──────────┴──────────── PC 2 ─────────────────┴───────┐
                        │  same stack: opponent policy, or a human player      │
                        └──────────────────────────────────────────────────────┘
```

Rules that hold the tiers together:

- **The tick loop never blocks on the cloud.** On timeout, rate limit or outage the policy
  continues with the last intent.
- **Numbers and positions come only from local perception.** DeepSeek supplies semantics
  (what a popup says, what an unfamiliar screen is, plan text), never coordinates or bar fill.
- **Every brief and intent carries the game-hour it was produced at.** Consumers see its age.
- **DeepSeek outputs intents, not unit orders**, because a thinking call can span days of
  game time at speed 5.
- **Generator and judge:** DeepSeek proposes candidate plans; they become the options of a
  Jev Choice question scored against fresh local state.
- **Confidence routing:** low Jev confidence escalates to DeepSeek with thinking, optionally
  pausing the game while waiting.

## Components

### Arena mod
- Two mirrored countries, about 48 provinces, three approach routes, capitals, contested
  central victory points, symmetric supply.
- Designed for perception: whole map on one screen, fixed camera and zoom, high-contrast
  counters, all own divisions in one army so the army panel lists them at fixed positions.
- **Timeout rule:** at day 90 the side holding more victory points wins. Draws should be rare;
  a league draw rate above about 30% is treated as a design bug.
- **Reset:** a scripted "reset arena" decision the agent clicks. It respawns units and restores
  province control without loading a save. Residual state (manpower, equipment, experience)
  is fingerprinted.
- Curriculum: 2–4 infantry divisions per side, then 12, then mixed infantry and armour.
  Scenario variants for crossings, breakthroughs, defence, retreat, encirclement and supply
  disruption, split into train, validation and held-out before training.

### Local perception
Classical computer vision, no neural network in v0.1. Output is a `PlayerObservation` with a
confidence per field. Because only the screen is read, fog of war is correct by construction.
Accuracy is audited against text save files captured at the same moment
(`hoi4_agent/eval/savefile.py`); the same pairing later provides free labels if a learned
reader (small CNN) is needed.

### Executor
Deterministic translation of an `Order` into input: select unit or stack, right-click target,
then visually confirm the order took (movement arrow present). Produces the `OrderReceipt`.
Also handles speed keys, pause and the reset decision. Records click timing; enforces no cap.

### Tier 1: learned policy
The existing recurrent policy (`hoi4_agent/arena/policy.py`, about 355k parameters).
Additions for v0.1: game speed, Jev intent, enemy stack count, perception confidence and brief
age as inputs; set-speed and pause as actions; an auxiliary intent head.
Decisions are paced in game time, not wall-clock time. Discounting uses elapsed game hours
(`gamma_per_hour = 0.9995`), which already handles uneven gaps at high speed.

### Tier 2: Jev
Text-only typed judgments with calibrated probabilities. Measured here: 20/22 HOI4 decisions
correct, median 357 ms, both errors below 0.5 confidence, five questions in one call at no extra
latency, about $0.00002 per call. Cannot learn and is weak at multi-step numeric reasoning.

### Tier 3: DeepSeek V4.1 Flash
Multimodal and generative. Measured here: reading and counting strong (OCR of counters, city,
menu, arrow label all correct); pointing unreliable (errors of 5–20% of the image); bar fill
unreadable (images compress to about 250–330 tokens); 1.3 s median without thinking, up to
13 s with. About $0.0001 per image call.

## Agent families (one harness, one leaderboard)

| Family | Composition | Training needed |
|---|---|---|
| Random / scripted | Executor with trivial logic | None; sanity floor |
| Cloud | Local perception + Jev + DeepSeek + scripted intent executor | None; first end-to-end number |
| Structured | Local perception + learned policy | Simulator RL, behaviour cloning |
| Hierarchical | Structured + Jev intent + DeepSeek brief | As above, goal-conditioned |
| End-to-end (later) | Pixels to mouse, learned perception fused with policy | Cloned from the families above and human play |

Every tier must earn its place by ablation: policy alone, plus Jev, plus Jev and DeepSeek.
Results are reported per game speed and in two tracks: pause allowed and pause forbidden.

## Opponents and evaluation

- **Fixed panel:** built-in AI variants (`ai_strategy` configurations, handicaps). All families
  face the same panel. Wins, losses and draws reported separately with confidence intervals;
  side-swapped pairs; held-out scenarios.
- **Self-play league over LAN:** one agent per PC in a multiplayer match. The existing league
  code applies (frozen historical opponents, 50/25/25 sampling, permanent evaluation panel).
  A TCP coordinator syncs episode start, reset, outcome and the pause/speed convention.
- **Human versus model:** human on one PC, agent on the other, from the first playable version.
- Gate: a candidate must beat each panel opponent outright, not merely improve on its
  initialization; improvement must reproduce across seeds and survive held-out opponents.

## Training pipeline

1. **Simulator** of arena land combat, labelled as a simulator, emitting the same
   `PlayerObservation` contract with measured perception noise injected. This is where
   reinforcement learning from scratch happens.
2. **Distillation:** label simulator states with Jev intent probabilities; pre-train the intent head.
3. **Behaviour cloning** from recorded human play: screen plus input hook, with the executor
   mapping run in reverse to recover `Order`s.
4. **Real-HOI4 fine-tuning and acceptance**, in parallel across PCs. Real HOI4 alone decides
   acceptance; simulator results never count as strength evidence.

## Repository status

| Area | State |
|---|---|
| `arena/contracts.py`, `actions.py`, `policy.py`, `learner.py`, `returns.py`, `trajectory.py`, `league.py`, `evaluation.py` | Exists, tested; bugs from the first review fixed. Needs the v0.1 input/action additions and a stack `count` field. |
| Vision stack (`perception/`, `io/`, `controller/`, `calibration.py`, `templates/`) | Exists for construction and research; to be extended to counters, bars and the army panel. |
| `brain/openai_compat.py` | Exists; DeepSeek is OpenAI-compatible. A Jev backend is to be added. |
| `native/`, `arena/protocol.py`, `arena/session.py`, `arena/probe.py` | Parked, not deleted. |
| Arena mod, executor for unit orders, save-file audit, LAN coordinator, simulator, recorder | To build. |

## Build order

1. Arena mod, perception-friendly, with reset decision and victory-point timeout.
2. Local counter/bar/army-panel reader; save-file accuracy gate.
3. Order executor with visual confirmation; order-success gate at each game speed.
4. Random and Cloud families against the AI panel: first end-to-end result, no training.
5. LAN coordinator; multiplayer stability gate (no desyncs with the mod, matching checksums).
6. Simulator, Jev distillation, human recorder, behaviour cloning, reinforcement learning.
7. Hierarchical family and ablations; then the end-to-end family.

## Open questions

- Pause and speed convention in LAN matches (host controls speed; pause is shared).
- Confirm in game that the lower counter bar is strength rather than supply.
- Multiplayer desync risk with a custom map mod.
- How closely the simulator must match HOI4 combat for policies to transfer.
- DeepSeek latency and pricing during its peak hours; behaviour under rate limits.
- Whether synthetic input can reach a non-foreground HOI4 window (would allow a human to use
  the agent's PC meanwhile; not required).

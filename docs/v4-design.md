# HOI4-AI v4 — design summary

The in-repo record of the load-bearing design ideas. (The original build plan
lived outside the repo; this summary stands alone.)

## The one principle

**Use the model for judgment, never for plumbing.** The VLM decides *what* from
pixels; deterministic code does *how*, *verify*, and all plumbing.

## What that buys us (vs v3)

1. **Typed, hotkey-first tool API** — the model emits closed-enum intents
   (`build_in_state(RUHR)`), never raw coordinates. Malformed/out-of-range
   actions are unrepresentable.
2. **Categorical-not-spatial** — spatial choices become a pick from a list; the
   executor owns the calibrated click-point. M1 needs ~zero live pixel grounding.
3. **Closed loop + action-as-assertion** — every tool re-perceives and asserts
   its effect (queue grew by 1) or raises a typed error. Bounded retry → reset
   macro → halt-and-flag. Never silent-loop. Nothing returns `None`.
4. **Verifier independent of the actor** — robust mode verifies with deterministic
   perception, not by re-asking the VLM that acted.

### Where (3) and (4) are weaker than they sound

Both are load-bearing claims, and both are currently qualified. Stated here so
nobody has to rediscover it from the source:

- **The assertion is cardinality, not identity.** `queue+1` holds whether the
  click landed on Ruhr or on Bavaria, and which state gets the factory is the
  whole content of the playbook. No perceived quantity distinguishes a correct
  state click from a wrong one, and `Calibration` carries no camera or zoom
  state. The only identity check that exists is offline:
  `save-audit --trace` pairs the trace's intended states against the save's
  construction lines. Until that runs, no claim about *where* the agent built
  is evidence-backed.
- **The verifying reads are the model's.** `construction_queue` and
  `idle_research_slots` are COUNTS of what a crop shows, not numbers the UI
  prints, so no glyph or OCR tier can produce them. "Deterministic verification"
  means deterministic *arithmetic* over a model read. Preflight names any such
  field M0 has not yet measured.
5. **Observed-not-assumed pause; date-driven cadence.**
6. **Externalized playbook memory** — an ordered, idempotent Germany-1936 goal
   queue with persisted plan-state, instead of re-deriving strategy each frame.
7. **Observability is the product** — one replayable JSONL trace per cycle; an
   offline replay harness re-runs saved frames through a new prompt/model.

## Model (researched June 2026)

Candidates are configured as named `[llm.profiles.*]` entries and switched by
name; `config/agent.toml` is the authority on which is active.

- **Shipped default: the `gemma4-cloud` profile** (`gemma4:31b-cloud`, Ollama's
  free cloud tier) — interim only. It is there so bring-up isn't blocked on VRAM,
  and it is to be retired once the M0 shootout (`eval --all-profiles`) names a
  **local** profile clearing ≥90% on the read tasks.
- **Local candidates**: `gemma4-local` (`gemma4:e4b-it-qat`, ~6 GB, native Ollama
  vision) and `qwen3vl-4b`. These carry the dominant load: read crops + pick from
  enums.
- **Grounding specialist: Holo 3.1** (`holo31-9b` / `holo31-4b`) — best open GUI
  grounders in the class, but Qwen-VL fine-tunes; Ollama mishandles imported
  Qwen-VL vision GGUFs, so they run via the OpenAI-compatible backend.
- M0 measures candidates on real HOI4 crops before committing.

## Build order

M0 (eval + scaffold; gates everything) → M1a Windows ctypes I/O + smoke test →
M1b perception + tools + calibrate → M1c controller + playbook + trace →
M1d live bring-up runbook.

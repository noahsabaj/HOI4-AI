# HOI4-AI v4 — design summary

The authoritative, detailed plan lives at
`.claude/plans/rebuild-the-entire-project-typed-wren.md`. This file is a short,
in-repo pointer plus the load-bearing ideas.

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
5. **Observed-not-assumed pause; date-driven cadence.**
6. **Externalized playbook memory** — an ordered, idempotent Germany-1936 goal
   queue with persisted plan-state, instead of re-deriving strategy each frame.
7. **Observability is the product** — one replayable JSONL trace per cycle; an
   offline replay harness re-runs saved frames through a new prompt/model.

## Model (researched June 2026)

- **Default: `gemma4:e4b`** (Google) — newest fitting multimodal, *native* Ollama
  vision + tool calling, ~8 GB-friendly. Carries our dominant load: read crops +
  pick from enums.
- **Deferred grounding specialist: `Hcompany/Holo1.5-7B`** (Apache-2.0) — best
  open GUI grounder in the class, but a Qwen2.5-VL fine-tune; Ollama mishandles
  imported Qwen-VL vision GGUFs, so run it via the OpenAI-compatible backend.
- M0 measures candidates on real HOI4 crops before committing.

## Build order

M0 (eval + scaffold; gates everything) → M1a Windows ctypes I/O + smoke test →
M1b perception + tools + calibrate → M1c controller + playbook + trace →
M1d live bring-up runbook.

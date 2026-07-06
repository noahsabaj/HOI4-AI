# HOI4-AI — working notes for coding agents

## Gates (all three before any commit)

```
python -m pytest tests/ && python -m ruff check . && python -m mypy hoi4_agent
```

Never pipe gate output through `head`/`tail` — the pipe eats the exit code. Chain
with `&&` or check exit codes explicitly.

## Standing conventions

- **Interactive operator flows reuse the step-wizard engine** in
  `hoi4_agent/cli/wizard.py` (`Step` + `Wizard`: Enter=capture, B/←=back,
  K=keep-on-revisit, S=skip, Q=abort-without-writing; results written only after the
  final step). Never build a raw `input()` prompt sequence — one operator mistake in a
  linear flow costs a restart, in the wizard it costs one recapture.
  `cli/calibrate.py` is the reference integration: pure step composition +
  `--only` subset filtering, unit-tested (`tests/test_wizard.py`); the single-key
  console I/O (`_read_event`, msvcrt with a line-based fallback) stays a thin separate
  layer. If the engine lacks a step kind (e.g. multi-select), extend `wizard.py` with
  tests rather than forking a bespoke loop.
- **Perception never guesses.** Unreadable/ambiguous reads return `None` (uncertain)
  at every tier (glyphs → OCR → VLM); a misread must never become a confident value.
- **Core dependencies stay minimal** (mss, Pillow, numpy, requests). No opencv /
  pydantic / pywin32 in core — optional extras only (`[ocr]`), enforced by a
  subprocess test.
- Claims in docs/comments must match mechanics; config options that are accepted but
  inert should fail loudly instead.

# HOI4-AI v4

A Windows-native, **closed-loop** AI agent that plays *Hearts of Iron IV* by sight.

Core principle: **use the model for judgment, never for plumbing.** A local vision
model decides *what* to do (which state to build in, which tech to research) from
screenshots; deterministic code handles *how* (hotkeys), *verifies* every action
actually happened, and manages all plumbing (menu state, the in-game date, pause).

This is a research instrument: every cycle is logged as a replayable JSONL trace,
and perception is measured offline before the live loop is trusted.

## Why a rebuild

v3 was a pure open-loop, coordinate-clicking, Linux/xdotool agent that could not
tell success from a softlock, swallowed every error, and was configured with a
model that does not exist. v4 inverts the fragile parts. See
[the design plan](../.claude/plans/rebuild-the-entire-project-typed-wren.md) and
[docs/v4-design.md](docs/v4-design.md).

## Requirements

- Windows 11, Python **3.11+** (tested on 3.14).
- Deps: `pip install -e .` (mss, Pillow, numpy, requests). `pip install -e ".[dev]"` for tests.
- A local VLM runtime: **Ollama** (default, native vision) with `gemma4:e4b`,
  or an OpenAI-compatible server (LM Studio / llama.cpp) for grounding models.
- Hearts of Iron IV running at **2560×1440 borderless** (configurable).

## Layout

```
hoi4_agent/      the package (errors, enums, schemas, config, geometry,
                 io, perception, brain, tools, controller, playbook, trace, eval, cli)
config/          agent.toml, playbooks/, (calibration.toml is generated)
templates/       calibration ROI template PNGs
tests/           offline test suite (mocked backends + fake LLM)
```

## Usage

```bash
# Offline (no game, no model needed):
python -m pytest                       # full test suite
python -m hoi4_agent.cli.main smoke-test --offline   # end-to-end with fakes

# Live (after installing HOI4 + Ollama):
python -m hoi4_agent.cli.main smoke-test   # validate the Windows I/O layer
python -m hoi4_agent.cli.main calibrate    # one-time: record ROIs + click-points
python -m hoi4_agent.cli.main eval         # M0: measure model perception on crops
python -m hoi4_agent.cli.main run          # play Germany-1936 construction + research
```

## License

See [LICENSE](LICENSE).

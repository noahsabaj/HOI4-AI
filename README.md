# HOI4-AI v3

Pure-vision AI agent that plays Hearts of Iron IV using a local VLM.

## Requirements

- Python 3.10+
- Ollama with `qwen3.5:35b` model
- xdotool (`sudo apt install xdotool`)
- HOI4 running on Linux (X11)

## Setup

```bash
pip install -r hoi4-ai-v3/requirements.txt
```

## Run

1. Launch HOI4, start a new game as Germany (1936)
2. Run the agent:

```bash
cd hoi4-ai-v3
python agent.py
```

3. Watch the agent play. Press Ctrl+C to stop.

## Architecture

Screenshot → Qwen3.5:35b (Ollama) → JSON action → xdotool → repeat

See `docs/superpowers/specs/2026-03-15-hoi4-ai-v3-design.md` for full design.

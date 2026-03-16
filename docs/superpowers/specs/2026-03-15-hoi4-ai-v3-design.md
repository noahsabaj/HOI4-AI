# HOI4-AI v3: Pure Vision Agent — Design Spec

## Overview

A pure-vision AI agent that plays Hearts of Iron IV autonomously using a single local VLM (Qwen3.5:35b via Ollama). The agent sees the game through screenshots, reasons about strategy, and executes actions through keyboard/mouse input — the same way a human plays.

**Milestone 1 scope:** Construction and research management only (Germany, 1936 start).

## Constraints

- **Entirely free** — no API costs, no cloud services
- **Runs locally** — RTX 4060 Ti (8GB VRAM), Linux (X11/Cinnamon)
- **No training** — uses Qwen3.5:35b as-is via Ollama
- **Pure vision** — no save file parsing, no memory reading, no OCR libraries. Screenshots only.
- **Single model** — Qwen3.5:35b handles both visual understanding and strategic reasoning

## Architecture

```
    ┌─────────────┐
    │  HOI4 Game   │ (paused)
    │  Screenshot  │───► mss (~5ms, zero GPU)
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │ Qwen3.5:35b  │  Sees screenshot.
    │  via Ollama   │  Understands game state.
    │  (MoE, 3B    │  Decides next action.
    │   active)     │  Outputs structured JSON.
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │   Executor    │  xdotool (primary)
    │  (zero GPU)   │  python-evdev (fallback)
    └──────────────┘
           │
           ▼
       Unpause, wait N game-days, repeat
```

## Core Loop

Each decision cycle:

1. Pause HOI4 (spacebar)
2. Screenshot the screen via `mss`
3. Send screenshot + system prompt to Qwen3.5:35b via Ollama API (`/api/chat` with image)
4. Model returns structured JSON action
5. Executor sends input via xdotool
6. Wait ~500ms for UI animation
7. If model needs more actions (multi-step menu navigation), repeat from step 2
8. When model returns `{"action": "done"}`, close any open menus (Escape), unpause (spacebar)
9. Wait for N in-game days at speed 4, then repeat

**Sub-step limit:** Max 15 actions per cycle to prevent infinite loops. When the limit is hit, send Escape twice (to close any open menus) before unpausing.

**Cycle timing:** After unpausing, the agent waits a fixed wall-clock duration (configurable, default ~10 seconds at speed 4, which approximates 7 game-days). This is intentionally simple — exact day-counting would require reading the date from the screen on every check. If timing drifts, the agent self-corrects on the next cycle by observing the actual date in the screenshot and adjusting its strategy accordingly.

**Error handling:** Invalid JSON → retry once → skip cycle (with Escape cleanup). If 3 consecutive cycles fail, the agent performs a full reset: spam Escape 3x, press F1 (return to default map mode), then resume.

**Logging:** Every screenshot saved as JPEG (smaller than PNG) + model response as JSON to `logs/cycle_{N}/`. Logs rotate: keep last 100 cycles, delete older ones to prevent disk bloat.

## Decision Cycle Example

Building a civilian factory in early 1936:

```
Sub-step 1: Screenshot (main map) → model: {"action": "key", "key": "w"}      → Open construction
Sub-step 2: Screenshot (menu open) → model: {"action": "click", "x": 340, "y": 280} → Select civ factory
Sub-step 3: Screenshot (civ selected) → model: {"action": "click", "x": 890, "y": 410} → Click Ruhr
Sub-step 4: Screenshot (factory queued) → model: {"action": "done"}            → End cycle, unpause
```

## Model Output Format

The model outputs exactly one JSON object per sub-step:

```json
{"action": "click", "x": 720, "y": 405, "description": "Click Ruhr state"}
```
```json
{"action": "key", "key": "w", "description": "Open construction menu"}
```
```json
{"action": "done", "description": "No more actions this cycle"}
```

The `description` field is for logging/debugging only.

## System Prompt

Stored in `prompts/system.md`. Contains:

1. **Role & objective** — Expert HOI4 player, Germany, 1936, maximize industry + tech
2. **Screen resolution & coordinate space** — The model receives screenshots resized to 1280x720. It outputs coordinates in that space. The system prompt states: "The screenshot is 1280x720. Output click coordinates in this space." The executor scales to window-relative coordinates: `win_x = (model_x / 1280) * window_width`. xdotool's `--window` flag handles the offset automatically.
3. **HOI4 hotkey reference** — W (construction), E (production), T (research), Space (pause), Escape (close menu), F1-F5 (map modes)
4. **Menu navigation knowledge** — How construction and research screens work, what to click and in what order
5. **Strategic principles (milestone 1):**
   - 1936-1937: Prioritize civilian factories
   - Target high-slot states: Ruhr, Saxony, Rhineland
   - Research priority: Industry → Electronics → Land Doctrine
   - Switch to military factories early-mid 1937
   - Never leave research slots or building queues idle
6. **Output format** — Strict JSON schema, one action per response

## Project Structure

```
hoi4-ai-v3/
├── requirements.txt         # mss, requests, pyyaml (+ python-evdev if needed)
├── config.yaml              # Resolution, Ollama endpoint, timing, game speed
├── agent.py                 # Main loop: pause → screenshot → think → act → unpause
├── vision.py                # Screenshot capture (mss) + base64 encoding for Ollama
├── executor.py              # Input dispatch via xdotool (or python-evdev fallback)
├── prompts/
│   └── system.md            # Full system prompt with HOI4 knowledge
└── logs/                    # Auto-generated: screenshots + model responses per cycle
```

### agent.py (~150 lines)
- Main decision loop
- Manages conversation history: within a cycle, keeps all sub-step screenshots + responses so the model remembers what it just did. Between cycles, history is cleared to avoid context bloat (each screenshot consumes significant tokens). The system prompt persists across all calls.
- Calls Ollama `/api/chat` endpoint with images
- Parses JSON responses (extracts JSON from model output, tolerates markdown fencing)
- Dispatches to executor
- Handles cycle limits, error recovery, and consecutive-failure resets

### vision.py (~50 lines)
- Screenshot via `mss`
- Find HOI4 game window via `xdotool search --name`
- Capture window region only (not full screen)
- Resize to model-friendly resolution (e.g., 1280x720)
- Encode to base64 for Ollama API

### executor.py (~80 lines)
- `execute_click(x, y)` — moves mouse and clicks via xdotool, coordinates relative to game window
- `execute_key(key)` — sends keystroke via xdotool
- Window offset and coordinate scaling: takes model coordinates (1280x720 space) and transforms to native window coordinates
- Fallback path for python-evdev: controlled by config flag `executor.backend: "xdotool" | "evdev"`. Not auto-detected — manually switched if xdotool doesn't work with HOI4

### config.yaml
```yaml
ollama:
  model: "qwen3.5:35b"
  endpoint: "http://localhost:11434"

game:
  window_title: "Hearts of Iron"
  speed: 4                    # Game speed during play (1-5)
  days_between_cycles: 7      # How many game-days to run between decisions

timing:
  action_delay_ms: 500        # Wait after each action for UI to respond
  max_substeps: 15            # Max actions per decision cycle
  cycle_pause_ms: 1000        # Wait after unpausing before timing game-days

display:
  capture_width: 1280         # Resize screenshot width for model
  capture_height: 720         # Resize screenshot height for model

executor:
  backend: "xdotool"          # "xdotool" or "evdev"

logging:
  max_cycles: 100             # Keep last N cycles, delete older
  screenshot_format: "jpeg"   # jpeg (small) or png (lossless)
```

## Latency Budget

Qwen3.5:35b is a 35B-parameter MoE model (3B active). On the RTX 4060 Ti with 8GB VRAM, Ollama will keep active parameters in VRAM and offload the rest to system RAM. Expected per-sub-step inference: **5-30 seconds** depending on prompt length and image complexity.

With up to 15 sub-steps per cycle (typically 3-5 for construction, 3-5 for research), a cycle takes **15-150 seconds**. This is acceptable because the game is paused during the entire cycle. Between cycles, the game runs unpaused for ~10 seconds.

**First implementation task:** Benchmark actual inference time with a single HOI4 screenshot before building the full loop. If latency exceeds 60s per sub-step, mitigation options: reduce capture resolution below 1280x720, use shorter system prompt, or switch to a faster quantization.

## Ollama API Integration

The agent calls Ollama's chat endpoint with vision:

```json
POST http://localhost:11434/api/chat
{
  "model": "qwen3.5:35b",
  "messages": [
    {"role": "system", "content": "<system prompt from prompts/system.md>"},
    {"role": "user", "content": "What is your next action?", "images": ["<base64 screenshot>"]}
  ],
  "stream": false,
  "format": {
    "type": "object",
    "properties": {
      "action": {"type": "string", "enum": ["click", "key", "done"]},
      "x": {"type": "integer"},
      "y": {"type": "integer"},
      "key": {"type": "string"},
      "description": {"type": "string"}
    },
    "required": ["action", "description"]
  },
  "options": {"temperature": 0}
}
```

The `"format"` parameter accepts a JSON schema (not just `"json"`) to enforce the exact action structure. Ollama constrains token generation to match the schema. The `temperature: 0` ensures deterministic output. The agent still validates the parsed JSON as a fallback.

## Input Execution (Linux)

**Primary: xdotool**
- `xdotool windowactivate --sync {wid}` — focus the game window
- `xdotool mousemove --window {wid} {x} {y} click --window {wid} 1` — window-relative click
- `xdotool key --window {wid} {key}` — send keystroke
- `xdotool getwindowgeometry --shell {wid}` — machine-parseable position/size
- Installed via `sudo apt install xdotool`
- Works on X11 (user's environment)

**Fallback: python-evdev + uinput**
- Kernel-level input injection if HOI4 rejects synthetic X11 events
- Requires uinput group membership or sudo
- Only used if xdotool fails

**Window management:**
- `xdotool search --name "Hearts of Iron"` to find window ID
- `xdotool getwindowgeometry {wid}` for position and size
- All coordinates offset relative to window origin

## Dependencies

```
mss          # Screenshot capture
requests     # Ollama API calls
pyyaml       # Config loading
Pillow       # Image resize and encoding
```

System: `xdotool`, Ollama with `qwen3.5:35b` model.

Note: python-evdev fallback is deferred past milestone 1. The config supports `executor.backend: "evdev"` but only `"xdotool"` is implemented initially.

No PyTorch. No transformers. No FAISS. No LMDB. Minimal.

## Success Criteria

**Competent (minimum bar):**
- AI navigates construction and research menus without getting stuck
- Queues factories and assigns research correctly
- Runs for 2+ in-game years without crashing or softlocking

**Good (human-level):**
- Prioritizes civilian factories before military (correct early-game meta)
- Picks high-slot states over low-slot ones
- Researches Industry and Electronics early
- Doesn't leave research slots empty

**Superhuman (the goal):**
- Optimally fills every building slot with zero idle time
- Perfect research timing — new tech starts the instant one finishes
- Never wastes a day of construction or research capacity
- Correct civ-to-mil switchover timing

## Future Milestones (Out of Scope for v3.0)

- **Milestone 2:** Production line management (equipment, templates)
- **Milestone 3:** Focus tree and political power decisions
- **Milestone 4:** Military — division deployment, front lines, battle plans
- **Milestone 5:** Diplomacy, trade, intelligence
- **Milestone 6:** Full game autonomy
- **Milestone 7:** Multi-country support, difficulty scaling

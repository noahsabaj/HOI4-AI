# HOI4-AI v3 Milestone 1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure-vision AI agent that manages construction and research in HOI4 using Qwen3.5:35b via Ollama.

**Architecture:** Screenshot → Ollama VLM → structured JSON action → xdotool execution. Game pauses during each decision cycle. Single model handles both visual understanding and strategic reasoning.

**Tech Stack:** Python 3.10+, mss, requests, pyyaml, xdotool, Ollama (qwen3.5:35b)

**Spec:** `docs/superpowers/specs/2026-03-15-hoi4-ai-v3-design.md`

---

## File Map

| File | Responsibility |
|---|---|
| `hoi4-ai-v3/requirements.txt` | Python dependencies |
| `hoi4-ai-v3/config.yaml` | All configuration (Ollama, timing, display, executor) |
| `hoi4-ai-v3/vision.py` | Screenshot capture, window detection, resize, base64 encode |
| `hoi4-ai-v3/executor.py` | xdotool input dispatch, coordinate scaling |
| `hoi4-ai-v3/agent.py` | Main loop, Ollama API calls, JSON parsing, cycle management |
| `hoi4-ai-v3/prompts/system.md` | System prompt with HOI4 knowledge |
| `tests/test_vision.py` | Tests for vision module |
| `tests/test_executor.py` | Tests for executor module |
| `tests/test_agent.py` | Tests for agent JSON parsing and cycle logic |

---

## Chunk 1: Project Scaffolding + Vision Module

### Task 1: Project scaffolding

**Files:**
- Create: `hoi4-ai-v3/requirements.txt`
- Create: `hoi4-ai-v3/config.yaml`

- [ ] **Step 1: Create requirements.txt**

```
mss>=9.0.0
requests>=2.31.0
pyyaml>=6.0
Pillow>=10.0.0
```

- [ ] **Step 2: Create config.yaml**

```yaml
ollama:
  model: "qwen3.5:35b"
  endpoint: "http://localhost:11434"

game:
  window_title: "Hearts of Iron"
  speed: 4
  cycle_wait_seconds: 10

timing:
  action_delay_ms: 500
  max_substeps: 15
  max_consecutive_failures: 3

display:
  capture_width: 1280
  capture_height: 720

executor:
  backend: "xdotool"

logging:
  max_cycles: 100
  screenshot_format: "jpeg"
```

- [ ] **Step 3: Install dependencies and verify**

Run: `cd /home/nsabaj/Code/HOI4-AI && pip install -r hoi4-ai-v3/requirements.txt`
Expected: All packages install successfully.

- [ ] **Step 4: Commit scaffolding**

```bash
git add hoi4-ai-v3/requirements.txt hoi4-ai-v3/config.yaml
git commit -m "feat: add v3 project scaffolding with config and dependencies"
```

---

### Task 2: Vision module — window detection and screenshot capture

**Files:**
- Create: `hoi4-ai-v3/vision.py`
- Create: `tests/test_vision.py`

- [ ] **Step 1: Write failing tests for vision module**

```python
# tests/test_vision.py
import subprocess
import pytest
from unittest.mock import patch, MagicMock
import vision


class TestFindGameWindow:
    def test_returns_window_info_when_found(self):
        mock_result = MagicMock()
        mock_result.stdout = "12345678\n"
        mock_result.returncode = 0

        mock_geo = MagicMock()
        mock_geo.stdout = "Window 12345678\n  Position: 100,200 (screen: 0)\n  Geometry: 1920x1080\n"
        mock_geo.returncode = 0

        with patch("subprocess.run", side_effect=[mock_result, mock_geo]):
            info = vision.find_game_window("Hearts of Iron")

        assert info["window_id"] == "12345678"
        assert info["x"] == 100
        assert info["y"] == 200
        assert info["width"] == 1920
        assert info["height"] == 1080

    def test_returns_none_when_not_found(self):
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            info = vision.find_game_window("Hearts of Iron")

        assert info is None


class TestCaptureScreenshot:
    def test_returns_base64_string(self):
        with patch.object(vision, "mss") as mock_mss_mod:
            mock_sct = MagicMock()
            mock_pixel_data = MagicMock()
            mock_pixel_data.rgb = b"\x00" * (100 * 100 * 3)
            mock_pixel_data.size = (100, 100)
            mock_pixel_data.width = 100
            mock_pixel_data.height = 100
            mock_sct.grab.return_value = mock_pixel_data
            mock_mss_mod.mss.return_value.__enter__ = MagicMock(return_value=mock_sct)
            mock_mss_mod.mss.return_value.__exit__ = MagicMock(return_value=False)

            window_info = {"x": 0, "y": 0, "width": 100, "height": 100}
            result = vision.capture_screenshot(window_info, 1280, 720)

        assert isinstance(result["base64"], str)
        assert len(result["base64"]) > 0
        assert result["width"] == 1280
        assert result["height"] == 720
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/nsabaj/Code/HOI4-AI && python -m pytest tests/test_vision.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vision'` (conftest.py doesn't exist yet)

- [ ] **Step 3: Implement vision.py**

```python
# hoi4-ai-v3/vision.py
"""Screenshot capture and window detection for HOI4-AI v3."""

import base64
import io
import subprocess
import re

import mss
from PIL import Image


def find_game_window(title: str) -> dict | None:
    """Find the HOI4 game window by title. Returns window info dict or None."""
    try:
        result = subprocess.run(
            ["xdotool", "search", "--name", title],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None

        window_id = result.stdout.strip().split("\n")[0]

        # Use --shell for machine-parseable output:
        # WINDOW=12345678
        # X=100
        # Y=200
        # WIDTH=1920
        # HEIGHT=1080
        # SCREEN=0
        geo = subprocess.run(
            ["xdotool", "getwindowgeometry", "--shell", window_id],
            capture_output=True, text=True, timeout=5
        )
        if geo.returncode != 0:
            return None

        # Parse KEY=VALUE lines
        vals = {}
        for line in geo.stdout.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                vals[k] = v

        if not all(k in vals for k in ("X", "Y", "WIDTH", "HEIGHT")):
            return None

        return {
            "window_id": window_id,
            "x": int(vals["X"]),
            "y": int(vals["Y"]),
            "width": int(vals["WIDTH"]),
            "height": int(vals["HEIGHT"]),
        }
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def capture_screenshot(
    window_info: dict, target_width: int = 1280, target_height: int = 720,
    fmt: str = "jpeg"
) -> dict:
    """Capture the game window and return resized base64-encoded image."""
    monitor = {
        "left": window_info["x"],
        "top": window_info["y"],
        "width": window_info["width"],
        "height": window_info["height"],
    }

    with mss.mss() as sct:
        raw = sct.grab(monitor)
        # Use BGRX raw format (more efficient than .rgb which copies + converts)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

    img = img.resize((target_width, target_height), Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG" if fmt == "jpeg" else "PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "base64": encoded,
        "width": target_width,
        "height": target_height,
        "image": img,
    }
```

- [ ] **Step 4: Add Python path setup so tests find the module**

Create `tests/conftest.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "hoi4-ai-v3"))
```

Create `hoi4-ai-v3/__init__.py` (empty file):

```python
# hoi4-ai-v3/__init__.py
```

Tests already use `import vision` so no import changes needed.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/nsabaj/Code/HOI4-AI && python -m pytest tests/test_vision.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit vision module**

```bash
git add hoi4-ai-v3/vision.py hoi4-ai-v3/__init__.py tests/test_vision.py tests/conftest.py
git commit -m "feat: add vision module with window detection and screenshot capture"
```

---

## Chunk 2: Executor Module

### Task 3: Executor — xdotool input dispatch with coordinate scaling

**Files:**
- Create: `hoi4-ai-v3/executor.py`
- Create: `tests/test_executor.py`

- [ ] **Step 1: Write failing tests for executor**

```python
# tests/test_executor.py
from unittest.mock import patch, call
import executor


class TestCoordinateScaling:
    def test_scales_model_coords_to_window_relative(self):
        window_info = {"window_id": "123", "x": 100, "y": 200, "width": 1920, "height": 1080}
        model_x, model_y = 640, 360  # center of 1280x720

        win_x, win_y = executor.scale_coordinates(
            model_x, model_y, 1280, 720, window_info
        )

        # Window-relative: 640/1280 * 1920 = 960, 360/720 * 1080 = 540
        # No window offset added (xdotool --window handles that)
        assert win_x == 960
        assert win_y == 540

    def test_origin_maps_to_zero(self):
        window_info = {"window_id": "123", "x": 50, "y": 75, "width": 1920, "height": 1080}

        win_x, win_y = executor.scale_coordinates(0, 0, 1280, 720, window_info)

        # Window-relative origin is (0,0) regardless of window position
        assert win_x == 0
        assert win_y == 0


class TestExecuteClick:
    @patch("subprocess.run")
    def test_calls_xdotool_with_scaled_coords(self, mock_run):
        window_info = {"window_id": "123", "x": 0, "y": 0, "width": 1280, "height": 720}

        executor.execute_click(640, 360, window_info, 1280, 720)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "xdotool" in cmd
        assert "mousemove" in cmd
        assert "click" in cmd


class TestExecuteKey:
    @patch("subprocess.run")
    def test_calls_xdotool_key(self, mock_run):
        window_info = {"window_id": "123", "x": 0, "y": 0, "width": 1280, "height": 720}

        executor.execute_key("w", window_info)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "xdotool" in cmd
        assert "key" in cmd


class TestDispatchAction:
    @patch.object(executor, "execute_click")
    def test_dispatches_click_action(self, mock_click):
        window_info = {"window_id": "123", "x": 0, "y": 0, "width": 1280, "height": 720}
        action = {"action": "click", "x": 500, "y": 300, "description": "test click"}

        executor.dispatch_action(action, window_info, 1280, 720)

        mock_click.assert_called_once_with(500, 300, window_info, 1280, 720)

    @patch.object(executor, "execute_key")
    def test_dispatches_key_action(self, mock_key):
        window_info = {"window_id": "123", "x": 0, "y": 0, "width": 1280, "height": 720}
        action = {"action": "key", "key": "w", "description": "test key"}

        executor.dispatch_action(action, window_info, 1280, 720)

        mock_key.assert_called_once_with("w", window_info)

    def test_done_action_returns_true(self):
        window_info = {"window_id": "123", "x": 0, "y": 0, "width": 1280, "height": 720}
        action = {"action": "done", "description": "end cycle"}

        result = executor.dispatch_action(action, window_info, 1280, 720)

        assert result is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/nsabaj/Code/HOI4-AI && python -m pytest tests/test_executor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'executor'`

- [ ] **Step 3: Implement executor.py**

```python
# hoi4-ai-v3/executor.py
"""Input execution for HOI4-AI v3. Translates model actions to xdotool commands."""

import subprocess
import time


def scale_coordinates(
    model_x: int, model_y: int,
    model_width: int, model_height: int,
    window_info: dict
) -> tuple[int, int]:
    """Scale coordinates from model space (1280x720) to window-relative coordinates.

    xdotool mousemove --window takes coordinates relative to the window origin,
    so we only scale by the resolution ratio — no window offset needed.
    """
    win_x = int((model_x / model_width) * window_info["width"])
    win_y = int((model_y / model_height) * window_info["height"])
    return win_x, win_y


def execute_click(
    model_x: int, model_y: int,
    window_info: dict,
    model_width: int = 1280, model_height: int = 720
) -> None:
    """Move mouse to scaled window-relative coordinates and click."""
    win_x, win_y = scale_coordinates(
        model_x, model_y, model_width, model_height, window_info
    )
    wid = window_info["window_id"]
    # Activate window, move mouse relative to it, and click
    subprocess.run(["xdotool", "windowactivate", "--sync", wid], timeout=5)
    subprocess.run(
        ["xdotool", "mousemove", "--window", wid, str(win_x), str(win_y),
         "click", "--window", wid, "1"],
        timeout=5
    )


def execute_key(key: str, window_info: dict) -> None:
    """Send a keystroke to the game window."""
    subprocess.run(
        ["xdotool", "key", "--window", window_info["window_id"], key],
        timeout=5
    )


def escape_cleanup(window_info: dict, presses: int = 2) -> None:
    """Press Escape N times to close any open menus."""
    for _ in range(presses):
        execute_key("Escape", window_info)
        time.sleep(0.3)


def full_reset(window_info: dict) -> None:
    """Full reset: spam Escape, return to default map mode."""
    escape_cleanup(window_info, presses=3)
    execute_key("F1", window_info)
    time.sleep(0.5)


def dispatch_action(
    action: dict, window_info: dict,
    model_width: int = 1280, model_height: int = 720
) -> bool:
    """Execute a single action. Returns True if action is 'done'."""
    action_type = action.get("action", "")

    if action_type == "click":
        execute_click(action["x"], action["y"], window_info, model_width, model_height)
        return False

    elif action_type == "key":
        execute_key(action["key"], window_info)
        return False

    elif action_type == "done":
        return True

    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/nsabaj/Code/HOI4-AI && python -m pytest tests/test_executor.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit executor module**

```bash
git add hoi4-ai-v3/executor.py tests/test_executor.py
git commit -m "feat: add executor module with xdotool dispatch and coordinate scaling"
```

---

## Chunk 3: Agent Core + System Prompt

### Task 4: Agent — JSON parsing and Ollama API integration

**Files:**
- Create: `hoi4-ai-v3/agent.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write failing tests for JSON parsing**

```python
# tests/test_agent.py
import json
import pytest
from unittest.mock import patch, MagicMock
import agent


class TestParseModelResponse:
    def test_parses_clean_json(self):
        raw = '{"action": "click", "x": 100, "y": 200, "description": "test"}'
        result = agent.parse_model_response(raw)
        assert result == {"action": "click", "x": 100, "y": 200, "description": "test"}

    def test_extracts_json_from_markdown_fenced(self):
        raw = 'Here is my action:\n```json\n{"action": "key", "key": "w", "description": "open menu"}\n```'
        result = agent.parse_model_response(raw)
        assert result == {"action": "key", "key": "w", "description": "open menu"}

    def test_extracts_json_from_surrounding_text(self):
        raw = 'I will click here: {"action": "click", "x": 50, "y": 60, "description": "click"} done.'
        result = agent.parse_model_response(raw)
        assert result["action"] == "click"
        assert result["x"] == 50

    def test_returns_none_on_garbage(self):
        raw = "I don't know what to do"
        result = agent.parse_model_response(raw)
        assert result is None

    def test_returns_none_on_empty(self):
        result = agent.parse_model_response("")
        assert result is None


class TestCallOllama:
    @patch("requests.post")
    def test_sends_correct_request(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": '{"action": "done", "description": "nothing to do"}'}
        }
        mock_post.return_value = mock_resp

        result = agent.call_ollama(
            endpoint="http://localhost:11434",
            model="qwen3.5:35b",
            system_prompt="You are an expert.",
            messages=[{"role": "user", "content": "What next?", "images": ["abc123"]}]
        )

        assert result == '{"action": "done", "description": "nothing to do"}'
        mock_post.assert_called_once()
        body = mock_post.call_args[1]["json"]
        assert body["model"] == "qwen3.5:35b"
        assert body["stream"] is False
        assert body["format"]["type"] == "object"  # JSON schema, not just "json"
        assert "action" in body["format"]["properties"]
        assert body["messages"][0]["role"] == "system"

    @patch("requests.post")
    def test_returns_none_on_http_error(self, mock_post):
        mock_post.side_effect = Exception("connection refused")

        result = agent.call_ollama(
            endpoint="http://localhost:11434",
            model="qwen3.5:35b",
            system_prompt="test",
            messages=[]
        )

        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/nsabaj/Code/HOI4-AI && python -m pytest tests/test_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent'`

- [ ] **Step 3: Implement agent.py**

```python
# hoi4-ai-v3/agent.py
"""Main agent loop for HOI4-AI v3."""

import json
import os
import re
import shutil
import time
from pathlib import Path

import requests
import yaml

import vision
import executor


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_system_prompt(path: str = "prompts/system.md") -> str:
    """Load the system prompt from file."""
    with open(path) as f:
        return f.read()


def parse_model_response(raw: str) -> dict | None:
    """Extract JSON action from model response. Tolerates markdown fencing and surrounding text."""
    if not raw or not raw.strip():
        return None

    # Try direct parse first
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding JSON object in text
    brace_match = re.search(r"\{[^{}]*\}", raw)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# JSON schema enforcing our exact action format.
# Ollama's structured output mode will constrain the model to this schema.
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["click", "key", "done"]
        },
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "key": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["action", "description"],
}


def call_ollama(
    endpoint: str, model: str, system_prompt: str, messages: list
) -> str | None:
    """Call Ollama chat API with vision. Returns raw response content or None."""
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    try:
        resp = requests.post(
            f"{endpoint}/api/chat",
            json={
                "model": model,
                "messages": full_messages,
                "stream": False,
                "format": ACTION_SCHEMA,
                "options": {"temperature": 0},
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception:
        return None


def save_cycle_log(
    log_dir: Path, cycle_num: int, substep: int,
    screenshot_b64: str, response: str, fmt: str = "jpeg"
) -> None:
    """Save screenshot and model response for debugging."""
    import base64

    cycle_dir = log_dir / f"cycle_{cycle_num:04d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    ext = "jpg" if fmt == "jpeg" else "png"
    img_path = cycle_dir / f"step_{substep:02d}.{ext}"
    with open(img_path, "wb") as f:
        f.write(base64.b64decode(screenshot_b64))

    resp_path = cycle_dir / f"step_{substep:02d}.json"
    with open(resp_path, "w") as f:
        f.write(response or "null")


def rotate_logs(log_dir: Path, max_cycles: int) -> None:
    """Remove oldest cycle logs if over the limit."""
    cycle_dirs = sorted(log_dir.glob("cycle_*"))
    while len(cycle_dirs) > max_cycles:
        shutil.rmtree(cycle_dirs.pop(0))


def run_cycle(
    config: dict, system_prompt: str, window_info: dict,
    cycle_num: int, log_dir: Path
) -> bool:
    """Run one decision cycle. Returns True if successful, False on failure."""
    messages = []
    cfg_display = config["display"]
    cfg_timing = config["timing"]
    cfg_ollama = config["ollama"]
    cfg_logging = config["logging"]

    for substep in range(cfg_timing["max_substeps"]):
        # Screenshot
        shot = vision.capture_screenshot(
            window_info, cfg_display["capture_width"], cfg_display["capture_height"],
            fmt=cfg_logging["screenshot_format"]
        )

        # Build message with image
        user_msg = {
            "role": "user",
            "content": "Here is the current game screenshot. What is your next action?",
            "images": [shot["base64"]],
        }
        messages.append(user_msg)

        # Call model
        raw_response = call_ollama(
            cfg_ollama["endpoint"], cfg_ollama["model"],
            system_prompt, messages
        )

        # Log
        save_cycle_log(
            log_dir, cycle_num, substep,
            shot["base64"], raw_response or "", cfg_logging["screenshot_format"]
        )

        if raw_response is None:
            return False

        # Parse response
        action = parse_model_response(raw_response)
        if action is None:
            # Retry once
            raw_response = call_ollama(
                cfg_ollama["endpoint"], cfg_ollama["model"],
                system_prompt, messages
            )
            if raw_response:
                action = parse_model_response(raw_response)
            if action is None:
                return False

        # Track assistant response in conversation
        messages.append({"role": "assistant", "content": json.dumps(action)})

        # Execute
        is_done = executor.dispatch_action(
            action, window_info,
            cfg_display["capture_width"], cfg_display["capture_height"]
        )

        if is_done:
            return True

        # Wait for UI
        time.sleep(cfg_timing["action_delay_ms"] / 1000)

    # Hit substep limit — cleanup
    executor.escape_cleanup(window_info)
    return True


def main() -> None:
    """Main entry point. Runs the agent loop."""
    os.chdir(Path(__file__).parent)

    config = load_config()
    system_prompt = load_system_prompt()
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    print("HOI4-AI v3 — Waiting for game window...")
    window_info = None
    while window_info is None:
        window_info = vision.find_game_window(config["game"]["window_title"])
        if window_info is None:
            time.sleep(2)

    print(f"Found game window: {window_info}")
    print("Starting agent loop. Press Ctrl+C to stop.")

    cycle_num = 0
    consecutive_failures = 0

    try:
        while True:
            # Pause the game
            executor.execute_key("space", window_info)
            time.sleep(0.5)

            # Run decision cycle
            success = run_cycle(config, system_prompt, window_info, cycle_num, log_dir)

            if success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                print(f"Cycle {cycle_num} failed ({consecutive_failures} consecutive)")
                executor.escape_cleanup(window_info)

                if consecutive_failures >= config["timing"]["max_consecutive_failures"]:
                    print("Too many failures — performing full reset")
                    executor.full_reset(window_info)
                    consecutive_failures = 0

            # Unpause and wait
            executor.execute_key("space", window_info)
            time.sleep(config["game"]["cycle_wait_seconds"])

            # Rotate logs
            rotate_logs(log_dir, config["logging"]["max_cycles"])

            cycle_num += 1

    except KeyboardInterrupt:
        print("\nAgent stopped.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/nsabaj/Code/HOI4-AI && python -m pytest tests/test_agent.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit agent module**

```bash
git add hoi4-ai-v3/agent.py tests/test_agent.py
git commit -m "feat: add agent module with Ollama integration and decision loop"
```

---

### Task 5: System prompt

**Files:**
- Create: `hoi4-ai-v3/prompts/system.md`

- [ ] **Step 1: Create prompts directory and write the system prompt**

Run: `mkdir -p /home/nsabaj/Code/HOI4-AI/hoi4-ai-v3/prompts`

```markdown
You are an expert Hearts of Iron IV player controlling Germany from the 1936 start. Your goal is to maximize industrial output and technological advantage.

## Your Capabilities

You see a screenshot of the game (1280x720 resolution). You decide the next action and output it as a single JSON object. The game is paused while you think.

## Output Format

You MUST output exactly one JSON object per response. No other text. Pick one:

Click somewhere:
{"action": "click", "x": <0-1280>, "y": <0-720>, "description": "<what you're clicking>"}

Press a key:
{"action": "key", "key": "<key>", "description": "<why>"}

End this decision cycle (nothing more to do right now):
{"action": "done", "description": "<summary of what you accomplished>"}

## Coordinate Space

The screenshot is 1280x720 pixels. Top-left is (0,0). Bottom-right is (1280,720). Output click coordinates in this space.

## HOI4 Hotkeys

- W — Open construction menu
- T — Open research screen
- Escape — Close current menu / go back
- F1 — Default map mode
- Space — Pause / unpause (the agent handles this, you don't need to)

## Menu Navigation

### Construction (W)
1. Press W to open the construction menu
2. On the left panel, click the building type (civilian factory, military factory, etc.)
3. Click on a state on the map to queue construction there
4. States with more free building slots are better targets
5. You can queue multiple buildings by clicking multiple states

### Research (T)
1. Press T to open the research screen
2. You see rows of technology icons organized by category
3. Click on an available (non-greyed-out) technology to assign a research slot
4. You have limited research slots — fill them all, never leave one idle
5. Technologies with a bonus (green icon/reduced time) should be prioritized

## Strategic Principles (1936-1938)

### Construction Priority
- **1936 to mid-1937:** Build CIVILIAN factories. You need economic base first.
  - Best states: Ruhr (most slots), Saxony, Rhineland, Westfalen
  - Pick states with the most available building slots
- **Mid-1937 onward:** Switch to MILITARY factories
  - Same priority: high-slot states first
- **Never leave your construction queue idle.** If a factory finishes, queue another immediately.

### Research Priority
1. Industry technologies (Construction I, II, III — faster building)
2. Electronics (Radar, encryption — research speed bonus)
3. Land Doctrine (Superior Firepower or Mobile Warfare)
4. Infantry equipment and artillery upgrades
- **Never leave a research slot empty.** Always have something researching.

## Decision Making

Each cycle, look at the screenshot and decide:
1. Is there a construction slot available? → Open construction (W), queue a factory
2. Is there a research slot empty? → Open research (T), assign a tech
3. Is everything already queued and researching? → {"action": "done"}

Check construction FIRST, then research. If both are full, you're done for this cycle.

## Important Rules

- Output ONLY the JSON object. No explanation, no commentary.
- One action per response. You will get a new screenshot after each action.
- If you are unsure what you see, describe what you observe in the "description" field and output {"action": "done"} to avoid misclicks.
- Click precisely on UI elements you can clearly identify.
```

- [ ] **Step 2: Commit system prompt**

```bash
git add hoi4-ai-v3/prompts/system.md
git commit -m "feat: add system prompt with HOI4 construction and research knowledge"
```

---

## Chunk 4: Integration and First Run

### Task 6: Latency benchmark

**Files:**
- None (manual test)

- [ ] **Step 1: Start Ollama and verify model is loaded**

Run: `ollama list | grep qwen3.5`
Expected: Shows `qwen3.5:35b` in the list.

- [ ] **Step 2: Run a quick benchmark with a test image**

Create a one-off test script:

```python
# hoi4-ai-v3/benchmark.py
"""Quick latency benchmark for Ollama vision inference."""
import base64
import time
import requests
from PIL import Image
import io

# Create a test image (solid color, simulating a game screenshot)
img = Image.new("RGB", (1280, 720), color=(50, 50, 80))
buf = io.BytesIO()
img.save(buf, format="JPEG")
b64 = base64.b64encode(buf.getvalue()).decode()

system = "You are an expert HOI4 player. Output a single JSON action."
prompt = "What do you see in this screenshot? If unsure, output a done action."

# Same schema the agent will use
action_schema = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["click", "key", "done"]},
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "key": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["action", "description"],
}

print("Sending request to Ollama...")
start = time.time()

resp = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "qwen3.5:35b",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt, "images": [b64]},
        ],
        "stream": False,
        "format": action_schema,
        "options": {"temperature": 0},
    },
    timeout=300,
)

elapsed = time.time() - start
print(f"Status: {resp.status_code}")
print(f"Latency: {elapsed:.1f}s")
print(f"Response: {resp.json()['message']['content'][:200]}")
```

Run: `cd /home/nsabaj/Code/HOI4-AI/hoi4-ai-v3 && python benchmark.py`
Expected: Response within 5-60 seconds. Note the actual latency. If >60s, we'll need to reduce resolution or prompt length.

- [ ] **Step 3: Record latency and adjust config if needed**

If latency > 30s: reduce `display.capture_width` to 960 and `display.capture_height` to 540 in config.yaml.
If latency > 60s: also shorten the system prompt and consider a smaller quantization of the model.
If latency < 30s: proceed as-is.

- [ ] **Step 4: Commit benchmark script**

```bash
git add hoi4-ai-v3/benchmark.py
git commit -m "feat: add Ollama latency benchmark script"
```

---

### Task 7: End-to-end integration test (without game running)

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test that mocks everything**

```python
# tests/test_integration.py
"""Integration test for the full agent cycle (all components mocked)."""
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import agent


class TestFullCycle:
    def test_complete_cycle_with_done_action(self, tmp_path):
        """Simulate a cycle where model immediately says done."""
        config = {
            "ollama": {"model": "qwen3.5:35b", "endpoint": "http://localhost:11434"},
            "display": {"capture_width": 1280, "capture_height": 720},
            "timing": {"max_substeps": 15, "action_delay_ms": 0},
            "logging": {"max_cycles": 10, "screenshot_format": "jpeg"},
        }
        window_info = {"window_id": "123", "x": 0, "y": 0, "width": 1280, "height": 720}
        system_prompt = "You are a test agent."

        mock_shot = {"base64": "dGVzdA==", "width": 1280, "height": 720, "image": None}
        mock_response = '{"action": "done", "description": "nothing to do"}'

        with patch("agent.vision.capture_screenshot", return_value=mock_shot), \
             patch("agent.call_ollama", return_value=mock_response), \
             patch("agent.executor.dispatch_action", return_value=True):

            result = agent.run_cycle(config, system_prompt, window_info, 0, tmp_path)

        assert result is True

    def test_multi_step_cycle(self, tmp_path):
        """Simulate a cycle with click then done."""
        config = {
            "ollama": {"model": "qwen3.5:35b", "endpoint": "http://localhost:11434"},
            "display": {"capture_width": 1280, "capture_height": 720},
            "timing": {"max_substeps": 15, "action_delay_ms": 0},
            "logging": {"max_cycles": 10, "screenshot_format": "jpeg"},
        }
        window_info = {"window_id": "123", "x": 0, "y": 0, "width": 1280, "height": 720}

        mock_shot = {"base64": "dGVzdA==", "width": 1280, "height": 720, "image": None}
        responses = [
            '{"action": "key", "key": "w", "description": "open construction"}',
            '{"action": "click", "x": 400, "y": 300, "description": "select civ factory"}',
            '{"action": "done", "description": "factory queued"}',
        ]

        with patch("agent.vision.capture_screenshot", return_value=mock_shot), \
             patch("agent.call_ollama", side_effect=responses), \
             patch("agent.executor.dispatch_action", side_effect=[False, False, True]):

            result = agent.run_cycle(config, "test", window_info, 0, tmp_path)

        assert result is True
```

- [ ] **Step 2: Run integration tests**

Run: `cd /home/nsabaj/Code/HOI4-AI && python -m pytest tests/test_integration.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit integration tests**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full agent cycle"
```

---

### Task 8: First live run against HOI4

**Files:**
- None (manual test)

- [ ] **Step 1: Launch HOI4 and start a new game as Germany (1936)**

Start the game, get to the main gameplay screen with the map visible. Pause the game.

- [ ] **Step 2: Run the agent**

Run: `cd /home/nsabaj/Code/HOI4-AI/hoi4-ai-v3 && python agent.py`
Expected: Agent finds the game window, begins decision cycles. Watch the logs/ directory and terminal output.

- [ ] **Step 3: Observe 3-5 cycles**

Watch the agent:
- Does it correctly pause/unpause?
- Does it open the construction menu?
- Does it click reasonable locations?
- Does it return to the map after menu interactions?
- Check `logs/cycle_0000/` for screenshots and responses.

- [ ] **Step 4: Note issues and iterate on system prompt**

Common first-run issues:
- Model outputs explanation text instead of JSON → tighten system prompt
- Click coordinates are wrong → check coordinate scaling in executor
- Model doesn't understand HOI4 UI → add more specific navigation hints to prompt
- Model gets stuck in a menu → verify Escape cleanup works

Document findings and adjust `prompts/system.md` accordingly.

- [ ] **Step 5: Commit any prompt/config adjustments**

```bash
git add -A hoi4-ai-v3/
git commit -m "fix: adjust system prompt and config based on first live run"
```

---

## Chunk 5: Polish and Cleanup

### Task 9: Update .gitignore and project README

**Files:**
- Modify: `.gitignore`
- Create: `README.md`

- [ ] **Step 1: Update .gitignore for v3**

Add to `.gitignore`:
```
hoi4-ai-v3/logs/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 2: Write minimal README**

```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore README.md
git commit -m "docs: add README and update gitignore for v3"
```

---

### Task 10: Final commit — clean slate with all v3 code

- [ ] **Step 1: Run all tests one final time**

Run: `cd /home/nsabaj/Code/HOI4-AI && python -m pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 2: Verify project structure is clean**

Run: `find hoi4-ai-v3/ -type f | sort`
Expected:
```
hoi4-ai-v3/__init__.py
hoi4-ai-v3/agent.py
hoi4-ai-v3/benchmark.py
hoi4-ai-v3/config.yaml
hoi4-ai-v3/executor.py
hoi4-ai-v3/prompts/system.md
hoi4-ai-v3/requirements.txt
hoi4-ai-v3/vision.py
```

- [ ] **Step 3: Create a tagged commit for v3.0-milestone1**

```bash
git tag -a v3.0-m1 -m "HOI4-AI v3 Milestone 1: Construction + Research agent"
```

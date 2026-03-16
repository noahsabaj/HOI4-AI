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
            "enum": ["click", "key", "done", "unpause"]
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

    if not executor.check_xdotool():
        print("ERROR: xdotool is not installed. Install with: sudo apt install xdotool")
        return

    print(f"Found game window: {window_info}")
    print("Starting agent loop. Press Ctrl+C to stop.")
    print("Pause the game before starting — the agent takes over from there.")

    cycle_num = 0
    consecutive_failures = 0

    try:
        while True:
            # First cycle: game should already be paused by the user.
            # Subsequent cycles: we just paused it at the bottom of the loop.
            # Either way, game is paused here — run the decision cycle.
            print(f"\n--- Cycle {cycle_num} ---")

            success = run_cycle(config, system_prompt, window_info, cycle_num, log_dir)

            if success:
                consecutive_failures = 0
                print(f"Cycle {cycle_num} complete.")
            else:
                consecutive_failures += 1
                print(f"Cycle {cycle_num} failed ({consecutive_failures} consecutive)")
                executor.escape_cleanup(window_info)

                if consecutive_failures >= config["timing"]["max_consecutive_failures"]:
                    print("Too many failures — performing full reset")
                    executor.full_reset(window_info)
                    consecutive_failures = 0

            # Unpause, let the game run, then pause again
            executor.execute_key("space", window_info)  # unpause
            print(f"Game running for {config['game']['cycle_wait_seconds']}s...")
            time.sleep(config["game"]["cycle_wait_seconds"])
            executor.execute_key("space", window_info)  # pause
            time.sleep(0.5)

            # Rotate logs
            rotate_logs(log_dir, config["logging"]["max_cycles"])

            cycle_num += 1

    except KeyboardInterrupt:
        print("\nAgent stopped.")


if __name__ == "__main__":
    main()

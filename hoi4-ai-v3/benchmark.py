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
        "model": "qwen3.5:9b",
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

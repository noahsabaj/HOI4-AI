"""Run several LIVE bring-up matches: relaunch + session camera + one match, per agent and repetition."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
out_dir = ROOT / "artifacts/live_matches"
out_dir.mkdir(parents=True, exist_ok=True)
agents, repeats, speed_ups = sys.argv[1].split(","), int(sys.argv[2]), int(sys.argv[3])
summary = []
for repeat in range(repeats):
    for agent in agents:
        name = f"{agent.replace(':', '-')}_{repeat + 1}"
        session = out_dir / f"{name}_camera.json"
        setup = subprocess.run([PY, "scripts/live_setup.py", "artifacts/previews/camera.json", str(session)],
                               cwd=ROOT, capture_output=True, text=True)
        if setup.returncode != 0:
            summary.append({"match": name, "error": "setup: " + setup.stdout.strip().splitlines()[-1]})
            print(summary[-1], flush=True)
            continue
        subprocess.run([PY, "scripts/live_match.py", agent, str(session), str(out_dir / f"{name}.json"), "12",
                        str(speed_ups)], cwd=ROOT, capture_output=True, text=True)
        result = json.loads((out_dir / f"{name}.json").read_text())
        records = result["records"]
        summary.append({"match": name, "outcome": result["outcome"], "decisions": result["decisions"],
                        "orders_sent": sum(r["executed"] == "sent" for r in records),
                        "blind_frames": sum(r["own"] == 0 and r["enemy_visible"] == 0 for r in records),
                        "minutes": result["minutes"]})
        print(summary[-1], flush=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
subprocess.run(["taskkill", "/IM", "hoi4.exe", "/F"], capture_output=True)

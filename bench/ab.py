"""Run several bench_decide.py runs in turn in one job: `ab.py a|b label[:opt=json[;opt=json]] ...`.

Each variant runs in its own process, one after another, so they share the GPU's
conditions of the moment as closely as separate processes can.
"""

import json
import subprocess
import sys

tower = sys.argv[1]
checkpoint = {"a": "artifacts/learned/bc5/epoch-0000.pt", "b": "artifacts/bench/bc5-4b.pt"}[tower]
for spec in sys.argv[2:]:
    label, _, opts = spec.partition(":")
    command = [
        sys.executable,
        "scripts/bench_decide.py",
        "run",
        "--checkpoint",
        checkpoint,
        "--label",
        f"{tower}-{label}",
        "--reference",
        f"bench/reference-{tower}.pt",
        "--output",
        f"bench/results/{tower}-{label}.json",
    ]
    for opt in filter(None, opts.split(";")):
        command += ["--option", opt]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode:
        print(label, "FAILED", done.stderr[-3000:], flush=True)
        continue
    r = json.loads(done.stdout[done.stdout.index("{") :])
    g = r.get("gate", {})
    p = r.get("profile") if isinstance(r.get("profile"), dict) else {}
    print(
        f"{r['label']}: b2b {r['back_to_back']['p50_ms']}/{r['back_to_back']['p95_ms']} "
        f"paced {r['paced']['p50_ms']}/{r['paced']['p95_ms']} gpu {p.get('gpu_ms')} "
        f"launches {p.get('launches')} syncs {r.get('syncs_per_decision')} "
        f"gate pass={g.get('pass')} same={g.get('same_actions')} agree={g.get('slot_agreement')} exact={g.get('exact')} diff={g.get('max_diff')} "
        f"split={r.get('split_ms')} load={r['load_s']} warm={r['warm_s']} peak={r['peak_mib']}",
        flush=True,
    )

"""Run the whole held-out TEST phase in order (resumable; safe to restart at any time).

  python -m eval.run_test_phase

Order: B1, B2, v2, v1 (so the key results exist first), then the v2 stability repeat on a fixed random subset
(gateway cache bypassed), then the v2 cache rerun on another fixed subset. Each step is `python -m eval.run`
with --results-dir results/v2; finished scenarios are skipped, API failures are retried.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = "inject/manifest_test.json"
BASE = ["--manifest", M, "--results-dir", "results/v2"]
STEPS = [
    ("B1", ["--model", "baseline_topcontrib", "--run-name", "test_b1"], {}),
    ("B2", ["--model", "baseline_scan", "--run-name", "test_b2"], {}),
    ("v2", ["--model", "nvidia", "--agent-version", "v2", "--run-name", "test_v2"], {}),
    ("v1", ["--model", "nvidia", "--agent-version", "v1", "--run-name", "test_v1"], {}),
    ("v2 repeat", ["--model", "nvidia", "--agent-version", "v2", "--run-name", "test_v2_repeat", "--no-cache",
                   "--only-file", "results/v2/repeat_ids.json"], {}),
    # cache hits never reach NVIDIA, so the client may go at the gateway's per-key pace
    ("v2 cache rerun", ["--model", "nvidia", "--agent-version", "v2", "--run-name", "test_v2", "--cache-rerun",
                        "--only-file", "results/v2/cache_ids.json"], {"NVIDIA_REQUESTS_PER_MIN": "20"}),
]


def main():
    for name, args, env in STEPS:
        # rerun each step until it has no API failures left (max 3 passes), so outages do not leave gaps
        for attempt in range(3):
            print(f"=== {name} (pass {attempt + 1})", flush=True)
            r = subprocess.run([sys.executable, "-u", "-m", "eval.run", *args, *BASE], cwd=ROOT,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8", **env})
            out = subprocess.run([sys.executable, "-c", (
                "import json,sys;from pathlib import Path;"
                f"p=Path('results/v2/{'runs_cache' if '--cache-rerun' in args else 'runs'}/{args[args.index('--run-name') + 1]}.jsonl');"
                "rows={};[rows.__setitem__(json.loads(l)['id'],json.loads(l)) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()] if p.exists() else None;"
                "print(sum(1 for r in rows.values() if str(r.get('error') or '').startswith('llm error')))")],
                cwd=ROOT, capture_output=True, text=True)
            left = int((out.stdout.strip() or "0").splitlines()[-1])
            print(f"=== {name}: exit {r.returncode}, API failures left {left}", flush=True)
            if left == 0:
                break
    print("=== TEST phase complete", flush=True)


if __name__ == "__main__":
    main()

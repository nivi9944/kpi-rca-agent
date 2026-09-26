"""Run the whole held-out TEST phase in order (resumable; safe to restart at any time).

  python -m eval.run_test_phase

Order: v2, v1, the v2 stability repeat (fixed random subset, gateway cache bypassed), then the v2 cache rerun
(another fixed subset). B1 and B2 are run separately (seconds). Each step is `python -m eval.run` with
--results-dir results/v2 and --workers N, where N is read from results/v2/workers.txt before each step.

Budget guard (time budget 18 h from the phase start, recorded in results/v2/phase_start.txt):
- during every LLM step, every 5 min: if a new API failure appears, or upstream 429s exceed 3 per completed
  scenario over the last 15 min, the step is stopped, workers drop to 1, an alert is written to
  results/v2/ALERT.txt, and the step resumes;
- after v2: if elapsed + projected remaining time exceeds the budget and 429s stayed under 1 per scenario
  during v2, workers go up to 3.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results" / "v2"
M = "inject/manifest_test.json"
BASE = ["--manifest", M, "--results-dir", "results/v2"]
BUDGET_H = 18.0
WORKERS = V2 / "workers.txt"
ALERT = V2 / "ALERT.txt"
START = V2 / "phase_start.txt"
LOG = V2 / "budget_guard.log"
STEPS = [
    ("v2", ["--model", "nvidia", "--agent-version", "v2", "--run-name", "test_v2"], {}, 170),
    ("v1", ["--model", "nvidia", "--agent-version", "v1", "--run-name", "test_v1"], {}, 170),
    ("v2 repeat", ["--model", "nvidia", "--agent-version", "v2", "--run-name", "test_v2_repeat", "--no-cache",
                   "--only-file", "results/v2/repeat_ids.json"], {}, 50),
    # cache hits never reach NVIDIA, so the client may go at the gateway's per-key pace
    ("v2 cache rerun", ["--model", "nvidia", "--agent-version", "v2", "--run-name", "test_v2", "--cache-rerun",
                        "--only-file", "results/v2/cache_ids.json"], {"NVIDIA_REQUESTS_PER_MIN": "20"}, 30),
]


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def workers() -> int:
    try:
        return max(1, int(WORKERS.read_text().strip()))
    except (OSError, ValueError):
        return 2


def set_workers(n: int, why: str) -> None:
    WORKERS.write_text(str(n))
    log(f"workers -> {n}: {why}")


def run_file(args: list[str]) -> Path:
    name = args[args.index("--run-name") + 1]
    return V2 / ("runs_cache" if "--cache-rerun" in args else "runs") / f"{name}.jsonl"


def rows(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def api_failures(path: Path) -> int:
    return sum(1 for r in rows(path).values() if str(r.get("error") or "").startswith("llm error"))


def upstream_429s(since: str) -> int:
    try:
        out = subprocess.run(["docker", "logs", "--since", since, "llm-gateway-gateway-1"], capture_output=True,
                             text=True, timeout=30, encoding="utf-8", errors="replace")
        return (out.stdout + out.stderr).count("429 Too Many")
    except Exception:
        return 0


def run_step(name: str, args: list[str], env: dict) -> None:
    path = run_file(args)
    llm = "--cache-rerun" not in args
    for _attempt in range(6):  # bounded: an outage cannot loop forever
        n = workers()
        log(f"start {name} with {n} workers ({len(rows(path))} rows so far)")
        p = subprocess.Popen([sys.executable, "-u", "-m", "eval.run", *args, *BASE, "--workers", str(n)], cwd=ROOT,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8", **env})
        fails0, hist = api_failures(path), []  # hist: (time, rows done)
        restart = False
        while p.poll() is None:
            time.sleep(300)
            if not llm or p.poll() is not None:
                continue
            hist.append((time.time(), len(rows(path))))
            hist = [h for h in hist if time.time() - h[0] <= 15 * 60 + 30]
            done_15 = hist[-1][1] - hist[0][1] if len(hist) > 1 else 0
            r429 = upstream_429s("15m")
            # at 1 worker there is nothing to drop to (Ultra gets steady 429s even then), so a restart would
            # only kill the scenario in progress: spikes only count with more than 1 worker
            spike = n > 1 and len(hist) >= 3 and r429 > 3 * max(1, done_15)
            if api_failures(path) > fails0 or spike:
                why = "new API failure" if api_failures(path) > fails0 else f"429 spike ({r429} in 15 min for {done_15} scenarios)"
                p.terminate()
                p.wait()
                if n > 1:
                    set_workers(1, why)
                ALERT.write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S} {name}: {why}; workers now {workers()}\n")
                restart = True
                break
        if not restart:
            left = api_failures(path)
            log(f"end {name}: exit {p.returncode}, rows {len(rows(path))}, API failures left {left}")
            if left == 0 or "--cache-rerun" in args:
                return
            # retry API failures (resume skips finished scenarios)
    ALERT.write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S} {name}: gave up after 6 attempts; "
                     f"{api_failures(path)} API failures left\n")
    log(f"gave up on {name} after 6 attempts")


def project_after_v2() -> None:
    t0 = datetime.fromisoformat(START.read_text().strip())
    elapsed_h = (datetime.now() - t0).total_seconds() / 3600
    v2 = rows(V2 / "runs" / "test_v2.jsonl")
    n = workers()
    per_scen_wall_h = elapsed_h / max(1, len(v2))  # wall-clock per scenario at the current worker count
    remaining_h = per_scen_wall_h * (170 + 50) + 0.4  # v1 + repeat, plus the cache rerun
    total = elapsed_h + remaining_h
    r429 = upstream_429s(f"{max(1, int(elapsed_h * 60))}m")
    per_scen_429 = r429 / max(1, len(v2))
    log(f"projection after v2: elapsed {elapsed_h:.1f} h, remaining {remaining_h:.1f} h, total {total:.1f} h "
        f"(budget {BUDGET_H} h); upstream 429s per scenario {per_scen_429:.2f}; workers {n}")
    if total > BUDGET_H and per_scen_429 < 1 and n < 3:
        set_workers(3, f"projected {total:.1f} h > {BUDGET_H} h and 429s {per_scen_429:.2f}/scenario < 1")


def main():
    V2.mkdir(parents=True, exist_ok=True)
    if not START.exists():
        START.write_text(datetime.now().isoformat(timespec="seconds"))
    if not WORKERS.exists():
        set_workers(2, "initial")
    for name, args, env, _ in STEPS:
        run_step(name, args, env)
        if name == "v2":
            project_after_v2()
    log("TEST phase complete")


if __name__ == "__main__":
    main()

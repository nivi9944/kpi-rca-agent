"""v3 run, fully automatic, 1 worker on every phase, hard 15-hour wall-clock ceiling from results/v3/run_start.txt.

  python -m eval.run_v3

Sequence: DEV B2 (v3 engine) -> DEV v3 -> DEV verdict (written to DECISIONS.md) -> TEST B2 (v3 engine) -> TEST v3
-> metrics (eval.metrics_v3) -> draft V3_SUMMARY.md -> wait up to 20 min for a final summary (results/v3/FINALIZED)
-> close the progress window and sleep the laptop.
Every phase passes --deadline = start + 15 h - 45 min (reserve for metrics and the summary); the runner then stops
starting new scenarios that would not finish in time. Everything finished is saved (resumable JSONL).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "results" / "v3"
LOG = V3 / "orchestrator.log"
STATUS = V3 / "status.json"
ALERT = V3 / "ALERT.txt"
CEILING_H, RESERVE_MIN = 15.0, 45
TEST_M = "inject/manifest_test.json"

PHASES = [
    ("DEV: B2 (v3 engine)", ["--model", "baseline_scan", "--run-name", "b2_dev", "--results-dir", "results/v3/dev"]),
    ("DEV: v3", ["--model", "nvidia", "--agent-version", "v3", "--run-name", "v3_dev", "--results-dir", "results/v3/dev"]),
    ("TEST: B2 (v3 engine)", ["--model", "baseline_scan", "--run-name", "test_b2_v3", "--results-dir", "results/v3",
                              "--manifest", TEST_M]),
    ("TEST: v3", ["--model", "nvidia", "--agent-version", "v3", "--run-name", "test_v3", "--results-dir", "results/v3",
                  "--manifest", TEST_M]),
]


def start_time() -> datetime:
    return datetime.fromisoformat((V3 / "run_start.txt").read_text().strip())


def deadline() -> datetime:
    return start_time() + timedelta(hours=CEILING_H) - timedelta(minutes=RESERVE_MIN)


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_status(**kw) -> None:
    st = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    st.update(kw)
    STATUS.write_text(json.dumps(st, indent=2), encoding="utf-8")


def rows(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def run_phase(name: str, args: list[str]) -> None:
    rd = Path(args[args.index("--results-dir") + 1])
    path = ROOT / rd / "runs" / f"{args[args.index('--run-name') + 1]}.jsonl"
    set_status(phase=name, phase_started=datetime.now().isoformat(timespec="seconds"))
    for attempt in range(3):  # re-pass only to retry API failures (resume skips finished scenarios)
        log(f"start {name} (pass {attempt + 1}, 1 worker, deadline {deadline():%Y-%m-%d %H:%M})")
        out = subprocess.run([sys.executable, "-u", "-m", "eval.run", *args, "--workers", "1", "--engine", "v3",
                              "--deadline", deadline().isoformat(timespec="seconds")], cwd=ROOT,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8", "KPI_ENGINE": "v3"},
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        (V3 / "logs").mkdir(exist_ok=True)
        with open(V3 / "logs" / f"{name.replace(':', '').replace(' ', '_')}.log", "a", encoding="utf-8") as f:
            f.write(out.stdout + out.stderr)
        r = rows(path)
        api = sum(1 for x in r.values() if str(x.get("error") or "").startswith("llm error"))
        stopped = "STOPPED (time budget)" in out.stdout
        log(f"end {name}: exit {out.returncode}, rows {len(r)}, API failures {api}, time-budget stop {stopped}")
        if out.returncode != 0:
            ALERT.write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S} {name}: exit {out.returncode}; see results/v3/logs\n")
        if stopped:
            set_status(time_budget_stop={"phase": name, "completed": len(r)})
            return
        if api == 0:
            return
    ALERT.write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S} {name}: API failures remain after 3 passes\n")


def dev_verdict() -> None:
    from eval.metrics_v3 import dev_verdict as dv
    text = dv()
    p = ROOT / "DECISIONS.md"
    s = p.read_text(encoding="utf-8")
    marker = "## v3 DEV verdict"
    if marker in s:
        s = s[:s.index(marker)].rstrip() + "\n"
    anchor = "\n## Model and provider"
    s = s.replace(anchor, "\n" + text.rstrip() + "\n" + anchor, 1) if anchor in s else s + "\n" + text
    p.write_text(s, encoding="utf-8")
    log("DEV verdict written to DECISIONS.md")


def finish() -> None:
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*progress_v3.ps1*' } | "
                    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"])
    log("progress window closed; sleeping the laptop")
    subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])


def main():
    V3.mkdir(parents=True, exist_ok=True)
    log(f"v3 run start {start_time():%Y-%m-%d %H:%M:%S}, ceiling {CEILING_H} h, deadline for new scenarios {deadline():%Y-%m-%d %H:%M}")
    for name, args in PHASES:
        if datetime.now() >= deadline():
            set_status(time_budget_stop={"phase": name, "completed": len(rows(ROOT / args[args.index('--results-dir') + 1] / 'runs' / f"{args[args.index('--run-name') + 1]}.jsonl"))})
            log(f"skip {name}: past the deadline")
            continue
        run_phase(name, args)
        if name == "DEV: v3":
            dev_verdict()
    set_status(phase="Metrics and summary")
    from eval.metrics_v3 import main as metrics, draft_summary
    metrics()
    draft_summary()
    log("metrics computed, draft V3_SUMMARY.md written")
    set_status(phase="Done", done=datetime.now().isoformat(timespec="seconds"))
    # give the interactive session up to 20 minutes to write the final summary, but never pass the ceiling
    limit = min(datetime.now() + timedelta(minutes=20), start_time() + timedelta(hours=CEILING_H))
    while datetime.now() < limit and not (V3 / "FINALIZED").exists():
        time.sleep(30)
    log("final summary marker found" if (V3 / "FINALIZED").exists() else "no final-summary marker; keeping the draft")
    finish()


if __name__ == "__main__":
    main()

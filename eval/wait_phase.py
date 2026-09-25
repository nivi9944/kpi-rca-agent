"""Block until the current evaluation phase finishes (or something needs attention). Prints one line.

  python -m eval.wait_phase

Reads only results/progress_status.txt (written by eval.progress). Exits when the phase name changes or
all phases are complete, when a failed run appears, or when there has been no progress for 30 minutes.
"""
from __future__ import annotations

import time
from pathlib import Path

STATUS = Path(__file__).resolve().parents[1] / "results" / "progress_status.txt"
STALL_S = 30 * 60


def read() -> tuple[str, str, int, str]:
    s = STATUS.read_text(encoding="utf-8").strip() if STATUS.exists() else ""
    parts = [p.strip() for p in s.split("|")]
    if len(parts) < 4:
        return s, "", 0, s
    phase, done = parts[1].rsplit(" ", 1)
    failed = int(parts[3].split()[-1]) if parts[3].startswith("failed") else 0
    return phase, done, failed, s


def main():
    phase0, done0, _, _ = read()
    last_change, last_done = time.time(), done0
    while True:
        time.sleep(60)
        phase, done, failed, line = read()
        if "all phases complete" in line or (phase and phase != phase0):
            print(f"PHASE DONE: {phase0} -> now: {line}")
            return
        if failed:
            print(f"ATTENTION failed runs: {line}")
            return
        if done != last_done:
            last_done, last_change = done, time.time()
        elif time.time() - last_change > STALL_S:
            print(f"ATTENTION no progress for 30 min: {line}")
            return


if __name__ == "__main__":
    main()

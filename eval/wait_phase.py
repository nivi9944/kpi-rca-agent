"""Block until the current evaluation phase finishes or needs attention, then print one line.

  python -m eval.wait_phase [--until-all]

Polls results/progress_status.txt (written by eval.progress) once a minute. Exits when the phase name
changes (with --until-all, only when all phases are complete), when a failed run appears, when
results/v2/ALERT.txt is updated, or after 30 minutes without progress.
"""
from __future__ import annotations

import time
from pathlib import Path

STATUS = Path(__file__).resolve().parents[1] / "results" / "progress_status.txt"
STALL_S = 30 * 60


def read() -> tuple[str, str, int, str]:
    """(phase name, "done/total", failed runs, raw line); a line without 4 fields is returned as the phase."""
    s = STATUS.read_text(encoding="utf-8").strip() if STATUS.exists() else ""
    parts = [p.strip() for p in s.split("|")]
    if len(parts) < 4:
        return s, "", 0, s
    phase, done = parts[1].rsplit(" ", 1)
    failed = int(parts[3].split()[-1]) if parts[3].startswith("failed") else 0
    return phase, done, failed, s


ALERT = STATUS.parent / "v2" / "ALERT.txt"


def main():
    import sys
    until_all = "--until-all" in sys.argv
    alert0 = ALERT.stat().st_mtime if ALERT.exists() else 0
    phase0, done0, _, _ = read()
    last_change, last_done = time.time(), done0
    while True:
        time.sleep(60)
        phase, done, failed, line = read()
        if ALERT.exists() and ALERT.stat().st_mtime != alert0:
            print(f"ATTENTION alert: {ALERT.read_text(encoding='utf-8').strip()} | {line}")
            return
        if "all phases complete" in line or (not until_all and phase and phase != phase0):
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

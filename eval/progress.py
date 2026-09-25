"""Live progress window for the long evaluation runs.

  python -m eval.progress            (print once)
  python -m eval.progress --watch    (refresh every 30 s)

Reads only the run JSONL files and the gateway log counts. Writes a one-line status to
results/progress_status.txt on every refresh.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results" / "v2"
DEV_MANIFEST = ROOT / "inject" / "manifest.json"
TEST_MANIFEST = ROOT / "inject" / "manifest_test.json"
STATUS = ROOT / "results" / "progress_status.txt"
REFRESH_S = 30
WORKERS = V2 / "workers.txt"


def _workers() -> int:
    try:
        return max(1, int(WORKERS.read_text().strip()))
    except (OSError, ValueError):
        return 1

# (phase, run file, manifest, optional subset file) in execution order
PHASES = [
    ("DEV round 2: v2-final", ROOT / "results" / "runs" / "nvidia_v2final_dev.jsonl", DEV_MANIFEST, None),
    ("TEST: B1", V2 / "runs" / "test_b1.jsonl", TEST_MANIFEST, None),
    ("TEST: B2", V2 / "runs" / "test_b2.jsonl", TEST_MANIFEST, None),
    ("TEST: v2", V2 / "runs" / "test_v2.jsonl", TEST_MANIFEST, None),
    ("TEST: v1", V2 / "runs" / "test_v1.jsonl", TEST_MANIFEST, None),
    ("TEST: v2 repeat", V2 / "runs" / "test_v2_repeat.jsonl", TEST_MANIFEST, V2 / "repeat_ids.json"),
    ("TEST: v2 cache rerun", V2 / "runs_cache" / "test_v2.jsonl", TEST_MANIFEST, V2 / "cache_ids.json"),
]
LLM_PHASES = {"DEV round 2: v2-final", "TEST: v2", "TEST: v1", "TEST: v2 repeat", "TEST: v2 cache rerun"}


def _rows(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    out[r["id"]] = r  # latest attempt per scenario
                except (json.JSONDecodeError, KeyError):
                    pass
    return out


def _ids(manifest: Path, subset: Path | None) -> list[str]:
    if not manifest.exists():
        return []
    ids = [s["id"] for s in json.loads(manifest.read_text(encoding="utf-8"))["scenarios"]]
    if subset is not None:
        if not subset.exists():
            return []
        keep = set(json.loads(subset.read_text(encoding="utf-8")))
        ids = [i for i in ids if i in keep]
    return ids


def _gateway_429s() -> str:
    try:
        out = subprocess.run(["docker", "logs", "--since", "5m", "llm-gateway-gateway-1"], capture_output=True,
                             text=True, timeout=20, encoding="utf-8", errors="replace")
        text = out.stdout + out.stderr
        return str(text.count("429 Too Many"))
    except Exception:
        return "n/a"


def _bar(done: int, total: int, width: int = 30) -> str:
    f = int(width * done / total) if total else 0
    return "[" + "#" * f + "-" * (width - f) + "]"


def snapshot() -> tuple[str, str]:
    lines, status = [], ""
    phase_info = []
    for name, path, man, subset in PHASES:
        ids = _ids(man, subset)
        rows = _rows(path)
        done = [i for i in ids if i in rows and not str(rows[i].get("error") or "").startswith("llm error")]
        phase_info.append((name, path, ids, rows, done))
    current = next((p for p in phase_info if p[2] and len(p[4]) < len(p[2])), None)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"KPI RCA agent: evaluation progress        updated {now}   (refresh {REFRESH_S} s)")
    lines.append("=" * 88)
    nw = _workers()
    lines.append(f"Gateway: upstream 429s in the last 5 min: {_gateway_429s()}    workers: {nw}"
                 f"    (ETAs are wall-clock: time per scenario / workers)")
    lines.append("")
    avg_min_llm = []
    for name, path, ids, rows, done in phase_info:
        total = len(ids)
        if not total:
            lines.append(f"{name:24} (not built yet)")
            continue
        rs = [rows[i] for i in done]
        lat = [r.get("latency_s") or 0 for r in rs]
        avg_min = (sum(lat) / len(lat) / 60) if lat else None
        if name in LLM_PHASES and avg_min:
            avg_min_llm.append(avg_min)
        mark = "  <== running" if current and name == current[0] else ""
        lines.append(f"{name:24} {_bar(len(done), total)} {len(done):4}/{total:<4}"
                     f"{'' if avg_min is None else f'  {avg_min:.2f} min/scenario'}{mark}")
    lines.append("")
    if current:
        name, path, ids, rows, done = current
        rs = [rows[i] for i in done]
        planted = [r for r in rs if not r["score"]["is_control"]]
        t1 = sum(bool(r["score"]["top1"]) for r in planted)
        t3 = sum(bool(r["score"]["top3"]) for r in planted)
        api_fail = sum(1 for r in rs if str(r.get("error") or "").startswith("llm error"))
        model_fail = sum(1 for r in rs if r.get("error") and not str(r.get("error")).startswith("llm error"))
        failed = api_fail
        nxt = next((i for i in ids if i not in done), "-")
        lat = [r.get("latency_s") or 0 for r in rs]
        avg_min = (sum(lat) / len(lat) / 60) if lat else (sum(avg_min_llm) / len(avg_min_llm) if avg_min_llm else 2.5)
        if name not in LLM_PHASES:
            avg_min = 0.02
        eta_run = (len(ids) - len(done)) * avg_min / nw
        # whole TEST phase: remaining scenarios of every LLM phase at the observed pace
        pace = sum(avg_min_llm) / len(avg_min_llm) if avg_min_llm else avg_min
        rem = 0.0
        for n2, _, ids2, _, done2 in phase_info:
            if n2.startswith("TEST") and ids2:
                rem += (len(ids2) - len(done2)) * (pace if n2 in LLM_PHASES else 0.02) * (0.1 if "cache" in n2 else 1) / nw
        pct = lambda k: f"{100 * k / len(planted):.1f}%" if planted else "n/a"
        lines.append(f"Current run: {name}   scenario in progress: {nxt}")
        lines.append(f"  running top-1 {pct(t1)} ({t1}/{len(planted)} planted)   top-3 {pct(t3)}   "
                     f"API failures {api_fail}   model failures (no report) {model_fail}")
        lines.append(f"  avg {avg_min:.2f} min/scenario   ETA this run {eta_run / 60:.1f} h   ETA whole TEST phase {rem / 60:.1f} h")
        lines.append("  last 5 scenarios:")
        for r in rs[-5:]:
            s = r["score"]
            res = ("no alarm (correct)" if not s["false_alarm"] else "FALSE ALARM") if s["is_control"] else \
                ("correct" if s["top1"] else "wrong")
            lines.append(f"    {r['id']:34} {res}{'  ERROR: ' + str(r['error'])[:40] if r.get('error') else ''}")
        status = (f"{now} | {name} {len(done)}/{len(ids)} | top-1 {pct(t1)} | failed {failed} | workers {nw} | "
                  f"ETA run {eta_run / 60:.1f} h, TEST {rem / 60:.1f} h | next {nxt}")
    else:
        lines.append("All phases complete.")
        status = f"{now} | all phases complete"
    return "\n".join(lines), status


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args(argv)
    while True:
        text, status = snapshot()
        if args.watch:
            os.system("cls" if os.name == "nt" else "clear")
        print(text, flush=True)
        try:
            STATUS.write_text(status + "\n", encoding="utf-8")
        except OSError:
            pass
        if not args.watch:
            break
        time.sleep(REFRESH_S)


if __name__ == "__main__":
    main()

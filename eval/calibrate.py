"""Calibrate the segment history-z threshold on NATURAL weeks only (no planted anomalies).

For each metric and each week in the scenario pool (clean data), scan all segments and count the
week as a "natural alarm" if any segment passes: classical BH q <= 0.05 AND |hist_z| >= T.
We pick the smallest T whose natural alarm rate is <= 10% across metrics. The planted scenarios are
never looked at here, so the threshold is not tuned on the evaluation set.

Usage: python -m eval.calibrate   -> results/calibration.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inject.scenarios import CONTROL_METRICS, PLAN, candidate_weeks
from metrics.metrics import default_store, metric_def
from tools.segscan import SCAN_DIMS, segment_history_z
from tools.stats import Q, bh, run_test

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [3.0, 3.5, 3.75, 4.0, 4.25, 4.5, 5.0, 6.0]
TARGET_RATE = 0.10


def week_max_z(store, metric, week) -> list[tuple[float, float]]:
    """(|hist_z|, p) for the top-3 segments per dimension and effect."""
    m = metric_def(metric)
    effects = ["share"] if m["kind"] == "sum" else ["rate", "share"]
    out = []
    for dim in SCAN_DIMS:
        for eff in effects:
            h = segment_history_z(store, metric, dim, week, eff).head(3)
            for r in h.itertuples():
                p = run_test(store, metric, dim, r.segment, week, effect=eff)["p_value"]
                out.append((abs(r.hist_z), float(p)))
    return out


def main() -> dict:
    store = default_store()
    metrics = sorted({p[1] for p in PLAN} | set(CONTROL_METRICS))
    rows = []
    for metric in metrics:
        for w in candidate_weeks(store, metric, max_abs_z=None):
            zs = week_max_z(store, metric, w)
            q = bh([p for _, p in zs])
            rows.append({"metric": metric, "week": w.strftime("%Y-%m-%d"),
                         "pairs": [(z, qq) for (z, _), qq in zip(zs, q)]})
    table = {}
    for t in THRESHOLDS:
        alarms = [any(z >= t and qq <= Q for z, qq in r["pairs"]) for r in rows]
        per_metric = {}
        for metric in metrics:
            a = [al for al, r in zip(alarms, rows) if r["metric"] == metric]
            per_metric[metric] = round(float(np.mean(a)), 3)
        table[str(t)] = {"overall": round(float(np.mean(alarms)), 3), "by_metric": per_metric}
    chosen = next((t for t in THRESHOLDS if table[str(t)]["overall"] <= TARGET_RATE), THRESHOLDS[-1])
    res = {"n_week_metric_pairs": len(rows), "target_natural_alarm_rate": TARGET_RATE,
           "chosen_hist_z": chosen, "natural_alarm_rate_by_threshold": table}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "calibration.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))

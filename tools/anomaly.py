"""Anomaly detection with a robust z-score.

For each week t we compute the change versus the average of the previous 4 weeks:
  sum metrics  (GMV, Orders):   c_t = value_t / mean(prev 4) - 1          (relative change)
  mean metrics (rates, AOV...):  c_t = value_t - mean(prev 4)              (absolute change)
Then we score c_t against the previous 8 changes with the median and MAD (robust to outliers):
  z_t = (c_t - median(c_{t-8..t-1})) / (1.4826 * MAD)
A week is flagged when |z_t| >= 3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metrics.metrics import metric_def, to_week
from tools.context import Investigation

Z_THRESHOLD = 3.0
BASE_WEEKS = 4
TRAIL = 8
MIN_N = 30  # a mean metric needs at least this many rows in a week to be scored


def score_series(store, metric: str, filters: dict | None = None) -> pd.DataFrame:
    m = metric_def(metric)
    s = store.weekly(metric, filters).copy()
    base = s["value"].shift(1).rolling(BASE_WEEKS, min_periods=BASE_WEEKS).mean()
    if m["kind"] == "sum":
        s["change"] = s["value"] / base - 1.0
    else:
        s["change"] = s["value"] - base
        s.loc[s["n"] < MIN_N, "change"] = np.nan
    s["baseline"] = base
    med = s["change"].shift(1).rolling(TRAIL, min_periods=6).median()
    mad = s["change"].shift(1).rolling(TRAIL, min_periods=6).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    # floor the scale so a flat history does not create infinite z-scores
    floor = 0.01 if m["kind"] == "sum" else max(1e-4, 0.002 * float(np.nanmedian(np.abs(s["value"]))))
    scale = np.maximum(1.4826 * mad, floor)
    s["z"] = (s["change"] - med) / scale
    s["is_anomaly"] = s["z"].abs() >= Z_THRESHOLD
    return s


def detect_anomalies(inv: Investigation, metric: str, week: str | None = None,
                     start: str | None = None, end: str | None = None,
                     filters: dict | None = None) -> dict:
    m = metric_def(metric)
    s = score_series(inv.store, metric, filters)
    out: dict = {"metric": metric, "method": f"robust z of change vs prev {BASE_WEEKS}-week avg, scored against trailing {TRAIL} weeks (median/MAD); flag |z|>={Z_THRESHOLD}",
                 "change_type": "relative" if m["kind"] == "sum" else "absolute"}
    if filters:
        out["filters"] = filters
    if week:
        w = to_week(week)
        row = s[s["week"] == w]
        if row.empty:
            raise ValueError(f"Week {week} not in data")
        r = row.iloc[0]
        chg_pct = (r["value"] / r["baseline"] - 1) * 100 if r["baseline"] else None
        out["target_week"] = {
            "week": w, "value": r["value"], "baseline_prev4_avg": r["baseline"],
            "change_pct": chg_pct, "change_abs": r["value"] - r["baseline"],
            "z_score": r["z"], "is_anomaly": bool(r["is_anomaly"]) if pd.notna(r["z"]) else False,
            "direction": "down" if r["value"] < r["baseline"] else "up",
        }
    lo = to_week(start) if start else s["week"].min()
    hi = to_week(end) if end else s["week"].max()
    flagged = s[(s["week"] >= lo) & (s["week"] <= hi) & s["is_anomaly"]]
    out["flagged_weeks"] = [
        {"week": r.week, "value": r.value, "z_score": r.z} for r in flagged.itertuples()
    ][:20]
    return out

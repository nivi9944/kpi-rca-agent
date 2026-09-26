"""Segment scan: find segments whose change in the target week is unusual versus their OWN history.

Why: a problem in one segment (one state, one category) can be invisible in the total, and
classical tests (z-test, chi-square) treat every order as independent, so on real weekly data they
call almost every large segment "significant" (weekly data is over-dispersed).
So each segment gets a robust history z-score, computed exactly like the headline detector:
  x_t     = segment share of rows (effect='share') or segment metric value (effect='rate';
            money metrics use log(1+value) so single big-ticket orders do not dominate)
  c_t     = x_t - mean(x over prev 4 weeks)
  hist_z  = (c_t - median(c over trailing 8 weeks)) / (1.4826 * MAD)
A root cause must pass BOTH: classical test q <= 0.05 (after Benjamini-Hochberg) AND |hist_z| >= HIST_Z.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metrics.metrics import dimension_col, metric_def, to_week
from tools.context import Investigation

from tools.engine import hist_z as _hist_z  # noqa: E402

# threshold from the calibration table on natural (un-planted) weeks only (results/calibration.json):
# v2 used 4.25 (8.2% natural alarm rate), v3 uses 4.0 (11.2%); see tools/engine.py
HIST_Z = 4.0
MIN_WEEKLY_N = 20  # a segment needs this many rows per week (on average) to be scored
BASE, TRAIL = 4, 16
SCAN_DIMS = ["customer_state", "product_category", "seller_state", "main_payment_type", "is_repeat_customer"]


def _smoothed_base(panel: pd.DataFrame, n) -> pd.DataFrame:
    """Expected proportion for this week = prev-4-week mean, pulled slightly toward 1/2 (Laplace
    smoothing), so a rate of exactly 0 or 1 in the baseline still has a sensible noise level."""
    base = panel.shift(1).rolling(BASE, min_periods=1).mean().fillna(panel)
    return (base * n + 1) / (n + 2)


def _robust_z(panel: pd.DataFrame, noise_sd: pd.DataFrame, floor: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = panel.shift(1).rolling(BASE, min_periods=BASE).mean()
    chg = panel - base
    hist = chg.shift(1)
    med = hist.rolling(TRAIL, min_periods=6).median()
    mad = (hist - med).abs().rolling(TRAIL, min_periods=6).median()
    # scale = the larger of (a) the segment's own historical spread (robust MAD) and
    # (b) the sampling noise expected from its size this week (so a tiny segment cannot look
    #     "unusual" from a handful of orders), and (c) a small absolute floor.
    samp = noise_sd * np.sqrt(1 + 1 / BASE)
    scale = np.maximum(np.maximum(1.4826 * mad, samp), floor)
    return (chg - med) / scale, chg


def segment_history_z(store, metric: str, dimension: str, week, effect: str | None = None,
                      filters: dict | None = None) -> pd.DataFrame:
    """One row per segment: hist_z, change, current and typical value, weekly support."""
    m = metric_def(metric)
    effect = effect or ("share" if m["kind"] == "sum" else "rate")
    col = dimension_col(dimension, m["table"])
    df = store.rows(metric, filters)
    if effect == "rate" and m.get("money"):
        # money values are heavy-tailed (a few big-ticket orders dominate a week), so rate scans
        # for money metrics run on log(1 + value): a detection device only, reports use real R$
        df = df.assign(value=np.log1p(df["value"].clip(lower=0)))
    df = df.assign(_seg=df[col].astype(str))
    weeks = pd.Index(store.weeks(), name="week")
    g = df.groupby(["week", "_seg"])["value"]
    tot = g.sum().unstack(fill_value=0.0).reindex(weeks).fillna(0.0)
    cnt = g.count().unstack(fill_value=0).reindex(weeks).fillna(0.0)
    if effect == "share":
        # share of ROWS (orders or items), not of value: value shares swing with a few big orders
        panel = cnt.div(cnt.sum(axis=1).replace(0, np.nan), axis=0)
        n_all = cnt.sum(axis=1)
        p0 = _smoothed_base(panel, n_all.to_frame().values)
        noise = np.sqrt((p0 * (1 - p0)).div(n_all.replace(0, np.nan), axis=0))
        floor = 0.002
    else:
        panel = tot / cnt.replace(0, np.nan)
        panel = panel.where(cnt >= 5)
        if m.get("binary"):
            p0 = _smoothed_base(panel, cnt.values)
            noise = np.sqrt(p0 * (1 - p0) / cnt.replace(0, np.nan))
        else:
            sq = df.assign(_sq=df["value"] ** 2).groupby(["week", "_seg"])["_sq"].sum().unstack(fill_value=0.0).reindex(weeks).fillna(0.0)
            var = (sq / cnt.replace(0, np.nan) - panel ** 2).clip(lower=0)
            noise = np.sqrt(var / cnt.replace(0, np.nan))
        floor = max(1e-4, 0.002 * float(np.nanmedian(np.abs(df["value"]))))
    z, chg = _robust_z(panel, noise.fillna(0.0), floor)
    w = to_week(week)
    prev = [w - pd.Timedelta(weeks=i) for i in range(BASE, 0, -1)]
    support = cnt.loc[prev + [w]].mean()
    out = pd.DataFrame({"segment": panel.columns, "hist_z": z.loc[w].values, "change": chg.loc[w].values,
                        "current": panel.loc[w].values, "baseline_prev4": panel.loc[prev].mean().values,
                        "weekly_n": support.values})
    out = out[out["weekly_n"] >= MIN_WEEKLY_N].dropna(subset=["hist_z"])
    out["effect"] = effect
    return out.sort_values("hist_z", key=np.abs, ascending=False).reset_index(drop=True)


def scan_segments(inv: Investigation, metric: str, week: str, dimensions: list[str] | None = None,
                  top_k: int = 3) -> dict:
    """Scan every dimension; return the segments that moved most unusually vs their own history.

    For ratio metrics both share (mix) and rate are scanned."""
    from tools.drilldown import drill_table

    m = metric_def(metric)
    effects = ["share"] if m["kind"] == "sum" else ["rate", "share"]
    found = []
    for dim in dimensions or SCAN_DIMS:
        t, _ = drill_table(inv.store, metric, dim, week, min_support=0)
        contrib = dict(zip(t["segment"].astype(str), t["contribution_pct"]))
        for eff in effects:
            h = segment_history_z(inv.store, metric, dim, week, eff)
            for r in h.head(top_k).itertuples():
                found.append({"dimension": dim, "segment": r.segment, "effect_scanned": eff,
                              "hist_z": r.hist_z, "value_current": r.current, "value_baseline_prev4": r.baseline_prev4,
                              "contribution_pct": contrib.get(r.segment), "weekly_n": r.weekly_n,
                              "unusual": bool(abs(r.hist_z) >= _hist_z())})
    found.sort(key=lambda d: -abs(d["hist_z"]))
    return {"metric": metric, "week": to_week(week), "hist_z_threshold": _hist_z(),
            "value_meaning": "share of the metric (share scans) or segment metric value (rate scans)",
            "candidates": found[:12],
            "n_unusual": sum(d["unusual"] for d in found),
            "next_step": "Confirm each unusual candidate with significance_test before calling it a root cause."}

"""Drill-down: which segments explain the change, ranked by contribution.

Sum metrics (GMV, Orders):
  delta_s = value_s(week) - avg value_s(baseline weeks);  contribution_s = delta_s / sum(delta) x 100

Mean / ratio metrics (AOV, rates, averages): exact mix vs rate split.
  Overall value R = sum_s w_s * r_s  (w = segment share of rows, r = segment mean)
  R1 - R0 = sum_s (w1_s - w0_s) * (r0_s - R0)   [mix: the segment's share changed]
          + sum_s  w1_s * (r1_s - r0_s)          [rate: behaviour inside the segment changed]
  Centering the mix term on R0 does not change its sum (shares sum to 1) but makes each
  segment's mix term meaningful: gaining share only lowers R if the segment is below average.
"""
from __future__ import annotations

import numpy as np

from metrics.metrics import metric_def, to_week
from tools.context import Investigation

SMALL = "(small segments)"


def drill_table(store, metric: str, dimension: str, week, n_baseline: int = 4,
                filters: dict | None = None, min_support: int = 30):
    m = metric_def(metric)
    t = store.segment_table(metric, dimension, week, n_baseline, filters)
    if m["kind"] == "sum":
        support = np.maximum(t["n1"], t["n0"])
    else:
        support = np.minimum(t["n1"], t["n0"] / n_baseline)
    small = support < min_support
    if small.any() and (~small).any():
        agg = t[small][["sum1", "n1", "sum0", "n0"]].sum()
        t = t[~small].copy()
        t.loc[len(t)] = {"segment": SMALL, **agg.to_dict(),
                         "mean1": agg["sum1"] / agg["n1"] if agg["n1"] else np.nan,
                         "mean0": agg["sum0"] / agg["n0"] if agg["n0"] else np.nan}
    t = t.reset_index(drop=True)
    if m["kind"] == "sum":
        t["delta"] = t["sum1"] - t["sum0"]
        total0, total1 = t["sum0"].sum(), t["sum1"].sum()
        mix_total = rate_total = None
    else:
        N0, N1 = t["n0"].sum(), t["n1"].sum()
        R0 = t["sum0"].sum() / N0
        R1 = t["sum1"].sum() / N1
        t["w0"], t["w1"] = t["n0"] / N0, t["n1"] / N1
        r0 = t["mean0"].fillna(R0)
        r1 = t["mean1"].fillna(r0)
        t["mix"] = (t["w1"] - t["w0"]) * (r0 - R0)
        t["rate"] = t["w1"] * (r1 - r0)
        t["delta"] = t["mix"] + t["rate"]
        total0, total1 = R0, R1
        mix_total, rate_total = float(t["mix"].sum()), float(t["rate"].sum())
    total_delta = float(total1 - total0)
    sign = 1.0 if total_delta >= 0 else -1.0
    t["contribution_pct"] = t["delta"] / total_delta * 100 if total_delta else 0.0
    t = t.assign(_k=t["delta"] * sign).sort_values("_k", ascending=False).drop(columns="_k")
    return t, {"total_baseline": total0, "total_current": total1, "total_delta": total_delta,
               "mix_total": mix_total, "rate_total": rate_total}


def drill_down(inv: Investigation, metric: str, dimension: str, week: str, n_baseline: int = 4,
               filters: dict | None = None, top_k: int = 5, min_support: int = 30) -> dict:
    m = metric_def(metric)
    t, tot = drill_table(inv.store, metric, dimension, week, n_baseline, filters, min_support)
    rows = []
    for r in t.head(top_k).itertuples():
        d = {"segment": r.segment, "contribution_pct": r.contribution_pct, "delta": r.delta}
        if m["kind"] == "sum":
            d.update({"baseline": r.sum0, "current": r.sum1,
                      "segment_change_pct": (r.sum1 / r.sum0 - 1) * 100 if r.sum0 else None})
        else:
            d.update({"baseline_value": r.mean0, "current_value": r.mean1,
                      "share_baseline_pct": r.w0 * 100, "share_current_pct": r.w1 * 100,
                      "mix_effect": r.mix, "rate_effect": r.rate,
                      "effect": "mix" if abs(r.mix) > abs(r.rate) else "rate",
                      "n_current": r.n1})
        rows.append(d)
    out = {"metric": metric, "dimension": dimension, "week": to_week(week),
           "baseline_def": f"prev {n_baseline}-week avg", **tot,
           "total_change_pct": (tot["total_current"] / tot["total_baseline"] - 1) * 100 if tot["total_baseline"] else None,
           "n_segments": int(len(t)), "top_segments": rows}
    if filters:
        out["filters"] = filters
    if m["kind"] == "mean":
        out["note"] = "delta = mix_effect + rate_effect; contributions sum to 100% over all segments"
    return out

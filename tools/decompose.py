"""Driver decomposition with a multiplicative (log) split.

If Y = A x B then ln(Y1/Y0) = ln(A1/A0) + ln(B1/B0), so the share of the change owed to A is
ln(A1/A0) / ln(Y1/Y0). This split is exact and does not depend on the order of the factors.
  GMV = Orders x AOV
  AOV = Items per order x Value per item
  any mean metric = numerator total / denominator count
"""
from __future__ import annotations

import math

from metrics.metrics import metric_def, to_week
from tools.context import Investigation


def _split(y0, y1, parts: dict) -> dict:
    """parts: name -> (v0, v1). Returns driver effects in units of Y."""
    dy = y1 - y0
    ly = math.log(y1 / y0) if y0 > 0 and y1 > 0 else 0.0
    drivers = []
    for name, (a0, a1) in parts.items():
        la = math.log(a1 / a0) if a0 > 0 and a1 > 0 else 0.0
        share = la / ly if ly != 0 else 0.0
        drivers.append({"driver": name, "baseline": a0, "current": a1,
                        "change_pct": (a1 / a0 - 1) * 100 if a0 else None,
                        "effect": dy * share, "share_of_change_pct": share * 100})
    drivers.sort(key=lambda d: -abs(d["effect"]))
    return {"baseline": y0, "current": y1, "change": dy,
            "change_pct": (y1 / y0 - 1) * 100 if y0 else None, "drivers": drivers}


def decompose_metric(inv: Investigation, metric: str, week: str, n_baseline: int = 4,
                     filters: dict | None = None) -> dict:
    st = inv.store
    w1 = to_week(week)
    w0 = st.baseline_weeks(w1, n_baseline)
    m = metric_def(metric)
    if metric in ("gmv", "aov"):
        o = st.rows("aov", filters)  # non-canceled orders with order_value
        cur, base = o[o["week"] == w1], o[o["week"].isin(w0)]
        O1, O0 = len(cur), len(base) / n_baseline
        G1, G0 = cur["order_value"].sum(), base["order_value"].sum() / n_baseline
        I1, I0 = cur["n_items"].sum(), base["n_items"].sum() / n_baseline
        if metric == "gmv":
            res = _split(G0, G1, {"orders": (O0, O1), "aov": (G0 / O0, G1 / O1)})
            res["formula"] = "GMV = Orders x AOV"
        else:
            res = _split(G0 / O0, G1 / O1, {"items_per_order": (I0 / O0, I1 / O1),
                                             "value_per_item": (G0 / I0, G1 / I1)})
            res["formula"] = "AOV = Items per order x Value per item"
    elif m["kind"] == "mean":
        r = st.rows(metric, filters)
        cur, base = r[r["week"] == w1], r[r["week"].isin(w0)]
        n1, n0 = len(cur), len(base) / n_baseline
        t1, t0 = cur["value"].sum(), base["value"].sum() / n_baseline
        res = _split(t0 / n0, t1 / n1, {"numerator_total": (t0, t1), "denominator_count": (n0, n1)})
        # denominator enters with a negative sign in the log identity
        for d in res["drivers"]:
            if d["driver"] == "denominator_count":
                d["effect"] = -d["effect"]
                d["share_of_change_pct"] = -d["share_of_change_pct"]
        res["formula"] = f"{metric} = numerator total / denominator count"
    else:
        return {"metric": metric, "week": w1, "note": "Orders is a base count with no multiplicative drivers; use drill_down."}
    res.update({"metric": metric, "week": w1, "baseline_def": f"prev {n_baseline}-week avg"})
    return res

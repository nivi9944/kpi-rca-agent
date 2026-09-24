"""Business impact of one segment's change, in R$ where the metric allows it.

  GMV   : segment GMV(week) - segment avg GMV(baseline)
  Orders: segment order change x baseline AOV (overall)
  AOV   : segment's contribution to the AOV change (mix + rate) x orders in the week
  others: no R$ figure; reports the change in affected orders, e.g. extra late deliveries
"""
from __future__ import annotations

from metrics.metrics import metric_def, to_week
from tools.context import Investigation
from tools.drilldown import drill_table


def estimate_impact(inv: Investigation, metric: str, dimension: str, segment: str, week: str,
                    n_baseline: int = 4, filters: dict | None = None) -> dict:
    m = metric_def(metric)
    st = inv.store
    t, tot = drill_table(st, metric, dimension, week, n_baseline, filters, min_support=0)
    row = t[t["segment"].astype(str) == str(segment)]
    if row.empty:
        raise ValueError(f"Segment {segment!r} not found for {dimension}")
    r = row.iloc[0]
    out = {"metric": metric, "dimension": dimension, "segment": str(segment), "week": to_week(week),
           "segment_delta": float(r["delta"]), "contribution_pct": float(r["contribution_pct"])}
    if metric == "gmv":
        out["impact_brl"] = float(r["delta"])
        out["method"] = "segment GMV(week) - segment avg weekly GMV(baseline)"
    elif metric == "orders":
        base = st.period_value("aov", st.baseline_weeks(week, n_baseline), filters)["value"]
        out["baseline_aov_brl"] = base
        out["impact_brl"] = float(r["delta"]) * base
        out["method"] = "segment order change x baseline AOV"
    elif metric == "aov":
        n1 = float(t["n1"].sum())
        out["orders_in_week"] = n1
        out["impact_brl"] = float(r["delta"]) * n1
        out["method"] = "segment contribution to AOV change x orders in week"
    else:
        out["impact_brl"] = None
        out["affected_rows_change"] = float(r["n1"]) * (float(r["mean1"]) - float(r["mean0"])) if r["n1"] and r["n0"] else None
        out["method"] = f"no R$ value for {metric}; affected_rows_change = n_week x (segment rate now - baseline rate)"
        out["unit"] = m["unit"]
    return out

"""Deterministic baselines (no LLM). The agent is compared against both, honestly.

B1 "top_contribution" (the simple rule from the spec):
   1. detect_anomalies on the headline metric; if not flagged -> "no anomaly".
   2. drill_down on every dimension; report the single segment with the largest contribution.
B2 "scan_and_test" (a strong rule that uses the same tools the agent has):
   1. scan_segments over all dimensions (rate and share).
   2. significance_test the unusual candidates (BH across all tests).
   3. report significant ones ranked by |hist_z|; if none, "no significant root cause".
"""
from __future__ import annotations

from metrics.metrics import metric_def
from tools.context import Investigation
from tools.registry import execute
from tools.segscan import SCAN_DIMS


def _cause(rank, dim, seg, contribution, effect, p=None, evid=None):
    return {"rank": rank, "dimension": dim, "segment": str(seg), "contribution_pct": contribution,
            "effect": effect, "p_value": p, "evidence_ids": evid or []}


def top_contribution(inv: Investigation, metric: str, week: str) -> dict:
    det = execute(inv, "detect_anomalies", {"metric": metric, "week": week})
    tw = det["target_week"]
    report = {"metric": metric, "week": week, "anomaly_confirmed": bool(tw["is_anomaly"]),
              "change_pct": tw["change_pct"], "root_causes": [], "evidence_ids": [det["evidence_id"]]}
    if not tw["is_anomaly"]:
        return report
    cands = []
    for dim in SCAN_DIMS:
        d = execute(inv, "drill_down", {"metric": metric, "dimension": dim, "week": week, "top_k": 3})
        for s in d.get("top_segments", []):
            if s["segment"] == "(small segments)":
                continue
            cands.append((s["contribution_pct"] or 0, dim, s, d["evidence_id"]))
    cands.sort(key=lambda x: -x[0])
    for i, (c, dim, s, eid) in enumerate(cands[:3], 1):
        report["root_causes"].append(_cause(i, dim, s["segment"], c, s.get("effect"), evid=[eid]))
    return report


def scan_and_test(inv: Investigation, metric: str, week: str) -> dict:
    m = metric_def(metric)
    det = execute(inv, "detect_anomalies", {"metric": metric, "week": week})
    scan = execute(inv, "scan_segments", {"metric": metric, "week": week})
    tw = det["target_week"]
    sig = []
    for c in scan["candidates"]:
        if not c["unusual"]:
            continue
        effect = "share" if c["effect_scanned"] == "share" else "rate"
        t = execute(inv, "significance_test", {"metric": metric, "dimension": c["dimension"],
                                               "segment": c["segment"], "week": week, "effect": effect})
        if t.get("significant"):
            eff = None if m["kind"] == "sum" else ("mix" if effect == "share" else "rate")
            sig.append((abs(c["hist_z"]), c, eff, t))
    sig.sort(key=lambda x: -x[0])
    report = {"metric": metric, "week": week,
              "anomaly_confirmed": bool(tw["is_anomaly"]) or bool(sig),
              "change_pct": tw["change_pct"], "root_causes": [],
              "evidence_ids": [det["evidence_id"], scan["evidence_id"]]}
    seen = set()
    for _, c, eff, t in sig:
        key = (c["dimension"], c["segment"])
        if key in seen:
            continue
        seen.add(key)
        report["root_causes"].append(_cause(len(seen), c["dimension"], c["segment"], c["contribution_pct"], eff,
                                            t["p_value"], [scan["evidence_id"], t["evidence_id"]]))
        if len(seen) == 3:
            break
    return report


BASELINES = {"baseline_topcontrib": top_contribution, "baseline_scan": scan_and_test}

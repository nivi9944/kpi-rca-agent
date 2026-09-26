"""v2 deterministic guards, applied after the report passes the grounding verifier.

1. Drop guard: a root cause without a significant test (on the same dimension and segment) is removed.
   The LLM's order is kept. (Optional `reorder=True` uses the evidence re-ranker instead, which was tried
   on DEV and rejected; see DECISIONS.md.)
2. Consistency guard: if at least one root cause backed by a significant test remains, anomaly_confirmed is
   set to true; if none remains, root_causes is empty.
"""
from __future__ import annotations

from agent.rerank import _best, _tests_for, rerank


def apply_guards(report: dict, evidence: dict, reorder: bool = False) -> tuple[dict, dict]:
    rcs = sorted(report.get("root_causes") or [], key=lambda r: r.get("rank", 99))
    before = [f"{c.get('dimension')}={c.get('segment')}" for c in rcs]
    if reorder:
        out, log = rerank(report, evidence)
    else:
        kept = [c for c in rcs if (_best(_tests_for(c, evidence)) or {}).get("significant")]
        out = {**report, "root_causes": [{**c, "rank": i} for i, c in enumerate(kept, 1)]}
        after = [f"{c.get('dimension')}={c.get('segment')}" for c in out["root_causes"]]
        log = {"before": before, "after": after, "dropped": [k for k in before if k not in after],
               "changed": before != after}
    confirmed_before = bool(report.get("anomaly_confirmed"))
    if out["root_causes"]:
        out["anomaly_confirmed"] = True
    log["confirmed_set_true"] = bool(out["root_causes"]) and not confirmed_before
    log["reorder"] = reorder
    return out, log


# ---- v3: candidate guard, enforced at submit_report time
def _norm(x) -> str:
    return str(x).strip().lower()


def significant_candidates(evidence: dict) -> set[tuple[str, str]]:
    """(dimension, segment) pairs this investigation itself verified as significant (BH q and history check)."""
    out = set()
    for e in evidence.values():
        r = e.get("result") or {}
        if e.get("tool") == "significance_test" and r.get("significant"):
            out.add((_norm(r.get("dimension")), _norm(r.get("segment"))))
    return out


def candidate_violations(report: dict, evidence: dict) -> list[str]:
    allowed = significant_candidates(evidence)
    return [f"{c.get('dimension')}={c.get('segment')}" for c in report.get("root_causes") or []
            if (_norm(c.get("dimension")), _norm(c.get("segment"))) not in allowed]


def enforce_candidates(report: dict, evidence: dict) -> tuple[dict, list[str]]:
    """Drop causes that are not among the investigation's own significant candidates (ranks renumbered)."""
    allowed = significant_candidates(evidence)
    rcs = sorted(report.get("root_causes") or [], key=lambda r: r.get("rank", 99))
    keep = [c for c in rcs if (_norm(c.get("dimension")), _norm(c.get("segment"))) in allowed]
    dropped = [f"{c.get('dimension')}={c.get('segment')}" for c in rcs if c not in keep]
    return {**report, "root_causes": [{**c, "rank": i} for i, c in enumerate(keep, 1)]}, dropped

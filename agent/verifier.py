"""Grounding verifier: every number in the report must appear in a cited tool output.

How it works:
1. Extract numbers from the narrative, and the numeric fields of each root cause
   (contribution_pct, p_value, impact_brl) plus change_pct.
2. Collect every number from the tool outputs the report cites (report + root-cause evidence_ids).
   If nothing is cited, all evidence is used but the report is marked as citing nothing.
3. A report number is grounded if some evidence number matches it within rounding tolerance
   (the precision the report wrote it with), allowing sign flips ("fell 18%" vs -18.0) and
   fraction <-> percent (0.052 vs 5.2%).
Ignored: dates (2018-03-12), ranks, and small integers 0-10 used as counts or list markers.
"""
from __future__ import annotations

import re

from tools.context import Investigation

NUM_RE = re.compile(r"(?<![\w.])[-+]?(?:R\$\s?)?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
EVID_RE = re.compile(r"\be\d+\b")


def _decimals(tok: str) -> int:
    if "e" in tok.lower():
        return 12
    return len(tok.split(".")[1]) if "." in tok else 0


def extract_numbers(text: str) -> list[tuple[float, int, str]]:
    text = DATE_RE.sub(" ", text or "")
    text = EVID_RE.sub(" ", text)
    out = []
    for m in NUM_RE.finditer(text):
        tok = m.group(0).replace("R$", "").replace(" ", "").replace(",", "")
        try:
            v = float(tok)
        except ValueError:
            continue
        if "." not in tok and "e" not in tok.lower() and 0 <= abs(v) <= 10:
            continue  # small integers: counts, ranks, "top 3"
        out.append((v, _decimals(tok), m.group(0)))
    return out


def _walk_numbers(obj, acc: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk_numbers(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _walk_numbers(v, acc)
    elif isinstance(obj, str):
        for v, _, _ in extract_numbers(obj):
            acc.append(v)


def is_grounded(value: float, decimals: int, evidence: list[float]) -> bool:
    """True if some evidence number equals `value` up to the precision it was written with."""
    tol = 0.5 * 10 ** (-decimals) + 1e-12 if decimals < 12 else 0.0
    for x in evidence:
        for cand in (x, -x, x * 100, -x * 100):
            if abs(value - cand) <= tol:
                return True  # e.g. "18.2" vs -18.2345, "0.0004" vs 0.000412
            if cand != 0 and abs(value - cand) <= 0.02 * abs(cand) and (decimals >= 12 or abs(value) < 1e-3):
                return True  # scientific notation / tiny p-values: 2% relative
            if abs(value) >= 1000 and abs(value - cand) <= 0.005 * abs(cand):
                return True  # "R$ 41,250" vs 41249.87
    return False


def verify(report: dict, inv: Investigation) -> dict:
    cited = set(report.get("evidence_ids") or [])
    for rc in report.get("root_causes") or []:
        cited |= set(rc.get("evidence_ids") or [])
    cited |= set(EVID_RE.findall(report.get("narrative") or ""))
    cited = {c for c in cited if c in inv.evidence}
    pool_ids = cited if cited else set(inv.evidence)
    evidence: list[float] = []
    for eid in pool_ids:
        _walk_numbers(inv.evidence[eid]["result"], evidence)

    claims: list[tuple[str, float, int]] = []
    for v, d, raw in extract_numbers(report.get("narrative") or ""):
        claims.append((f"narrative:{raw}", v, d))
    if report.get("change_pct") is not None:
        claims.append(("change_pct", float(report["change_pct"]), _decimals(repr(float(report["change_pct"])))))
    for rc in report.get("root_causes") or []:
        for k in ("contribution_pct", "p_value", "impact_brl"):
            if rc.get(k) is not None:
                v = float(rc[k])
                claims.append((f"root_cause[{rc.get('rank')}].{k}", v, _decimals(repr(v))))

    bad = [name for name, v, d in claims if not is_grounded(v, d, evidence)]
    n = len(claims)
    return {"n_numbers": n, "n_grounded": n - len(bad), "ungrounded": bad,
            "grounding_rate_pct": round(100.0 * (n - len(bad)) / n, 1) if n else 100.0,
            "cites_evidence": bool(cited)}

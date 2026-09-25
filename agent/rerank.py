"""v2: evidence-based final ranking of the root causes the LLM submitted.

The LLM still chooses the candidate causes, their effect labels and the narrative. After its report passes the
grounding verifier, the causes are re-ordered by the strength of their significance evidence:
significant first, then larger |hist_z|, then smaller q. A cause with no significant test is dropped.

Which test counts for a cause: significance_test evidence on the same dimension and segment, preferring
tests the cause cites, then tests whose effect matches the cause's label (mix = share, rate = rate).
"""
from __future__ import annotations

EFFECT_OF = {"mix": "share", "rate": "rate", "volume": "volume"}


def _tests_for(rc: dict, evidence: dict) -> list[dict]:
    out = []
    cited = set(rc.get("evidence_ids") or [])
    want = EFFECT_OF.get(str(rc.get("effect")))
    for eid, e in evidence.items():
        if e.get("tool") != "significance_test":
            continue
        r = e.get("result") or {}
        if str(r.get("dimension")) != str(rc.get("dimension")) or str(r.get("segment")) != str(rc.get("segment")):
            continue
        out.append({"eid": eid, "cited": eid in cited, "effect_match": r.get("effect_tested") == want,
                     "significant": bool(r.get("significant")), "hist_z": r.get("hist_z"),
                     "q": r.get("q_value_bh")})
    return out


def _best(tests: list[dict]) -> dict | None:
    if not tests:
        return None
    # prefer cited, then effect-matching tests; among those take the strongest evidence
    return sorted(tests, key=lambda t: (not t["cited"], not t["effect_match"], *_strength(t)))[0]


def _strength(t: dict | None) -> tuple:
    if not t:
        return (True, 0.0, 1.0)
    hz = abs(t["hist_z"]) if t.get("hist_z") is not None else 0.0
    q = t["q"] if t.get("q") is not None else 1.0
    return (not t["significant"], -hz, q)


def rerank(report: dict, evidence: dict) -> tuple[dict, dict]:
    """Return (report with re-ranked root causes, log of what changed)."""
    rcs = sorted(report.get("root_causes") or [], key=lambda r: r.get("rank", 99))
    scored = [(rc, _best(_tests_for(rc, evidence))) for rc in rcs]
    kept = [(rc, t) for rc, t in scored if t and t["significant"]]
    dropped = [f"{rc.get('dimension')}={rc.get('segment')}" for rc, t in scored if not (t and t["significant"])]
    kept.sort(key=lambda x: _strength(x[1]))  # stable: ties keep the LLM's order
    new = []
    for i, (rc, t) in enumerate(kept, 1):
        new.append({**rc, "rank": i})
    before = [f"{rc.get('dimension')}={rc.get('segment')}" for rc in rcs]
    after = [f"{rc.get('dimension')}={rc.get('segment')}" for rc in new]
    return {**report, "root_causes": new}, {"before": before, "after": after, "dropped": dropped,
                                            "changed": before != after}

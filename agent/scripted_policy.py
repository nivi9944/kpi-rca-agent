"""A deterministic fake LLM that follows the investigation recipe. Used ONLY to test the agent loop,
the verifier and the eval pipeline end to end without an API key. Its runs go to trial_results/,
never to results/.
"""
from __future__ import annotations

import json
import re

from agent.llm_client import ScriptedLLM


def _tool_outputs(messages):
    outs = []
    for m in messages:
        if m.get("role") == "tool":
            try:
                outs.append(json.loads(m["content"]))
            except (json.JSONDecodeError, TypeError):
                outs.append({})
    return outs


def _call(name, args, i):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def policy(messages):
    task = next(m["content"] for m in messages if m["role"] == "user")
    metric = re.search(r"\((\w+)\) changed", task).group(1)
    week = re.search(r"week of (\d{4}-\d{2}-\d{2})", task).group(1)
    outs = [o for o in _tool_outputs(messages) if "evidence_id" in o]
    i = len(outs)
    if i == 0:
        return _call("detect_anomalies", {"metric": metric, "week": week}, i)
    if i == 1:
        return _call("scan_segments", {"metric": metric, "week": week}, i)
    scan = outs[1]
    unusual = [c for c in scan.get("candidates", []) if c.get("unusual")][:3]
    tests = [o for o in outs[2:] if "q_value_bh" in o]
    if len(tests) < len(unusual):
        c = unusual[len(tests)]
        return _call("significance_test", {"metric": metric, "dimension": c["dimension"], "segment": c["segment"],
                                           "week": week, "effect": "share" if c["effect_scanned"] == "share" else "rate"}, i)
    sig = [t for t in tests if t.get("significant")]
    impacts = [o for o in outs if "impact_brl" in o]
    if sig and not impacts:
        t = sig[0]
        return _call("estimate_impact", {"metric": metric, "dimension": t["dimension"], "segment": t["segment"],
                                         "week": week}, i)
    det = outs[0]["target_week"]
    rcs = []
    for r, t in enumerate(sig[:3], 1):
        eff = "volume" if metric in ("gmv", "orders") else ("mix" if t["effect_tested"] == "share" else "rate")
        imp = next((o for o in impacts if o["segment"] == t["segment"] and o["dimension"] == t["dimension"]), None)
        rcs.append({"rank": r, "dimension": t["dimension"], "segment": t["segment"], "effect": eff,
                    "p_value": t["p_value"], "contribution_pct": imp["contribution_pct"] if imp else None,
                    "impact_brl": imp["impact_brl"] if imp else None, "confidence": "high",
                    "evidence_ids": [t["evidence_id"]] + ([imp["evidence_id"]] if imp else [])})
    if rcs:
        top = rcs[0]
        narrative = (f"{metric} moved {det['change_pct']}% versus the previous 4-week average ({outs[0]['evidence_id']}). "
                     f"The main driver is {top['dimension']}={top['segment']} (p={top['p_value']}, {top['evidence_ids'][0]}).")
    else:
        narrative = f"{metric} moved {det['change_pct']}% ({outs[0]['evidence_id']}); no significant root cause found."
    report = {"metric": metric, "week": week, "anomaly_confirmed": bool(det["is_anomaly"] or rcs),
              "change_pct": det["change_pct"], "evidence_ids": [outs[0]["evidence_id"], scan["evidence_id"]],
              "root_causes": rcs, "narrative": narrative,
              "recommended_actions": ["Review the flagged segment with its owner."] if rcs else []}
    return _call("submit_report", report, i)


def policy_llm():
    return ScriptedLLM(policy)

"""Scoring: compare a report with the planted ground truth.

Top-1   : rank-1 root cause matches dimension AND segment (AND effect for mix/rate scenarios)
Top-3   : ground truth is among the top 3 root causes
False alarm (controls only): the report confirms an anomaly AND names at least one root cause
Detection recall (planted only): anomaly_confirmed is true
"""
from __future__ import annotations


def cause_matches(rc: dict, gt: dict) -> bool:
    if str(rc.get("dimension")) != str(gt["dimension"]) or str(rc.get("segment")) != str(gt["segment"]):
        return False
    if gt.get("effect") in ("mix", "rate"):
        return rc.get("effect") == gt["effect"]
    return True


def score_one(sc: dict, report: dict | None) -> dict:
    gt = sc["ground_truth"]
    rep = report or {}
    rcs = sorted(rep.get("root_causes") or [], key=lambda r: r.get("rank", 99))
    confirmed = bool(rep.get("anomaly_confirmed"))
    out = {"id": sc["id"], "type": sc["type"], "severity": sc["severity"], "is_control": sc["is_control"],
           "failed_run": report is None or bool(rep.get("error"))}
    if sc["is_control"]:
        out["false_alarm"] = bool(confirmed and rcs)
        out["top1"] = out["top3"] = None
        out["detected"] = None
    else:
        out["detected"] = confirmed
        out["top1"] = bool(confirmed and rcs and cause_matches(rcs[0], gt))
        out["top3"] = bool(confirmed and any(cause_matches(r, gt) for r in rcs[:3]))
        out["false_alarm"] = None
    return out


def pct(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(100.0 * sum(bool(x) for x in xs) / len(xs), 1) if xs else None


def mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def summarize(rows: list[dict]) -> dict:
    planted = [r for r in rows if not r["is_control"]]
    controls = [r for r in rows if r["is_control"]]
    return {
        "n_scenarios": len(rows), "n_planted": len(planted), "n_controls": len(controls),
        "top1_accuracy_pct": pct(r["top1"] for r in planted),
        "top3_accuracy_pct": pct(r["top3"] for r in planted),
        "detection_recall_pct": pct(r["detected"] for r in planted),
        "false_alarm_rate_pct": pct(r["false_alarm"] for r in controls),
        "failed_runs": sum(r["failed_run"] for r in rows),
        "grounding_rate_pct": mean(r.get("grounding_rate_pct") for r in rows),
        "ungrounded_reports": sum(1 for r in rows if r.get("ungrounded")),
        "avg_steps": mean(r.get("steps") for r in rows),
        "avg_tokens": mean(r.get("total_tokens") for r in rows),
        "avg_latency_s": mean(r.get("latency_s") for r in rows),
        "avg_est_cost_usd": mean(r.get("est_cost_usd") for r in rows),
    }


def by_type(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r["type"], r["severity"]) for r in rows})
    for t, s in keys:
        g = [r for r in rows if r["type"] == t and r["severity"] == s]
        out.append({"type": t, "severity": s, "n": len(g),
                    "top1_pct": pct(r["top1"] for r in g), "top3_pct": pct(r["top3"] for r in g),
                    "detected_pct": pct(r["detected"] for r in g), "false_alarm_pct": pct(r["false_alarm"] for r in g)})
    return out

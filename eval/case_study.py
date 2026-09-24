"""Real-world case study (not part of the accuracy score): run the deterministic tools on real weeks.

  python -m eval.case_study   -> results/case_study.json

Weeks: Black Friday (2017-11-20) and the Brazilian truckers' strike (week of 2018-05-21).
We only report what the data shows; nothing here is planted.
"""
from __future__ import annotations

import json
from pathlib import Path

from baseline.rules import scan_and_test
from tools.context import Investigation
from tools.registry import execute

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("black_friday_2017", "gmv", "2017-11-20"),
    ("black_friday_2017", "on_time_rate", "2017-11-20"),
    ("truckers_strike_2018", "gmv", "2018-05-21"),
    ("truckers_strike_2018", "orders", "2018-05-21"),
]


def main() -> dict:
    out = {}
    for name, metric, week in CASES:
        inv = Investigation(chart_dir=ROOT / "results" / "case_study_charts")
        det = execute(inv, "detect_anomalies", {"metric": metric, "week": week})["target_week"]
        dec = execute(inv, "decompose_metric", {"metric": metric, "week": week})
        drills = {d: execute(inv, "drill_down", {"metric": metric, "dimension": d, "week": week, "top_k": 3})["top_segments"]
                  for d in ("customer_state", "product_category", "seller_state")}
        rep = scan_and_test(Investigation(), metric, week)
        execute(inv, "make_chart", {"kind": "series", "metric": metric, "week": week})
        out[f"{name}:{metric}"] = {"week": week, "headline": det, "drivers": dec.get("drivers"),
                                   "top_segments_by_dimension": drills,
                                   "significant_segments_scan": rep["root_causes"]}
    path = ROOT / "results" / "case_study.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    res = main()
    for k, v in res.items():
        h = v["headline"]
        print(k, f"change {h['change_pct']:.1f}% z={h['z_score']:.1f}",
              [(r["dimension"], r["segment"]) for r in v["significant_segments_scan"]])

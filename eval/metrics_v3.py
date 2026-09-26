"""v3 metrics: DEV verdict vs v2-final, TEST metrics for v3 against frozen v1/v2/B1 and the rerun B2 (v3 engine).

  python -m eval.metrics_v3         writes results/v3/*.json, *.csv, *.png

Uses the v2 metric definitions (eval/metrics_v2.py): Wilson and week-clustered bootstrap CIs, McNemar, precision /
recall / F1, specificity, detection by severity, effect labels, impact MAPE, grounding audit. If the 15-hour ceiling
stopped TEST v3 early, every investigator is also scored on exactly the scenarios v3 completed (paired comparison).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import eval.metrics_v2 as m2
from eval.metrics_v2 import evaluate, mcnemar, read_runs, wilson
from inject.scenarios import load_manifest

ROOT = Path(__file__).resolve().parents[1]
V2, V3 = ROOT / "results" / "v2", ROOT / "results" / "v3"
TEST = {"v3": V3 / "runs" / "test_v3.jsonl", "B2 (v3 engine)": V3 / "runs" / "test_b2_v3.jsonl",
        "v2": V2 / "runs" / "test_v2.jsonl", "v1": V2 / "runs" / "test_v1.jsonl",
        "B2 (v2 engine)": V2 / "runs" / "test_b2.jsonl", "B1": V2 / "runs" / "test_b1.jsonl"}
LABELS = {"v3": "Agent v3 (after TEST-informed changes)", "B2 (v3 engine)": "B2: scan + test, v3 engine",
          "v2": "Agent v2 (frozen)", "v1": "Agent v1 (frozen)", "B2 (v2 engine)": "B2: scan + test, v2 engine",
          "B1": "B1: top contribution"}
ORDER = list(TEST)
PAIRS = [("v3", "v1"), ("v3", "B2 (v3 engine)"), ("v3", "v2"), ("v3", "B2 (v2 engine)"), ("v2", "v1")]


def _base():
    from metrics.metrics import default_store
    return default_store()


def _strip(s: dict) -> dict:
    s = dict(s)
    s.pop("_top1_list", None)
    return s


# ------------------------------------------------------------------ DEV verdict
def dev_verdict() -> str:
    base = _base()
    man = {s["id"]: s for s in load_manifest()}
    v2f = read_runs(ROOT / "results" / "runs" / "nvidia_v2final_dev.jsonl")
    v3 = read_runs(V3 / "dev" / "runs" / "v3_dev.jsonl")
    b2o = read_runs(ROOT / "results" / "runs" / "baseline_scan.jsonl")
    b2n = read_runs(V3 / "dev" / "runs" / "b2_dev.jsonl")
    common = sorted(set(v2f) & set(v3))
    if not common:
        out = "## v3 DEV verdict\n\nNo v3 DEV scenario completed (time budget); no verdict.\n"
        V3.mkdir(parents=True, exist_ok=True)
        (V3 / "dev_verdict.md").write_text(out, encoding="utf-8")
        return out
    sub = lambda r: {i: r[i] for i in common if i in r}
    a, b = _strip(evaluate(sub(v2f), man, base)), _strip(evaluate(sub(v3), man, base))
    ob = _strip(evaluate(sub(b2o), man, base))
    nb = _strip(evaluate(sub(b2n), man, base)) if b2n else {}
    metrics = [("Top-1", lambda s: s["top1"]["pct"]), ("Precision", lambda s: s["precision"]["pct"]),
               ("Recall", lambda s: s["recall"]["pct"]), ("F1", lambda s: s["f1_pct"]),
               ("Detection recall", lambda s: s["detection_recall"]["pct"]),
               ("Price-drop top-1", lambda s: s["top1_by_type"].get("price_drop", {}).get("pct"))]
    lines = ["## v3 DEV verdict", "",
             f"Same {len(common)} DEV scenarios; rule: PASS if v3 is at least as good as v2-final. "
             f"v3 DEV runs: {len(v3)} of 50 completed. Written automatically by eval/run_v3.py.", "",
             "| Metric | v2-final (DEV) | v3 (DEV) | Delta | Verdict |", "|---|---|---|---|---|"]
    for name, f in metrics:
        x, y = f(a), f(b)
        if x is None or y is None:
            lines.append(f"| {name} | {x} | {y} | n/a | n/a |")
            continue
        d = round(y - x, 1)
        lines.append(f"| {name} | {x}% | {y}% | {'+' if d >= 0 else ''}{d} pts | {'PASS' if y >= x else 'FAIL'} |")
    if nb:
        lines += ["", f"For reference, B2 on DEV: top-1 {ob['top1']['pct']}% (v2 engine) vs {nb['top1']['pct']}% (v3 engine); "
                  f"detection {ob['detection_recall']['pct']}% vs {nb['detection_recall']['pct']}%; "
                  f"specificity {ob['specificity']['pct']}% vs {nb['specificity']['pct']}%."]
    out = "\n".join(lines) + "\n"
    V3.mkdir(parents=True, exist_ok=True)
    (V3 / "dev_verdict.md").write_text(out, encoding="utf-8")
    (V3 / "dev_summary.json").write_text(json.dumps({"v2_final": a, "v3": b, "b2_v2_engine": ob, "b2_v3_engine": nb,
                                                     "n_common": len(common)}, indent=2), encoding="utf-8")
    return out


# ------------------------------------------------------------------ TEST metrics
def main() -> dict:
    base = _base()
    man = {s["id"]: s for s in load_manifest(ROOT / "inject" / "manifest_test.json")}
    runs = {k: read_runs(p) for k, p in TEST.items()}
    runs = {k: v for k, v in runs.items() if v}
    v3_ids = sorted(runs.get("v3", {}))
    partial = 0 < len(v3_ids) < len(man)
    out = {"v3_completed": len(v3_ids), "test_total": len(man), "partial": partial}
    for scope, ids in [("all", None), ("v3_completed", v3_ids if partial else None)]:
        if scope == "v3_completed" and not partial:
            continue
        summ, t1 = {}, {}
        for k, r in runs.items():
            rr = r if ids is None else {i: r[i] for i in ids if i in r}
            s = evaluate(rr, man, base)
            t1[k] = s.pop("_top1_list")
            s["label"] = LABELS[k]
            summ[k] = s
        tests = {}
        for a, b in PAIRS:
            if a in t1 and b in t1:
                common = sorted(set(t1[a]) & set(t1[b]))
                tests[f"{a} vs {b}"] = {"n": len(common), **mcnemar([t1[a][i] for i in common], [t1[b][i] for i in common])}
        out[scope] = {"summary": summ, "mcnemar": tests}
    V3.mkdir(parents=True, exist_ok=True)
    (V3 / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    head = out["v3_completed"] if partial and "v3_completed" in out else out["all"]
    with open(V3 / "by_type_severity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["investigator", "type", "severity", "n", "top1_pct", "ci_lo", "ci_hi"])
        for k in [x for x in ORDER if x in head["summary"]]:
            for key, v in head["summary"][k]["top1_by_type_severity"].items():
                t, s_ = key.split("|")
                w.writerow([k, t, s_, v["n"], v["pct"], v["lo"], v["hi"]])
    # plots: reuse the v2 plotting code with v3 labels and output folder
    m2.V2, m2.ORDER, m2.LABELS = V3, [k for k in ORDER if k in head["summary"]], LABELS
    m2.COLORS = {"v3": "#2F855A", "B2 (v3 engine)": "#718096", "v2": "#2B6CB0", "v1": "#DD6B20",
                 "B2 (v2 engine)": "#A0AEC0", "B1": "#CBD5E0"}
    m2.plot_all(head["summary"])
    return out


# ------------------------------------------------------------------ draft summary (fallback)
def draft_summary() -> None:
    S = json.loads((V3 / "summary.json").read_text(encoding="utf-8"))
    st = json.loads((V3 / "status.json").read_text(encoding="utf-8")) if (V3 / "status.json").exists() else {}
    head = S.get("v3_completed") or S["all"]
    L = ["# V3 summary (draft, written automatically; local only, gitignored)", "",
         f"TEST v3 completed: {S['v3_completed']} of {S['test_total']}"
         + (f" (stopped for the 15-hour time budget, not an error: {st.get('time_budget_stop')})" if st.get("time_budget_stop") else ""), "",
         "| Investigator | Top-1 | Clustered CI | Top-3 | Precision | Recall | F1 | Specificity | Detection |", "|---|---|---|---|---|---|---|---|---|"]
    for k in [x for x in ORDER if x in head["summary"]]:
        s = head["summary"][k]; w = s["week_clustered_ci"]
        L.append(f"| {s['label']} | {s['top1']['pct']}% ({s['top1']['k']}/{s['top1']['n']}) | [{w['top1']['lo']}, {w['top1']['hi']}] | "
                 f"{s['top3']['pct']}% | {s['precision']['pct']}% | {s['recall']['pct']}% | {s['f1_pct']} | {s['specificity']['pct']}% | {s['detection_recall']['pct']}% |")
    L += ["", "McNemar: " + "; ".join(f"{k}: {v['only_first_correct']} vs {v['only_second_correct']}, p = {v['p_value']}" for k, v in head["mcnemar"].items()), "",
          "DEV verdict: see results/v3/dev_verdict.md and DECISIONS.md."]
    Path(ROOT / "V3_SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    o = main()
    h = o.get("v3_completed") or o["all"]
    for k in [x for x in ORDER if x in h["summary"]]:
        s = h["summary"][k]
        print(f"{k:16} top1 {s['top1']['pct']} ({s['top1']['k']}/{s['top1']['n']}) F1 {s['f1_pct']} det {s['detection_recall']['pct']} spec {s['specificity']['pct']}")
    print(h["mcnemar"])

"""Fill the README results section and print draft resume bullets, ONLY from files in results/.

  python -m eval.readme
Replaces the text between <!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ORDER = ["nvidia", "mistral", "mistral-medium", "gemini", "ollama", "baseline_scan", "baseline_topcontrib"]
AGENTS = ("nvidia", "mistral", "mistral-medium", "gemini", "ollama")
FREE = ("nvidia",)  # free tier with no list price: cost is reported as $0 actual, with tokens per run


def _load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else {}


def fmt(v, suffix=""):
    return "n/a" if v is None else f"{v}{suffix}"


def cost_cell(m, cost) -> str:
    if m in FREE:
        return "$0 (free tier)"
    return "$" + format(cost, ".4f") + " est." if cost is not None and m in AGENTS else "n/a"


def build() -> tuple[str, list[str]]:
    summ = {**_load("eval_summary.json"), **_load("baseline_summary.json")}
    models = [m for m in ORDER if m in summ]
    lines = ["| Investigator | Top-1 | Top-3 | Detection recall | False alarms (controls) | Grounding | Avg tool calls | Avg tokens / run | Avg latency | Cost / run |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        s = summ[m]
        cost = s.get("avg_est_cost_usd")
        lines.append(f"| {s['label']} | {fmt(s['top1_accuracy_pct'], '%')} | {fmt(s['top3_accuracy_pct'], '%')} | "
                     f"{fmt(s['detection_recall_pct'], '%')} | {fmt(s['false_alarm_rate_pct'], '%')} | "
                     f"{fmt(s.get('grounding_rate_pct'), '%') if m in AGENTS else 'n/a (no text)'} | "
                     f"{fmt(s.get('avg_steps'))} | {fmt(s.get('avg_tokens')) if m in AGENTS else 'n/a'} | "
                     f"{fmt(s.get('avg_latency_s'), ' s')} | {cost_cell(m, cost)} |")
    n = summ[models[0]] if models else {}
    lines.insert(0, f"Scenarios: **{n.get('n_planted', '?')} planted anomalies + {n.get('n_controls', '?')} controls** "
                    "(inject/manifest.json). Source files: `results/eval_summary.json`, `results/baseline_summary.json`.\n")
    # by type table (top-1) for each model
    rows = list(csv.DictReader(open(RES / "eval_by_type.csv", encoding="utf-8"))) if (RES / "eval_by_type.csv").exists() else []
    if rows:
        types = []
        for r in rows:
            k = (r["type"], r["severity"])
            if k not in types:
                types.append(k)
        full = [m for m in models if not summ[m].get("is_subset")]  # subsets (smoke tests) are too small per type
        models = full
        head = "| Type | Severity | n | " + " | ".join(summ[m]["label"] for m in models) + " |"
        lines += ["", "**Top-1 accuracy by scenario type and severity** (controls: false-alarm rate)", "", head,
                  "|---|---|---|" + "---|" * len(models)]
        for t, sev in types:
            cells = []
            nn = ""
            for m in models:
                r = next((x for x in rows if x["model"] == m and x["type"] == t and x["severity"] == sev), None)
                if r:
                    nn = r["n"]
                    val = r["false_alarm_pct"] if t.startswith("control") else r["top1_pct"]
                    cells.append(f"{val}%" if val else "n/a")
                else:
                    cells.append("n/a")
            lines.append(f"| {t} | {sev} | {nn} | " + " | ".join(cells) + " |")
    cal = _load("calibration.json")
    if cal:
        t = str(cal["chosen_hist_z"])
        lines += ["", f"Segment-scan threshold |hist_z| >= {cal['chosen_hist_z']} was calibrated on "
                      f"{cal['n_week_metric_pairs']} natural (un-planted) week x metric pairs: "
                      f"{round(100 * cal['natural_alarm_rate_by_threshold'][t]['overall'], 1)}% natural alarm rate "
                      "(`results/calibration.json`)."]
    return "\n".join(lines), bullets(summ)


def bullets(summ) -> list[str]:
    a = summ.get("nvidia")  # headline agent model
    b = summ.get("baseline_scan")
    if not a:
        return ["(run the agent evaluation first: python -m eval.run --model nvidia)"]
    out = [
        f"Built an LLM agent with **10** typed tools that diagnoses KPI drops (GMV, AOV, on-time %) across **98K** Olist orders.",
        f"Reached **{a['top1_accuracy_pct']}%** top-1 root-cause accuracy on **{a['n_planted']}** planted anomalies vs "
        f"**{b['top1_accuracy_pct'] if b else '?'}%** for a rule-based baseline.",
        f"Verified **{a['grounding_rate_pct']}%** of report figures against tool outputs, with **{a['false_alarm_rate_pct']}%** "
        f"false alarms on control weeks.",
        f"Diagnosed each KPI change in **{a['avg_latency_s']} s** using **~{a['avg_tokens']}** tokens per run via my LLM gateway.",
    ]
    return [f"{x}  ({len(re.sub(r'[*]', '', x))} chars)" for x in out]


def main():
    text, bl = build()
    readme = ROOT / "README.md"
    s = readme.read_text(encoding="utf-8")
    s = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
               "<!-- RESULTS:START -->\n" + text.replace("\\", "\\\\") + "\n<!-- RESULTS:END -->", s, flags=re.S)
    readme.write_text(s, encoding="utf-8")
    print(text)
    print("\nDraft resume bullets (from results/):")
    for b in bl:
        print(" -", b)


if __name__ == "__main__":
    main()

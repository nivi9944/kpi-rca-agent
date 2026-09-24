"""Build every results file from results/runs/*.jsonl (the ONLY source of resume numbers).

  python -m eval.report_tables
Outputs: results/eval_summary.json, baseline_summary.json, eval_by_type.csv, model_comparison.json,
         accuracy_by_severity.png, agent_vs_baseline.png, examples/*.md
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval.score import by_type, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
AGENTS = ["nvidia", "mistral", "mistral-medium", "gemini", "ollama"]
BASELINES = ["baseline_topcontrib", "baseline_scan"]
LABELS = {"nvidia": "Agent (Nemotron 3 Ultra, NVIDIA)", "mistral": "Agent (Mistral Small 4)", "mistral-medium": "Agent (Mistral Medium 3.5)", "gemini": "Agent (Gemini Flash)", "ollama": "Agent (Qwen 2.5 local)",
          "baseline_topcontrib": "Baseline B1: top contribution", "baseline_scan": "Baseline B2: scan + test"}
TYPE_ORDER = ["volume_drop", "price_drop", "delivery_delay", "cancellation_spike", "review_drop", "mix_shift",
              "control_clean", "control_noise"]


def read_jsonl(p: Path) -> tuple[list[dict], int]:
    """Latest row per scenario, plus the number of infra-failed attempts (error, no report) in the file.

    Failed attempts are retried by eval.run and stay in the file; only the latest attempt is scored, so a
    retried scenario is never counted twice, and the failed attempts are still reported."""
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    infra = sum(1 for r in rows if r.get("error") and not r.get("report"))
    latest = {}
    for r in rows:
        latest[r["id"]] = r
    return list(latest.values()), infra


def load_runs(res: Path = RES) -> dict[str, list[dict]]:
    return {p.stem: read_jsonl(p)[0] for p in sorted((res / "runs").glob("*.jsonl"))}


def infra_failures(res: Path = RES, subdir: str = "runs") -> dict[str, int]:
    return {p.stem: read_jsonl(p)[1] for p in sorted((res / subdir).glob("*.jsonl"))}


def score_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        s = dict(r["score"])
        for k in ("steps", "total_tokens", "latency_s", "est_cost_usd", "grounding_rate_pct", "ungrounded"):
            s[k] = r.get(k)
        out.append(s)
    return out


def write_example(r: dict, path: Path) -> None:
    rep = r.get("report") or {}
    lines = [f"# {r['id']} ({r['model']})", "", f"**Metric:** {rep.get('metric')}  ",
             f"**Week:** {rep.get('week')}  ", f"**Anomaly confirmed:** {rep.get('anomaly_confirmed')}  ",
             f"**Change:** {rep.get('change_pct')}%", "", "## Root causes", ""]
    for rc in rep.get("root_causes") or []:
        lines.append(f"{rc.get('rank')}. {rc.get('dimension')} = {rc.get('segment')} | effect {rc.get('effect')} | "
                     f"contribution {rc.get('contribution_pct')}% | p {rc.get('p_value')} | impact R$ {rc.get('impact_brl')} | "
                     f"evidence {', '.join(rc.get('evidence_ids') or [])}")
    lines += ["", "## Narrative", "", rep.get("narrative") or "(baseline: no narrative)", "",
              "## Actions", ""] + [f"- {a}" for a in rep.get("recommended_actions") or []]
    if r.get("trace"):
        lines += ["", "## Tool-call trace", ""]
        for ev in r["trace"]:
            if ev.get("type") == "tool":
                lines.append(f"- step {ev['step']}: `{ev['tool']}` {ev['args']} -> {ev.get('evidence_id') or ev.get('error')}")
    if r.get("verification"):
        v = r["verification"]
        lines += ["", f"**Grounding:** {v['n_grounded']}/{v['n_numbers']} numbers verified ({v['grounding_rate_pct']}%)"]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(res: Path = RES) -> dict:
    runs = load_runs(res)
    if not runs:
        print("no runs yet")
        return {}
    summaries = {m: summarize(score_rows(rows)) for m, rows in runs.items()}
    infra = infra_failures(res)
    from inject.scenarios import load_manifest
    n_full = len(load_manifest())
    for m, s in summaries.items():
        s["infra_failed_attempts"] = infra.get(m, 0)
        # a model run on fewer scenarios than the manifest (e.g. a smoke test) is kept and labelled, but left
        # out of the charts and the per-type table so it is never read as a full result
        s["is_subset"] = s["n_scenarios"] < n_full
        s["label"] = LABELS.get(m, m) + (f" (subset: {s['n_scenarios']} of {n_full} scenarios)" if s["is_subset"] else "")
    agent_s = {m: summaries[m] for m in AGENTS if m in summaries}
    base_s = {m: summaries[m] for m in BASELINES if m in summaries}
    if agent_s:
        (res / "eval_summary.json").write_text(json.dumps(agent_s, indent=2))
    if base_s:
        (res / "baseline_summary.json").write_text(json.dumps(base_s, indent=2))
    if len(agent_s) > 1:
        (res / "model_comparison.json").write_text(json.dumps(
            {m: {k: s[k] for k in ("top1_accuracy_pct", "top3_accuracy_pct", "false_alarm_rate_pct",
                                   "grounding_rate_pct", "avg_latency_s", "avg_tokens")} for m, s in agent_s.items()}, indent=2))

    # by type x severity, all models side by side
    with open(res / "eval_by_type.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "type", "severity", "n", "top1_pct", "top3_pct", "detected_pct", "false_alarm_pct"])
        for m, rows in runs.items():
            for t in sorted(by_type(score_rows(rows)), key=lambda x: (TYPE_ORDER.index(x["type"]) if x["type"] in TYPE_ORDER else 99, x["severity"])):
                w.writerow([m, t["type"], t["severity"], t["n"], t["top1_pct"], t["top3_pct"], t["detected_pct"], t["false_alarm_pct"]])

    # plot 1: accuracy by severity
    models = [m for m in AGENTS + BASELINES if m in runs and not summaries[m]["is_subset"]]
    sev = ["small", "medium", "large"]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=120)
    width = 0.8 / max(1, len(models))
    colors = ["#2B6CB0", "#DD6B20", "#718096", "#A0AEC0"]
    for i, m in enumerate(models):
        rows = [r for r in score_rows(runs[m]) if not r["is_control"]]
        vals = []
        for s_ in sev:
            g = [r["top1"] for r in rows if r["severity"] == s_]
            vals.append(100 * sum(bool(x) for x in g) / len(g) if g else 0)
        ax.bar([x + i * width for x in range(3)], vals, width, label=summaries[m]["label"], color=colors[i % 4])
    ax.set_xticks([x + width * (len(models) - 1) / 2 for x in range(3)], sev)
    ax.set_ylabel("top-1 accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Root-cause top-1 accuracy by severity")
    ax.legend(frameon=False, fontsize=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(res / "accuracy_by_severity.png")
    plt.close(fig)

    # plot 2: headline metrics per model
    keys = [("top1_accuracy_pct", "Top-1"), ("top3_accuracy_pct", "Top-3"),
            ("detection_recall_pct", "Detection"), ("false_alarm_rate_pct", "False alarm")]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=120)
    for i, m in enumerate(models):
        vals = [summaries[m][k] or 0 for k, _ in keys]
        ax.bar([x + i * width for x in range(len(keys))], vals, width, label=LABELS.get(m, m), color=colors[i % 4])
    ax.set_xticks([x + width * (len(models) - 1) / 2 for x in range(len(keys))], [lab for _, lab in keys])
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title("Agent vs rule-based baselines (planted anomalies + controls)")
    ax.legend(frameon=False, fontsize=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(res / "agent_vs_baseline.png")
    plt.close(fig)

    # gateway cache hit rate on a second, identical pass (python -m eval.run --model gemini --cache-rerun)
    cache = {}
    for p in sorted((res / "runs_cache").glob("*.jsonl")) if (res / "runs_cache").exists() else []:
        rows, infra_c = read_jsonl(p)
        calls = sum(r.get("llm_calls") or 0 for r in rows)
        hits = sum(r.get("cache_hits") or 0 for r in rows)
        first = {r["id"]: r for r in runs.get(p.stem, [])}
        cache[p.stem] = {"scenarios": len(rows), "infra_failed_attempts": infra_c, "llm_calls": calls, "cache_hits": hits,
                         "hit_rate_pct": round(100 * hits / calls, 1) if calls else None,
                         "avg_latency_first_s": round(sum(first[r["id"]]["latency_s"] for r in rows if r["id"] in first) / max(1, sum(r["id"] in first for r in rows)), 2),
                         "avg_latency_rerun_s": round(sum(r.get("latency_s") or 0 for r in rows) / max(1, len(rows)), 2)}
    if cache:
        (res / "cache_rerun.json").write_text(json.dumps(cache, indent=2))

    # three example reports from the best available agent run (fall back to a baseline)
    ex_model = next((m for m in AGENTS if m in runs), None) or next(iter(runs))
    exdir = res / "examples"
    exdir.mkdir(exist_ok=True)
    for old in exdir.glob("*.md"):
        old.unlink()
    picks = []
    for want in ("mix_shift", "delivery_delay", "control_clean"):
        cand = [r for r in runs[ex_model] if r["id"].split("_", 1)[1].startswith(want) and r.get("report")]
        good = [r for r in cand if r["score"].get("top1") or r["score"].get("false_alarm") is False]
        if good or cand:
            picks.append((good or cand)[-1])
    for r in picks:
        write_example(r, exdir / f"{r['id']}_{ex_model}.md")
    print(json.dumps(summaries, indent=2))
    return summaries


if __name__ == "__main__":
    main()

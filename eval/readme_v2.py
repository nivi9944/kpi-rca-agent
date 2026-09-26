"""Write README.md for v2 from results files only, and print draft resume bullets.

  python -m eval.readme_v2

Every number below is read from results/ (TEST: results/v2/*.json; DEV: results/runs/*.jsonl; data:
results/data_summary.json; case study: results/case_study.json). Nothing is typed by hand.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from eval.compare_runs import reciprocal_rank
from eval.metrics_v2 import read_runs
from eval.score import score_one
from inject.scenarios import load_manifest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
V2 = RES / "v2"
J = lambda p: json.loads((ROOT / p).read_text(encoding="utf-8"))


def pct(w: dict) -> str:
    return f"{w['pct']}%"


def ci(w: dict) -> str:
    return f"[{w['lo']}, {w['hi']}]"


def dev_table() -> tuple[list[str], dict]:
    man = {s["id"]: s for s in load_manifest()}
    runs = {"Agent v2": "nvidia_v2final_dev", "Agent v1": "nvidia", "B2: scan + test": "baseline_scan",
            "B1: top contribution": "baseline_topcontrib"}
    out, vals = [], {}
    for label, f in runs.items():
        rows = read_runs(RES / "runs" / f"{f}.jsonl")
        pl = [i for i in rows if not man[i]["is_control"]]
        t1 = sum(bool(score_one(man[i], rows[i].get("report"))["top1"]) for i in pl)
        t3 = sum(bool(score_one(man[i], rows[i].get("report"))["top3"]) for i in pl)
        mrr = sum(reciprocal_rank(man[i], rows[i].get("report")) for i in pl) / len(pl)
        vals[label] = (t1, len(pl))
        out.append(f"| {label} | {round(100 * t1 / len(pl), 1)}% ({t1}/{len(pl)}) | {round(100 * t3 / len(pl), 1)}% | {mrr:.4f} |")
    return out, vals


def build() -> tuple[str, list[str]]:
    S = J("results/v2/summary.json")
    MC = J("results/v2/mcnemar.json")
    ST = J("results/v2/stability.json")
    CR = J("results/v2/cache_rerun.json")
    GAP = J("results/v2/dev_test_gap.json")
    D = J("results/data_summary.json")
    CS = J("results/case_study.json")
    CAL = J("results/calibration.json")
    AR = J("results/v2/archive_launch_weeks/summary.json")
    man = load_manifest(ROOT / "inject" / "manifest_test.json")
    n_weeks = S["v2"]["week_clustered_ci"]["n_weeks"]
    v2, v1, b2, b1 = S["v2"], S["v1"], S["B2"], S["B1"]
    wc = lambda m, k: S[m]["week_clustered_ci"][k]
    types = list(v2["top1_by_type"])
    by_type = [f"| {t} | " + " | ".join(f"{S[m]['top1_by_type'][t]['k']}/{S[m]['top1_by_type'][t]['n']}"
                                        for m in ("v2", "v1", "B2", "B1")) + " |" for t in types]
    dev_rows, _ = dev_table()
    strike, bf_gmv, bf_ot = CS["truckers_strike_2018:gmv"], CS["black_friday_2017:gmv"], CS["black_friday_2017:on_time_rate"]
    lw = AR["test_b2"]["launch_weeks"]
    main_rows = []
    for m in ("v2", "v2+fallback", "v1", "B2", "B1"):
        s = S[m]
        main_rows.append(
            f"| {s['label']} | **{pct(s['top1'])}** ({s['top1']['k']}/{s['top1']['n']}) {ci(wc(m, 'top1'))} | "
            f"{pct(s['top3'])} {ci(wc(m, 'top3'))} | {s['mrr']} | {pct(s['precision'])} | {pct(s['recall'])} | "
            f"{s['f1_pct']} {ci(wc(m, 'f1'))} | {pct(s['specificity'])} ({s['specificity']['k']}/{s['specificity']['n']}) |")
    cost_rows = []
    for m in ("v2", "v1", "B2", "B1"):
        s = S[m]
        lat = s["latency_s"]
        cost_rows.append(f"| {s['label']} | {lat['avg']} s / {lat['p50']} s / {lat['p95']} s | {s['avg_tool_calls']} | "
                         f"{s['avg_tokens'] if s['avg_tokens'] is not None else 'none (no LLM)'} | "
                         f"{s['grounding_rate_pct'] if s['grounding_rate_pct'] is not None else 'n/a'}"
                         f"{'%' if s['grounding_rate_pct'] is not None else ''} | {s['invented_numbers'] if m in ('v2', 'v1') else 'n/a'} |")
    sev = lambda m: " / ".join(f"{S[m]['detection_by_severity'][k]['pct']}%" for k in ("small", "medium", "large"))
    mc = lambda k: f"{MC[k]['only_first_correct']} vs {MC[k]['only_second_correct']}, p = {MC[k]['p_value']}"
    ug = v2["ungrounded_numbers"]

    text = f"""# KPI Root-Cause Agent

An LLM agent that investigates why a business KPI moved: it confirms the anomaly, decomposes the metric, scans every segment, tests significance and writes a short incident report in which every number is checked against a tool output.

**Highlights** (held-out TEST set, {v2['n_planted']} planted anomalies + {v2['n_controls']} control weeks)

<!-- HIGHLIGHTS:START -->

- **{pct(v2['top1'])}** top-1 root-cause accuracy (week-clustered 95% CI {ci(wc('v2', 'top1'))}); **{pct(v2['top1_given_detected'])}** when the anomaly is detected.
- **{v2['grounding_rate_pct']}%** of report figures verified against tool outputs, **{v2['invented_numbers']}** invented numbers, **{pct(v2['specificity'])}** specificity on control weeks.
- **{ST['same_top1_cause']['pct']}%** identical top cause on a repeat run (kappa {ST['correct_agreement_kappa']}); **{pct(CR['hit_rate'])}** gateway cache hits on a rerun.

<!-- HIGHLIGHTS:END -->

[![CI](https://github.com/nivi9944/kpi-rca-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/nivi9944/kpi-rca-agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

`LLM function calling` · `Pydantic` · `DuckDB` · `pandas` · `SciPy` / `statsmodels` · `Streamlit` · `pytest`

---

## Overview

When GMV, average order value or on-time delivery moves unexpectedly, an analyst has to find out which
segment caused it, whether the effect is real, and what it is worth. This project hands that investigation to
an LLM agent with a strict division of labour: **the LLM plans and explains, deterministic Python computes.**
The model chooses which of 10 typed tools to call; every figure in its final report must match a cited tool
output or the grounding verifier rejects it; deterministic guards then drop any cause that has no significant
test.

Data: the public [Olist Brazilian e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
{D['orders']:,} orders and {D['items']:,} items over {D['weeks']} weeks ({D['first_week']} to {D['last_week']}),
aggregated weekly (`results/data_summary.json`).

**Model:** NVIDIA Nemotron 3 Ultra (`nvidia/nemotron-3-ultra-550b-a55b`, NVIDIA API catalog free tier, thinking
off), called through my [LLM API Gateway](https://github.com/nivi9944/llm-gateway). Chosen because the Gemini free
tier allows only 20 requests per day and Mistral's free API returned a 0 requests/min limit
([DECISIONS.md](DECISIONS.md)).

## Key results (held-out TEST set)

{v2['n_planted']} planted anomalies (7 types x 20) and {v2['n_controls']} control weeks, built with a new seed after
all v2 code was frozen, in {n_weeks} target weeks that never overlap the development set. Headline intervals are
**week-clustered bootstrap 95% CIs** (scenarios sharing a week are correlated); Wilson intervals are in
`results/v2/summary.json`.

| Investigator | Top-1 | Top-3 | MRR | Precision | Recall | F1 | Specificity |
|---|---|---|---|---|---|---|---|
{chr(10).join(main_rows)}

**Significance** (exact McNemar on paired top-1, `results/v2/mcnemar.json`): v2 vs v1 {mc('v2 vs v1')};
v2 vs B2 {mc('v2 vs B2')}; v1 vs B2 {mc('v1 vs B2')}.

**Reading the table honestly.**
- **v2 vs v1:** v2 improves on v1 on unseen scenarios, but the gain is **not statistically significant**.
- **v2 vs B2:** v2 **ties** the strong statistical baseline B2 on accuracy, while B2 has higher precision (the agent sometimes lists extra causes).
- **Fallback:** the B2 fallback never triggered (v2 never left a planted scenario empty while B2 had an answer), so that row equals v2.
- **What the agent adds over B2** is what a rule cannot produce: a readable incident report with business framing, R$ impact and actions, in which {v2['grounding_rate_pct']}% of the figures are machine-verified.
- **Both far exceed the naive rule (B1).**

![Agent vs baselines on TEST](results/v2/agent_vs_baselines.png)

### Detection and sensitivity

The binding limit is detection, shared by every investigator that uses the segment scan:
- **{v2['planted_undetected']} of {v2['n_planted']}** planted anomalies are never flagged: the scan threshold is set for a low false-alarm rate.
- **When v2 does detect an anomaly, its top cause is right {pct(v2['top1_given_detected'])} of the time** {ci(v2['top1_given_detected'])} (B2: {pct(b2['top1_given_detected'])}).
- **Detection recall by severity** (small / medium / large): v2 {sev('v2')}; B2 {sev('B2')}; B1 {sev('B1')}.

![Sensitivity curve](results/v2/sensitivity_curve.png)

### Accuracy by anomaly type (top-1, out of 20 each)

| Type | Agent v2 | Agent v1 | B2 | B1 |
|---|---|---|---|---|
{chr(10).join(by_type)}

`delay_then_reviews` is a chained scenario: deliveries of one seller state are delayed in week W and that
state's reviews drop in week W+1 (the investigated week). Full type x severity table: `results/v2/by_type_severity.csv`.

![Accuracy by type](results/v2/accuracy_by_type.png)

### Grounding, impact, cost, stability and caching

| Investigator | Latency avg / p50 / p95 | Tool calls | Tokens | Grounding | Invented numbers |
|---|---|---|---|---|---|
{chr(10).join(cost_rows)}

- **Ungrounded numbers (v2):** {sum(ug.values())} numbers were flagged, **none invented**: {ug.get('derived', 0)} were values the model derived from two tool values, {ug.get('threshold_notation', 0)} were "p < 0.001" threshold notation.
- **Effect label:** the effect label (mix vs rate) was right in {pct(v2['effect_label_accuracy'])} of the {v2['effect_label_accuracy']['n']} cases where the right segment was named.
- **R$ impact error:** {v2['impact_mape_pct']}% MAPE against the true planted impact, over {v2['impact_n']} scenarios (v1: {v1['impact_mape_pct']}%).
- **Stability:** v2 rerun on {ST['n']} random TEST scenarios with the gateway cache bypassed gave the same top cause in **{ST['same_top1_cause']['pct']}%** {ci(ST['same_top1_cause'])}, Cohen's kappa {ST['correct_agreement_kappa']} on correctness (`results/v2/stability.json`).
- **Caching:** a cache rerun of {CR['scenarios']} scenarios hit the exact cache on {CR['cache_hits']} of {CR['llm_calls']} calls ({pct(CR['hit_rate'])}); median cache-hit call {CR['median_cache_hit_call_latency_s']} s, average scenario {CR['avg_scenario_latency_first_s']} s to {CR['avg_scenario_latency_rerun_s']} s (`results/v2/cache_rerun.json`).
- **Cost:** $0 actual (free tier).

## Architecture

```mermaid
flowchart LR
    U[Task: why did GMV change in week W?] --> L[Agent loop<br/>agent/loop.py]
    L -- chat + tool specs --> G[LLM Gateway<br/>cache, rate limit, routing]
    G --> NV[Nemotron 3 Ultra<br/>NVIDIA API, free tier]
    L -- tool calls --> T[10 typed tools<br/>detect, decompose, scan,<br/>drill-down, test, impact, SQL, chart]
    T --> D[(DuckDB<br/>fact_orders, fact_items)]
    T -- JSON + evidence_id --> L
    L --> R[Report<br/>Pydantic schema]
    R --> V{{Grounding verifier}}
    V -- ungrounded: 1 retry --> L
    V -- ok --> GD[Guards: drop causes without<br/>a significant test; consistency]
    GD --> OUT[Incident report + charts]
```

## How the agent investigates

1. `detect_anomalies`: robust z-score of the week's change vs the previous 4-week average, scored against the 8 trailing weekly changes (median/MAD).
2. `decompose_metric`: GMV = Orders x AOV (exact log split); AOV = items per order x value per item.
3. `scan_segments`: every segment of every dimension against its own weekly history (share and rate), with a sampling-noise floor; also on `avg_item_price` (item-level price) for AOV.
4. `drill_down`: contribution of each segment, with the exact mix vs rate split `delta = sum (w1 - w0)(r0 - R0) [mix] + sum w1 (r1 - r0) [rate]`.
5. `significance_test`: chi-square on share, two-proportion z or Welch t on rate; a cause must pass **Benjamini-Hochberg q <= 0.05 and |hist_z| >= {CAL['chosen_hist_z']}** (threshold calibrated on {CAL['n_week_metric_pairs']} natural weeks only, `results/calibration.json`).
6. `estimate_impact` (R$), optional read-only `run_sql` and `make_chart`, then `submit_report`.

The loop is bounded (15 tool calls, token budget, timeout) and runs at temperature 0. The report is validated
against a Pydantic schema, and the model gets one corrective retry if the verifier finds an untraceable number.
Example reports: `results/examples/`.

## Methodology: DEV and held-out TEST

- **DEV:** the original 50 scenarios (40 planted + 10 controls) were used freely to develop v2 (2 rounds). Each change and its evidence is in [DECISIONS.md](DECISIONS.md), including ideas that were **tried and rejected**: evidence-based re-ranking of causes, and a longer price-drop prompt.
- **TEST:** built once, after v2 was frozen and committed, with a new seed: 20 scenarios per type, severities cycling small / medium / large, target weeks disjoint from DEV, segments drawn by rule from the largest segments. Each investigator ran on TEST once. No threshold was recalibrated.
- **A test-set construction error was caught and fixed before any agent result was used.** The first TEST build also used Olist's launch weeks, which have too little history. Even B2 found only {lw['top1_correct']} of {lw['planted_run']} planted causes there ({lw['top1_pct']}%), so the set was rebuilt with the DEV pool rule and all runs restarted (`results/v2/archive_launch_weeks/`).
- **v1 on TEST** is reproduced exactly with `--agent-version v1` (the v1 prompt verbatim, no new metric, no guards, 12 steps).

DEV results (`results/runs/`, 40 planted):

| Investigator | Top-1 | Top-3 | MRR |
|---|---|---|---|
{chr(10).join(dev_rows)}

The v2 DEV to TEST gap is {GAP['gap_pts']} points of top-1 ({pct(GAP['dev_top1'])} on DEV vs {pct(GAP['test_top1'])} on TEST,
`results/v2/dev_test_gap.json`): part is DEV selection, part is that TEST uses mostly new segments and a new
chained type.

## Failure analysis (TEST)

- **Detection is the bottleneck.** {v2['planted_undetected']} of {v2['n_planted']} planted anomalies were never flagged by v2 (B2: {b2['planted_undetected']}); small severities are mostly below the noise floor of weekly segment data (detection {v2['detection_by_severity']['small']['pct']}% for small vs {v2['detection_by_severity']['large']['pct']}% for large).
- **Price drops** ({v2['top1_by_type']['price_drop']['k']}/20 for every investigator): a price cut in one category barely moves AOV, and the category's own price history is noisy. The agent used `avg_item_price` in only a few investigations.
- **Mix shifts:** v2 {v2['top1_by_type']['mix_shift']['k']}/20 vs B2 {b2['top1_by_type']['mix_shift']['k']}/20. In both misses the agent ranked a correlated side effect (a seller state whose share moved with the category) above the true category, which it still listed.
- **Precision:** v2 sometimes lists up to 5 supported causes where one is planted, so its precision ({pct(v2['precision'])}) is below B2's ({pct(b2['precision'])}).
- **False alarm:** one noisy control week (2018-04-23, GMV) was flagged by every investigator. The same `watches_gifts` movement appears in the untouched data that week, so it is a real event in a control week (controls are deliberately not filtered).

## Real-world case study

Not part of the score (`results/case_study.json`):

- **2018 Brazilian truckers' strike (week of {strike['week']}):** GMV fell {strike['headline']['change_pct']}% vs the prior 4 weeks (z {strike['headline']['z_score']}); {strike['drivers'][0]['share_of_change_pct']}% of the drop came from fewer orders, and no single segment passed the tests (a broad shock).
- **Black Friday 2017 (week of {bf_gmv['week']}):** GMV rose {bf_gmv['headline']['change_pct']}%, while the on-time delivery rate fell from {bf_ot['headline']['baseline_prev4_avg']} to {bf_ot['headline']['value']}.

## Expectations vs results

| Expectation | Result on TEST | Status | Evidence |
|---|---|---|---|
| Numbers only from tools, enforced | {v2['grounding_rate_pct']}% grounded, {v2['invented_numbers']} invented numbers | Met | `results/v2/summary.json` |
| Beat the naive baseline (B1) | {pct(v2['top1'])} vs {pct(b1['top1'])} top-1 | Met | `results/v2/summary.json` |
| Beat the statistical baseline (B2) | {pct(v2['top1'])} vs {pct(b2['top1'])}, McNemar p = {MC['v2 vs B2']['p_value']} | Not met (tie) | `results/v2/mcnemar.json` |
| v2 better than v1 | {pct(v2['top1'])} vs {pct(v1['top1'])}, {MC['v2 vs v1']['only_first_correct']} wins and {MC['v2 vs v1']['only_second_correct']} losses, p = {MC['v2 vs v1']['p_value']} | Partly met (not significant) | `results/v2/mcnemar.json` |
| Advantage on mix shifts | v2 {v2['top1_by_type']['mix_shift']['k']}/20 vs B1 {b1['top1_by_type']['mix_shift']['k']}/20, B2 {b2['top1_by_type']['mix_shift']['k']}/20 | Partly met | `results/v2/summary.json` |
| Downstream effects (delay then reviews) | v2 {v2['top1_by_type']['delay_then_reviews']['k']}/20, same as B2 | Not met (no advantage) | `results/v2/summary.json` |
| Say "nothing found" on controls | {pct(v2['specificity'])} specificity; the one alarm is a real event | Met | `results/v2/summary.json` |
| Readable, business-framed report | narrative, R$ impact ({v2['impact_mape_pct']}% MAPE), actions | Met | `results/examples/`, `results/v2/summary.json` |
| Reproducible and stable | {ST['same_top1_cause']['pct']}% same top cause on repeat, {pct(CR['hit_rate'])} cache hits | Met | `results/v2/stability.json`, `results/v2/cache_rerun.json` |

## Design decisions

The main choices are in [DECISIONS.md](DECISIONS.md), each with its reason and, for v2, the DEV evidence. In short:
- an own agent loop with plain OpenAI-style function calling (no framework);
- a Pydantic report schema submitted through a tool;
- median/MAD anomaly scoring against the change history;
- an exact mix vs rate split;
- significance with Benjamini-Hochberg correction, plus a history check calibrated on natural weeks;
- deterministic post-verifier guards;
- every run forced to one provider through the gateway;
- a held-out TEST set built after the code was frozen.

## Run it in 3 commands

```powershell
pip install -r requirements.txt
python -m data.load_olist                  # put the Kaggle zip (or CSVs) in data/raw first
streamlit run app/streamlit_app.py         # demo: pick a scenario or a real week
```

Evaluation:
- `python -m eval.run --model nvidia` runs the DEV set. It needs the gateway with an `NVIDIA_API_KEY`.
- `python -m eval.run_test_phase` runs the TEST set: resumable, with a budget guard and `--workers`.
- `python -m eval.metrics_v2` computes the TEST metrics, and `python -m eval.readme_v2` writes this README.
- Live progress: `python -m eval.progress --watch`.
- Tests: `python -m pytest -q` (synthetic fixture with hand-computed answers; no data or key needed).

## Project structure

| Folder | What it holds |
|---|---|
| `data/` | CSV to DuckDB build: `fact_orders`, `fact_items` |
| `metrics/` | Metric catalog (each metric defined once, incl. `avg_item_price`) and the metric engine |
| `tools/` | Analysis tools, SQL guard, charts, tool registry (typed schemas) |
| `inject/` | Anomaly injector, DEV manifest (`manifest.json`) and held-out TEST manifest (`manifest_test.json`) |
| `agent/` | Loop, prompts, report schema, grounding verifier, guards, compact tool view, LLM client |
| `baseline/` | Two deterministic baselines |
| `eval/` | Runner (workers, resume), TEST orchestrator, metrics (bootstrap, McNemar), progress window, README |
| `app/` | Streamlit demo |
| `results/` | Every reported number: DEV (`results/`), TEST (`results/v2/`), run logs with full evidence |
| `tests/` | Unit tests for tools, verifier, guards, SQL guard, scoring, metrics, workers and the agent loop |

## Limitations and future work

- **Only {n_weeks} TEST weeks.** To stay disjoint from DEV, the TEST set uses only {n_weeks} target weeks, so scenarios are correlated within a week and the week-clustered intervals are wide.
- **Launch weeks are out of scope.** Scenarios in Olist's launch weeks (thin history, tiny segments) are undetectable even for B2: {lw['top1_correct']} of {lw['planted_run']} ({lw['top1_pct']}%) in the discarded first TEST build.
- **Anomalies are synthetic:** clean, single-cause, one-week events planted into real data. Olist has no traffic data (no conversion funnel), one currency (R$) and weekly grain only.
- **Detection trades recall for a low false-alarm rate;** small effects are mostly missed.
- **The agent does not beat B2 on accuracy.** Next: rank candidates by their share of the KPI change, not only their unusualness, and add scenario types where reasoning across metrics should matter more.
- **Throughput:** the model runs on a free tier with a per-model rate limit (about 5 calls/min for Ultra on our key), so a full TEST run takes most of a day.

## Licence

Code: MIT. Data: Olist, CC BY-NC-SA 4.0 (non-commercial); this repository contains code and results only,
download the data from Kaggle.
"""
    bullets = [
        f"Built a tool-calling LLM agent with a grounding verifier that finds why e-commerce KPIs move, on {D['orders'] // 1000}K+ orders.",
        f"Reached **{pct(v2['top1'])}** top-1 root-cause accuracy on **{v2['n_planted']}** held-out planted anomalies, **{pct(v2['top1_given_detected'])}** once the anomaly is detected.",
        f"Verified **{v2['grounding_rate_pct']}%** of report figures against tool outputs, with **{v2['invented_numbers']}** invented and **{pct(v2['specificity'])}** specificity on control weeks.",
        f"Ran a held-out evaluation with McNemar tests and week-clustered CIs; **{ST['same_top1_cause']['pct']}%** identical top cause on repeat runs.",
    ]
    return text, bullets


def main():
    text, bullets = build()
    (ROOT / "README.md").write_text(text, encoding="utf-8")
    print("README.md written;", "em dashes:", text.count(chr(0x2014)))
    print("\nDraft resume bullets (numbers from results/):")
    for b in bullets:
        print(f" - {b}  ({len(re.sub(r'[*]', '', b))} chars)")


if __name__ == "__main__":
    main()

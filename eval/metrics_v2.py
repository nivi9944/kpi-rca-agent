"""v2 evaluation metrics on the held-out TEST set (inject/manifest_test.json). Reads run files only.

  python -m eval.metrics_v2          writes results/v2/*.json, *.csv, *.png and results/data_summary.json

Investigators: B1, B2 (rules), v1 and v2 (the agent, Nemotron 3 Ultra), and "v2 + fallback" (scored offline:
the v2 report, but when it names no significant cause and B2 found one, B2's top cause is used, flagged
source=fallback). v2 and v2 + fallback are always separate rows.

Definitions (planted = scenarios with a ground truth; causes count only when the anomaly is confirmed):
  top-1 / top-3 / MRR     rank of the first cause matching the truth (dimension + segment, + effect for mix/rate)
  precision / recall / F1 correct cause = TP; every wrong or extra cause on a planted scenario = FP;
                          planted scenario without the correct cause = FN; flagged control = FP
  specificity             share of control weeks with no confirmed root cause
  detection recall        planted scenarios with anomaly_confirmed = true, by severity
  effect-label accuracy   among planted mix/rate scenarios where the right segment was named, share with the
                          right effect label
  impact MAPE             |impact_brl - true planted impact| / |true|, true = the same R$ quantity computed on the
                          injected minus the clean data in the target week (GMV and AOV scenarios only)
  grounding               mean grounding rate; ungrounded numbers classified as threshold notation, derived
                          from two tool values, or invented
All proportions carry 95% Wilson intervals. Because the TEST set has only 13 distinct target weeks and
scenarios sharing a week are correlated, top-1, top-3 and F1 also get week-clustered bootstrap 95% intervals
(resample weeks with replacement, 2,000 draws, fixed seed); these are the headline intervals.
McNemar (exact) on paired top-1.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from agent.verifier import EVID_RE, _walk_numbers, extract_numbers, is_grounded  # noqa: E402
from eval.score import cause_matches, score_one  # noqa: E402
from inject.scenarios import apply_scenario, load_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
V2 = RES / "v2"
TEST_MANIFEST = ROOT / "inject" / "manifest_test.json"
RUNS = {"B1": "test_b1", "B2": "test_b2", "v1": "test_v1", "v2": "test_v2"}
LABELS = {"B1": "Baseline B1: top contribution", "B2": "Baseline B2: scan + test", "v1": "Agent v1",
          "v2": "Agent v2", "v2+fallback": "Agent v2 + B2 fallback"}
ORDER = ["v2", "v2+fallback", "v1", "B2", "B1"]
SEVS = ["small", "medium", "large"]


# ------------------------------------------------------------------ loading and derived investigators
def read_runs(path: Path) -> dict:
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def causes(rep: dict | None) -> list[dict]:
    rep = rep or {}
    if not rep.get("anomaly_confirmed"):
        return []
    return sorted(rep.get("root_causes") or [], key=lambda c: c.get("rank", 99))


def with_fallback(v2: dict, b2: dict) -> dict:
    out = {}
    for i, r in v2.items():
        rep = r.get("report")
        b = (b2.get(i) or {}).get("report")
        if not causes(rep) and causes(b):
            top = {**causes(b)[0], "rank": 1, "source": "fallback"}
            rep = {**(rep or {}), "anomaly_confirmed": True, "root_causes": [top]}
            out[i] = {**r, "report": rep, "fallback_used": True}
        else:
            out[i] = {**r, "fallback_used": False}
    return out


# ------------------------------------------------------------------ statistics
def wilson(k: int, n: int, z: float = 1.96) -> dict:
    if n == 0:
        return {"pct": None, "lo": None, "hi": None, "k": 0, "n": 0}
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"pct": round(100 * p, 1), "lo": round(100 * (c - h), 1), "hi": round(100 * (c + h), 1), "k": k, "n": n}


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    p = binomtest(min(only_a, only_b), n, 0.5).pvalue if n else 1.0
    return {"only_first_correct": only_a, "only_second_correct": only_b, "p_value": round(float(p), 4)}


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


BOOT_DRAWS, BOOT_SEED = 2000, 20270103


def cluster_bootstrap(recs: list[dict]) -> dict:
    """recs: one per scenario with week, planted, top1, top3, tp, fp, fn. Resample weeks, recompute metrics."""
    import numpy as np
    weeks = sorted({r["week"] for r in recs})
    by = {w: [r for r in recs if r["week"] == w] for w in weeks}
    rng = np.random.default_rng(BOOT_SEED)
    out = {"top1": [], "top3": [], "f1": []}
    for _ in range(BOOT_DRAWS):
        sample = [r for w in rng.choice(len(weeks), len(weeks), replace=True) for r in by[weeks[w]]]
        pl = [r for r in sample if r["planted"]]
        if pl:
            out["top1"].append(100 * sum(r["top1"] for r in pl) / len(pl))
            out["top3"].append(100 * sum(r["top3"] for r in pl) / len(pl))
        tp, fp, fn = (sum(r[k] for r in sample) for k in ("tp", "fp", "fn"))
        if tp:
            pr, rc = tp / (tp + fp), tp / (tp + fn)
            out["f1"].append(100 * 2 * pr * rc / (pr + rc))
        else:
            out["f1"].append(0.0)
    res = {"n_weeks": len(weeks), "draws": BOOT_DRAWS}
    for k, v in out.items():
        res[k] = {"lo": round(float(np.percentile(v, 2.5)), 1), "hi": round(float(np.percentile(v, 97.5)), 1)} if v else None
    return res


def pctl(xs: list[float], q: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return round(xs[f] + (xs[c] - xs[f]) * (k - f), 2)


# ------------------------------------------------------------------ impact ground truth
_TRUE_IMPACT: dict = {}


def true_impact(sc: dict, base) -> float | None:
    if sc["id"] in _TRUE_IMPACT:
        return _TRUE_IMPACT[sc["id"]]
    val = None
    if sc["metric"] in ("gmv", "aov"):
        inj = apply_scenario(base, sc)
        w = sc["week"]
        g0, g1 = base.weekly("gmv"), inj.weekly("gmv")
        g0 = float(g0.loc[g0["week"] == w, "value"].iloc[0])
        g1 = float(g1.loc[g1["week"] == w, "value"].iloc[0])
        if sc["metric"] == "gmv":
            val = g1 - g0
        else:
            a0, a1 = base.weekly("aov"), inj.weekly("aov")
            a0 = float(a0.loc[a0["week"] == w, "value"].iloc[0])
            a1r = a1.loc[a1["week"] == w]
            val = (float(a1r["value"].iloc[0]) - a0) * float(a1r["n"].iloc[0])
    _TRUE_IMPACT[sc["id"]] = val
    return val


# ------------------------------------------------------------------ ungrounded-number classification
def classify_ungrounded(row: dict) -> Counter:
    out = Counter()
    ver = row.get("verification") or {}
    bad = ver.get("ungrounded") or []
    if not bad:
        return out
    rep = row.get("report_raw") or row.get("report") or {}
    narrative = rep.get("narrative") or ""
    cited = set(rep.get("evidence_ids") or []) | set(EVID_RE.findall(narrative))
    for rc in rep.get("root_causes") or []:
        cited |= set(rc.get("evidence_ids") or [])
    ev = row.get("evidence") or {}
    pool = [e for e in cited if e in ev] or list(ev)
    nums: list[float] = []
    for e in pool:
        _walk_numbers(ev[e]["result"], nums)
    nums = sorted({round(x, 10) for x in nums if x == x and abs(x) < 1e12})[:400]
    derived = set()
    for i, a in enumerate(nums):
        for b in nums[i + 1:]:
            derived.update((a + b, a - b, b - a))
            if b:
                derived.update((100 * (a - b) / b, 100 * a / b))
            if a:
                derived.update((100 * (b - a) / a, 100 * b / a))
    derived = list(derived)
    for name in bad:
        raw = name.split(":", 1)[1] if name.startswith("narrative:") else None
        if raw is None:
            out["field"] += 1
            continue
        parsed = extract_numbers(raw)
        if not parsed:
            out["invented"] += 1
            continue
        v, d, _ = parsed[0]
        if any(f"<{s}{raw}" in narrative for s in ("", " ", "= ", "=")) and v in (0.05, 0.01, 0.001, 0.0001):
            out["threshold_notation"] += 1
        elif is_grounded(v, d, derived):
            out["derived"] += 1
        else:
            out["invented"] += 1
    return out


# ------------------------------------------------------------------ per-investigator metrics
def rr(sc: dict, rep: dict | None) -> float:
    for k, c in enumerate(causes(rep), 1):
        if cause_matches(c, sc["ground_truth"]):
            return 1.0 / k
    return 0.0


def evaluate(rows: dict, man: dict, base) -> dict:
    ids = [i for i in man if i in rows]
    planted = [i for i in ids if not man[i]["is_control"]]
    controls = [i for i in ids if man[i]["is_control"]]
    sc_ = {i: score_one(man[i], rows[i].get("report")) for i in ids}
    top1 = [bool(sc_[i]["top1"]) for i in planted]
    top3 = [bool(sc_[i]["top3"]) for i in planted]
    tp = fp = fn = 0
    recs = []
    for i in planted:
        cs = causes(rows[i].get("report"))
        hit = any(cause_matches(c, man[i]["ground_truth"]) for c in cs)
        tp += hit
        fn += not hit
        fp += len(cs) - (1 if hit else 0)
        recs.append({"week": man[i]["week"], "planted": True, "top1": bool(sc_[i]["top1"]),
                     "top3": bool(sc_[i]["top3"]), "tp": int(hit), "fn": int(not hit), "fp": len(cs) - int(hit)})
    fa = [bool(sc_[i]["false_alarm"]) for i in controls]
    fp += sum(fa)
    for i in controls:
        recs.append({"week": man[i]["week"], "planted": False, "top1": False, "top3": False, "tp": 0, "fn": 0,
                     "fp": int(bool(sc_[i]["false_alarm"]))})
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    det = {s: wilson(sum(bool(sc_[i]["detected"]) for i in planted if man[i]["severity"] == s),
                     sum(1 for i in planted if man[i]["severity"] == s)) for s in SEVS}
    # effect label: mix/rate scenarios where the right segment was named
    lab_k = lab_n = 0
    for i in planted:
        gt = man[i]["ground_truth"]
        if gt.get("effect") not in ("mix", "rate"):
            continue
        seg = [c for c in causes(rows[i].get("report"))
               if str(c.get("dimension")) == gt["dimension"] and str(c.get("segment")) == gt["segment"]]
        if seg:
            lab_n += 1
            lab_k += seg[0].get("effect") == gt["effect"]
    # impact error
    ape = []
    for i in planted:
        gt = man[i]["ground_truth"]
        m = [c for c in causes(rows[i].get("report")) if cause_matches(c, gt)]
        if m and m[0].get("impact_brl") is not None:
            t = true_impact(man[i], base)
            if t:
                ape.append(abs(float(m[0]["impact_brl"]) - t) / abs(t))
    # grounding and cost
    g = [rows[i].get("grounding_rate_pct") for i in ids if rows[i].get("grounding_rate_pct") is not None]
    ug = Counter()
    for i in ids:
        ug += classify_ungrounded(rows[i])
    lat = [rows[i].get("latency_s") for i in ids]
    tok = [rows[i].get("total_tokens") for i in ids if rows[i].get("total_tokens") is not None]
    steps = [rows[i].get("steps") for i in ids if rows[i].get("steps") is not None]
    by = defaultdict(list)
    for i in planted:
        by[(man[i]["type"], man[i]["severity"])].append(bool(sc_[i]["top1"]))
    by_type = defaultdict(list)
    for i in planted:
        by_type[man[i]["type"]].append(bool(sc_[i]["top1"]))
    return {
        "n_scenarios": len(ids), "n_planted": len(planted), "n_controls": len(controls),
        "failed_runs": sum(1 for i in ids if rows[i].get("error")),
        "top1": wilson(sum(top1), len(top1)), "top3": wilson(sum(top3), len(top3)),
        "mrr": round(sum(rr(man[i], rows[i].get("report")) for i in planted) / len(planted), 4) if planted else None,
        "precision": wilson(tp, tp + fp), "recall": wilson(tp, tp + fn),
        "f1_pct": round(100 * f1, 1) if f1 else None, "tp": tp, "fp": fp, "fn": fn,
        "week_clustered_ci": cluster_bootstrap(recs),
        "specificity": wilson(len(fa) - sum(fa), len(fa)), "false_alarm": wilson(sum(fa), len(fa)),
        "detection_recall": wilson(sum(bool(sc_[i]["detected"]) for i in planted), len(planted)),
        "detection_by_severity": det,
        "top1_given_detected": wilson(sum(bool(sc_[i]["top1"]) for i in planted if sc_[i]["detected"]),
                                      sum(1 for i in planted if sc_[i]["detected"])),
        "planted_undetected": sum(1 for i in planted if not sc_[i]["detected"]),
        "causes_per_planted_report": dict(sorted(Counter(len(causes(rows[i].get("report"))) for i in planted).items())),
        "effect_label_accuracy": wilson(lab_k, lab_n),
        "impact_mape_pct": round(100 * sum(ape) / len(ape), 1) if ape else None, "impact_n": len(ape),
        "grounding_rate_pct": round(sum(g) / len(g), 2) if g else None,
        "ungrounded_reports": sum(1 for i in ids if rows[i].get("ungrounded")),
        "ungrounded_numbers": dict(ug), "invented_numbers": ug.get("invented", 0),
        "latency_s": {"avg": round(sum(lat) / len(lat), 2) if lat else None, "p50": pctl(lat, 0.5), "p95": pctl(lat, 0.95)},
        "avg_tokens": round(sum(tok) / len(tok), 1) if tok else None,
        "avg_tool_calls": round(sum(steps) / len(steps), 2) if steps else None,
        "top1_by_type": {t: wilson(sum(v), len(v)) for t, v in sorted(by_type.items())},
        "top1_by_type_severity": {f"{t}|{s}": wilson(sum(v), len(v)) for (t, s), v in sorted(by.items())},
        "fallback_used": sum(1 for i in ids if rows[i].get("fallback_used")),
        "_top1_list": {i: bool(sc_[i]["top1"]) for i in planted},
    }


# ------------------------------------------------------------------ plots
COLORS = {"v2": "#2B6CB0", "v2+fallback": "#63B3ED", "v1": "#DD6B20", "B2": "#718096", "B1": "#A0AEC0"}


def plot_all(summ: dict) -> None:
    inv = [m for m in ORDER if m in summ]
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=120)
    for m in inv:
        d = summ[m]["detection_by_severity"]
        ys = [d[s]["pct"] or 0 for s in SEVS]
        ax.errorbar(range(3), ys, yerr=[[y - (d[s]["lo"] or 0) for y, s in zip(ys, SEVS)],
                                        [(d[s]["hi"] or 0) - y for y, s in zip(ys, SEVS)]],
                    marker="o", capsize=3, label=LABELS[m], color=COLORS[m])
    ax.set_xticks(range(3), SEVS)
    ax.set_ylim(0, 105)
    ax.set_ylabel("detection recall (%)")
    ax.set_title("Sensitivity: detection recall by severity (TEST, 95% CI)")
    ax.legend(frameon=False, fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(V2 / "sensitivity_curve.png")
    plt.close(fig)

    keys = [("top1", "Top-1"), ("top3", "Top-3"), ("recall", "Recall"), ("precision", "Precision"),
            ("specificity", "Specificity")]
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=120)
    w = 0.8 / len(inv)
    for k, m in enumerate(inv):
        vals = [summ[m][key]["pct"] or 0 for key, _ in keys]
        lo = [v - (summ[m][key]["lo"] or 0) for v, (key, _) in zip(vals, keys)]
        hi = [(summ[m][key]["hi"] or 0) - v for v, (key, _) in zip(vals, keys)]
        ax.bar([x + k * w for x in range(len(keys))], vals, w, yerr=[lo, hi], capsize=2, label=LABELS[m],
               color=COLORS[m])
    ax.set_xticks([x + w * (len(inv) - 1) / 2 for x in range(len(keys))], [lab for _, lab in keys])
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    ax.set_title("Agent vs baselines on the held-out TEST set (95% CI)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(V2 / "agent_vs_baselines.png")
    plt.close(fig)

    types = sorted({t for m in inv for t in summ[m]["top1_by_type"]})
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=120)
    for k, m in enumerate(inv):
        vals = [(summ[m]["top1_by_type"].get(t) or {}).get("pct") or 0 for t in types]
        ax.bar([x + k * w for x in range(len(types))], vals, w, label=LABELS[m], color=COLORS[m])
    ax.set_xticks([x + w * (len(inv) - 1) / 2 for x in range(len(types))],
                  [t.replace("_", "\n") for t in types], fontsize=7)
    ax.set_ylim(0, 105)
    ax.set_ylabel("top-1 (%)")
    ax.set_title("Top-1 accuracy by anomaly type (TEST)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(V2 / "accuracy_by_type.png")
    plt.close(fig)


# ------------------------------------------------------------------ main
def data_summary() -> dict:
    import duckdb
    con = duckdb.connect(str(ROOT / "data" / "olist.duckdb"), read_only=True)
    q = lambda s: con.execute(s).fetchone()
    lo, hi = q("SELECT MIN(week), MAX(week) FROM fact_orders")
    out = {"orders": q("SELECT COUNT(*) FROM fact_orders")[0], "items": q("SELECT COUNT(*) FROM fact_items")[0],
           "weeks": q("SELECT COUNT(DISTINCT week) FROM fact_orders")[0],
           "first_week": str(lo), "last_week": str(hi), "source": "Olist Brazilian e-commerce (Kaggle)"}
    (RES / "data_summary.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> dict:
    from metrics.metrics import default_store
    base = default_store()
    man = {s["id"]: s for s in load_manifest(TEST_MANIFEST)}
    runs = {k: read_runs(V2 / "runs" / f"{v}.jsonl") for k, v in RUNS.items()}
    runs = {k: v for k, v in runs.items() if v}
    if "v2" in runs and "B2" in runs:
        runs["v2+fallback"] = with_fallback(runs["v2"], runs["B2"])
    summ = {m: evaluate(r, man, base) for m, r in runs.items()}
    t1 = {m: s.pop("_top1_list") for m, s in summ.items()}
    tests = {}
    for a, b in [("v2", "B2"), ("v2", "v1"), ("v2+fallback", "B2"), ("v2+fallback", "v1"), ("v1", "B2")]:
        if a in t1 and b in t1:
            common = sorted(set(t1[a]) & set(t1[b]))
            tests[f"{a} vs {b}"] = {"n": len(common), **mcnemar([t1[a][i] for i in common], [t1[b][i] for i in common])}
    # stability: v2 repeat on a fixed random subset, gateway cache bypassed
    stab = {}
    rep = read_runs(V2 / "runs" / "test_v2_repeat.jsonl")
    if rep and "v2" in runs:
        common = sorted(set(rep) & set(runs["v2"]))
        top = lambda r: (lambda c: f"{c[0].get('dimension')}={c[0].get('segment')}" if c else "none")(causes(r.get("report")))
        same = [top(rep[i]) == top(runs["v2"][i]) for i in common]
        a = [bool(score_one(man[i], runs["v2"][i].get("report"))["top1"] or score_one(man[i], runs["v2"][i].get("report"))["false_alarm"] is False) for i in common]
        b = [bool(score_one(man[i], rep[i].get("report"))["top1"] or score_one(man[i], rep[i].get("report"))["false_alarm"] is False) for i in common]
        stab = {"n": len(common), "same_top1_cause": wilson(sum(same), len(same)), "correct_agreement_kappa": kappa(a, b),
                "correct_first_run": sum(a), "correct_repeat": sum(b)}
    cache = {}
    cr = read_runs(V2 / "runs_cache" / "test_v2.jsonl")
    if cr:
        calls = sum(r.get("llm_calls") or 0 for r in cr.values())
        hits = sum(r.get("cache_hits") or 0 for r in cr.values())
        per_call = [r["llm_latency_s"] / r["llm_calls"] for r in cr.values()
                    if r.get("llm_calls") and r.get("cache_hits") == r.get("llm_calls")]
        first = [runs["v2"][i]["latency_s"] for i in cr if "v2" in runs and i in runs["v2"]]
        cache = {"scenarios": len(cr), "llm_calls": calls, "cache_hits": hits, "hit_rate": wilson(hits, calls),
                 "median_cache_hit_call_latency_s": pctl(per_call, 0.5),
                 "avg_scenario_latency_first_s": round(sum(first) / len(first), 2) if first else None,
                 "avg_scenario_latency_rerun_s": round(sum(r["latency_s"] for r in cr.values()) / len(cr), 2)}
    gap = {}
    dev = read_runs(RES / "runs" / "nvidia_v2final_dev.jsonl")
    if dev and "v2" in summ:
        dman = {s["id"]: s for s in load_manifest()}
        dp = [i for i in dev if not dman[i]["is_control"]]
        k = sum(bool(score_one(dman[i], dev[i].get("report"))["top1"]) for i in dp)
        gap = {"dev_top1": wilson(k, len(dp)), "test_top1": summ["v2"]["top1"],
               "gap_pts": round((wilson(k, len(dp))["pct"] or 0) - (summ["v2"]["top1"]["pct"] or 0), 1)}
    V2.mkdir(parents=True, exist_ok=True)
    for m, s in summ.items():
        s["label"] = LABELS[m]
    (V2 / "summary.json").write_text(json.dumps(summ, indent=2))
    (V2 / "mcnemar.json").write_text(json.dumps(tests, indent=2))
    (V2 / "stability.json").write_text(json.dumps(stab, indent=2))
    (V2 / "cache_rerun.json").write_text(json.dumps(cache, indent=2))
    (V2 / "dev_test_gap.json").write_text(json.dumps(gap, indent=2))
    with open(V2 / "by_type_severity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["investigator", "type", "severity", "n", "top1_pct", "ci_lo", "ci_hi"])
        for m in [x for x in ORDER if x in summ]:
            for key, v in summ[m]["top1_by_type_severity"].items():
                t, s_ = key.split("|")
                w.writerow([m, t, s_, v["n"], v["pct"], v["lo"], v["hi"]])
    plot_all(summ)
    data_summary()
    return {"summary": summ, "mcnemar": tests, "stability": stab, "cache": cache, "gap": gap}


if __name__ == "__main__":
    out = main()
    for m in [x for x in ORDER if x in out["summary"]]:
        s = out["summary"][m]
        wc = s["week_clustered_ci"]
        print(f"{m:12} top1 {s['top1']['pct']} clustered [{wc['top1']['lo']}, {wc['top1']['hi']}] "
              f"Wilson [{s['top1']['lo']}, {s['top1']['hi']}]  top3 {s['top3']['pct']}  "
              f"MRR {s['mrr']}  F1 {s['f1_pct']}  spec {s['specificity']['pct']}  failed {s['failed_runs']}")
    print("McNemar:", out["mcnemar"])

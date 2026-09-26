"""Significance tests with Benjamini-Hochberg correction.

Which test:
  effect="share" (default for GMV / Orders, and for mix effects): did the segment's share of rows
      change between the baseline weeks and the target week?  2x2 chi-square
      (segment vs rest) x (baseline vs week). Robust to overall growth or seasonality,
      because it asks whether this segment moved differently from everything else.
  effect="rate" (default for mean metrics):
      binary metrics (on-time, cancellation, repeat) -> two-proportion z-test, week vs baseline
      money metrics (AOV, avg item price), v3 engine -> Yuen trimmed-mean t-test (20% trimmed; robust to big tickets)
      continuous metrics (AOV, delay, review)        -> Welch t-test (Mann-Whitney U reported too)

Benjamini-Hochberg: every test in one investigation is added to a list; q-values are recomputed
over all tests so far. A segment is significant when q <= 0.05. Testing many segments at 0.05
would otherwise produce false positives by chance alone (20 tests -> ~1 false hit expected).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

from metrics.metrics import dimension_col, metric_def, to_week
from tools.context import Investigation

Q = 0.05
MIN_N = 5


def bh(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    return list(multipletests(pvals, alpha=Q, method="fdr_bh")[1])


def _rows(store, metric, dimension, segment, week, n_baseline, filters):
    m = metric_def(metric)
    col = dimension_col(dimension, m["table"])
    df = store.rows(metric, filters)
    w1 = to_week(week)
    w0 = store.baseline_weeks(w1, n_baseline)
    if m["table"] == "items" and m["kind"] == "sum":
        # share tests count orders, not items (one order = one trial)
        df = df.drop_duplicates(["order_id", col])
    in_seg = df[col].astype(str) == str(segment)
    cur = df["week"] == w1
    base = df["week"].isin(w0)
    return df, in_seg, cur, base


def run_test(store, metric: str, dimension: str, segment: str, week, n_baseline: int = 4,
             effect: str | None = None, filters: dict | None = None) -> dict:
    m = metric_def(metric)
    effect = effect or ("share" if m["kind"] == "sum" else "rate")
    df, in_seg, cur, base = _rows(store, metric, dimension, segment, week, n_baseline, filters)
    res = {"metric": metric, "dimension": dimension, "segment": str(segment), "effect_tested": effect}
    if effect in ("share", "mix"):
        a, b = int((in_seg & cur).sum()), int((~in_seg & cur).sum())
        c, d = int((in_seg & base).sum()), int((~in_seg & base).sum())
        table = np.array([[a, b], [c, d]])
        if min(a + c, b + d) < MIN_N or min(a + b, c + d) < MIN_N:
            return {**res, "test": "insufficient data", "p_value": 1.0, "n_current": a, "n_baseline": c}
        chi2, p, _, _ = sps.chi2_contingency(table, correction=False)
        res.update({"test": "chi-square 2x2 on segment share (segment vs rest, week vs baseline)",
                    "share_current_pct": a / (a + b) * 100, "share_baseline_pct": c / (c + d) * 100,
                    "n_current": a, "n_baseline": c, "statistic": chi2, "p_value": p})
        return res
    x1 = df.loc[in_seg & cur, "value"].dropna().to_numpy()
    x0 = df.loc[in_seg & base, "value"].dropna().to_numpy()
    if len(x1) < MIN_N or len(x0) < MIN_N:
        return {**res, "test": "insufficient data", "p_value": 1.0, "n_current": len(x1), "n_baseline": len(x0)}
    res.update({"n_current": len(x1), "n_baseline": len(x0),
                "mean_current": float(x1.mean()), "mean_baseline": float(x0.mean())})
    if m.get("binary"):
        k1, k0 = int(x1.sum()), int(x0.sum())
        if (k1 + k0 == 0) or (k1 + k0 == len(x1) + len(x0)):
            return {**res, "test": "two-proportion z-test", "p_value": 1.0, "statistic": 0.0}
        z, p = proportions_ztest([k1, k0], [len(x1), len(x0)])
        res.update({"test": "two-proportion z-test", "statistic": z, "p_value": p})
    else:
        from tools.engine import TRIM, money_rate_test
        u_p = sps.mannwhitneyu(x1, x0, alternative="two-sided").pvalue
        if m.get("money") and money_rate_test() == "yuen":
            # v3: a few big-ticket orders dominate a week's mean price; the 20% trimmed mean is robust to them
            t, p = sps.ttest_ind(x1, x0, equal_var=False, trim=TRIM)
            res.update({"test": f"Yuen trimmed-mean t-test ({int(TRIM * 100)}% trimmed; money metric)", "statistic": t,
                        "p_value": p, "mann_whitney_p": u_p,
                        "trimmed_mean_current": float(sps.trim_mean(x1, TRIM)),
                        "trimmed_mean_baseline": float(sps.trim_mean(x0, TRIM))})
        else:
            t, p = sps.ttest_ind(x1, x0, equal_var=False)
            res.update({"test": "Welch t-test", "statistic": t, "p_value": p, "mann_whitney_p": u_p})
    return res


def significance_test(inv: Investigation, metric: str, dimension: str, segment: str, week: str,
                      n_baseline: int = 4, effect: str | None = None, filters: dict | None = None) -> dict:
    from tools.engine import hist_z
    from tools.segscan import segment_history_z

    res = run_test(inv.store, metric, dimension, segment, week, n_baseline, effect, filters)
    p = float(res["p_value"]) if res["p_value"] == res["p_value"] else 1.0
    inv.tests.append({"key": f"{metric}|{dimension}={segment}|{res['effect_tested']}", "p_value": p})
    q = bh([t["p_value"] for t in inv.tests])
    h = segment_history_z(inv.store, metric, dimension, week, res["effect_tested"], filters)
    h = h[h["segment"] == str(segment)]
    hz = float(h["hist_z"].iloc[0]) if len(h) else None
    res["q_value_bh"] = q[-1]
    res["n_tests_so_far"] = len(inv.tests)
    res["hist_z"] = hz
    res["significant"] = bool(q[-1] <= Q and hz is not None and abs(hz) >= hist_z())
    res["rule"] = (f"significant = BH q<={Q} (over all {len(inv.tests)} tests in this investigation) "
                   f"AND |hist_z|>={hist_z()} (change unusual vs the segment's own weekly history)")
    return res

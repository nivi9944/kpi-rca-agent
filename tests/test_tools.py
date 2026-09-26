"""Each analysis tool against answers worked out by hand on the synthetic fixture."""
import math

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

from tests.fixtures.synth import N_WEEKS, TARGET, make_store
from tools.anomaly import detect_anomalies, score_series
from tools.context import Investigation
from tools.decompose import decompose_metric
from tools.drilldown import drill_down
from tools.impact import estimate_impact
from tools.stats import bh, run_test, significance_test

LAST = N_WEEKS - 1
W = TARGET.strftime("%Y-%m-%d")


def inv_for(overrides=None):
    return Investigation(store=make_store(overrides))


# ---------- metric layer
def test_weekly_values_match_hand_calculation():
    st = make_store()
    assert st.weekly("gmv")["value"].iloc[0] == 7500
    assert st.weekly("orders")["value"].iloc[0] == 100
    assert st.weekly("aov")["value"].iloc[0] == 75
    assert st.weekly("on_time_rate")["value"].iloc[0] == pytest.approx(0.86)


# ---------- drill-down: sum metric
def test_drilldown_sum_contribution():
    # state A loses half its orders in the target week: 30 orders (15 x at 100, 15 y at 50) = -2250
    inv = inv_for({LAST: {"n_a": 30}})
    d = drill_down(inv, "gmv", "customer_state", W)
    assert d["total_delta"] == pytest.approx(-2250)
    top = d["top_segments"][0]
    assert top["segment"] == "A" and top["contribution_pct"] == pytest.approx(100.0)
    assert top["baseline"] == pytest.approx(4500) and top["current"] == pytest.approx(2250)


# ---------- drill-down: mix vs rate split (exact)
def test_rate_effect():
    # B's on-time rate falls 0.8 -> 0.5, shares unchanged: R1 = .6*.9 + .4*.5 = .74, delta = -.12
    inv = inv_for({LAST: {"ontime_b": 0.5}})
    d = drill_down(inv, "on_time_rate", "customer_state", W)
    assert d["total_delta"] == pytest.approx(-0.12)
    assert d["mix_total"] == pytest.approx(0.0, abs=1e-12)
    assert d["rate_total"] == pytest.approx(-0.12)
    b = d["top_segments"][0]
    assert b["segment"] == "B" and b["effect"] == "rate"
    assert b["rate_effect"] == pytest.approx(-0.12) and b["contribution_pct"] == pytest.approx(100.0)


def test_mix_effect_simpson_style():
    # shares flip to A 40 / B 60 with UNCHANGED rates: R1 = .4*.9 + .6*.8 = .84, delta = -.02
    # centered mix: A (0.4-0.6)*(0.9-0.86) = -0.008 (40%), B (0.6-0.4)*(0.8-0.86) = -0.012 (60%)
    inv = inv_for({LAST: {"n_a": 40, "n_b": 60}})
    d = drill_down(inv, "on_time_rate", "customer_state", W)
    assert d["total_delta"] == pytest.approx(-0.02)
    assert d["rate_total"] == pytest.approx(0.0, abs=1e-12)
    seg = {s["segment"]: s for s in d["top_segments"]}
    assert seg["B"]["mix_effect"] == pytest.approx(-0.012)
    assert seg["A"]["mix_effect"] == pytest.approx(-0.008)
    assert seg["B"]["contribution_pct"] == pytest.approx(60.0)
    assert seg["B"]["effect"] == "mix"


def test_mix_plus_rate_sum_to_total():
    inv = inv_for({LAST: {"n_a": 45, "n_b": 70, "ontime_a": 0.7}})
    d = drill_down(inv, "on_time_rate", "customer_state", W, top_k=10)
    assert d["mix_total"] + d["rate_total"] == pytest.approx(d["total_delta"])
    assert sum(s["contribution_pct"] for s in d["top_segments"]) == pytest.approx(100.0)


# ---------- decomposition
def test_decompose_gmv_log_split_exact():
    # orders 100 -> 70 (A loses 30), GMV 7500 -> 5250
    inv = inv_for({LAST: {"n_a": 30}})
    d = decompose_metric(inv, "gmv", W)
    eff = {x["driver"]: x["effect"] for x in d["drivers"]}
    assert d["change"] == pytest.approx(-2250)
    assert eff["orders"] + eff["aov"] == pytest.approx(-2250)
    # AOV unchanged (A loses x and y equally) -> everything is the orders effect
    assert eff["aov"] == pytest.approx(0.0, abs=1e-6)
    share = math.log(70 / 100) / math.log(5250 / 7500)
    assert eff["orders"] == pytest.approx(-2250 * share)


# ---------- anomaly detection
def test_flat_series_not_flagged_and_drop_flagged():
    clean = detect_anomalies(inv_for(), "gmv", week=W)
    assert clean["target_week"]["is_anomaly"] is False
    drop = detect_anomalies(inv_for({LAST: {"n_a": 30}}), "gmv", week=W)
    assert drop["target_week"]["is_anomaly"] is True
    assert drop["target_week"]["change_pct"] == pytest.approx(-30.0)
    assert drop["target_week"]["direction"] == "down"


def test_robust_z_uses_trailing_history():
    s = score_series(make_store(), "orders")
    assert s["z"].iloc[-1] == pytest.approx(0.0)


# ---------- significance
def test_two_proportion_matches_statsmodels():
    inv = inv_for({LAST: {"ontime_b": 0.5}})
    r = run_test(inv.store, "on_time_rate", "customer_state", "B", W, effect="rate")
    z, p = proportions_ztest([20, 4 * 32], [40, 160])
    assert r["test"] == "two-proportion z-test"
    assert r["p_value"] == pytest.approx(p)


def test_share_test_detects_volume_drop_but_not_uniform_drop():
    hit = run_test(make_store({LAST: {"n_a": 30}}), "gmv", "customer_state", "A", W)
    assert hit["p_value"] < 0.01
    uniform = run_test(make_store({LAST: {"n_a": 30, "n_b": 20}}), "gmv", "customer_state", "A", W)
    assert uniform["p_value"] > 0.5  # everything fell by the same share: A is not special


def test_benjamini_hochberg_matches_statsmodels():
    p = [0.001, 0.02, 0.03, 0.2, 0.5]
    assert bh(p) == pytest.approx(list(multipletests(p, method="fdr_bh")[1]))


def test_bh_accumulates_within_investigation():
    inv = inv_for({LAST: {"ontime_b": 0.5}})
    significance_test(inv, "on_time_rate", "customer_state", "B", W)
    r2 = significance_test(inv, "on_time_rate", "customer_state", "A", W)
    assert r2["n_tests_so_far"] == 2 and len(inv.tests) == 2


# ---------- impact
def test_gmv_impact_equals_segment_delta():
    inv = inv_for({LAST: {"n_a": 30}})
    r = estimate_impact(inv, "gmv", "customer_state", "A", W)
    assert r["impact_brl"] == pytest.approx(-2250)


def test_non_money_metric_reports_affected_rows():
    inv = inv_for({LAST: {"ontime_b": 0.5}})
    r = estimate_impact(inv, "on_time_rate", "customer_state", "B", W)
    assert r["impact_brl"] is None
    assert r["affected_rows_change"] == pytest.approx(40 * (0.5 - 0.8))


def test_evidence_ids_are_sequential():
    from tools.registry import execute
    inv = inv_for()
    a = execute(inv, "detect_anomalies", {"metric": "gmv", "week": W})
    b = execute(inv, "drill_down", {"metric": "gmv", "dimension": "customer_state", "week": W})
    assert (a["evidence_id"], b["evidence_id"]) == ("e1", "e2")
    assert set(inv.evidence) == {"e1", "e2"}


def test_registry_rejects_bad_args_without_raising():
    from tools.registry import execute
    inv = inv_for()
    assert "error" in execute(inv, "drill_down", {"metric": "not_a_metric", "dimension": "x", "week": W})
    assert "error" in execute(inv, "no_such_tool", {})
    assert "error" in execute(inv, "drill_down", "{not json")


def test_openai_tool_specs_are_plain_json_schema():
    import json
    from tools.registry import openai_tools
    specs = openai_tools()
    s = json.dumps(specs)
    assert "$ref" not in s and "$defs" not in s
    assert {t["function"]["name"] for t in specs} >= {"detect_anomalies", "drill_down", "significance_test"}
    assert np.all([t["function"]["parameters"]["type"] == "object" for t in specs])


# ---------- segment scan (history z)
def test_segment_scan_flags_the_changed_segment_only():
    from tools.segscan import HIST_Z, segment_history_z
    h = segment_history_z(make_store({LAST: {"ontime_b": 0.3}}), "on_time_rate", "customer_state", W, "rate")
    z = dict(zip(h["segment"], h["hist_z"]))
    assert abs(z["B"]) >= HIST_Z and abs(z["A"]) < 1


def test_segment_scan_share_for_volume_drop():
    from tools.segscan import HIST_Z, segment_history_z
    h = segment_history_z(make_store({LAST: {"n_a": 10}}), "gmv", "customer_state", W, "share")
    assert h.iloc[0]["segment"] in ("A", "B") and abs(h.iloc[0]["hist_z"]) >= HIST_Z


def test_v3_engine_threshold_is_4_and_v2_is_reproducible():
    from tools import engine
    from tools.segscan import scan_segments
    from tools.context import Investigation
    from tests.fixtures.synth import TARGET, make_store
    assert engine.engine_name() == "v3" and engine.hist_z() == 4.0
    inv = Investigation(store=make_store())
    assert scan_segments(inv, "gmv", TARGET.strftime("%Y-%m-%d"))["hist_z_threshold"] == 4.0
    engine.set_engine("v2")
    try:
        assert engine.hist_z() == 4.25 and engine.money_rate_test() == "welch"
        assert scan_segments(Investigation(store=make_store()), "gmv", TARGET.strftime("%Y-%m-%d"))["hist_z_threshold"] == 4.25
    finally:
        engine.set_engine("v3")


def test_v3_money_rate_test_is_trimmed_mean_and_robust_to_big_tickets():
    import numpy as np
    import pandas as pd
    import scipy.stats as sps

    from tools import engine
    from tools.stats import run_test
    from tests.fixtures.synth import TARGET, make_store
    st = make_store()
    w = TARGET.strftime("%Y-%m-%d")
    # AOV rate test on state A: v3 uses Yuen's trimmed-mean test, v2 keeps Welch
    r3 = run_test(st, "aov", "customer_state", "A", w, effect="rate")
    assert r3["test"].startswith("Yuen") and "trimmed_mean_current" in r3
    engine.set_engine("v2")
    try:
        assert run_test(st, "aov", "customer_state", "A", w, effect="rate")["test"] == "Welch t-test"
    finally:
        engine.set_engine("v3")
    # the reason: one huge order hides a 30% price cut from the plain mean, not from the trimmed mean
    rng = np.random.default_rng(0)
    base = rng.normal(100, 5, 60)
    cut = np.r_[rng.normal(70, 5, 59), 20000.0]
    assert sps.ttest_ind(cut, base, equal_var=False).pvalue > 0.05
    assert sps.ttest_ind(cut, base, equal_var=False, trim=0.2).pvalue < 1e-10
    assert run_test(st, "on_time_rate", "customer_state", "A", w)["test"] == "two-proportion z-test"  # binary unchanged

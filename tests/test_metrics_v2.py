import pytest

from eval.metrics_v2 import classify_ungrounded, kappa, mcnemar, wilson, with_fallback


def test_wilson_interval_known_values():
    w = wilson(21, 40)
    assert w["pct"] == 52.5 and w["lo"] == pytest.approx(37.5, abs=0.1) and w["hi"] == pytest.approx(67.1, abs=0.1)
    assert wilson(0, 0)["pct"] is None and wilson(10, 10)["hi"] == 100.0


def test_mcnemar_counts_discordant_pairs_only():
    a = [True, True, False, False, True]
    b = [True, False, True, False, False]
    m = mcnemar(a, b)
    assert m["only_first_correct"] == 2 and m["only_second_correct"] == 1 and 0 < m["p_value"] <= 1
    assert mcnemar([True], [True])["p_value"] == 1.0


def test_kappa_perfect_and_chance():
    assert kappa([True, False, True], [True, False, True]) == 1.0
    assert kappa([True, True, False, False], [True, False, True, False]) == 0.0


def test_fallback_only_fills_empty_v2_reports_and_is_flagged():
    cause = {"rank": 1, "dimension": "seller_state", "segment": "SP", "effect": "rate"}
    v2 = {"a": {"report": {"anomaly_confirmed": False, "root_causes": []}},
          "b": {"report": {"anomaly_confirmed": True, "root_causes": [{**cause, "segment": "RJ"}]}}}
    b2 = {"a": {"report": {"anomaly_confirmed": True, "root_causes": [cause]}},
          "b": {"report": {"anomaly_confirmed": True, "root_causes": [cause]}}}
    out = with_fallback(v2, b2)
    assert out["a"]["fallback_used"] and out["a"]["report"]["root_causes"][0]["source"] == "fallback"
    assert not out["b"]["fallback_used"] and out["b"]["report"]["root_causes"][0]["segment"] == "RJ"


def test_ungrounded_numbers_are_classified():
    ev = {"e1": {"tool": "t", "args": {}, "result": {"a": 3.948, "b": 2.256, "p": 0.0000012}}}
    row = {"evidence": ev, "verification": {"ungrounded": ["narrative:0.001", "narrative:1.692", "narrative:77.7"]},
           "report": {"narrative": "p < 0.001; fell by 1.692 stars; up 77.7 (e1)", "evidence_ids": ["e1"]}}
    c = classify_ungrounded(row)
    assert c == {"threshold_notation": 1, "derived": 1, "invented": 1}


def test_week_clustered_bootstrap_is_reproducible_and_wider_when_weeks_differ():
    from eval.metrics_v2 import cluster_bootstrap
    recs = [{"week": f"w{w}", "planted": True, "top1": w < 3, "top3": w < 3, "tp": int(w < 3), "fn": int(w >= 3),
             "fp": 0} for w in range(6) for _ in range(5)]  # whole weeks right or wrong: strong clustering
    a, b = cluster_bootstrap(recs), cluster_bootstrap(recs)
    assert a == b and a["n_weeks"] == 6
    assert a["top1"]["lo"] < 50 < a["top1"]["hi"] and a["top1"]["hi"] - a["top1"]["lo"] > 40

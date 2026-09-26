"""v2: evidence-based re-ranking, the avg_item_price metric, the v1 reproduction switch and the v2 prompt."""
import json

import pytest

from agent.llm_client import ScriptedLLM
from agent.loop import run_agent
from agent.prompts import SYSTEM_PROMPT_V1, SYSTEM_PROMPT_V2
from agent.rerank import rerank
from tests.fixtures.synth import LAST, TARGET, WEEK0, make_store
from tools.context import Investigation
from tools.registry import execute, openai_tools

W = TARGET.strftime("%Y-%m-%d")


def sig_evidence(tests):
    ev = {}
    for i, (dim, seg, eff, sig, hz, q) in enumerate(tests, 1):
        ev[f"e{i}"] = {"tool": "significance_test", "args": {}, "result": {
            "dimension": dim, "segment": seg, "effect_tested": eff, "significant": sig, "hist_z": hz,
            "q_value_bh": q}}
    return ev


def rc(dim, seg, eff, rank, ids):
    return {"rank": rank, "dimension": dim, "segment": seg, "effect": eff, "evidence_ids": ids}


def test_rerank_orders_by_evidence_and_drops_unsupported_causes():
    ev = sig_evidence([("is_repeat_customer", "0", "rate", True, -4.4, 0.02),
                        ("product_category", "telephony", "share", True, 5.6, 1e-23),
                        ("main_payment_type", "boleto", "rate", False, -3.0, 0.2)])
    report = {"root_causes": [rc("is_repeat_customer", "0", "rate", 1, ["e1"]),
                              rc("product_category", "telephony", "mix", 2, ["e2"]),
                              rc("main_payment_type", "boleto", "rate", 3, ["e3"]),
                              rc("seller_state", "SP", "rate", 4, [])]}
    new, log = rerank(report, ev)
    assert [(c["segment"], c["rank"]) for c in new["root_causes"]] == [("telephony", 1), ("0", 2)]
    assert log["changed"] and log["dropped"] == ["main_payment_type=boleto", "seller_state=SP"]
    assert new["root_causes"][0]["effect"] == "mix"  # the LLM's label is kept


def test_rerank_prefers_cited_then_effect_matching_test_and_keeps_ties_in_llm_order():
    ev = sig_evidence([("seller_state", "PR", "rate", True, 9.0, 1e-5),     # not the label's effect
                        ("seller_state", "PR", "share", False, 2.0, 0.3),    # cited, matches "mix": not significant
                        ("customer_state", "RJ", "rate", True, 5.0, 1e-3),
                        ("customer_state", "MG", "rate", True, 5.0, 1e-3)])
    report = {"root_causes": [rc("seller_state", "PR", "mix", 1, ["e2"]), rc("customer_state", "MG", "rate", 2, ["e4"]),
                              rc("customer_state", "RJ", "rate", 3, ["e3"])]}
    new, log = rerank(report, ev)
    assert [c["segment"] for c in new["root_causes"]] == ["MG", "RJ"]  # tie keeps the LLM's order
    assert log["dropped"] == ["seller_state=PR"]


def test_rerank_with_no_causes_is_a_no_op():
    new, log = rerank({"root_causes": []}, {})
    assert new["root_causes"] == [] and not log["changed"]


def test_avg_item_price_is_the_item_level_mean_price():
    s = make_store().weekly("avg_item_price")
    # fixture: half the items are category x (price 90), half category y (price 40)
    assert s[s["week"] == WEEK0]["value"].iloc[0] == pytest.approx(65.0)


def test_v1_switch_hides_the_new_metric_everywhere():
    inv = Investigation(store=make_store(), hidden_metrics={"avg_item_price"})
    specs = json.dumps(openai_tools(hidden_metrics=inv.hidden_metrics))
    assert "avg_item_price" not in specs and "avg_item_price" in json.dumps(openai_tools())
    assert "avg_item_price" not in execute(inv, "list_metrics", {})["metrics"]
    assert "error" in execute(inv, "detect_anomalies", {"metric": "avg_item_price", "week": W})
    ok = execute(Investigation(store=make_store()), "detect_anomalies", {"metric": "avg_item_price", "week": W})
    assert "error" not in ok


def test_prompts_v1_is_unchanged_and_v2_adds_two_rules():
    assert "avg_item_price" not in SYSTEM_PROMPT_V1 and "7. Rank root_causes" not in SYSTEM_PROMPT_V1
    assert SYSTEM_PROMPT_V2.startswith(SYSTEM_PROMPT_V1.split("Investigation id:")[0].rstrip())
    assert "7. Rank root_causes by evidence strength" in SYSTEM_PROMPT_V2 and "8. For AOV" in SYSTEM_PROMPT_V2


def _call(name, args, i):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_loop_applies_rerank_only_in_v2(version):
    report = {"metric": "gmv", "week": W, "anomaly_confirmed": True, "change_pct": -30.0, "evidence_ids": ["e1"],
              "root_causes": [{"rank": 1, "dimension": "customer_state", "segment": "A", "effect": "volume",
                               "evidence_ids": ["e1"]}], "narrative": "GMV fell (e1)."}
    llm = ScriptedLLM([_call("detect_anomalies", {"metric": "gmv", "week": W}, 0), _call("submit_report", report, 1)])
    r = run_agent(llm, Investigation(store=make_store({LAST: {"n_a": 30}})), "task", version=version)
    assert r.version == version
    if version == "v1":
        assert r.rerank is None and len(r.report["root_causes"]) == 1
    else:  # no significance test was run, so v2 drops the cause
        assert r.rerank["dropped"] == ["customer_state=A"] and r.report["root_causes"] == []


# ---- v2 final: guards (drop unsupported causes, keep LLM order; consistency) and the step limit
from agent.guards import apply_guards  # noqa: E402
from agent.loop import V2_MAX_STEPS, AgentConfig  # noqa: E402


def test_drop_guard_keeps_llm_order_and_removes_untested_causes():
    ev = sig_evidence([("seller_state", "SP", "rate", True, 4.5, 1e-3),
                       ("product_category", "garden_tools", "share", True, 9.0, 1e-9)])
    report = {"anomaly_confirmed": True, "root_causes": [rc("seller_state", "SP", "rate", 1, ["e1"]),
                                                         rc("product_category", "garden_tools", "mix", 2, ["e2"]),
                                                         rc("customer_state", "RJ", "rate", 3, [])]}
    out, log = apply_guards(report, ev)
    assert [c["segment"] for c in out["root_causes"]] == ["SP", "garden_tools"]  # no re-ordering by hist_z
    assert log["dropped"] == ["customer_state=RJ"] and not log["reorder"]
    reordered, _ = apply_guards(report, ev, reorder=True)
    assert reordered["root_causes"][0]["segment"] == "garden_tools"  # the rejected option, still available


def test_consistency_guard_sets_confirmed_or_empties_causes():
    ev = sig_evidence([("seller_state", "SP", "rate", True, 4.5, 1e-3), ("seller_state", "MG", "rate", False, 1.0, 0.4)])
    out, log = apply_guards({"anomaly_confirmed": False, "root_causes": [rc("seller_state", "SP", "rate", 1, ["e1"])]}, ev)
    assert out["anomaly_confirmed"] is True and log["confirmed_set_true"]
    out, log = apply_guards({"anomaly_confirmed": True, "root_causes": [rc("seller_state", "MG", "rate", 1, ["e2"])]}, ev)
    assert out["root_causes"] == [] and out["anomaly_confirmed"] is True  # headline flag left to the model
    out, _ = apply_guards({"anomaly_confirmed": False, "root_causes": []}, ev)
    assert out["anomaly_confirmed"] is False and out["root_causes"] == []


def test_v2_uses_15_steps_and_v1_keeps_12():
    assert V2_MAX_STEPS == 15 and AgentConfig().max_steps == 12


def test_test_manifest_uses_the_dev_pool_and_never_dev_weeks():
    import pandas as pd

    from inject.scenarios import POOL_START, load_manifest
    from inject.test_set import MIN_HISTORY_WEEKS, TEST_MANIFEST
    if not TEST_MANIFEST.exists():
        pytest.skip("TEST manifest not built")
    test, dev = load_manifest(TEST_MANIFEST), {s["week"] for s in load_manifest()}
    weeks = {s["week"] for s in test}
    assert not weeks & dev                                   # held out: no DEV week is reused
    assert min(weeks) >= POOL_START                          # same pool as DEV (no launch weeks)
    first = pd.Timestamp("2017-01-02")                       # first week of the data window
    assert min((pd.Timestamp(w) - first).days // 7 for w in weeks) >= MIN_HISTORY_WEEKS


# ---- v3: candidate guard (only causes the investigation itself verified as significant)
from agent.guards import candidate_violations, enforce_candidates, significant_candidates  # noqa: E402


def test_candidate_guard_allows_only_significant_tested_segments():
    ev = sig_evidence([("seller_state", "SP", "rate", True, 4.5, 1e-3), ("seller_state", "MG", "rate", False, 1.0, 0.4)])
    assert significant_candidates(ev) == {("seller_state", "sp")}
    report = {"root_causes": [rc("seller_state", " sp ", "rate", 1, []), rc("seller_state", "MG", "rate", 2, []),
                              rc("customer_state", "RJ", "rate", 3, [])]}
    assert candidate_violations(report, ev) == ["seller_state=MG", "customer_state=RJ"]  # case/space-insensitive
    out, dropped = enforce_candidates(report, ev)
    assert [(c["segment"], c["rank"]) for c in out["root_causes"]] == [(" sp ", 1)] and len(dropped) == 2


def test_v3_loop_warns_once_then_drops_unverified_causes():
    report = {"metric": "gmv", "week": W, "anomaly_confirmed": True, "change_pct": -30.0, "evidence_ids": ["e1"],
              "root_causes": [{"rank": 1, "dimension": "customer_state", "segment": "A", "effect": "volume",
                               "evidence_ids": ["e1"]}], "narrative": "GMV fell (e1)."}
    llm = ScriptedLLM([_call("detect_anomalies", {"metric": "gmv", "week": W}, 0), _call("submit_report", report, 1),
                       _call("submit_report", report, 2)])  # ignores the warning and resubmits
    r = run_agent(llm, Investigation(store=make_store({LAST: {"n_a": 30}})), "task", version="v3")
    assert r.candidate_guard == {"warned": ["customer_state=A"], "dropped": ["customer_state=A"]}
    assert r.report["root_causes"] == [] and r.llm_calls == 3

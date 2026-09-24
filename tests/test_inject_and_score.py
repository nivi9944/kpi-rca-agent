import pandas as pd
import pytest

from eval.score import cause_matches, score_one, summarize
from inject.scenarios import apply_scenario
from tests.fixtures.synth import TARGET, make_store

W = TARGET.strftime("%Y-%m-%d")


def sc(typ, params, seed=1):
    return {"id": "t", "type": typ, "week": W, "params": params, "seed": seed}


def test_volume_drop_removes_share_of_segment_only():
    base = make_store()
    st = apply_scenario(base, sc("volume_drop", {"dimension": "customer_state", "segment": "A", "pct": 0.5}))
    wk = st.orders[st.orders["week"] == TARGET]
    assert (wk["customer_state"] == "A").sum() == 30 and (wk["customer_state"] == "B").sum() == 40
    assert len(st.items) == len(st.orders)
    assert len(base.orders) == 2000  # base untouched


def test_price_drop_updates_items_and_order_value():
    st = apply_scenario(make_store(), sc("price_drop", {"dimension": "product_category", "segment": "x", "pct": 0.5}))
    it = st.items[(st.items["week"] == TARGET) & (st.items["product_category"] == "x")]
    assert it["price"].iloc[0] == pytest.approx(45.0) and it["item_value"].iloc[0] == pytest.approx(55.0)
    o = st.orders.set_index("order_id").loc[it["order_id"].iloc[0]]
    assert o["order_value"] == pytest.approx(55.0)


def test_delivery_delay_turns_orders_late():
    st = apply_scenario(make_store(), sc("delivery_delay", {"dimension": "seller_state", "segment": "B", "days": 4}))
    wk = st.orders[(st.orders["week"] == TARGET) & (st.orders["main_seller_state"] == "B")]
    assert wk["is_on_time"].sum() == 0  # on-time orders had delay -3 -> +1 (late); late ones 2 -> 6
    assert (wk["delay_days"] >= 0).all()


def test_mix_shift_adds_orders_from_segment_with_unchanged_rate():
    st = apply_scenario(make_store(), sc("mix_shift", {"dimension": "product_category", "segment": "y", "pct": 0.2}))
    wk = st.orders[st.orders["week"] == TARGET]
    assert len(wk) == 120 and (wk["main_category"] == "y").sum() == 70
    assert wk.loc[wk["main_category"] == "y", "order_value"].eq(50).all()


def test_control_clean_is_identical():
    base = make_store()
    st = apply_scenario(base, sc("control_clean", {}))
    pd.testing.assert_frame_equal(st.orders, base.orders)


def test_cause_matching_rules():
    gt = {"dimension": "product_category", "segment": "x", "effect": "mix"}
    assert cause_matches({"dimension": "product_category", "segment": "x", "effect": "mix"}, gt)
    assert not cause_matches({"dimension": "product_category", "segment": "x", "effect": "rate"}, gt)
    gt2 = {"dimension": "customer_state", "segment": "SP", "effect": None}
    assert cause_matches({"dimension": "customer_state", "segment": "SP", "effect": "volume"}, gt2)


def test_scoring_controls_and_planted():
    planted = {"id": "a", "type": "volume_drop", "severity": "large", "is_control": False,
               "ground_truth": {"anomaly": True, "dimension": "customer_state", "segment": "SP", "effect": None}}
    control = {"id": "b", "type": "control_clean", "severity": "none", "is_control": True,
               "ground_truth": {"anomaly": False, "dimension": None, "segment": None, "effect": None}}
    hit = {"anomaly_confirmed": True, "root_causes": [{"rank": 2, "dimension": "x", "segment": "y"},
                                                      {"rank": 1, "dimension": "customer_state", "segment": "SP"}]}
    s1 = score_one(planted, hit)
    assert s1["top1"] and s1["top3"] and s1["detected"]
    s2 = score_one(control, {"anomaly_confirmed": True, "root_causes": [{"rank": 1}]})
    assert s2["false_alarm"] is True
    s3 = score_one(control, {"anomaly_confirmed": False, "root_causes": []})
    summ = summarize([s1, s2, s3])
    assert summ["top1_accuracy_pct"] == 100.0 and summ["false_alarm_rate_pct"] == 50.0

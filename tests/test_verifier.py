from agent.verifier import extract_numbers, verify
from tools.context import Investigation
from tests.fixtures.synth import make_store


def inv_with(result: dict) -> Investigation:
    inv = Investigation(store=make_store())
    inv.record("drill_down", {}, result)
    return inv


def test_extract_ignores_dates_evidence_ids_and_small_ints():
    nums = [v for v, _, _ in extract_numbers("In week 2018-03-12 (e5) GMV fell 18.2% across 3 states, R$ 41,250 lost")]
    assert nums == [18.2, 41250.0]


def test_grounded_report_passes():
    inv = inv_with({"change_pct": -18.2345, "impact_brl": -41249.87, "p_value": 0.000412})
    rep = {"narrative": "GMV fell 18.2% (e1), losing R$ 41,250; p=0.0004.", "evidence_ids": ["e1"],
           "root_causes": [{"rank": 1, "contribution_pct": None, "p_value": 0.000412, "impact_brl": -41249.87,
                            "evidence_ids": ["e1"]}]}
    v = verify(rep, inv)
    assert v["ungrounded"] == [] and v["grounding_rate_pct"] == 100.0


def test_invented_number_is_caught():
    inv = inv_with({"change_pct": -18.2345})
    rep = {"narrative": "GMV fell 18.2% and SP drove 71.4% of it (e1).", "evidence_ids": ["e1"], "root_causes": []}
    v = verify(rep, inv)
    assert v["ungrounded"] == ["narrative:71.4"]
    assert v["grounding_rate_pct"] == 50.0


def test_fraction_to_percent_and_sign_flip_allowed():
    inv = inv_with({"total_delta": -0.052})
    rep = {"narrative": "On-time delivery fell 5.2 points (e1).", "evidence_ids": ["e1"], "root_causes": []}
    assert verify(rep, inv)["ungrounded"] == []


def test_only_cited_evidence_counts():
    inv = Investigation(store=make_store())
    inv.record("a", {}, {"x": 12.5})
    inv.record("b", {}, {"y": 99.9})
    rep = {"narrative": "It rose 99.9 (e1).", "evidence_ids": ["e1"], "root_causes": []}
    assert verify(rep, inv)["ungrounded"] == ["narrative:99.9"]

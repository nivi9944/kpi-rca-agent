import json

from agent.compact import Compactor, dumps

RULE = "significant = BH q<=0.05 AND |hist_z|>=4.25 (change unusual vs history)"


def sig(eid, segment, n):
    return {"evidence_id": eid, "metric": "gmv", "segment": segment, "test": "chi-square",
            "p_value": 0.01, "n_current": n, "rule": RULE, "direction": "up"}


def test_echoes_and_repeated_guidance_are_dropped_numbers_kept():
    c = Compactor()
    first = c.compact(sig("e1", "RJ", 10), {"metric": "gmv", "segment": "RJ"})
    # echoes of the call's own arguments go; the first guidance text stays
    assert "metric" not in first and "segment" not in first
    assert first["rule"] == RULE and first["evidence_id"] == "e1"
    second = c.compact(sig("e2", "SP", 20), {"metric": "gmv", "segment": "SP"})
    assert "rule" not in second                        # exact repeat of long guidance text
    assert second["test"] == "chi-square" and second["direction"] == "up"  # short values always stay
    assert second["p_value"] == 0.01 and second["n_current"] == 20


def test_non_echo_values_and_new_text_are_kept():
    c = Compactor()
    out = c.compact({"evidence_id": "e1", "metric": "aov", "note": RULE}, {"metric": "gmv"})
    assert out["metric"] == "aov"                      # differs from the argument: not an echo
    out2 = c.compact({"evidence_id": "e2", "note": RULE + " (2 tests)"}, {})
    assert "note" in out2                              # different text: not a duplicate


def test_compact_json_has_no_spaces():
    assert dumps({"a": [1, 2], "b": "x y"}) == '{"a":[1,2],"b":"x y"}'
    assert json.loads(dumps({"a": 1.5})) == {"a": 1.5}

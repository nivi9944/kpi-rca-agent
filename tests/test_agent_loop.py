"""The agent loop with a scripted fake LLM: tool execution, error recovery, verifier retry, step cap."""
import json

from agent.llm_client import ScriptedLLM
from agent.loop import AgentConfig, run_agent
from tests.fixtures.synth import LAST, TARGET, make_store
from tools.context import Investigation

W = TARGET.strftime("%Y-%m-%d")


def call(name, args, i=0):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function",
         "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}]}


def good_report(narrative="GMV fell 30.0% (e1).", **kw):
    r = {"metric": "gmv", "week": W, "anomaly_confirmed": True, "change_pct": -30.0, "evidence_ids": ["e1"],
         "root_causes": [{"rank": 1, "dimension": "customer_state", "segment": "A", "effect": "volume",
                          "evidence_ids": ["e1"]}], "narrative": narrative}
    r.update(kw)
    return r


def inv():
    return Investigation(store=make_store({LAST: {"n_a": 30}}))


def test_happy_path_records_evidence_and_accepts_report():
    llm = ScriptedLLM([call("detect_anomalies", {"metric": "gmv", "week": W}), call("submit_report", good_report(), 1)])
    # v1: the submitted report is kept as is (v2 would drop this cause, which has no significance test)
    r = run_agent(llm, inv(), "task", version="v1")
    assert r.error is None and r.report["root_causes"][0]["segment"] == "A"
    assert r.steps == 1 and r.verification["grounding_rate_pct"] == 100.0 and not r.ungrounded


def test_tool_errors_and_bad_json_go_back_to_model():
    llm = ScriptedLLM([call("no_such_tool", {}), call("drill_down", "{bad json", 1),
                       call("detect_anomalies", {"metric": "gmv", "week": W}, 2),
                       call("submit_report", good_report(), 3)])
    r = run_agent(llm, inv(), "task")
    errs = [t for t in r.trace if t.get("type") == "tool" and t.get("error")]
    assert len(errs) == 2 and r.report is not None


def test_invalid_report_schema_is_rejected_then_fixed():
    llm = ScriptedLLM([call("detect_anomalies", {"metric": "gmv", "week": W}),
                       call("submit_report", {"metric": "gmv"}, 1),  # missing required fields
                       call("submit_report", good_report(), 2)])
    r = run_agent(llm, inv(), "task")
    assert any(t["type"] == "report_invalid" for t in r.trace) and r.report is not None


def test_ungrounded_report_gets_one_correction_then_is_flagged():
    bad = good_report(narrative="GMV fell 30.0% and 87.3% came from A (e1).")
    llm = ScriptedLLM([call("detect_anomalies", {"metric": "gmv", "week": W}),
                       call("submit_report", bad, 1), call("submit_report", bad, 2)])
    r = run_agent(llm, inv(), "task")
    assert r.corrected and r.ungrounded
    assert r.verification["ungrounded"] == ["narrative:87.3"]


def test_ungrounded_report_fixed_after_correction():
    bad = good_report(narrative="GMV fell 30.0% and 87.3% came from A (e1).")
    llm = ScriptedLLM([call("detect_anomalies", {"metric": "gmv", "week": W}),
                       call("submit_report", bad, 1), call("submit_report", good_report(), 2)])
    r = run_agent(llm, inv(), "task")
    assert r.corrected and not r.ungrounded


def test_step_cap_forces_submit():
    script = [call("detect_anomalies", {"metric": "gmv", "week": W}, i) for i in range(3)]
    script.append(call("submit_report", good_report(), 9))
    llm = ScriptedLLM(script)
    r = run_agent(llm, inv(), "task", cfg=AgentConfig(max_steps=3))
    assert r.steps == 3 and r.report is not None


def test_no_tool_call_gets_a_nudge():
    llm = ScriptedLLM([{"role": "assistant", "content": "thinking..."},
                       call("detect_anomalies", {"metric": "gmv", "week": W}),
                       call("submit_report", good_report(), 1)])
    r = run_agent(llm, inv(), "task")
    assert r.trace[0]["type"] == "no_tool_call" and r.report is not None

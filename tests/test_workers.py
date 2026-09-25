"""Parallel evaluation: workers=1 and workers=2 must give identical scored rows (order-independent)."""
import json

from agent.llm_client import ScriptedLLM
from eval.run import done_ids, run_scenarios
from tests.fixtures.synth import TARGET, make_store

W = TARGET.strftime("%Y-%m-%d")


def policy(messages):
    """Stateless scripted LLM: detect first, then submit a report citing that evidence."""
    n_tool = sum(1 for m in messages if m.get("role") == "tool")
    if n_tool == 0:
        name, args = "detect_anomalies", {"metric": "gmv", "week": W}
    else:
        name, args = "submit_report", {
            "metric": "gmv", "week": W, "anomaly_confirmed": True, "change_pct": -30.0, "evidence_ids": ["e1"],
            "root_causes": [{"rank": 1, "dimension": "customer_state", "segment": "A", "effect": "volume",
                             "evidence_ids": ["e1"]}], "narrative": "GMV changed (e1)."}
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{n_tool}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def scenarios():
    out = []
    for k in range(6):
        planted = k % 2 == 0
        out.append({"id": f"w{k}", "type": "volume_drop" if planted else "control_clean",
                    "severity": "large" if planted else "none", "metric": "gmv", "week": W,
                    "params": {"dimension": "customer_state", "segment": "A", "pct": 0.5} if planted else {},
                    "seed": 100 + k, "is_control": not planted,
                    "ground_truth": {"anomaly": planted, "dimension": "customer_state" if planted else None,
                                     "segment": "A" if planted else None, "effect": None}})
    return out


def run(tmp_path, workers):
    path = tmp_path / f"w{workers}.jsonl"
    run_scenarios("scripted", scenarios(), make_store(), path, lambda: ScriptedLLM(policy), workers=workers,
                  version="v1")
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    keep = lambda r: {"id": r["id"], "score": r["score"], "report": r["report"], "steps": r["steps"],
                      "verification": r["verification"]}
    return sorted((keep(r) for r in rows), key=lambda r: r["id"]), path


def test_workers_give_identical_scored_rows_and_resume_still_works(tmp_path):
    one, p1 = run(tmp_path, 1)
    two, p2 = run(tmp_path, 2)
    assert len(one) == 6 and one == two
    assert done_ids(p2) == {s["id"] for s in scenarios()}  # one line per finished scenario: resume skips them all

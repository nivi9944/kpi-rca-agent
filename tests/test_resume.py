import json

from eval.report_tables import read_jsonl
from eval.run import done_ids


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_infra_failure_is_retried_and_counted(tmp_path):
    p = tmp_path / "gemini.jsonl"
    write_rows(p, [
        {"id": "a", "report": {"x": 1}, "error": None},
        {"id": "b", "report": None, "error": "llm error: 502"},       # infra failure: retry
        {"id": "c", "report": {"x": 1}, "error": "ungrounded"},        # has a report: keep, do not retry
        {"id": "d", "report": None, "error": "no report after step limit"},  # model failure: a result
    ])
    assert done_ids(p) == {"a", "c", "d"}

    # the retry succeeds and is appended; only the latest row per scenario is scored
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "b", "report": {"x": 2}, "error": None}) + "\n")
    rows, infra = read_jsonl(p)
    assert done_ids(p) == {"a", "b", "c", "d"}
    assert sorted(r["id"] for r in rows) == ["a", "b", "c", "d"]
    assert next(r for r in rows if r["id"] == "b")["report"] == {"x": 2}
    assert infra == 1

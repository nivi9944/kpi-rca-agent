"""Compare two run files scenario by scenario (used for the DEV checkpoint).

  python -m eval.compare_runs results/runs/nvidia.jsonl results/runs/nvidia_v2_dev.jsonl [--manifest PATH]

Prints top-1, top-3, MRR (planted), false alarms (controls), top-1 by type, and every scenario whose top-1
or false-alarm outcome changed, with the answers and (for v2) the re-ranking log.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from eval.report_tables import read_jsonl
from eval.score import cause_matches
from inject.scenarios import load_manifest


def reciprocal_rank(sc: dict, report: dict | None) -> float:
    rep = report or {}
    if sc["is_control"] or not rep.get("anomaly_confirmed"):
        return 0.0
    rcs = sorted(rep.get("root_causes") or [], key=lambda r: r.get("rank", 99))
    for i, rc in enumerate(rcs, 1):
        if cause_matches(rc, sc["ground_truth"]):
            return 1.0 / i
    return 0.0


def summary(rows: dict, man: dict) -> dict:
    planted = [i for i in rows if not man[i]["is_control"]]
    controls = [i for i in rows if man[i]["is_control"]]
    by_type = defaultdict(list)
    for i in planted:
        by_type[man[i]["type"]].append(bool(rows[i]["score"]["top1"]))
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None
    return {"n": len(rows), "top1": pct([bool(rows[i]["score"]["top1"]) for i in planted]),
            "top3": pct([bool(rows[i]["score"]["top3"]) for i in planted]),
            "mrr": round(sum(reciprocal_rank(man[i], rows[i].get("report")) for i in planted) / len(planted), 4)
            if planted else None,
            "false_alarm": pct([bool(rows[i]["score"]["false_alarm"]) for i in controls]),
            "by_type_top1": {t: pct(v) for t, v in sorted(by_type.items())}}


def answer(r: dict) -> str:
    rep = r.get("report") or {}
    rcs = sorted(rep.get("root_causes") or [], key=lambda x: x.get("rank", 99))
    return f"confirmed={rep.get('anomaly_confirmed')} " + "; ".join(
        f"{c.get('dimension')}={c.get('segment')}/{c.get('effect')}" for c in rcs[:3])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--manifest")
    args = ap.parse_args(argv)
    man = {s["id"]: s for s in load_manifest(args.manifest)}
    A = {r["id"]: r for r in read_jsonl(Path(args.a))[0]}
    B = {r["id"]: r for r in read_jsonl(Path(args.b))[0]}
    common = sorted(set(A) & set(B) & set(man))
    sa, sb = summary({i: A[i] for i in common}, man), summary({i: B[i] for i in common}, man)
    print(f"A = {args.a}\nB = {args.b}\ncommon scenarios: {len(common)}")
    for k in ("top1", "top3", "mrr", "false_alarm"):
        print(f"  {k:12} A {sa[k]}   B {sb[k]}")
    for t in sa["by_type_top1"]:
        print(f"  top1 {t:20} A {sa['by_type_top1'][t]}   B {sb['by_type_top1'].get(t)}")
    print("changed scenarios:")
    for i in common:
        a, b = A[i]["score"], B[i]["score"]
        if a["top1"] != b["top1"] or a.get("false_alarm") != b.get("false_alarm"):
            gt = man[i]["ground_truth"]
            print(f"  {i}: top1 {a['top1']} -> {b['top1']}, false_alarm {a.get('false_alarm')} -> {b.get('false_alarm')}"
                  f" | truth {gt.get('dimension')}={gt.get('segment')}/{gt.get('effect')}")
            print(f"      A: {answer(A[i])}\n      B: {answer(B[i])}")
            if B[i].get("rerank"):
                print(f"      B rerank: {B[i]['rerank']}")


if __name__ == "__main__":
    main()

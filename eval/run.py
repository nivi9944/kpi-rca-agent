"""Run an evaluation over all scenarios in inject/manifest.json.

  python -m eval.run --model baseline_topcontrib     (no LLM)
  python -m eval.run --model baseline_scan           (no LLM)
  python -m eval.run --model nvidia                  (agent via the gateway, Nemotron 3 Ultra on NVIDIA; headline)
  python -m eval.run --model mistral                 (agent via the gateway, Mistral Small 4)
  python -m eval.run --model mistral-medium          (agent via the gateway, Mistral Medium)
  python -m eval.run --model gemini                  (agent via the gateway, forced to Gemini)
  python -m eval.run --model ollama                  (agent via the gateway, forced to local Qwen)
  python -m eval.run --model scripted                (pipeline check with a fake LLM; NOT a result)

Resumable: appends one JSON line per scenario to results/runs/<model>.jsonl and skips scenarios already
there (delete the file, or pass --fresh, to start over). Rows with an error and no report (infrastructure
failures) are retried on the next run; the failed attempts stay in the file and are counted in the summary
as infra_failed_attempts. Throttled by REQUESTS_PER_MIN.

Spending guard: the estimated list-price spend of every run file in results/ (runs and runs_cache, all
models, smoke runs included) is printed after each scenario. The run stops cleanly before a scenario that
could push the total past BUDGET_USD (default 8). Gateway cache hits count as 0.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import traceback
from pathlib import Path

from baseline.rules import BASELINES
from eval.score import score_one
from inject.scenarios import apply_scenario, load_manifest, task_text
from metrics.metrics import default_store
from tools.context import Investigation
from tools.engine import engine_name

ROOT = Path(__file__).resolve().parents[1]


def out_dir(model: str) -> Path:
    # the scripted fake LLM only checks the pipeline: keep it out of results/
    return ROOT / ("trial_results" if model == "scripted" else "results")


def run_file(model: str, subdir: str = "runs") -> Path:
    return out_dir(model) / subdir / f"{model}.jsonl"


def row_spend(row: dict) -> float:
    """Estimated list-price spend of one row. Older rows without est_spend_usd count their full cost."""
    v = row.get("est_spend_usd")
    return float(v if v is not None else row.get("est_cost_usd") or 0.0)


def spend_by_file(res: Path | None = None) -> dict[str, float]:
    res = res or ROOT / "results"
    out = {}
    for sub in ("runs", "runs_cache"):
        for p in sorted((res / sub).glob("*.jsonl")) if (res / sub).exists() else []:
            rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
            out[f"{sub}/{p.name}"] = round(sum(row_spend(r) for r in rows), 6)
    return out


def worst_case_usd(model: str, path: Path) -> float:
    """Largest spend of one scenario so far for this model, or the token budget at list price if none yet."""
    seen = [row_spend(json.loads(x)) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] \
        if path.exists() else []
    if seen:
        return max(seen)
    from agent.llm_client import PRICES
    from agent.loop import AgentConfig
    p_in, p_out = PRICES.get(model, PRICES["gemini"])
    return AgentConfig().token_budget * p_in / 1e6 + 10_000 * p_out / 1e6


def is_infra_failure(row: dict) -> bool:
    """The LLM API failed after all retries (outage, quota) and there is no report: retried on the next run,
    and counted. Model failures (e.g. no report after the step limit) are results, not retried."""
    return str(row.get("error") or "").startswith("llm error") and not row.get("report")


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if not is_infra_failure(row):
                ids.add(row["id"])
        except (json.JSONDecodeError, KeyError):
            pass
    return ids


def progress_line(path: Path) -> str:
    """PROGRESS: scenarios with a report, infra failures, and running top-1 over planted scenarios so far
    (latest attempt per scenario, the same rows eval.report_tables scores)."""
    from eval.report_tables import read_jsonl
    rows, _ = read_jsonl(path)
    done = [r for r in rows if not is_infra_failure(r)]
    planted = [r for r in done if not r["score"]["is_control"]]
    top1 = f"{100 * sum(bool(r['score']['top1']) for r in planted) / len(planted):.1f}%" if planted else "n/a"
    return (f"PROGRESS: {len(done)} done, {len(rows) - len(done)} failed, "
            f"running top-1 {top1} over {len(planted)} planted")


def make_llm(model: str):
    if model == "scripted":
        from agent.scripted_policy import policy_llm
        return policy_llm()
    from agent.llm_client import ChatClient
    return ChatClient(model)


def run_one(model: str, sc: dict, base, llm=None, version: str = "v2", run_name: str | None = None,
            reorder: bool = False) -> dict:
    store = apply_scenario(base, sc)
    inv = Investigation(store=store, chart_dir=out_dir(model) / "charts" / model / sc["id"])
    t0 = time.time()
    row: dict = {"id": sc["id"], "model": run_name or model}
    try:
        if model in BASELINES:
            report = BASELINES[model](inv, sc["metric"], sc["week"])
            row.update({"report": report, "steps": len(inv.evidence), "latency_s": round(time.time() - t0, 3)})
        else:
            from agent.loop import run_agent
            r = run_agent(llm, inv, task_text(sc), inv_id=sc["id"], version=version, reorder=reorder)
            d = r.to_dict()
            row.update({"report": r.report, "error": r.error, "steps": r.steps, "llm_calls": r.llm_calls,
                        "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                        "total_tokens": r.total_tokens, "est_cost_usd": d["est_cost_usd"],
                        "est_spend_usd": d["est_spend_usd"], "est_price_usd_per_m": list(r.price_per_m),
                        "model_name": r.model_name, "served_models": r.served_models,
                        "agent_version": r.version, "rerank": r.rerank, "report_raw": r.report_raw,
                        "candidate_guard": r.candidate_guard, "engine": engine_name(),
                        "latency_s": round(r.latency_s, 2), "llm_latency_s": round(r.llm_latency_s, 2),
                        "cache_hits": r.cache_hits, "ungrounded": r.ungrounded, "corrected": r.corrected,
                        "grounding_rate_pct": (r.verification or {}).get("grounding_rate_pct"),
                        "verification": r.verification, "trace": r.trace})
    except Exception as e:
        row.update({"report": None, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2000:]})
    row["score"] = score_one(sc, row.get("report") if not row.get("error") or row.get("report") else None)
    row["evidence"] = {k: v for k, v in inv.evidence.items()}
    return row


def run_scenarios(model: str, scs: list[dict], base, path: Path, llm_factory, workers: int = 1,
                  version: str = "v2", run_name: str | None = None, reorder: bool = False,
                  before=None, after=None) -> list[dict]:
    """Run scenarios with `workers` threads. Each thread has its own LLM client (its own pacing); each scenario
    builds its own Investigation on its own copy of the data. One JSONL line per finished scenario, written under
    a lock, so resume works and results do not depend on order. `before(sc)` and `after(row)` run under the
    lock; either may return False to stop new scenarios from starting (budget or outage)."""
    lock, stop, local = threading.Lock(), threading.Event(), threading.local()
    rows: list[dict] = []

    def work(sc):
        if stop.is_set():
            return None
        with lock:
            if before and before(sc) is False:
                stop.set()
                return None
        if llm_factory is not None and not hasattr(local, "llm"):
            local.llm = llm_factory()
        row = run_one(model, sc, base, getattr(local, "llm", None), version, run_name, reorder)
        with lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
            rows.append(row)
            if after and after(row) is False:
                stop.set()
        return row

    if workers <= 1:
        for sc in scs:
            work(sc)
            if stop.is_set():
                break
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, scs))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["baseline_topcontrib", "baseline_scan", "nvidia", "nvidia-super", "mistral", "mistral-medium", "gemini", "ollama",
                             "scripted"])
    ap.add_argument("--only", nargs="*", help="scenario ids to run")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--cache-rerun", action="store_true",
                    help="write to results/runs_cache/ instead: a second pass to measure gateway cache hits")
    ap.add_argument("--max-infra-streak", type=int, default=2,
                    help="stop after this many scenarios in a row fail with no report (likely daily quota)")
    ap.add_argument("--agent-version", choices=["v1", "v2", "v3"], default="v2",
                    help="v1 reproduces the v1.0.0 agent (old prompt, no avg_item_price, no re-ranking)")
    ap.add_argument("--reorder", action="store_true",
                    help="v2 only: evidence re-ranking of causes (tried on DEV and rejected; off by default)")
    ap.add_argument("--run-name", help="results file stem (default: the model name)")
    ap.add_argument("--manifest", help="scenario manifest (default: inject/manifest.json, the DEV set)")
    ap.add_argument("--results-dir", help="base results folder (default: results/); the TEST set uses results/v2")
    ap.add_argument("--only-file", help="JSON list of scenario ids to run (fixed subsets: repeat, cache rerun)")
    ap.add_argument("--no-cache", action="store_true", help="bypass the gateway cache (stability repeats)")
    ap.add_argument("--engine", choices=["v2", "v3"], help="analysis engine (threshold, money test); default v3")
    ap.add_argument("--deadline", help="ISO time: do not start a scenario that would likely finish after it")
    ap.add_argument("--workers", type=int, default=int(os.getenv("EVAL_WORKERS", "1")),
                    help="scenarios run concurrently (each worker has its own LLM client and pacing)")
    args = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    budget = float(os.getenv("BUDGET_USD", "8"))
    guarded = args.model not in BASELINES and args.model != "scripted"

    if args.engine:
        from tools.engine import set_engine
        set_engine(args.engine)
    if args.no_cache:
        os.environ["LLM_CACHE_BYPASS"] = "true"
    base_dir = Path(args.results_dir) if args.results_dir else out_dir(args.model)
    path = base_dir / ("runs_cache" if args.cache_rerun else "runs") / f"{args.run_name or args.model}.jsonl"
    if args.only_file:
        args.only = (args.only or []) + json.loads(Path(args.only_file).read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh and path.exists():
        path.unlink()
    done = done_ids(path)
    scs = [s for s in load_manifest(args.manifest) if (not args.only or s["id"] in args.only) and s["id"] not in done]
    if args.limit:
        scs = scs[: args.limit]
    base = default_store()
    llm_factory = None if args.model in BASELINES else (lambda: make_llm(args.model))
    print(f"{args.model}: {len(done)} done, {len(scs)} to run -> {path} (workers {args.workers})", flush=True)
    state = {"i": 0, "streak": 0}

    def before(sc):
        if args.deadline:
            from datetime import datetime
            done_rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]                 if path.exists() else []
            lat = [r.get("latency_s") or 0 for r in done_rows if r.get("latency_s")]
            expected = (sum(lat) / len(lat)) if lat else 240.0
            left = (datetime.fromisoformat(args.deadline) - datetime.now()).total_seconds()
            if left < expected:
                print(f"STOPPED (time budget): {left / 60:.1f} min left before the deadline, an average scenario "
                      f"takes {expected / 60:.1f} min. Finished scenarios are saved; this is not an error.", flush=True)
                return False
        if guarded:
            spent, worst = sum(spend_by_file().values()), worst_case_usd(args.model, path)
            if spent + worst > budget:
                print(f"STOPPED (budget): estimated spend so far ${spent:.4f}; the next scenario could cost up to "
                      f"${worst:.4f}, which would pass BUDGET_USD=${budget:.2f}. Nothing else was sent.", flush=True)
                return False
        return True

    def after(row):
        state["i"] += 1
        i, s = state["i"], row["score"]
        print(f"[{i}/{len(scs)}] {row['id']:<32} top1={s['top1']} top3={s['top3']} "
              f"false_alarm={s['false_alarm']} steps={row.get('steps')} "
              f"err={(row.get('error') or '')[:80]}", flush=True)
        if guarded:
            print(f"    tokens {row.get('total_tokens')}; est. spend: this scenario ${row_spend(row):.4f}, "
                  f"all results/ ${sum(spend_by_file().values()):.4f} "
                  f"of ${budget:.2f} budget; model {row.get('model_name')} served by {row.get('served_models')}; "
                  f"{(row.get('latency_s') or 0) / 60:.1f} min", flush=True)
        if i % 10 == 0 or i == len(scs):
            print(progress_line(path), flush=True)
        # each failed scenario already retried for several minutes, so a per-minute limit would have cleared
        state["streak"] = state["streak"] + 1 if is_infra_failure(row) else 0
        if args.max_infra_streak and state["streak"] >= args.max_infra_streak:
            print(f"STOPPED: {state['streak']} scenarios in a row failed with no report (last error: "
                  f"{(row.get('error') or '')[:120]}). Likely the daily quota or an outage. "
                  f"Rerun the same command later; failed scenarios will be retried.", flush=True)
            return False
        return True

    run_scenarios(args.model, scs, base, path, llm_factory, args.workers, args.agent_version, args.run_name,
                  args.reorder, before, after)
    if not args.results_dir:  # the TEST set (results/v2) has its own metrics script
        from eval.report_tables import main as tables
        tables()


if __name__ == "__main__":
    main()

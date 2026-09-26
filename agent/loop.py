"""The agent loop (no framework): model proposes tool calls -> we execute -> append results -> repeat.

Bounded: MAX_STEPS tool-calling turns, a token budget, a wall-clock timeout, and one corrective retry
if the report fails the grounding check. Tool errors and malformed calls are returned to the model
as data so it can recover.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from agent.compact import Compactor, dumps
from agent.prompts import CANDIDATE_PROMPT, CORRECTION_PROMPT, SYSTEM_PROMPT_V1, SYSTEM_PROMPT_V2
from agent.guards import apply_guards, candidate_violations, enforce_candidates
from agent.schemas import Report
from agent.verifier import verify
from tools.context import Investigation
from tools.registry import _inline_refs, execute, openai_tools

MAX_TOOL_CHARS = 6000
PRICE_IN_PER_M, PRICE_OUT_PER_M = 0.75, 3.75  # default: Gemini Flash paid-tier list price, USD (estimate only)


def submit_tool() -> dict:
    return {"type": "function", "function": {
        "name": "submit_report",
        "description": "Submit the final investigation report (call exactly once, at the end).",
        "parameters": _inline_refs(Report.model_json_schema())}}


@dataclass
class AgentConfig:
    max_steps: int = int(os.getenv("MAX_STEPS", "12"))
    token_budget: int = 250_000
    timeout_s: float = 900.0


@dataclass
class AgentResult:
    report: dict | None
    verification: dict | None
    trace: list = field(default_factory=list)
    steps: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    llm_latency_s: float = 0.0
    cache_hits: int = 0
    ungrounded: bool = False
    corrected: bool = False
    error: str | None = None
    price_per_m: tuple = (PRICE_IN_PER_M, PRICE_OUT_PER_M)  # the model's paid list price (estimate only)
    model_name: str | None = None                      # model name the client asked for
    served_models: list = field(default_factory=list)  # model names in the provider responses
    billable_prompt_tokens: int = 0                    # tokens of calls that reached the provider (no cache hit)
    billable_completion_tokens: int = 0
    version: str = "v2"                                # agent version (v1 = the v1.0.0 behaviour)
    rerank: dict | None = None                         # v2: what the guards changed (dropped causes, confirmed flag)
    report_raw: dict | None = None                     # the report exactly as the LLM submitted it (before guards)
    candidate_guard: dict | None = None                # v3: causes the candidate guard warned about / dropped

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def est_cost_usd(self) -> float:
        p_in, p_out = self.price_per_m
        return (self.prompt_tokens * p_in + self.completion_tokens * p_out) / 1e6

    def est_spend_usd(self) -> float:
        """Estimated list-price spend of calls that reached the provider (gateway cache hits cost nothing)."""
        p_in, p_out = self.price_per_m
        return (self.billable_prompt_tokens * p_in + self.billable_completion_tokens * p_out) / 1e6

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["total_tokens"] = self.total_tokens
        d["est_cost_usd"] = round(self.est_cost_usd(), 6)
        d["est_spend_usd"] = round(self.est_spend_usd(), 6)
        return d


def _args(raw) -> dict:
    try:
        a = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return {}
    return a if isinstance(a, dict) else {}


def _truncate(obj: dict) -> str:
    s = dumps(obj)  # compact JSON: fewer tokens, same content
    return s if len(s) <= MAX_TOOL_CHARS else s[:MAX_TOOL_CHARS] + '..."(truncated)"'


V1_HIDDEN_METRICS = {"avg_item_price"}  # added in v2
V2_MAX_STEPS = 15  # 34 of 50 v2 DEV round-1 runs (68%) hit the v1 limit of 12 (DECISIONS.md)


def run_agent(llm, inv: Investigation, task: str, inv_id: str = "adhoc",
              cfg: AgentConfig | None = None, on_event=None, version: str = "v2",
              reorder: bool = False) -> AgentResult:
    cfg = cfg or (AgentConfig() if version == "v1" else AgentConfig(max_steps=V2_MAX_STEPS))
    if version == "v1":
        inv.hidden_metrics = set(V1_HIDDEN_METRICS)
    prompt = SYSTEM_PROMPT_V1 if version == "v1" else SYSTEM_PROMPT_V2
    tools = openai_tools([submit_tool()], inv.hidden_metrics)
    messages = [{"role": "system", "content": prompt.format(inv_id=inv_id)},
                {"role": "user", "content": task}]
    compactor = Compactor()  # the model sees compacted tool results; inv.evidence keeps them in full
    res = AgentResult(report=None, verification=None,
                      price_per_m=tuple(getattr(llm, "price", (PRICE_IN_PER_M, PRICE_OUT_PER_M))),
                      model_name=getattr(llm, "model", None), version=version)
    t_start = time.time()
    retried = False
    cand_warned = False
    final_forced = False

    def emit(ev):
        res.trace.append(ev)
        if on_event:
            on_event(ev)

    turn = 0
    while True:
        turn += 1
        if time.time() - t_start > cfg.timeout_s:
            res.error = "timeout"
            break
        if res.total_tokens > cfg.token_budget:
            res.error = "token budget exceeded"
            break
        if res.steps >= cfg.max_steps and not final_forced:
            final_forced = True
            messages.append({"role": "user", "content": "Step limit reached. Call submit_report now with the evidence you have."})
        if turn > cfg.max_steps + 6:
            res.error = "no report after step limit"
            break
        use_tools = [submit_tool()] if final_forced else tools
        try:
            r = llm.chat(messages, use_tools)
        except Exception as e:  # provider down after retries
            res.error = f"llm error: {type(e).__name__}: {e}"
            break
        res.llm_calls += 1
        res.prompt_tokens += r.usage.get("prompt_tokens", 0)
        res.completion_tokens += r.usage.get("completion_tokens", 0)
        res.llm_latency_s += r.latency_s
        if r.cache and r.cache.startswith("HIT"):
            res.cache_hits += 1
        else:
            res.billable_prompt_tokens += r.usage.get("prompt_tokens", 0)
            res.billable_completion_tokens += r.usage.get("completion_tokens", 0)
        served = getattr(r, "served_model", None)
        if served and served not in res.served_models:
            res.served_models.append(served)
        msg = r.message
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            emit({"type": "no_tool_call", "content": str(msg.get("content") or "")[:500]})
            messages.append({"role": "user", "content": "Please continue by calling a tool, or call submit_report to finish."})
            continue
        done = False
        for tc in calls:
            name = tc["function"]["name"]
            raw = tc["function"]["arguments"]
            if name == "submit_report":
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                    report = Report(**args).model_dump()
                except (ValidationError, json.JSONDecodeError, TypeError) as e:
                    out = {"error": f"Invalid report: {e}"}
                    emit({"type": "report_invalid", "error": str(e)[:500]})
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": _truncate(out)})
                    continue
                if version == "v3":  # v3 candidate guard: only causes this investigation verified as significant
                    bad = candidate_violations(report, inv.evidence)
                    if bad and not cand_warned:
                        cand_warned = True
                        res.candidate_guard = {"warned": bad, "dropped": []}
                        emit({"type": "candidate_guard", "violations": bad})
                        messages.append({"role": "tool", "tool_call_id": tc["id"],
                                         "content": CANDIDATE_PROMPT.format(bad=", ".join(bad))})
                        continue
                    if bad:
                        report, dropped = enforce_candidates(report, inv.evidence)
                        res.candidate_guard = {**(res.candidate_guard or {"warned": []}), "dropped": dropped}
                v = verify(report, inv)
                emit({"type": "report", "verification": v})
                if v["ungrounded"] and not retried:
                    retried = True
                    res.corrected = True
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": CORRECTION_PROMPT.format(bad=", ".join(v["ungrounded"]))})
                    continue
                res.report, res.verification = report, v
                if version != "v1":  # v2 guards (deterministic; the verifier already ran on the LLM's report)
                    res.report_raw = report
                    res.report, res.rerank = apply_guards(report, inv.evidence, reorder=reorder)
                res.ungrounded = bool(v["ungrounded"])
                done = True
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "Report accepted."})
                break
            res.steps += 1
            t0 = time.time()
            out = execute(inv, name, raw)
            emit({"type": "tool", "step": res.steps, "tool": name, "args": raw if isinstance(raw, str) else json.dumps(raw),
                  "evidence_id": out.get("evidence_id"), "error": out.get("error"), "seconds": round(time.time() - t0, 3)})
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": _truncate(compactor.compact(out, _args(raw)))})
        if done:
            break
    res.latency_s = time.time() - t_start
    return res

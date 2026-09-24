"""Streamlit demo: pick a metric + week (or a planted scenario), watch the tool calls, read the report.

  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from baseline.rules import scan_and_test, top_contribution  # noqa: E402
from inject.scenarios import apply_scenario, load_manifest, task_text  # noqa: E402
from metrics.metrics import default_store, load_catalog  # noqa: E402
from tools.context import Investigation  # noqa: E402
from tools.registry import execute  # noqa: E402

st.set_page_config(page_title="KPI Root-Cause Agent", layout="wide")
st.title("KPI Root-Cause Agent")
st.caption("Olist e-commerce (Brazil, R$). The LLM plans and explains; deterministic Python computes every number.")


@st.cache_resource
def base_store():
    return default_store()


cat = load_catalog()
with st.sidebar:
    mode = st.radio("Data", ["Planted scenario", "Real week"])
    if mode == "Planted scenario":
        scs = load_manifest()
        sc = st.selectbox("Scenario", scs, format_func=lambda s: f"{s['id']}  ({s['metric']}, {s['week']})")
        metric, week = sc["metric"], sc["week"]
        with st.expander("Ground truth (hidden from the agent)"):
            st.json(sc["ground_truth"])
    else:
        sc = None
        metric = st.selectbox("Metric", list(cat["metrics"]), format_func=lambda m: cat["metrics"][m]["label"])
        week = st.text_input("Week (YYYY-MM-DD)", "2018-05-21")
    runner = st.selectbox("Investigator", ["Agent (Nemotron via gateway)", "Agent (Mistral Small via gateway)", "Agent (Gemini via gateway)", "Agent (Qwen via gateway)",
                                           "Baseline B2: scan + test", "Baseline B1: top contribution"])
    go = st.button("Investigate", type="primary")

if go:
    store = apply_scenario(base_store(), sc) if sc else base_store()
    chart_dir = ROOT / "app_charts"
    inv = Investigation(store=store, chart_dir=chart_dir)
    left, right = st.columns([1, 1.3])
    with left:
        st.subheader("Tool-call trace")
        trace_box = st.container()
    report = None
    if runner.startswith("Agent"):
        from agent.llm_client import ChatClient
        from agent.loop import run_agent

        task = task_text(sc) if sc else task_text({"metric": metric, "week": week})

        def on_event(ev):
            with trace_box:
                if ev["type"] == "tool":
                    st.markdown(f"**{ev['step']}. `{ev['tool']}`** -> {ev.get('evidence_id') or ev.get('error')}")
                    st.code(ev["args"], language="json")
                elif ev["type"] == "report":
                    v = ev["verification"]
                    st.markdown(f"Report submitted. Grounding: **{v['n_grounded']}/{v['n_numbers']}** numbers verified")

        try:
            llm = ChatClient("ollama" if "Qwen" in runner else "gemini" if "Gemini" in runner
                             else "mistral" if "Mistral" in runner else "nvidia")
            with st.spinner("Agent investigating..."):
                res = run_agent(llm, inv, task, inv_id=(sc or {}).get("id", f"app-{metric}-{week}"), on_event=on_event)
            report = res.report
            st.sidebar.metric("Tool calls", res.steps)
            st.sidebar.metric("Tokens", res.total_tokens)
            st.sidebar.metric("Latency (s)", f"{res.latency_s:.1f}")
            if res.error:
                st.error(res.error)
        except Exception as e:  # gateway down etc.
            st.error(f"LLM not reachable ({e}). Start the gateway (docker compose up -d) or pick a baseline.")
    else:
        fn = scan_and_test if "B2" in runner else top_contribution
        report = fn(inv, metric, week)
        with trace_box:
            for eid, ev in inv.evidence.items():
                st.markdown(f"**{eid}. `{ev['tool']}`**")
                st.code(json.dumps(ev["args"]), language="json")

    with right:
        st.subheader("Report")
        if report:
            c1, c2 = st.columns(2)
            c1.metric("Anomaly confirmed", "Yes" if report.get("anomaly_confirmed") else "No")
            if report.get("change_pct") is not None:
                c2.metric(f"{cat['metrics'][metric]['label']} change", f"{report['change_pct']:.1f}%")
            if report.get("narrative"):
                st.write(report["narrative"])
            rcs = report.get("root_causes") or []
            if rcs:
                st.dataframe(rcs, width="stretch")
            else:
                st.info("No significant root cause found.")
            for a in report.get("recommended_actions") or []:
                st.markdown(f"- {a}")
            ch = execute(inv, "make_chart", {"kind": "series", "metric": metric, "week": week})
            st.image(ch["chart_path"])
            if rcs:
                ch2 = execute(inv, "make_chart", {"kind": "segments", "metric": metric, "week": week,
                                                  "dimension": rcs[0]["dimension"]})
                st.image(ch2["chart_path"])
            if sc:
                from eval.score import score_one
                st.caption(f"Score vs ground truth: {score_one(sc, report)}")
    with st.expander("All evidence (tool outputs)"):
        st.json({k: v["result"] for k, v in inv.evidence.items()})

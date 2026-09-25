"""Tool registry: typed (Pydantic) input schemas, OpenAI function-calling specs, and execution.

The agent loop only ever calls `execute(inv, name, raw_args)`. Inputs are validated by Pydantic
before any code runs; every output is recorded as evidence and gets an evidence_id.
"""
from __future__ import annotations

import json
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from metrics.metrics import load_catalog, to_week
from tools.anomaly import detect_anomalies
from tools.charts import make_chart
from tools.context import Investigation
from tools.decompose import decompose_metric
from tools.drilldown import drill_down
from tools.impact import estimate_impact
from tools.sql_guard import run_sql
from tools.segscan import scan_segments
from tools.stats import significance_test

MetricName = Literal["gmv", "orders", "aov", "avg_item_price", "on_time_rate", "avg_delay_days", "avg_review_score",
                     "cancellation_rate", "repeat_customer_rate"]
DimName = Literal["customer_state", "product_category", "seller_state", "seller_id",
                  "main_payment_type", "is_repeat_customer"]
WEEK_DESC = "Target week as YYYY-MM-DD (any day; normalised to the Monday that starts the week)"


class ListMetricsArgs(BaseModel):
    pass


class SeriesArgs(BaseModel):
    metric: MetricName
    start: Optional[str] = Field(None, description="First week YYYY-MM-DD")
    end: Optional[str] = Field(None, description="Last week YYYY-MM-DD")
    filter_dimension: Optional[DimName] = Field(None, description="Optional: restrict to one segment of this dimension")
    filter_segment: Optional[str] = Field(None, description="Segment value for filter_dimension, e.g. SP")


class DetectArgs(BaseModel):
    metric: MetricName
    week: Optional[str] = Field(None, description=WEEK_DESC)
    start: Optional[str] = None
    end: Optional[str] = None
    filter_dimension: Optional[DimName] = None
    filter_segment: Optional[str] = None


class DecomposeArgs(BaseModel):
    metric: MetricName
    week: str = Field(..., description=WEEK_DESC)
    filter_dimension: Optional[DimName] = None
    filter_segment: Optional[str] = None


class DrillArgs(BaseModel):
    metric: MetricName
    dimension: DimName
    week: str = Field(..., description=WEEK_DESC)
    filter_dimension: Optional[DimName] = Field(None, description="Optional: restrict to one segment first, to drill one level deeper")
    filter_segment: Optional[str] = Field(None, description="Segment value for filter_dimension, e.g. SP")
    top_k: int = Field(5, ge=1, le=15)


class SigArgs(BaseModel):
    metric: MetricName
    dimension: DimName
    segment: str
    week: str = Field(..., description=WEEK_DESC)
    effect: Optional[Literal["rate", "share"]] = Field(
        None, description="'share' tests whether the segment's share of orders changed (use for GMV/Orders and mix effects); 'rate' tests whether the metric inside the segment changed. Default: share for GMV/Orders, rate otherwise")
    filter_dimension: Optional[DimName] = None
    filter_segment: Optional[str] = None


class ImpactArgs(BaseModel):
    metric: MetricName
    dimension: DimName
    segment: str
    week: str = Field(..., description=WEEK_DESC)
    filter_dimension: Optional[DimName] = None
    filter_segment: Optional[str] = None


class SQLArgs(BaseModel):
    query: str = Field(..., description="One read-only SELECT over fact_orders / fact_items; max 1000 rows")


class ScanArgs(BaseModel):
    metric: MetricName
    week: str = Field(..., description=WEEK_DESC)


class ChartArgs(BaseModel):
    kind: Literal["series", "segments"]
    metric: MetricName
    week: str = Field(..., description=WEEK_DESC)
    dimension: Optional[DimName] = None


def _list_metrics(inv: Investigation) -> dict:
    cat = load_catalog()
    return {"metrics": {k: {"label": v["label"], "description": v["description"], "unit": v["unit"]}
                        for k, v in cat["metrics"].items() if k not in inv.hidden_metrics},
            "dimensions": list(cat["dimensions"]),
            "data_window": "weekly, 2017-01-02 to 2018-08-27 (Olist, R$)"}


def _series(inv: Investigation, metric, start=None, end=None, filters=None) -> dict:
    s = inv.store.weekly(metric, filters)
    if start:
        s = s[s["week"] >= to_week(start)]
    if end:
        s = s[s["week"] <= to_week(end)]
    s = s.tail(60)
    return {"metric": metric, "filters": filters or {},
            "series": [{"week": r.week, "value": r.value, "n": r.n} for r in s.itertuples()]}


TOOLS: dict[str, tuple[Callable, type[BaseModel], str]] = {
    "list_metrics": (_list_metrics, ListMetricsArgs, "List the available metrics and drill-down dimensions."),
    "get_metric_series": (_series, SeriesArgs, "Weekly values of a metric (optionally for one segment)."),
    "detect_anomalies": (detect_anomalies, DetectArgs,
                         "Check whether a week is anomalous for a metric (robust z-score vs trailing weeks). Always call this first with the target week."),
    "decompose_metric": (decompose_metric, DecomposeArgs,
                         "Split the change into drivers: GMV = Orders x AOV; AOV = items per order x value per item; rates = numerator / denominator."),
    "drill_down": (drill_down, DrillArgs,
                   "Rank segments of a dimension by their contribution (%) to the change vs the previous 4-week average. For ratio/rate metrics it also splits each segment into mix and rate effects."),
    "scan_segments": (scan_segments, ScanArgs,
                      "Scan all dimensions for segments whose change this week is unusual versus their own weekly history (catches problems hidden in the total). For ratio metrics scans both rate and share (mix)."),
    "significance_test": (significance_test, SigArgs,
                          "Test whether a segment's change is statistically significant. Returns p-value and a Benjamini-Hochberg q-value over all tests in this investigation."),
    "estimate_impact": (estimate_impact, ImpactArgs, "Business impact of a segment's change (R$ where possible)."),
    "run_sql": (run_sql, SQLArgs, "Ad-hoc read-only SQL on fact_orders and fact_items for checks the other tools do not cover."),
    "make_chart": (make_chart, ChartArgs, "Save a chart (series or segment contributions) for the report."),
}


def _inline_refs(schema: dict) -> dict:
    """Inline $defs so simple function-calling backends (Gemini, Ollama) accept the schema."""
    defs = schema.pop("$defs", {})

    def walk(x):
        if isinstance(x, dict):
            if "$ref" in x:
                return walk(dict(defs[x["$ref"].split("/")[-1]]))
            out = {}
            for k, v in x.items():
                if k in ("title", "default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
                    continue
                out[k] = walk(v)
            if "anyOf" in out:  # Optional[X] -> X (nullable handled by 'required')
                opts = [o for o in out["anyOf"] if o.get("type") != "null"]
                if len(opts) == 1:
                    rest = {k: v for k, v in out.items() if k != "anyOf"}
                    out = {**opts[0], **rest}
            if out.get("type") == "object" and "additionalProperties" in out and isinstance(out["additionalProperties"], dict):
                # dict[Literal, str] -> plain object with string values
                out["additionalProperties"] = {"type": "string"}
                out.pop("propertyNames", None)
            return out
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x

    return walk(schema)


def _hide(node, hidden: set):
    """Remove hidden metric names from every enum in a schema (v1 reproduction)."""
    if isinstance(node, dict):
        if isinstance(node.get("enum"), list):
            node["enum"] = [v for v in node["enum"] if v not in hidden]
        for v in node.values():
            _hide(v, hidden)
    elif isinstance(node, list):
        for v in node:
            _hide(v, hidden)
    return node


def openai_tools(extra: list[dict] | None = None, hidden_metrics: set | None = None) -> list[dict]:
    specs = []
    for name, (_, model, desc) in TOOLS.items():
        params = _hide(_inline_refs(model.model_json_schema()), set(hidden_metrics or ()))
        params.setdefault("properties", {})
        params["type"] = "object"
        specs.append({"type": "function", "function": {"name": name, "description": desc, "parameters": params}})
    return specs + (extra or [])


def execute(inv: Investigation, name: str, raw_args) -> dict:
    """Validate args, run the tool, record evidence. Errors are returned (not raised) so the LLM can recover."""
    if name not in TOOLS:
        return {"error": f"Unknown tool '{name}'. Available: {list(TOOLS)}"}
    fn, model, _ = TOOLS[name]
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
        parsed = model(**args).model_dump(exclude_none=True)
        fd, fs = parsed.pop("filter_dimension", None), parsed.pop("filter_segment", None)
        if fd and fs is not None:
            parsed["filters"] = {fd: fs}
    except (ValidationError, json.JSONDecodeError, TypeError) as e:
        return {"error": f"Invalid arguments for {name}: {e}"}
    if parsed.get("metric") in inv.hidden_metrics:
        return {"error": f"Unknown metric '{parsed['metric']}'."}
    try:
        result = fn(inv, **parsed)
    except Exception as e:  # tool errors go back to the model as data
        return {"error": f"{name} failed: {type(e).__name__}: {e}"}
    return inv.record(name, parsed, result)

"""Investigation context: the store being analysed plus the evidence log.

Every tool call is recorded as evidence with an id (e1, e2, ...). The agent must cite these ids,
and the verifier checks report numbers against exactly these outputs.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from metrics.metrics import Store, default_store


def clean(obj):
    """Make tool outputs JSON-safe and readable (round floats, dates to ISO strings)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        if f == 0:
            return 0.0
        if abs(f) >= 100:
            return round(f, 2)
        if abs(f) >= 1:
            return round(f, 4)
        # small numbers (p-values, rates): keep 4 significant digits
        return float(f"{f:.4g}")
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


@dataclass
class Investigation:
    store: Store = field(default_factory=default_store)
    chart_dir: Path = Path("charts")
    evidence: dict = field(default_factory=dict)
    tests: list = field(default_factory=list)  # [{"key":..., "p_value":...}] for BH correction
    _n: int = 0

    def record(self, tool: str, args: dict, result: dict) -> dict:
        self._n += 1
        eid = f"e{self._n}"
        out = clean(result)
        out = {"evidence_id": eid, **out}
        self.evidence[eid] = {"tool": tool, "args": clean(args), "result": out}
        return out

    def evidence_json(self, eid: str) -> str:
        return json.dumps(self.evidence.get(eid, {}).get("result", {}))

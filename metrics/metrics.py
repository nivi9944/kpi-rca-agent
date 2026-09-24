"""Metric layer: loads the fact tables and computes any catalog metric, by week and by segment.

All analysis tools go through this module, so every metric is defined exactly once
(in catalog.yaml) and computed by the same code everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "metrics" / "catalog.yaml"


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def metric_def(metric: str) -> dict:
    cat = load_catalog()["metrics"]
    if metric not in cat:
        raise ValueError(f"Unknown metric '{metric}'. Known: {sorted(cat)}")
    return cat[metric]


def dimension_col(dimension: str, table: str) -> str:
    dims = load_catalog()["dimensions"]
    if dimension not in dims:
        raise ValueError(f"Unknown dimension '{dimension}'. Known: {sorted(dims)}")
    return dims[dimension][table]


def to_week(w) -> pd.Timestamp:
    """Normalise any date-like to the Monday that starts its week."""
    ts = pd.Timestamp(w).normalize()
    return ts - pd.Timedelta(days=ts.weekday())


@dataclass
class Store:
    """In-memory copy of the two fact tables. Scenarios inject problems into a copy of this."""

    orders: pd.DataFrame
    items: pd.DataFrame
    name: str = "olist"
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_duckdb(cls, path: str | Path) -> "Store":
        con = duckdb.connect(str(path), read_only=True)
        orders = con.execute("SELECT * FROM fact_orders").df()
        items = con.execute("SELECT * FROM fact_items").df()
        con.close()
        return cls.from_frames(orders, items)

    @classmethod
    def from_frames(cls, orders: pd.DataFrame, items: pd.DataFrame, name: str = "olist") -> "Store":
        orders = orders.copy()
        items = items.copy()
        orders["week"] = pd.to_datetime(orders["week"])
        items["week"] = pd.to_datetime(items["week"])
        orders["one"] = 1
        items["one"] = 1
        return cls(orders=orders, items=items, name=name)

    def copy(self, name: str | None = None) -> "Store":
        return Store(orders=self.orders.copy(), items=self.items.copy(), name=name or self.name)

    def table(self, table: str) -> pd.DataFrame:
        return self.orders if table == "orders" else self.items

    # ---------- core computations ----------
    def rows(self, metric: str, filters: dict | None = None) -> pd.DataFrame:
        """Rows that feed a metric: filtered table with a 'value' column."""
        m = metric_def(metric)
        df = self.table(m["table"])
        if m.get("filter"):
            df = df.query(m["filter"])
        for dim, val in (filters or {}).items():
            col = dimension_col(dim, m["table"])
            df = df[df[col].astype(str) == str(val)]
        df = df.assign(value=df[m["value_col"]].astype(float))
        return df

    def weeks(self) -> list[pd.Timestamp]:
        return sorted(self.orders["week"].unique())

    def weekly(self, metric: str, filters: dict | None = None) -> pd.DataFrame:
        """Weekly series: week, value, n (rows), total (sum of values)."""
        m = metric_def(metric)
        df = self.rows(metric, filters)
        g = df.groupby("week")["value"].agg(["sum", "count"]).rename(columns={"sum": "total", "count": "n"})
        all_weeks = pd.Index(self.weeks(), name="week")
        g = g.reindex(all_weeks).fillna({"total": 0.0, "n": 0})
        if m["kind"] == "sum":
            g["value"] = g["total"]
        else:
            g["value"] = g["total"] / g["n"].replace(0, np.nan)
        return g.reset_index()

    @staticmethod
    def baseline_weeks(week, n_weeks: int = 4) -> list[pd.Timestamp]:
        w = to_week(week)
        return [w - pd.Timedelta(weeks=i) for i in range(n_weeks, 0, -1)]

    def period_value(self, metric: str, weeks: list, filters: dict | None = None) -> dict:
        """Value over a set of weeks. sum -> average weekly sum; mean -> pooled mean."""
        m = metric_def(metric)
        df = self.rows(metric, filters)
        wk = [to_week(w) for w in weeks]
        sub = df[df["week"].isin(wk)]
        n = len(sub)
        tot = float(sub["value"].sum())
        if m["kind"] == "sum":
            return {"value": tot / len(wk), "n": n / len(wk), "total": tot / len(wk)}
        return {"value": tot / n if n else float("nan"), "n": n, "total": tot}

    def segment_table(self, metric: str, dimension: str, week, n_baseline: int = 4,
                      filters: dict | None = None) -> pd.DataFrame:
        """Per-segment stats for the target week (suffix 1) vs the baseline weeks (suffix 0).

        For sum metrics, baseline figures are weekly averages (sum0, n0).
        For mean metrics, baseline figures are pooled (n0 = rows in all baseline weeks).
        """
        m = metric_def(metric)
        col = dimension_col(dimension, m["table"])
        df = self.rows(metric, filters)
        w1 = to_week(week)
        w0 = self.baseline_weeks(w1, n_baseline)
        cur = df[df["week"] == w1]
        base = df[df["week"].isin(w0)]
        a = cur.groupby(cur[col].astype(str))["value"].agg(["sum", "count"]).rename(columns={"sum": "sum1", "count": "n1"})
        b = base.groupby(base[col].astype(str))["value"].agg(["sum", "count"]).rename(columns={"sum": "sum0", "count": "n0"})
        t = a.join(b, how="outer").fillna(0.0)
        if m["kind"] == "sum":
            t["sum0"] = t["sum0"] / n_baseline
            t["n0"] = t["n0"] / n_baseline
        t["mean1"] = t["sum1"] / t["n1"].replace(0, np.nan)
        t["mean0"] = t["sum0"] / t["n0"].replace(0, np.nan)
        t.index.name = "segment"
        return t.reset_index()


_DEFAULT: Store | None = None


def default_store() -> Store:
    """The clean Olist store (loaded once)."""
    global _DEFAULT
    if _DEFAULT is None:
        import os
        path = os.getenv("DUCKDB_PATH", str(ROOT / "data" / "olist.duckdb"))
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        _DEFAULT = Store.from_duckdb(p)
    return _DEFAULT

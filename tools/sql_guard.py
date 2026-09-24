"""Read-only SQL for ad-hoc checks, with guards.

Rules: a single SELECT/WITH statement, no write or admin keywords, at most 1,000 rows returned,
and a timeout. Tables available: fact_orders, fact_items (the investigation's current data).
"""
from __future__ import annotations

import re
import threading

import duckdb

from tools.context import Investigation

MAX_ROWS = 1000
TIMEOUT_S = 10.0
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|install|load|export|import|"
    r"call|set|reset|checkpoint|vacuum|truncate|grant|revoke|begin|commit|rollback|replace|merge|"
    r"read_csv|read_csv_auto|read_parquet|read_json|read_json_auto|read_text|read_blob|glob|getenv)\b",
    re.IGNORECASE)


class SQLGuardError(ValueError):
    pass


def check_sql(query: str) -> str:
    q = re.sub(r"--[^\n]*", " ", query)
    q = re.sub(r"/\*.*?\*/", " ", q, flags=re.S).strip()
    q = q.rstrip(";").strip()
    if not q:
        raise SQLGuardError("Empty query")
    if ";" in q:
        raise SQLGuardError("Only a single statement is allowed")
    if not re.match(r"^(select|with)\b", q, re.IGNORECASE):
        raise SQLGuardError("Only SELECT (or WITH ... SELECT) queries are allowed")
    bad = FORBIDDEN.search(q)
    if bad:
        raise SQLGuardError(f"Keyword not allowed: {bad.group(0)}")
    return q


def run_sql(inv: Investigation, query: str) -> dict:
    q = check_sql(query)
    con = duckdb.connect(":memory:")
    con.register("fact_orders", inv.store.orders.drop(columns=["one"], errors="ignore"))
    con.register("fact_items", inv.store.items.drop(columns=["one"], errors="ignore"))
    timer = threading.Timer(TIMEOUT_S, con.interrupt)
    timer.start()
    try:
        cur = con.execute(q)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(MAX_ROWS + 1)
    except duckdb.InterruptException as e:
        raise SQLGuardError(f"Query timed out after {TIMEOUT_S}s") from e
    finally:
        timer.cancel()
        con.close()
    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]
    return {"columns": cols, "rows": [list(r) for r in rows], "n_rows": len(rows), "truncated": truncated}

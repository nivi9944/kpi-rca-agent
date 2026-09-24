"""Tiny synthetic dataset where the right answers are known by hand.

Every normal week has 100 orders, 1 item each:
  state A: 60 orders, state B: 40 orders
  category x (even i): item value 100;  category y (odd i): item value 50
  on-time: A 90% (first 54 of 60), B 80% (first 32 of 40)
  review score: 4 everywhere; payment: card; nothing canceled
So weekly GMV = A 4500 + B 3000 = 7500; AOV = 75; on-time = 0.86.
`overrides` lets a test change the composition of one week.
"""
from __future__ import annotations

import pandas as pd

from metrics.metrics import Store

WEEK0 = pd.Timestamp("2021-01-04")  # a Monday
N_WEEKS = 20
LAST = N_WEEKS - 1
TARGET = WEEK0 + pd.Timedelta(weeks=LAST)


def _week_orders(week, n_a=60, n_b=40, ontime_a=0.9, ontime_b=0.8, tag=""):
    rows = []
    for state, n, rate in (("A", n_a, ontime_a), ("B", n_b, ontime_b)):
        n_on = int(round(n * rate))
        for i in range(n):
            cat = "x" if i % 2 == 0 else "y"
            val = 100.0 if cat == "x" else 50.0
            on = 1 if i < n_on else 0
            rows.append({
                "order_id": f"{week:%Y%m%d}{tag}_{state}{i}", "week": week, "purchase_ts": week,
                "customer_unique_id": f"c_{week:%Y%m%d}_{state}{i}", "customer_state": state,
                "order_status": "delivered", "main_payment_type": "card", "n_items": 1, "order_value": val,
                "is_canceled": 0, "delivered_ts": week, "estimated_ts": week, "main_category": cat,
                "main_seller_state": state, "main_seller_id": f"s{state}", "review_score": 4.0,
                "is_delivered": 1, "is_on_time": on, "delay_days": -3.0 if on else 2.0, "is_repeat_customer": 0,
            })
    return rows


def make_store(overrides: dict | None = None) -> Store:
    overrides = overrides or {}
    rows = []
    for k in range(N_WEEKS):
        w = WEEK0 + pd.Timedelta(weeks=k)
        rows += _week_orders(w, **overrides.get(k, {}))
    orders = pd.DataFrame(rows)
    items = orders[["order_id", "week", "customer_state", "main_payment_type", "is_repeat_customer", "is_canceled"]].copy()
    items["product_category"] = orders["main_category"]
    items["seller_id"] = orders["main_seller_id"]
    items["seller_state"] = orders["main_seller_state"]
    items["price"] = orders["order_value"] - 10.0
    items["freight_value"] = 10.0
    items["item_value"] = orders["order_value"]
    return Store.from_frames(orders, items, name="synthetic")

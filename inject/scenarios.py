"""Anomaly injection: plant one known problem in one week of a copy of the clean data.

The planted problem is the ground truth used to score the agent and the baseline.

Usage:
  python -m inject.scenarios            writes inject/manifest.json (50 scenarios, fixed seeds)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics.metrics import Store, default_store, to_week
from tools.anomaly import score_series

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "inject" / "manifest.json"
MASTER_SEED = 20260924

# Severity per problem type (small / medium / large). Calibrated so that "small" is near the noise
# floor of the weekly totals and "large" is clearly visible (see DECISIONS.md). Olist delivers about
# 11 days BEFORE its estimate on average, so delays must be large to turn orders late.
SEVERITY = {
    "volume_drop":        {"small": 0.30, "medium": 0.50, "large": 0.80},  # share of segment orders removed
    "price_drop":         {"small": 0.20, "medium": 0.35, "large": 0.50},  # price cut in the category
    "delivery_delay":     {"small": 5,    "medium": 10,   "large": 15},    # days added to delivery
    "cancellation_spike": {"small": 0.10, "medium": 0.20, "large": 0.35},  # share of segment orders canceled
    "review_drop":        {"small": 0.20, "medium": 0.40, "large": 0.60},  # share of reviews set to 1 star
    "mix_shift":          {"small": 0.10, "medium": 0.20, "large": 0.35},  # extra orders (share of week) from a low-AOV category
}

# Weeks with known real-world events or data edges are never used as targets
EXCLUDED_RANGES = [
    ("2017-08-21", "2017-09-11"),  # a real AOV / delay spike late Aug 2017
    ("2017-11-06", "2018-01-08"),  # Black Friday and holiday season
    ("2018-01-15", "2018-03-26"),  # real delivery problems and cancellations in early 2018
    ("2018-05-07", "2018-06-11"),  # truckers' strike (late May 2018) and recovery
    ("2018-07-02", "2018-07-09"),  # World Cup dip
]
POOL_START, POOL_END = "2017-05-01", "2018-08-06"

# (type, metric, ground-truth dimension, targets, n scenarios)
PLAN = [
    ("volume_drop", "gmv", None, [("customer_state", "SP"), ("customer_state", "RJ"), ("customer_state", "MG"),
                                   ("product_category", "bed_bath_table"), ("product_category", "health_beauty"),
                                   ("product_category", "sports_leisure"), ("product_category", "watches_gifts"),
                                   ("customer_state", "RS")], 8),
    ("price_drop", "aov", "product_category", ["watches_gifts", "health_beauty", "computers_accessories",
                                               "cool_stuff", "sports_leisure", "auto", "bed_bath_table"], 7),
    ("delivery_delay", "on_time_rate", "seller_state", ["MG", "PR", "RJ", "SC", "MG", "PR", "SP"], 7),
    ("cancellation_spike", "cancellation_rate", "main_payment_type", ["boleto", "credit_card", "voucher",
                                                                      "boleto", "credit_card", "voucher"], 6),
    ("review_drop", "avg_review_score", "product_category", ["bed_bath_table", "health_beauty", "furniture_decor",
                                                             "housewares", "sports_leisure", "computers_accessories"], 6),
    ("mix_shift", "aov", "product_category", ["telephony", "electronics", "fashion_bags_accessories",
                                              "telephony", "electronics", "fashion_bags_accessories"], 6),
]
N_CONTROLS_CLEAN, N_CONTROLS_NOISE = 5, 5
CONTROL_METRICS = ["gmv", "aov", "on_time_rate", "cancellation_rate", "avg_review_score"]
SEV_CYCLE = ["small", "medium", "large"]


# ---------------------------------------------------------------- injections
def _recompute_order_values(orders: pd.DataFrame, items: pd.DataFrame, ids) -> None:
    sums = items[items["order_id"].isin(ids)].groupby("order_id")["item_value"].sum()
    m = orders["order_id"].isin(sums.index)
    orders.loc[m, "order_value"] = orders.loc[m, "order_id"].map(sums)


def apply_scenario(base: Store, sc: dict) -> Store:
    st = base.copy(name=sc["id"])
    o, it = st.orders, st.items
    w = to_week(sc["week"])
    rng = np.random.default_rng(sc["seed"])
    p = sc["params"]
    in_week = o["week"] == w
    t = sc["type"]

    if t == "volume_drop":
        col = "customer_state" if p["dimension"] == "customer_state" else "main_category"
        cand = o.loc[in_week & (o[col] == p["segment"]), "order_id"].to_numpy()
        drop = rng.choice(cand, size=int(round(len(cand) * p["pct"])), replace=False)
        st.orders = o[~o["order_id"].isin(drop)].reset_index(drop=True)
        st.items = it[~it["order_id"].isin(drop)].reset_index(drop=True)

    elif t == "price_drop":
        m = (it["week"] == w) & (it["product_category"] == p["segment"])
        it.loc[m, "price"] = it.loc[m, "price"] * (1 - p["pct"])
        it.loc[m, "item_value"] = it.loc[m, "price"] + it.loc[m, "freight_value"]
        _recompute_order_values(o, it, it.loc[m, "order_id"].unique())

    elif t == "delivery_delay":
        m = in_week & (o["main_seller_state"] == p["segment"]) & (o["is_delivered"] == 1)
        o.loc[m, "delivered_ts"] = o.loc[m, "delivered_ts"] + pd.Timedelta(days=p["days"])
        o.loc[m, "delay_days"] = o.loc[m, "delay_days"] + p["days"]
        o.loc[m, "is_on_time"] = (o.loc[m, "delay_days"] <= 0).astype(int)

    elif t == "delay_then_reviews":
        # chained (TEST set only): deliveries of one seller_state are delayed in week W, and reviews of that
        # seller_state's orders drop in week W+1 (the target week). Ground truth: that seller_state, rate.
        w0 = w - pd.Timedelta(weeks=1)
        m = (o["week"] == w0) & (o["main_seller_state"] == p["segment"]) & (o["is_delivered"] == 1)
        o.loc[m, "delivered_ts"] = o.loc[m, "delivered_ts"] + pd.Timedelta(days=p["days"])
        o.loc[m, "delay_days"] = o.loc[m, "delay_days"] + p["days"]
        o.loc[m, "is_on_time"] = (o.loc[m, "delay_days"] <= 0).astype(int)
        cand = o.index[in_week & (o["main_seller_state"] == p["segment"]) & o["review_score"].notna()]
        idx = rng.choice(cand, size=int(round(len(cand) * p["pct"])), replace=False)
        o.loc[idx, "review_score"] = 1.0

    elif t == "cancellation_spike":
        cand = o.loc[in_week & (o["main_payment_type"] == p["segment"]) & (o["is_canceled"] == 0), "order_id"].to_numpy()
        ids = rng.choice(cand, size=int(round(len(cand) * p["pct"])), replace=False)
        m = o["order_id"].isin(ids)
        o.loc[m, ["is_canceled", "is_delivered"]] = [1, 0]
        o.loc[m, "order_status"] = "canceled"
        o.loc[m, ["is_on_time", "delay_days", "delivered_ts"]] = np.nan
        it.loc[it["order_id"].isin(ids), "is_canceled"] = 1

    elif t == "review_drop":
        cand = o.index[in_week & (o["main_category"] == p["segment"]) & o["review_score"].notna()]
        idx = rng.choice(cand, size=int(round(len(cand) * p["pct"])), replace=False)
        o.loc[idx, "review_score"] = 1.0

    elif t == "mix_shift":
        n_week = int((in_week & (o["is_canceled"] == 0)).sum())
        src = o[in_week & (o["main_category"] == p["segment"]) & (o["is_canceled"] == 0)]
        k = int(round(n_week * p["pct"]))
        pick = src.sample(n=k, replace=True, random_state=sc["seed"]).copy()
        new_ids = [f"inj_{sc['id']}_{i}" for i in range(k)]
        mapping = pd.DataFrame({"order_id": pick["order_id"].to_numpy(), "new_id": new_ids})
        pick["order_id"] = new_ids
        new_items = mapping.merge(it, on="order_id").drop(columns="order_id").rename(columns={"new_id": "order_id"})
        st.orders = pd.concat([o, pick], ignore_index=True)
        st.items = pd.concat([it, new_items[it.columns]], ignore_index=True)

    elif t == "control_noise":
        # remove a small random share of ALL orders in the week (no segment is special)
        cand = o.loc[in_week, "order_id"].to_numpy()
        drop = rng.choice(cand, size=int(round(len(cand) * p["pct"])), replace=False)
        st.orders = o[~o["order_id"].isin(drop)].reset_index(drop=True)
        st.items = it[~it["order_id"].isin(drop)].reset_index(drop=True)

    elif t == "control_clean":
        pass
    else:
        raise ValueError(f"unknown scenario type {t}")
    st._cache.clear()
    return st


# ---------------------------------------------------------------- scenario plan
def _in_excluded(w: pd.Timestamp) -> bool:
    return any(pd.Timestamp(a) <= w <= pd.Timestamp(b) for a, b in EXCLUDED_RANGES)


def candidate_weeks(store: Store, metric: str, max_abs_z: float | None = 2.0) -> list[pd.Timestamp]:
    s = score_series(store, metric)
    s = s[(s["week"] >= to_week(POOL_START)) & (s["week"] <= to_week(POOL_END))]
    s = s[~s["week"].apply(_in_excluded)]
    if max_abs_z is not None:
        s = s[s["z"].abs() < max_abs_z]
    return list(s["week"])


def build_manifest(store: Store | None = None) -> list[dict]:
    store = store or default_store()
    rng = np.random.default_rng(MASTER_SEED)
    out: list[dict] = []
    i = 0
    for typ, metric, gt_dim, targets, n in PLAN:
        weeks = candidate_weeks(store, metric)
        chosen = rng.choice(len(weeks), size=n, replace=len(weeks) < n)
        for j in range(n):
            sev = SEV_CYCLE[j % 3]
            tgt = targets[j % len(targets)]
            dim, seg = tgt if isinstance(tgt, tuple) else (gt_dim, tgt)
            params = {"dimension": dim, "segment": seg}
            if typ == "delivery_delay":
                params["days"] = SEVERITY[typ][sev]
            else:
                params["pct"] = SEVERITY[typ][sev]
            effect = "mix" if typ == "mix_shift" else ("rate" if typ in ("price_drop", "delivery_delay",
                                                                          "cancellation_spike", "review_drop") else None)
            i += 1
            out.append({"id": f"s{i:02d}_{typ}_{sev}", "type": typ, "severity": sev, "metric": metric,
                        "week": weeks[chosen[j]].strftime("%Y-%m-%d"), "params": params,
                        "seed": int(MASTER_SEED + i), "is_control": False,
                        "ground_truth": {"anomaly": True, "dimension": dim, "segment": seg, "effect": effect}})
    # controls: weeks drawn from the same pool WITHOUT filtering on the detector (conservative)
    for k in range(N_CONTROLS_CLEAN + N_CONTROLS_NOISE):
        kind = "control_clean" if k < N_CONTROLS_CLEAN else "control_noise"
        metric = CONTROL_METRICS[k % len(CONTROL_METRICS)]
        weeks = candidate_weeks(store, metric, max_abs_z=None)
        wk = weeks[int(rng.integers(len(weeks)))]
        i += 1
        out.append({"id": f"s{i:02d}_{kind}", "type": kind, "severity": "none", "metric": metric,
                    "week": wk.strftime("%Y-%m-%d"), "params": {"pct": 0.05} if kind == "control_noise" else {},
                    "seed": int(MASTER_SEED + i), "is_control": True,
                    "ground_truth": {"anomaly": False, "dimension": None, "segment": None, "effect": None}})
    return out


def load_manifest(path=None) -> list[dict]:
    with open(path or MANIFEST, encoding="utf-8") as f:
        return json.load(f)["scenarios"]


def task_text(sc: dict) -> str:
    from metrics.metrics import metric_def
    return (f"Investigate why {metric_def(sc['metric'])['label']} ({sc['metric']}) changed in the week of "
            f"{sc['week']} compared with the previous 4-week average. If there is no significant change "
            f"or no significant root cause, say so.")


if __name__ == "__main__":
    scs = build_manifest()
    MANIFEST.write_text(json.dumps({"master_seed": MASTER_SEED, "n": len(scs), "scenarios": scs}, indent=2))
    from collections import Counter
    print(len(scs), Counter(s["type"] for s in scs))

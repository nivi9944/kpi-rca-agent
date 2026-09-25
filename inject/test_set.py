"""Held-out TEST set (built once, with a new master seed, after the v2 code was frozen on DEV).

  python -m inject.test_set      writes inject/manifest_test.json

Differences from the DEV manifest (inject/manifest.json), all by rule, none tuned on results:
- Target weeks never overlap DEV target weeks and come from the same pool as DEV (from POOL_START, 2017-05-01,
  with the same excluded event weeks), each with at least MIN_HISTORY_WEEKS of prior data. A first version
  also used Olist's launch weeks (2017-03-27 to 2017-04-24); scenarios there were undetectable even for the
  statistical baseline (thin, ramp-up history), so the set was rebuilt (results/v2/archive_launch_weeks).
- Target segments are drawn from the largest segments of each dimension (by volume in the whole window), so
  most TEST segments were never used on DEV.
- A planted scenario must change its target metric in the target week; otherwise its week is redrawn
  (a data-validity rule: an injection that touches no rows would be impossible to solve).
- 20 scenarios per planted type (severities cycle small / medium / large) including a new chained type,
  plus 15 clean and 15 noisy control weeks. Severities are the DEV per-type levels.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from inject.scenarios import (EXCLUDED_RANGES, MANIFEST, POOL_END, POOL_START, SEV_CYCLE, SEVERITY,
                              apply_scenario, load_manifest)
from metrics.metrics import default_store, to_week
from tools.anomaly import score_series

TEST_MANIFEST = MANIFEST.parent / "manifest_test.json"
MASTER_SEED_TEST = 20270101
TEST_POOL_START = POOL_START  # same pool as DEV (the launch weeks before it have too little history)
MIN_HISTORY_WEEKS = 8
N_PER_TYPE = 20
N_CLEAN, N_NOISE = 15, 15
CHAINED_SEVERITY = {"small": (5, 0.20), "medium": (10, 0.40), "large": (15, 0.60)}  # (delay days, share 1-star)
CONTROL_METRICS = ["gmv", "aov", "on_time_rate", "cancellation_rate", "avg_review_score"]


def _excluded(w: pd.Timestamp) -> bool:
    return any(pd.Timestamp(a) <= w <= pd.Timestamp(b) for a, b in EXCLUDED_RANGES)


def free_weeks(store, metric: str, dev_weeks: set, max_abs_z: float | None = 2.0) -> list[pd.Timestamp]:
    s = score_series(store, metric)
    s = s[(s["week"] >= to_week(TEST_POOL_START)) & (s["week"] <= to_week(POOL_END)) & s["z"].notna()]
    s = s[~s["week"].apply(_excluded) & ~s["week"].isin(dev_weeks)]
    first = store.orders["week"].min()
    s = s[(s["week"] - first).dt.days // 7 >= MIN_HISTORY_WEEKS]
    if max_abs_z is not None:
        s = s[s["z"].abs() < max_abs_z]
    return list(s["week"])


def top_segments(store, col: str, n: int, table: str = "orders", exclude=("", "unknown", "nan")) -> list[str]:
    df = store.orders if table == "orders" else store.items
    vc = df[df["is_canceled"] == 0][col].astype(str).value_counts()
    return [v for v in vc.index if v.lower() not in exclude][:n]


def low_aov_categories(store, n: int) -> list[str]:
    o = store.orders[store.orders["is_canceled"] == 0]
    g = o.groupby("main_category")["order_value"].agg(["mean", "size"])
    overall = o["order_value"].mean()
    big = g[g["size"] >= g["size"].sort_values(ascending=False).iloc[min(19, len(g) - 1)]]
    return [str(c) for c in big[big["mean"] < 0.8 * overall].sort_values("size", ascending=False).index[:n]]


def _changes_metric(store, sc: dict) -> bool:
    a = store.weekly(sc["metric"])
    b = apply_scenario(store, sc).weekly(sc["metric"])
    va = a.loc[a["week"] == to_week(sc["week"]), "value"]
    vb = b.loc[b["week"] == to_week(sc["week"]), "value"]
    return len(va) > 0 and len(vb) > 0 and abs(float(vb.iloc[0]) - float(va.iloc[0])) > 1e-12


def build_test_manifest(store=None) -> list[dict]:
    store = store or default_store()
    rng = np.random.default_rng(MASTER_SEED_TEST)
    dev_weeks = {to_week(s["week"]) for s in load_manifest()}
    cats = top_segments(store, "main_category", 12)
    plan = [  # (type, metric, [(dimension, segment), ...])
        ("volume_drop", "gmv", [("customer_state", s) for s in top_segments(store, "customer_state", 8)]
         + [("product_category", c) for c in cats[:8]]),
        ("price_drop", "aov", [("product_category", c) for c in cats]),
        ("delivery_delay", "on_time_rate", [("seller_state", s) for s in top_segments(store, "main_seller_state", 6)]),
        ("cancellation_spike", "cancellation_rate", [("main_payment_type", s) for s in
                                                     top_segments(store, "main_payment_type", 4)]),
        ("review_drop", "avg_review_score", [("product_category", c) for c in cats]),
        ("mix_shift", "aov", [("product_category", c) for c in low_aov_categories(store, 6)]),
        ("delay_then_reviews", "avg_review_score", [("seller_state", s) for s in
                                                    top_segments(store, "main_seller_state", 6)]),
    ]
    out, i, used = [], 0, set()
    for typ, metric, targets in plan:
        weeks = free_weeks(store, metric, dev_weeks)
        if typ == "delay_then_reviews":  # the delay week W (= target - 1) must not be an excluded week either
            weeks = [w for w in weeks if not _excluded(w - pd.Timedelta(weeks=1))]
        tg = [targets[k] for k in rng.permutation(len(targets))]
        for j in range(N_PER_TYPE):
            sev = SEV_CYCLE[j % 3]
            dim, seg = tg[j % len(tg)]
            params = {"dimension": dim, "segment": seg}
            if typ == "delivery_delay":
                params["days"] = SEVERITY[typ][sev]
            elif typ == "delay_then_reviews":
                params["days"], params["pct"] = CHAINED_SEVERITY[sev]
            else:
                params["pct"] = SEVERITY[typ][sev]
            effect = {"mix_shift": "mix", "volume_drop": None}.get(typ, "rate")
            i += 1
            for _ in range(100):  # distinct (type, segment, week), and the injection must change the metric
                wk = weeks[int(rng.integers(len(weeks)))]
                sc = {"id": f"t{i:03d}_{typ}_{sev}", "type": typ, "severity": sev, "metric": metric,
                      "week": wk.strftime("%Y-%m-%d"), "params": params, "seed": int(MASTER_SEED_TEST + i),
                      "is_control": False,
                      "ground_truth": {"anomaly": True, "dimension": dim, "segment": seg, "effect": effect}}
                if (typ, seg, wk) not in used and _changes_metric(store, sc):
                    break
            used.add((typ, seg, wk))
            out.append(sc)
    used_ctrl = set()
    for k in range(N_CLEAN + N_NOISE):
        kind = "control_clean" if k < N_CLEAN else "control_noise"
        metric = CONTROL_METRICS[k % len(CONTROL_METRICS)]
        weeks = free_weeks(store, metric, dev_weeks, max_abs_z=None)
        for _ in range(100):
            wk = weeks[int(rng.integers(len(weeks)))]
            if (kind, metric, wk) not in used_ctrl:
                break
        used_ctrl.add((kind, metric, wk))
        i += 1
        out.append({"id": f"t{i:03d}_{kind}", "type": kind, "severity": "none", "metric": metric,
                    "week": wk.strftime("%Y-%m-%d"), "params": {"pct": 0.05} if kind == "control_noise" else {},
                    "seed": int(MASTER_SEED_TEST + i), "is_control": True,
                    "ground_truth": {"anomaly": False, "dimension": None, "segment": None, "effect": None}})
    return out


if __name__ == "__main__":
    scs = build_test_manifest()
    TEST_MANIFEST.write_text(json.dumps({"master_seed": MASTER_SEED_TEST, "n": len(scs), "scenarios": scs}, indent=2))
    dev = {s["week"] for s in load_manifest()}
    print(len(scs), dict(Counter(s["type"] for s in scs)))
    print("distinct target weeks:", len({s["week"] for s in scs}), "| overlap with DEV weeks:",
          len({s["week"] for s in scs} & dev))
    print("severities:", dict(Counter(s["severity"] for s in scs)))

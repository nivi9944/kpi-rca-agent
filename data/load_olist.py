"""Build the DuckDB database from the raw Olist CSVs.

Usage:  python -m data.load_olist            (reads data/raw, writes data/olist.duckdb)

Creates two clean fact tables:
  fact_orders : one row per order
  fact_items  : one row per order item (carries the order-level dimensions too)
"""
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DEFAULT_DB = ROOT / "data" / "olist.duckdb"

START = "2017-01-01"
END = "2018-08-31"

REQUIRED = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]


def ensure_raw(raw: Path = RAW) -> None:
    """Unzip any Olist zip found in data/raw (or the project root) if CSVs are missing."""
    if all((raw / f).exists() for f in REQUIRED):
        return
    raw.mkdir(parents=True, exist_ok=True)
    zips = list(raw.glob("*.zip")) + list(ROOT.glob("*.zip")) + list(ROOT.parent.glob("*.zip"))
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            if any(n.endswith("olist_orders_dataset.csv") for n in zf.namelist()):
                for n in zf.namelist():
                    if n.endswith(".csv"):
                        (raw / Path(n).name).write_bytes(zf.read(n))
                break
    missing = [f for f in REQUIRED if not (raw / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Olist CSVs in {raw}: {missing}. Put the Kaggle zip in data/raw.")


BUILD_SQL = f"""
CREATE OR REPLACE TABLE orders_raw AS SELECT * FROM read_csv_auto('{{raw}}/olist_orders_dataset.csv', header=true);
CREATE OR REPLACE TABLE items_raw AS SELECT * FROM read_csv_auto('{{raw}}/olist_order_items_dataset.csv', header=true);
CREATE OR REPLACE TABLE payments_raw AS SELECT * FROM read_csv_auto('{{raw}}/olist_order_payments_dataset.csv', header=true);
CREATE OR REPLACE TABLE reviews_raw AS SELECT * FROM read_csv_auto('{{raw}}/olist_order_reviews_dataset.csv', header=true);
CREATE OR REPLACE TABLE customers_raw AS SELECT * FROM read_csv_auto('{{raw}}/olist_customers_dataset.csv', header=true, types={{'customer_zip_code_prefix':'VARCHAR'}});
CREATE OR REPLACE TABLE products_raw AS SELECT * FROM read_csv_auto('{{raw}}/olist_products_dataset.csv', header=true);
CREATE OR REPLACE TABLE sellers_raw AS SELECT * FROM read_csv_auto('{{raw}}/olist_sellers_dataset.csv', header=true, types={{'seller_zip_code_prefix':'VARCHAR'}});
CREATE OR REPLACE TABLE translation_raw AS SELECT * FROM read_csv_auto('{{raw}}/product_category_name_translation.csv', header=true);

-- item level, with English category ("other" when unknown)
CREATE OR REPLACE TABLE items_enriched AS
SELECT i.order_id, i.order_item_id, i.seller_id, s.seller_state,
       COALESCE(t.product_category_name_english, p.product_category_name, 'other') AS product_category,
       i.price, i.freight_value, i.price + i.freight_value AS item_value
FROM items_raw i
LEFT JOIN products_raw p USING (product_id)
LEFT JOIN translation_raw t ON t.product_category_name = p.product_category_name
LEFT JOIN sellers_raw s USING (seller_id);

-- payments: main payment type = the type with the largest total value in the order
CREATE OR REPLACE TABLE pay_main AS
SELECT order_id, payment_type AS main_payment_type FROM (
  SELECT order_id, payment_type, SUM(payment_value) v,
         ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY SUM(payment_value) DESC, payment_type) rn
  FROM payments_raw GROUP BY order_id, payment_type) WHERE rn = 1;

-- reviews: mean score if an order has several
CREATE OR REPLACE TABLE rev AS
SELECT order_id, AVG(review_score)::DOUBLE AS review_score FROM reviews_raw GROUP BY order_id;

-- order totals and "main" item attributes (the item group with the largest value)
CREATE OR REPLACE TABLE order_items_agg AS
SELECT order_id, COUNT(*) AS n_items, SUM(item_value) AS order_value FROM items_enriched GROUP BY order_id;

CREATE OR REPLACE TABLE order_main AS
SELECT order_id, product_category AS main_category, seller_state AS main_seller_state, seller_id AS main_seller_id FROM (
  SELECT order_id, product_category, seller_state, seller_id,
         ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY item_value DESC, order_item_id) rn
  FROM items_enriched) WHERE rn = 1;

CREATE OR REPLACE TABLE fact_orders AS
WITH base AS (
  SELECT o.order_id, o.order_purchase_timestamp AS purchase_ts,
         CAST(date_trunc('week', o.order_purchase_timestamp) AS DATE) AS week,
         c.customer_unique_id, c.customer_state, o.order_status,
         COALESCE(pm.main_payment_type, 'unknown') AS main_payment_type,
         oi.n_items, oi.order_value,
         (o.order_status IN ('canceled','unavailable'))::INTEGER AS is_canceled,
         o.order_delivered_customer_date AS delivered_ts,
         o.order_estimated_delivery_date AS estimated_ts,
         om.main_category, om.main_seller_state, om.main_seller_id,
         r.review_score
  FROM orders_raw o
  JOIN customers_raw c USING (customer_id)
  JOIN order_items_agg oi USING (order_id)        -- orders without items carry no GMV; dropped
  LEFT JOIN order_main om USING (order_id)
  LEFT JOIN pay_main pm USING (order_id)
  LEFT JOIN rev r USING (order_id)
)
SELECT *,
  (order_status = 'delivered' AND delivered_ts IS NOT NULL)::INTEGER AS is_delivered,
  CASE WHEN delivered_ts IS NULL THEN NULL
       ELSE (CAST(delivered_ts AS DATE) <= CAST(estimated_ts AS DATE))::INTEGER END AS is_on_time,
  CASE WHEN delivered_ts IS NULL THEN NULL
       ELSE date_diff('day', CAST(estimated_ts AS DATE), CAST(delivered_ts AS DATE)) END AS delay_days,
  (ROW_NUMBER() OVER (PARTITION BY customer_unique_id ORDER BY purchase_ts, order_id) > 1)::INTEGER AS is_repeat_customer
FROM base;

-- keep the analysis window only (repeat flag computed on full history first)
DELETE FROM fact_orders WHERE purchase_ts < TIMESTAMP '{START}' OR purchase_ts >= TIMESTAMP '{END}' + INTERVAL 1 DAY;

CREATE OR REPLACE TABLE fact_items AS
SELECT i.order_id, f.week, i.product_category, i.seller_id, i.seller_state,
       i.price, i.freight_value, i.item_value, f.customer_state, f.main_payment_type,
       f.is_repeat_customer, f.is_canceled
FROM items_enriched i JOIN fact_orders f USING (order_id);

DROP TABLE orders_raw; DROP TABLE items_raw; DROP TABLE payments_raw; DROP TABLE reviews_raw;
DROP TABLE customers_raw; DROP TABLE products_raw; DROP TABLE sellers_raw; DROP TABLE translation_raw;
DROP TABLE items_enriched; DROP TABLE pay_main; DROP TABLE rev; DROP TABLE order_items_agg; DROP TABLE order_main;
"""


def build(db_path: Path = DEFAULT_DB, raw: Path = RAW) -> dict:
    ensure_raw(raw)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    con.execute(BUILD_SQL.replace("{raw}", raw.as_posix()))
    stats = {
        "orders": con.execute("SELECT COUNT(*) FROM fact_orders").fetchone()[0],
        "items": con.execute("SELECT COUNT(*) FROM fact_items").fetchone()[0],
        "weeks": con.execute("SELECT COUNT(DISTINCT week) FROM fact_orders").fetchone()[0],
        "gmv_brl": round(con.execute("SELECT SUM(item_value) FROM fact_items WHERE is_canceled=0").fetchone()[0], 2),
    }
    con.close()
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.getenv("DUCKDB_PATH", str(DEFAULT_DB)))
    args = ap.parse_args()
    print(build(Path(args.db)))

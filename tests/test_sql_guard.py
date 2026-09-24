import pytest

from tests.fixtures.synth import make_store
from tools.context import Investigation
from tools.sql_guard import MAX_ROWS, SQLGuardError, check_sql, run_sql


@pytest.mark.parametrize("q", [
    "DROP TABLE fact_orders",
    "select 1; drop table fact_orders",
    "insert into fact_orders values (1)",
    "select * from read_csv_auto('secrets.csv')",
    "COPY fact_orders TO 'x.csv'",
    "attach 'other.db'",
    "",
    "pragma database_list",
])
def test_rejects_unsafe(q):
    with pytest.raises(SQLGuardError):
        check_sql(q)


def test_allows_select_and_with_and_strips_comments():
    assert check_sql("-- hi\nSELECT 1;") == "SELECT 1"
    assert check_sql("with t as (select 1 a) select a from t").lower().startswith("with")


def test_runs_on_investigation_data_with_row_limit():
    inv = Investigation(store=make_store())
    r = run_sql(inv, "select customer_state, count(*) n from fact_orders group by 1 order by 1")
    assert r["rows"] == [["A", 1200], ["B", 800]]
    big = run_sql(inv, "select * from fact_orders")
    assert big["n_rows"] == MAX_ROWS and big["truncated"] is True

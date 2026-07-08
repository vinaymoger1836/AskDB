"""Tests for schema introspection and read-only query execution."""

from __future__ import annotations

import pytest

from app.db import QueryError, get_schema, run_query
from data.seed import ensure_database


@pytest.fixture(scope="session", autouse=True)
def _demo_db() -> None:
    ensure_database()


def test_schema_lists_expected_tables() -> None:
    schema = get_schema()
    for table in ("customers", "products", "orders", "order_items"):
        assert table in schema
    # Structure only — no row data should leak into the schema text.
    assert "INSERT" not in schema.upper()


def test_run_query_returns_columns_and_rows() -> None:
    columns, rows = run_query("SELECT name, unit_price FROM products LIMIT 3")
    assert columns == ["name", "unit_price"]
    assert len(rows) == 3


def test_run_query_raises_on_bad_sql() -> None:
    with pytest.raises(QueryError):
        run_query("SELECT missing_column FROM products")


def test_read_only_connection_rejects_writes() -> None:
    # Backstop: even if a write slipped past the guardrail, the connection is ro.
    with pytest.raises(QueryError):
        run_query("DELETE FROM products")

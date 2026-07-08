"""Tests for the SQL guardrail — the module that makes AskDB safe.

These assert that generated SQL is treated as untrusted input: only single
read-only SELECT queries survive, and a LIMIT is always enforced.
"""

from __future__ import annotations

import pytest

from app.guardrails import GuardrailError, validate_and_prepare


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE products",
        "DELETE FROM products",
        "UPDATE products SET unit_price = 0",
        "INSERT INTO products (id) VALUES (1)",
        "CREATE TABLE x (a INT)",
        "ALTER TABLE products ADD COLUMN x INT",
        "PRAGMA table_info(products)",
        "ATTACH DATABASE 'evil.db' AS e",
    ],
)
def test_rejects_non_select_statements(sql: str) -> None:
    with pytest.raises(GuardrailError):
        validate_and_prepare(sql, max_limit=100)


def test_rejects_stacked_statements() -> None:
    with pytest.raises(GuardrailError) as exc:
        validate_and_prepare("SELECT 1; DROP TABLE products", max_limit=100)
    assert "one statement" in str(exc.value).lower()


def test_rejects_write_hidden_after_valid_select() -> None:
    # A SELECT that is immediately followed by a destructive statement must not
    # be waved through on the strength of its first token.
    with pytest.raises(GuardrailError):
        validate_and_prepare(
            "SELECT * FROM products; DELETE FROM products", max_limit=100
        )


@pytest.mark.parametrize("sql", ["", "   ", "not valid sql !!"])
def test_rejects_empty_or_unparseable(sql: str) -> None:
    with pytest.raises(GuardrailError):
        validate_and_prepare(sql, max_limit=100)


def test_injects_limit_when_absent() -> None:
    out = validate_and_prepare("SELECT * FROM products", max_limit=100)
    assert "LIMIT 100" in out.upper()


def test_clamps_limit_when_too_large() -> None:
    out = validate_and_prepare("SELECT * FROM products LIMIT 5000", max_limit=100)
    assert "LIMIT 100" in out.upper()
    assert "5000" not in out


def test_keeps_smaller_limit() -> None:
    out = validate_and_prepare("SELECT * FROM products LIMIT 5", max_limit=100)
    assert "LIMIT 5" in out.upper()
    assert "100" not in out


def test_allows_plain_select() -> None:
    out = validate_and_prepare(
        "SELECT name, unit_price FROM products WHERE category = 'Electronics'",
        max_limit=100,
    )
    assert out.upper().startswith("SELECT")


def test_allows_cte_with_select() -> None:
    out = validate_and_prepare(
        "WITH recent AS (SELECT id FROM orders) SELECT * FROM recent",
        max_limit=100,
    )
    assert out.upper().startswith("WITH")
    assert "LIMIT 100" in out.upper()


def test_allows_join_group_order() -> None:
    sql = (
        "SELECT p.name, SUM(oi.quantity) AS q "
        "FROM order_items oi JOIN products p ON p.id = oi.product_id "
        "GROUP BY p.name ORDER BY q DESC LIMIT 5"
    )
    out = validate_and_prepare(sql, max_limit=100)
    assert "LIMIT 5" in out.upper()

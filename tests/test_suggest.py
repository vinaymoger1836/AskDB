"""Tests for value suggestions on zero-row queries (against the seeded demo DB).

The demo data has known categorical values — customers.country in
{USA, UK, Germany, India, ...}, products.category in {Electronics, ...} — so a
deliberately misspelled filter should surface the real value as a suggestion.
"""

from __future__ import annotations

import pytest

from app.suggest import ValueSuggestion, apply_suggestion, suggest_values
from data.seed import ensure_database


@pytest.fixture(scope="module", autouse=True)
def _demo_db() -> None:
    """Ensure the seeded demo database exists before any suggestion test runs."""
    ensure_database()


def test_misspelled_value_suggests_the_real_one() -> None:
    sql = "SELECT name FROM customers WHERE country = 'Germny' LIMIT 5"
    suggestions = suggest_values(sql)
    assert suggestions
    assert suggestions[0].column == "country"
    assert "Germany" in suggestions[0].candidates


def test_wrong_case_value_is_suggested_in_its_real_case() -> None:
    # The filter would match nothing (SQLite '=' is case-sensitive on text), and
    # the suggestion returns the value in its actual stored casing.
    suggestions = suggest_values(
        "SELECT name FROM customers WHERE country = 'germany' LIMIT 5"
    )
    assert suggestions and "Germany" in suggestions[0].candidates


def test_qualified_column_through_a_join_resolves_the_table() -> None:
    sql = (
        "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id "
        "WHERE c.country = 'Indai' LIMIT 5"
    )
    suggestions = suggest_values(sql)
    assert suggestions and "India" in suggestions[0].candidates


def test_like_filter_with_a_typo_is_suggested() -> None:
    sql = "SELECT name FROM products WHERE category LIKE '%Electronik%' LIMIT 5"
    suggestions = suggest_values(sql)
    assert suggestions and "Electronics" in suggestions[0].candidates


def test_no_string_filter_yields_no_suggestions() -> None:
    # An aggregate with no textual filter has nothing to disambiguate.
    assert suggest_values("SELECT COUNT(*) FROM orders LIMIT 1") == []


def test_unknown_value_with_no_close_match_yields_nothing() -> None:
    sql = "SELECT name FROM customers WHERE country = 'Zzzzzzz' LIMIT 5"
    assert suggest_values(sql) == []


def test_unparseable_sql_is_handled_gracefully() -> None:
    assert suggest_values("this is not sql") == []


def test_apply_suggestion_swaps_the_literal() -> None:
    sql = "SELECT name FROM customers WHERE country = 'Germny' LIMIT 5"
    fixed = apply_suggestion(sql, "country", "Germny", "Germany")
    assert "Germany" in fixed
    assert "Germny" not in fixed


def test_apply_suggestion_preserves_like_wildcards() -> None:
    sql = "SELECT name FROM products WHERE category LIKE '%Electronik%' LIMIT 5"
    fixed = apply_suggestion(sql, "category", "%Electronik%", "Electronics")
    assert "%Electronics%" in fixed


def test_apply_suggestion_returns_sql_unchanged_when_no_filter_matches() -> None:
    sql = "SELECT name FROM customers LIMIT 5"
    assert apply_suggestion(sql, "country", "Germny", "Germany") == sql


def test_value_suggestion_to_dict_roundtrips() -> None:
    suggestion = ValueSuggestion("country", "Germny", ["Germany"])
    assert suggestion.to_dict() == {
        "column": "country",
        "given": "Germny",
        "candidates": ["Germany"],
    }

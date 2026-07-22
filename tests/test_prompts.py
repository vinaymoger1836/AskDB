"""Tests for prompt construction (no LLM calls — pure string assembly)."""

from __future__ import annotations

from app.prompts import build_explain_prompt, build_sql_prompt


def test_sql_prompt_carries_schema_and_question() -> None:
    messages = build_sql_prompt("CREATE TABLE t (a INT);", "how many rows?")
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "CREATE TABLE t" in user
    assert "how many rows?" in user


def test_sql_prompt_warns_against_bare_columns_with_aggregates() -> None:
    # Guards the fix for the "HAVING SUM(x) > 5 but SELECT shows a raw row"
    # class of bug: the system prompt must steer the model away from it.
    system = build_sql_prompt("schema", "q")[0]["content"].lower()
    assert "group by" in system
    assert "aggregate" in system
    assert "having" in system


def test_explain_prompt_carries_the_sql_and_asks_for_plain_english() -> None:
    sql = "SELECT name FROM products LIMIT 10"
    messages = build_explain_prompt(sql)
    assert messages[0]["role"] == "system"
    assert "plain english" in messages[0]["content"].lower()
    assert sql in messages[1]["content"]

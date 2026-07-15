"""Tests for the text-to-SQL agent, using a mocked LLM (no Groq calls).

The database is the real seeded SQLite demo (built on demand), so query
execution and the self-correction loop are exercised end to end.
"""

from __future__ import annotations

import pytest

from app import agent
from data.seed import ensure_database


@pytest.fixture(scope="session", autouse=True)
def _demo_db() -> None:
    """Ensure the seeded demo database exists before any agent test runs."""
    ensure_database()


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Start each test with an empty result cache so tests stay independent."""
    agent.clear_cache()


class FakeLLM:
    """A scripted LLM: returns queued SQL for SQL prompts, a fixed line for summaries."""

    def __init__(self, sql_responses: list[str]) -> None:
        self._sql = list(sql_responses)
        self.calls: list[list[dict[str, str]]] = []

    def __call__(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        system = messages[0]["content"].lower()
        if "sentence" in system:  # the summary prompt
            return "Here is your answer."
        return self._sql.pop(0)


def test_happy_path_returns_rows_and_summary() -> None:
    fake = FakeLLM(["SELECT name, unit_price FROM products"])
    result = agent.answer("list all products", llm=fake, max_retries=2)

    assert result.ok
    assert result.attempts == 1
    assert "name" in result.columns
    assert len(result.rows) > 0
    assert result.summary == "Here is your answer."
    assert "LIMIT" in (result.sql or "").upper()  # guardrail injected a limit


def test_retry_after_rejected_write() -> None:
    # First reply is a forbidden write (rejected by the guardrail), second is valid.
    fake = FakeLLM(["DROP TABLE products", "SELECT name FROM products"])
    result = agent.answer("show product names", llm=fake, max_retries=2)

    assert result.ok
    assert result.attempts == 2
    # The corrective prompt must carry the prior error back to the model.
    second_prompt = fake.calls[1][-1]["content"]
    assert "Error:" in second_prompt


def test_retry_after_sql_execution_error() -> None:
    # First reply references a non-existent column (QueryError), second is valid.
    fake = FakeLLM(
        ["SELECT nonexistent_column FROM products", "SELECT name FROM products"]
    )
    result = agent.answer("show product names", llm=fake, max_retries=2)

    assert result.ok
    assert result.attempts == 2


def test_gives_up_after_exhausting_retries() -> None:
    fake = FakeLLM(["DROP TABLE products"] * 3)
    result = agent.answer("do something bad", llm=fake, max_retries=2)

    assert not result.ok
    assert result.error is not None
    assert result.attempts == 3


def test_strips_markdown_code_fences() -> None:
    fake = FakeLLM(["```sql\nSELECT name FROM products\n```"])
    result = agent.answer("names please", llm=fake, max_retries=0)

    assert result.ok
    assert result.sql is not None


def test_empty_question_is_rejected() -> None:
    result = agent.answer("   ", llm=FakeLLM([]))
    assert not result.ok
    assert result.error is not None


def test_history_is_threaded_into_the_prompt() -> None:
    # A follow-up question should carry prior turns into the SQL prompt so the
    # model can resolve references like "break it down by month".
    fake = FakeLLM(["SELECT name FROM products"])
    history = [
        {"question": "top products by revenue", "sql": "SELECT name FROM products"}
    ]
    result = agent.answer(
        "break it down by month", llm=fake, max_retries=0, history=history
    )

    assert result.ok
    sql_prompt = fake.calls[0][-1]["content"]
    assert "top products by revenue" in sql_prompt
    assert "Conversation so far" in sql_prompt


def test_repeat_question_is_served_from_cache() -> None:
    # The same question a second time must not call the LLM again — the cached
    # result is returned instead (skipping both the SQL and summary calls).
    fake = FakeLLM(["SELECT name FROM products"])
    first = agent.answer("list product names", llm=fake, max_retries=0)
    calls_after_first = len(fake.calls)

    second = agent.answer("list product names", llm=fake, max_retries=0)

    assert first.ok and second.ok
    assert second.sql == first.sql
    assert len(fake.calls) == calls_after_first  # no further LLM calls
    # The cached copy is independent — mutating it must not corrupt the cache.
    second.rows.clear()
    third = agent.answer("list product names", llm=fake, max_retries=0)
    assert third.rows == first.rows


def test_use_cache_false_forces_fresh_generation() -> None:
    fake = FakeLLM(["SELECT name FROM products", "SELECT name FROM products"])
    agent.answer("all product names", llm=fake, max_retries=0)
    agent.answer("all product names", llm=fake, max_retries=0, use_cache=False)

    # Two SQL generations happened (the second bypassed the cache), so the
    # scripted SQL queue was consumed twice.
    assert fake._sql == []


def test_failures_are_not_cached() -> None:
    # A question that never succeeds must not poison the cache — a later valid
    # answer to the same question should still run and succeed.
    bad = FakeLLM(["DROP TABLE products"])
    failed = agent.answer("give me everything", llm=bad, max_retries=0)
    assert not failed.ok

    good = FakeLLM(["SELECT name FROM products"])
    recovered = agent.answer("give me everything", llm=good, max_retries=0)
    assert recovered.ok

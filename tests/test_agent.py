"""Tests for the text-to-SQL agent, using a mocked LLM (no Groq calls).

The database is the real seeded SQLite demo (built on demand), so query
execution and the self-correction loop are exercised end to end.
"""

from __future__ import annotations

import pytest

from app import agent, audit
from data.seed import ensure_database


@pytest.fixture(scope="session", autouse=True)
def _demo_db() -> None:
    """Ensure the seeded demo database exists before any agent test runs."""
    ensure_database()


@pytest.fixture(autouse=True)
def _clear_state() -> None:
    """Start each test with an empty result cache and audit log (independence)."""
    agent.clear_cache()
    audit.clear()


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


def test_cache_hit_sets_the_cached_flag() -> None:
    fake = FakeLLM(["SELECT name FROM products"])
    first = agent.answer("names of products", llm=fake, max_retries=0)
    second = agent.answer("names of products", llm=fake, max_retries=0)

    assert first.cached is False  # freshly generated
    assert second.cached is True  # served from the question cache


def test_run_sql_is_cached_on_repeat() -> None:
    first = agent.run_sql("SELECT name FROM products")
    second = agent.run_sql("SELECT name FROM products")

    assert first.ok and second.ok
    assert first.cached is False
    assert second.cached is True  # the SQL-level cache served the re-run
    assert second.rows == first.rows


def test_run_sql_reuses_a_query_the_agent_already_ran() -> None:
    # The question path and the edited-SQL path share one SQL cache: re-running a
    # query the agent generated earlier is a cache hit even via run_sql.
    fake = FakeLLM(["SELECT name FROM products"])
    answered = agent.answer("give product names", llm=fake, max_retries=0)

    reran = agent.run_sql(answered.sql or "")
    assert reran.cached is True
    assert reran.rows == answered.rows


def test_clear_cache_empties_the_sql_cache() -> None:
    agent.run_sql("SELECT name FROM products")
    agent.clear_cache()
    assert agent.run_sql("SELECT name FROM products").cached is False


def test_use_cache_false_forces_fresh_generation() -> None:
    fake = FakeLLM(["SELECT name FROM products", "SELECT name FROM products"])
    agent.answer("all product names", llm=fake, max_retries=0)
    agent.answer("all product names", llm=fake, max_retries=0, use_cache=False)

    # Two SQL generations happened (the second bypassed the cache), so the
    # scripted SQL queue was consumed twice.
    assert fake._sql == []


def test_ambiguous_question_returns_clarification_options() -> None:
    # The model declines to guess and offers concrete rephrasings instead.
    fake = FakeLLM(['CLARIFY: ["Revenue by product", "Units sold by product"]'])
    result = agent.answer("show me the best products", llm=fake, max_retries=2)

    assert result.needs_clarification
    assert result.clarification == ["Revenue by product", "Units sold by product"]
    assert not result.ok  # no query ran…
    assert result.error is None  # …but it isn't a failure either
    assert result.sql is None
    assert result.rows == []


def test_clarification_is_parsed_even_when_fenced() -> None:
    fake = FakeLLM(['```\nCLARIFY: ["Option A", "Option B"]\n```'])
    result = agent.answer("ambiguous", llm=fake, max_retries=0)
    assert result.clarification == ["Option A", "Option B"]


def test_clarification_can_be_disabled() -> None:
    # With clarify=False a 'CLARIFY:' reply is treated as (invalid) SQL, not a
    # question — so it's rejected by the guardrail rather than offered as options.
    fake = FakeLLM(['CLARIFY: ["a", "b"]'])
    result = agent.answer("ambiguous", llm=fake, max_retries=0, clarify=False)
    assert result.clarification is None
    assert not result.ok
    assert result.error is not None


def test_clarification_is_only_offered_on_the_first_attempt() -> None:
    # A first-attempt clarification short-circuits; the retry loop never runs, so
    # only one LLM call is made and no summary is requested.
    fake = FakeLLM(['CLARIFY: ["only interpretation that matters"]'])
    result = agent.answer("vague", llm=fake, max_retries=2)
    assert result.needs_clarification
    assert result.attempts == 1
    assert len(fake.calls) == 1


def test_run_sql_executes_valid_select() -> None:
    # No LLM is involved: user-supplied SQL runs straight through the guardrail.
    result = agent.run_sql("SELECT name FROM products")
    assert result.ok
    assert "name" in result.columns
    assert len(result.rows) > 0
    assert "LIMIT" in (result.sql or "").upper()  # guardrail injected a limit


def test_run_sql_rejects_a_write() -> None:
    result = agent.run_sql("DROP TABLE products")
    assert not result.ok
    assert result.error is not None
    assert result.sql == "DROP TABLE products"  # echoes what the user submitted


def test_run_sql_reports_execution_errors() -> None:
    result = agent.run_sql("SELECT nonexistent_column FROM products")
    assert not result.ok
    assert result.error is not None


def test_run_sql_rejects_empty_input() -> None:
    result = agent.run_sql("   ")
    assert not result.ok
    assert result.error is not None


def test_guardrail_block_is_audited() -> None:
    # A model-generated write is blocked, retried, and recorded in the audit log.
    fake = FakeLLM(["DROP TABLE products", "SELECT name FROM products"])
    result = agent.answer("show product names", llm=fake, max_retries=2)
    assert result.ok  # recovered on the retry

    events = audit.recent()
    assert events and events[0].source == "llm"
    assert "DROP" in events[0].sql.upper()


def test_edited_sql_block_is_audited() -> None:
    agent.run_sql("DELETE FROM products")
    events = audit.recent()
    assert events and events[0].source == "edited"
    assert "DELETE" in events[0].sql.upper()


def test_execution_error_is_not_audited() -> None:
    # A bad column is a QueryError, not a guardrail block — it must NOT be logged.
    result = agent.run_sql("SELECT nonexistent_column FROM products")
    assert not result.ok
    assert audit.recent() == []


def test_explain_sql_returns_the_models_explanation() -> None:
    calls: list[list[dict[str, str]]] = []

    def fake_llm(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return "It lists every product name, capped at 10 rows."

    text = agent.explain_sql("SELECT name FROM products LIMIT 10", llm=fake_llm)

    assert "product name" in text
    # The query under explanation must reach the prompt.
    assert "SELECT name FROM products" in calls[0][-1]["content"]


def test_explain_sql_handles_empty_input() -> None:
    # No LLM call should be made when there's nothing to explain.
    def boom(_messages: list[dict[str, str]]) -> str:
        raise AssertionError("LLM should not be called for empty SQL")

    assert agent.explain_sql("   ", llm=boom) == "There is no SQL to explain."


def test_explain_sql_wraps_llm_failure() -> None:
    def failing(_messages: list[dict[str, str]]) -> str:
        raise ConnectionError("groq down")

    with pytest.raises(RuntimeError, match="Could not explain"):
        agent.explain_sql("SELECT 1", llm=failing)


def test_failures_are_not_cached() -> None:
    # A question that never succeeds must not poison the cache — a later valid
    # answer to the same question should still run and succeed.
    bad = FakeLLM(["DROP TABLE products"])
    failed = agent.answer("give me everything", llm=bad, max_retries=0)
    assert not failed.ok

    good = FakeLLM(["SELECT name FROM products"])
    recovered = agent.answer("give me everything", llm=good, max_retries=0)
    assert recovered.ok

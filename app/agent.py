"""The text-to-SQL agent: generate -> validate -> execute -> self-correct.

`answer()` orchestrates one question end to end. The LLM is injected as a simple
callable so the loop (including the retry-on-error path) is fully testable without
hitting Groq.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from app.config import settings
from app.db import QueryError, get_schema, run_query
from app.guardrails import GuardrailError, validate_and_prepare
from app.prompts import build_sql_prompt, build_summary_prompt

logger = logging.getLogger(__name__)

# A message list is a list of {"role": ..., "content": ...} dicts.
Messages = list[dict[str, str]]
# The injectable LLM: takes chat messages, returns the model's text reply.
LLM = Callable[[Messages], str]

_FENCE_RE = re.compile(r"^\s*```(?:sql)?\s*|\s*```\s*$", re.IGNORECASE)


@dataclass
class AgentResult:
    """Outcome of answering one question."""

    question: str
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    summary: str | None = None
    attempts: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when a query ran successfully."""
        return self.error is None and self.sql is not None


def _strip_sql(text: str) -> str:
    """Remove surrounding markdown code fences from an LLM reply."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Drop the opening fence line and any trailing fence.
        cleaned = _FENCE_RE.sub("", cleaned)
    return cleaned.strip().rstrip(";").strip()


def _default_llm() -> LLM:
    """Build the production LLM caller backed by Groq (lazy import of langchain)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.llm import build_chat_model

    model = build_chat_model()

    def _call(messages: Messages) -> str:
        converted = [
            SystemMessage(m["content"])
            if m["role"] == "system"
            else HumanMessage(m["content"])
            for m in messages
        ]
        reply = model.invoke(converted)
        return reply.content if isinstance(reply.content, str) else str(reply.content)

    return _call


def answer(
    question: str,
    *,
    llm: LLM | None = None,
    schema: str | None = None,
    max_retries: int | None = None,
    max_limit: int | None = None,
) -> AgentResult:
    """Answer a natural-language question with a validated read-only SQL query.

    Loops up to `max_retries + 1` times, feeding any validation/execution error
    back to the model. Returns an AgentResult; on total failure `error` is set
    (the caller shows a friendly message rather than crashing).
    """
    question = (question or "").strip()
    if not question:
        return AgentResult(question=question, error="Please enter a question.")

    llm = llm or _default_llm()
    schema = schema if schema is not None else get_schema()
    retries = settings.agent_max_retries if max_retries is None else max_retries
    limit = settings.max_limit if max_limit is None else max_limit

    result = AgentResult(question=question)
    prior_sql: str | None = None
    prior_error: str | None = None

    for attempt in range(1, retries + 2):
        result.attempts = attempt
        try:
            messages = build_sql_prompt(schema, question, prior_sql, prior_error)
            raw = llm(messages)
        except Exception as exc:  # external call — surface, don't crash the app
            logger.error("LLM call failed on attempt %d: %s", attempt, exc)
            result.error = f"The language model could not be reached: {exc}"
            return result

        candidate = _strip_sql(raw)
        try:
            safe_sql = validate_and_prepare(candidate, max_limit=limit)
            columns, rows = run_query(safe_sql)
        except (GuardrailError, QueryError) as exc:
            logger.info("Attempt %d rejected/failed: %s", attempt, exc)
            prior_sql = candidate
            prior_error = str(exc)
            result.sql = candidate
            result.error = str(exc)
            continue

        # Success — fill in results and a summary.
        result.sql = safe_sql
        result.columns = columns
        result.rows = rows
        result.error = None
        result.summary = _summarise(llm, question, columns, rows)
        return result

    # Exhausted all attempts; result.error holds the last failure reason.
    logger.warning("Gave up after %d attempts: %s", result.attempts, result.error)
    return result


def _summarise(
    llm: LLM, question: str, columns: list[str], rows: list[tuple]
) -> str | None:
    """Ask the LLM for a one-line summary; never fail the whole request over it."""
    try:
        return llm(build_summary_prompt(question, columns, rows)).strip()
    except Exception as exc:  # a missing summary is non-fatal
        logger.warning("Summary generation failed: %s", exc)
        return None

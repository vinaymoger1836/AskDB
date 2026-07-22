"""FastAPI entry point for AskDB.

Exposes the text-to-SQL agent over HTTP. The Streamlit UI calls `/query`; the
`/schema` and `/health` endpoints are handy for debugging and uptime checks.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import agent
from app.config import ConfigError
from app.db import DatabaseError, get_schema
from app.logging_config import configure_logging
from data.seed import ensure_database

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Build the demo database on first launch if it isn't present yet."""
    try:
        ensure_database()
    except Exception as exc:  # never let seeding crash startup silently
        logger.error("Could not ensure demo database: %s", exc)
    yield


app = FastAPI(title="AskDB API", version="0.1.0", lifespan=lifespan)


class HistoryTurn(BaseModel):
    """A prior conversation turn: the question asked and the SQL that answered it."""

    question: str
    sql: str | None = None


class QueryRequest(BaseModel):
    """A natural-language question to answer, with optional conversation history."""

    question: str = Field(..., min_length=1, max_length=500)
    history: list[HistoryTurn] = Field(default_factory=list)


class ExplainRequest(BaseModel):
    """A validated SQL query to explain in plain English."""

    sql: str = Field(..., min_length=1, max_length=5000)


class ExplainResponse(BaseModel):
    """The plain-English explanation of a SQL query."""

    explanation: str


class QueryResponse(BaseModel):
    """The agent's answer: SQL, tabular result, summary, and diagnostics."""

    question: str
    sql: str | None = None
    columns: list[str] = []
    rows: list[list] = []
    summary: str | None = None
    attempts: int = 0
    error: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/schema")
def schema() -> dict[str, str]:
    """Return the demo database schema as DDL text."""
    try:
        return {"schema": get_schema()}
    except DatabaseError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Answer a question: generate, validate, execute, and summarise SQL."""
    try:
        result = agent.answer(
            request.question,
            history=[turn.model_dump() for turn in request.history],
        )
    except ConfigError as exc:
        # Missing GROQ_API_KEY, etc. — a configuration problem, not the user's fault.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except DatabaseError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return QueryResponse(
        question=result.question,
        sql=result.sql,
        columns=result.columns,
        rows=[list(r) for r in result.rows],
        summary=result.summary,
        attempts=result.attempts,
        error=result.error,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> ExplainResponse:
    """Return a plain-English explanation of a generated SQL query."""
    try:
        explanation = agent.explain_sql(request.sql)
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ExplainResponse(explanation=explanation)

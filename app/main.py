"""FastAPI entry point for AskDB.

Exposes the text-to-SQL agent over HTTP. The Streamlit UI calls `/query`; the
`/schema` and `/health` endpoints are handy for debugging and uptime checks.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app import agent
from app.config import ConfigError, settings
from app.db import DatabaseError, get_schema
from app.ingest import IngestError
from app.logging_config import configure_logging
from app.sources import SourceRegistry
from data.seed import ensure_database

configure_logging()
logger = logging.getLogger(__name__)

# Process-local registry of uploaded sources. Clients reference an upload only by
# the opaque ID returned from /upload — never by a filesystem path.
_sources = SourceRegistry()


def _resolve_source(source_id: str | None) -> str | None:
    """Map an optional source ID to a server-owned db_path (None ⇒ demo DB).

    Raises 404 if an ID is given but not registered — so a client can never coax
    the query engine toward an arbitrary path.
    """
    if not source_id:
        return None
    db_path = _sources.path(source_id)
    if db_path is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown source_id. Upload the file first via /upload.",
        )
    return db_path


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
    """A natural-language question to answer, with optional history and source."""

    question: str = Field(..., min_length=1, max_length=500)
    history: list[HistoryTurn] = Field(default_factory=list)
    source_id: str | None = Field(
        default=None,
        description="ID from /upload to query an uploaded file; omit for the demo DB.",
    )


class UploadResponse(BaseModel):
    """The result of ingesting an uploaded file into a queryable source."""

    source_id: str
    name: str
    tables: list[str]
    truncated: bool


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
def schema(source_id: str | None = None) -> dict[str, str]:
    """Return the schema DDL for the demo DB, or an uploaded source if given."""
    db_path = _resolve_source(source_id)
    try:
        return {"schema": get_schema(db_path)}
    except DatabaseError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    """Ingest a CSV/Excel file into a queryable source; return its opaque ID."""
    data = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_mb} MB upload limit.",
        )
    try:
        source_id, source = _sources.add_upload(file.filename or "", data)
    except IngestError as exc:
        # A bad/unsupported/empty file is the client's error, not a server fault.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return UploadResponse(
        source_id=source_id,
        name=source.name,
        tables=source.tables,
        truncated=source.truncated,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Answer a question: generate, validate, execute, and summarise SQL."""
    db_path = _resolve_source(request.source_id)
    try:
        result = agent.answer(
            request.question,
            history=[turn.model_dump() for turn in request.history],
            db_path=db_path,
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

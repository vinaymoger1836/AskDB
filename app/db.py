"""SQLite access: read-only connections, schema introspection, query execution.

The agent only ever executes SQL through `run_query`, which uses a **read-only**
connection (SQLite URI `mode=ro`) as a hard backstop behind the guardrail layer —
even if a write somehow slipped past validation, the connection itself would
reject it.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class DatabaseError(RuntimeError):
    """Raised when the database file is missing or cannot be opened."""


class QueryError(RuntimeError):
    """Raised when a query fails to execute (bad SQL, unknown column, etc.).

    The agent catches this to feed the error back to the LLM for self-correction.
    """


def _resolve_path(db_path: str | Path | None) -> Path:
    """Return the DB path, raising DatabaseError if it doesn't exist."""
    path = Path(db_path or settings.db_path)
    if not path.exists():
        raise DatabaseError(
            f"Database not found at {path}. Run `python -m data.seed` to build it."
        )
    return path


def get_connection(
    db_path: str | Path | None = None, read_only: bool = True
) -> sqlite3.Connection:
    """Open a SQLite connection. Read-only by default (URI mode=ro).

    Read-write is used only by the seeder; every query path uses read_only=True.
    """
    path = _resolve_path(db_path)
    if read_only:
        uri = f"file:{path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=settings.query_timeout_s)
    else:
        conn = sqlite3.connect(path, timeout=settings.query_timeout_s)
    conn.row_factory = sqlite3.Row
    return conn


def get_schema(db_path: str | Path | None = None) -> str:
    """Return the database schema as compact CREATE TABLE DDL text.

    This text is injected into the LLM prompt so it can ground SQL in real table
    and column names. Only structure is exposed — never row data.
    """
    conn = get_connection(db_path, read_only=True)
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()

        blocks: list[str] = []
        for (table_name,) in ((row["name"],) for row in tables):
            cols = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            col_lines = [
                f"    {col['name']} {col['type']}"
                + (" PRIMARY KEY" if col["pk"] else "")
                for col in cols
            ]
            blocks.append(
                f"CREATE TABLE {table_name} (\n" + ",\n".join(col_lines) + "\n);"
            )
        return "\n\n".join(blocks)
    finally:
        conn.close()


def run_query(
    sql: str, db_path: str | Path | None = None
) -> tuple[list[str], list[tuple]]:
    """Execute a (validated) read-only query. Returns (column_names, rows).

    Raises QueryError on any SQLite failure so the agent can retry with the error.
    """
    conn = get_connection(db_path, read_only=True)
    try:
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = [tuple(r) for r in cursor.fetchall()]
        return columns, rows
    except sqlite3.Error as exc:
        # Surface a clean message; the full SQL is logged at debug level only.
        logger.debug("Query failed: %s", sql)
        raise QueryError(str(exc)) from exc
    finally:
        conn.close()

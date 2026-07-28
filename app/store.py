"""Writable SQLite store for durable app state (audit log, saved queries).

The analytics databases are opened **read-only** — this is the one place the app
*writes* persistent data that must outlive a single request or a process reload:
the guardrail audit log and user-curated saved queries. It lives in its own
SQLite file (`STATE_DB_PATH`, default ``data/askdb_state.db``) so runtime writes
never mix with the read-only demo/query data.

Durability caveat: on ephemeral hosts (e.g. Hugging Face Spaces free tier) the
container filesystem resets on rebuild, so surviving a restart there requires
attaching persistent storage and pointing ``STATE_DB_PATH`` at it. On any host
with a stable disk (including local dev) the state survives process restarts.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

# Runtime override for the DB path. Tests point this at a temp file; None ⇒ use
# the configured default. Set via `use_path` so callers never touch it directly.
_db_path_override: str | None = None

# Idempotent schema: every table the state store owns. Applied on each connect
# (CREATE ... IF NOT EXISTS is cheap and guarantees the tables exist).
_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    source    TEXT NOT NULL,
    reason    TEXT NOT NULL,
    sql       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS saved_queries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id  TEXT NOT NULL,
    name       TEXT NOT NULL,
    question   TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE (source_id, name)
);
"""


def use_path(path: str | None) -> None:
    """Point the store at a specific DB file (tests); None restores the default."""
    global _db_path_override
    _db_path_override = path


def _path() -> Path:
    """Resolve the state DB path (runtime override wins over the configured default)."""
    return Path(_db_path_override or settings.state_db_path)


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Yield a short-lived read-write connection to the initialised state DB.

    A fresh connection per operation keeps writes thread-safe under FastAPI's
    sync threadpool. Commits on clean exit; rolls back if the block raises.
    """
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=settings.query_timeout_s)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

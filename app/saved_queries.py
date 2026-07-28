"""Durable store of user-curated saved queries (named questions to re-run).

Unlike the ephemeral chat thread and recent-questions list, saved queries are
favourites a user deliberately pins. They live in the writable state DB
(`app.store`) so they survive a process restart, and are keyed by source so demo
and uploaded-file favourites stay separate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from app import store

# Key used for the built-in demo DB, where the source id is None.
_DEMO_KEY = "demo"


@dataclass(frozen=True)
class SavedQuery:
    """One pinned question: its stable id, source, display name, and the question."""

    id: int
    source_id: str
    name: str
    question: str
    created_at: float

    def to_dict(self) -> dict:
        """JSON-serialisable view (used by the /saved-queries endpoints)."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "name": self.name,
            "question": self.question,
            "created_at": self.created_at,
        }


def _key(source_id: str | None) -> str:
    """Normalise a source id to a storage key (None ⇒ the demo DB)."""
    return source_id or _DEMO_KEY


def save(name: str, question: str, source_id: str | None = None) -> SavedQuery:
    """Pin `question` under `name` for a source; re-saving a name updates it."""
    name = name.strip()
    question = question.strip()
    if not name or not question:
        raise ValueError("A saved query needs a non-empty name and question.")
    key = _key(source_id)
    now = time.time()
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO saved_queries (source_id, name, question, created_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(source_id, name) DO UPDATE SET "
            "question = excluded.question, created_at = excluded.created_at",
            (key, name, question, now),
        )
        row = conn.execute(
            "SELECT id, source_id, name, question, created_at FROM saved_queries "
            "WHERE source_id = ? AND name = ?",
            (key, name),
        ).fetchone()
    return SavedQuery(**dict(row))


def list_for(source_id: str | None = None) -> list[SavedQuery]:
    """Return a source's saved queries, newest first."""
    key = _key(source_id)
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT id, source_id, name, question, created_at FROM saved_queries "
            "WHERE source_id = ? ORDER BY created_at DESC, id DESC",
            (key,),
        ).fetchall()
    return [SavedQuery(**dict(row)) for row in rows]


def delete(name: str, source_id: str | None = None) -> bool:
    """Remove a saved query by name; return True if a row was deleted."""
    key = _key(source_id)
    with store.connect() as conn:
        cursor = conn.execute(
            "DELETE FROM saved_queries WHERE source_id = ? AND name = ?",
            (key, name),
        )
        return cursor.rowcount > 0


def clear() -> None:
    """Empty all saved queries (used by tests)."""
    with store.connect() as conn:
        conn.execute("DELETE FROM saved_queries")

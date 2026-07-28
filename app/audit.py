"""Persistent audit log of guardrail rejections.

Whenever a generated or user-edited query is blocked by the guardrail
(`GuardrailError` — a write, a stacked statement, `SELECT ... INTO`, etc.), the
reason and the offending SQL are recorded here. This powers the UI's "guardrail
log" view: a concrete, live record that the safety layer is doing its job —
"the LLM's SQL is untrusted input, and here's every time we refused to run it."

Events are stored in the writable state DB (`app.store`) so the log survives a
process restart, and are pruned to a bounded ring of the newest `_MAX_EVENTS`.
The log is diagnostic only and never contains secrets (only SQL text and a
reason string).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from app import store

# Cap the log so it can't grow without bound; older rows are pruned on write.
_MAX_EVENTS = 100


@dataclass(frozen=True)
class AuditEvent:
    """One blocked query: when it happened, from where, why, and the SQL."""

    timestamp: float
    source: str  # "llm" (model-generated) or "edited" (user-edited SQL)
    reason: str
    sql: str

    def to_dict(self) -> dict:
        """JSON-serialisable view (used by the /audit endpoint)."""
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "reason": self.reason,
            "sql": self.sql,
        }


def record(source: str, reason: str, sql: str) -> None:
    """Append a guardrail-block event, pruning the log to the newest _MAX_EVENTS."""
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO audit_events (timestamp, source, reason, sql) "
            "VALUES (?, ?, ?, ?)",
            (time.time(), source, reason, (sql or "").strip()),
        )
        conn.execute(
            "DELETE FROM audit_events WHERE id NOT IN "
            "(SELECT id FROM audit_events ORDER BY id DESC LIMIT ?)",
            (_MAX_EVENTS,),
        )


def recent(limit: int = _MAX_EVENTS) -> list[AuditEvent]:
    """Return the most recent events, newest first (up to `limit`)."""
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT timestamp, source, reason, sql FROM audit_events "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        AuditEvent(
            timestamp=row["timestamp"],
            source=row["source"],
            reason=row["reason"],
            sql=row["sql"],
        )
        for row in rows
    ]


def clear() -> None:
    """Empty the audit log (used by tests and after a fresh start)."""
    with store.connect() as conn:
        conn.execute("DELETE FROM audit_events")

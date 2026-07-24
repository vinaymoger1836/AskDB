"""In-memory audit log of guardrail rejections.

Whenever a generated or user-edited query is blocked by the guardrail
(`GuardrailError` — a write, a stacked statement, `SELECT ... INTO`, etc.), the
reason and the offending SQL are recorded here. This powers the UI's "guardrail
log" view: a concrete, live record that the safety layer is doing its job —
"the LLM's SQL is untrusted input, and here's every time we refused to run it."

The log is a bounded, process-local ring buffer. It is diagnostic only — never
persisted, and it never contains secrets (only SQL text and a reason string).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

# Cap the buffer so a long-running process can't grow it without bound.
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


_events: deque[AuditEvent] = deque(maxlen=_MAX_EVENTS)


def record(source: str, reason: str, sql: str) -> None:
    """Append a guardrail-block event to the log (bounded ring buffer)."""
    _events.append(
        AuditEvent(
            timestamp=time.time(), source=source, reason=reason, sql=(sql or "").strip()
        )
    )


def recent(limit: int = _MAX_EVENTS) -> list[AuditEvent]:
    """Return the most recent events, newest first (up to `limit`)."""
    events = list(_events)[-limit:]
    events.reverse()
    return events


def clear() -> None:
    """Empty the audit log (used by tests and after a fresh start)."""
    _events.clear()

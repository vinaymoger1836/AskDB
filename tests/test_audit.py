"""Tests for the in-memory guardrail audit log."""

from __future__ import annotations

from app import audit


def test_record_and_recent_are_newest_first() -> None:
    audit.clear()
    audit.record("llm", "no writes allowed", "DROP TABLE t")
    audit.record("edited", "one statement only", "SELECT 1; SELECT 2")

    events = audit.recent()
    assert len(events) == 2
    assert events[0].source == "edited"  # newest first
    assert events[1].sql == "DROP TABLE t"


def test_ring_buffer_is_bounded() -> None:
    audit.clear()
    for i in range(audit._MAX_EVENTS + 10):
        audit.record("llm", "blocked", f"SELECT {i}")
    assert len(audit.recent(10_000)) == audit._MAX_EVENTS


def test_limit_caps_returned_events() -> None:
    audit.clear()
    for i in range(5):
        audit.record("llm", "blocked", f"SELECT {i}")
    assert len(audit.recent(2)) == 2


def test_to_dict_is_json_shaped() -> None:
    audit.clear()
    audit.record("llm", "no writes allowed", "  DROP TABLE t  ")
    event = audit.recent()[0].to_dict()
    assert set(event) == {"timestamp", "source", "reason", "sql"}
    assert event["sql"] == "DROP TABLE t"  # whitespace trimmed

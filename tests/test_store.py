"""Tests for the writable state-DB connection helper."""

from __future__ import annotations

import sqlite3

import pytest

from app import store


def test_connect_creates_the_state_tables() -> None:
    with store.connect() as conn:
        names = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert {"audit_events", "saved_queries"} <= names


def test_connect_commits_on_clean_exit() -> None:
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO audit_events (timestamp, source, reason, sql) "
            "VALUES (0, 'llm', 'x', 'SELECT 1')"
        )
    with store.connect() as conn:
        count = conn.execute("SELECT COUNT(*) AS n FROM audit_events").fetchone()["n"]
    assert count == 1


def test_connect_rolls_back_on_error() -> None:
    with pytest.raises(RuntimeError):
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO audit_events (timestamp, source, reason, sql) "
                "VALUES (0, 'llm', 'x', 'SELECT 1')"
            )
            raise RuntimeError("boom")
    with store.connect() as conn:
        count = conn.execute("SELECT COUNT(*) AS n FROM audit_events").fetchone()["n"]
    assert count == 0


def test_use_path_creates_parent_directories(tmp_path) -> None:
    nested = tmp_path / "a" / "b" / "state.db"
    store.use_path(str(nested))
    try:
        with store.connect() as conn:
            assert isinstance(conn, sqlite3.Connection)
        assert nested.exists()
    finally:
        store.use_path(None)

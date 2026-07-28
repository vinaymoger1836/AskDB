"""Shared pytest fixtures.

The writable state DB (`app.store`) is redirected to a per-test temp file so the
audit log and saved-query tests never touch the real ``data/askdb_state.db`` and
never see each other's rows.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app import store


@pytest.fixture(autouse=True)
def isolate_state_db(tmp_path) -> Iterator[None]:
    """Point the state store at a fresh temp DB for the duration of each test."""
    store.use_path(str(tmp_path / "state.db"))
    try:
        yield
    finally:
        store.use_path(None)

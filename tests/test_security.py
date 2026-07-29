"""Tests for optional API-key auth and per-IP rate limiting.

Both protections are off unless configured, so these tests monkeypatch
`security.settings` to switch them on and reset the shared limiter each time.
"""

from __future__ import annotations

import types

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import security
from app.main import app
from data.seed import ensure_database

client = TestClient(app)

_SELECT = {"sql": "SELECT name FROM products"}


@pytest.fixture(scope="module", autouse=True)
def _demo_db() -> None:
    """The demo DB must exist for /run-sql to execute."""
    ensure_database()


@pytest.fixture(autouse=True)
def _reset_limiter() -> None:
    """Each test starts and ends with an empty rate-limit table."""
    security._limiter.reset()
    yield
    security._limiter.reset()


def _configure(monkeypatch, *, api_key=None, rate_limit=0) -> None:
    """Point the security module at a throwaway settings object."""
    monkeypatch.setattr(
        security,
        "settings",
        types.SimpleNamespace(api_key=api_key, rate_limit_per_min=rate_limit),
    )


def test_open_api_allows_requests_without_a_key() -> None:
    # Default config: no key set, so the endpoint is open.
    assert client.post("/run-sql", json=_SELECT).status_code == 200


def test_missing_key_is_rejected_when_one_is_configured(monkeypatch) -> None:
    _configure(monkeypatch, api_key="s3cret")
    assert client.post("/run-sql", json=_SELECT).status_code == 401


def test_wrong_key_is_rejected(monkeypatch) -> None:
    _configure(monkeypatch, api_key="s3cret")
    resp = client.post("/run-sql", json=_SELECT, headers={"X-API-Key": "nope"})
    assert resp.status_code == 401


def test_correct_key_is_accepted(monkeypatch) -> None:
    _configure(monkeypatch, api_key="s3cret")
    resp = client.post("/run-sql", json=_SELECT, headers={"X-API-Key": "s3cret"})
    assert resp.status_code == 200


def test_health_stays_open_even_when_a_key_is_required(monkeypatch) -> None:
    # Liveness must not require auth (probes don't send the key).
    _configure(monkeypatch, api_key="s3cret")
    assert client.get("/health").status_code == 200


def test_rate_limit_returns_429_after_the_cap(monkeypatch) -> None:
    _configure(monkeypatch, rate_limit=2)
    assert client.post("/run-sql", json=_SELECT).status_code == 200
    assert client.post("/run-sql", json=_SELECT).status_code == 200
    third = client.post("/run-sql", json=_SELECT)
    assert third.status_code == 429
    assert "Retry-After" in third.headers


def test_rate_limit_of_zero_is_disabled() -> None:
    # The default (0) never blocks, however many requests arrive.
    security._limiter.reset()
    for _ in range(5):
        assert client.post("/run-sql", json=_SELECT).status_code == 200


def test_limiter_tracks_each_key_independently() -> None:
    limiter = security._FixedWindowLimiter()
    limiter.hit("1.1.1.1", limit=1)  # first for this IP — allowed
    with pytest.raises(HTTPException) as exc:
        limiter.hit("1.1.1.1", limit=1)  # second — blocked
    assert exc.value.status_code == 429
    limiter.hit("2.2.2.2", limit=1)  # a different IP is unaffected


def test_limiter_is_a_noop_when_limit_not_positive() -> None:
    limiter = security._FixedWindowLimiter()
    for _ in range(10):
        limiter.hit("x", limit=0)  # must never raise

"""Tests for central configuration and secret validation."""

from __future__ import annotations

import pytest

from app.config import ConfigError, Settings


def test_require_raises_when_secret_missing() -> None:
    s = Settings(groq_api_key=None)
    with pytest.raises(ConfigError) as exc:
        s.require("groq_api_key")
    assert "GROQ_API_KEY" in str(exc.value)


def test_require_passes_when_present() -> None:
    s = Settings(groq_api_key="sk-test")
    s.require("groq_api_key")  # should not raise


def test_defaults_are_applied() -> None:
    s = Settings()
    assert s.groq_model
    assert s.max_limit > 0
    assert s.agent_max_retries >= 0
    assert s.db_path.endswith(".db")

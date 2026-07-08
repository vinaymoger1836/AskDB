"""Central configuration: loads secrets and settings from the environment.

This is the ONLY place the app reads raw environment variables. Every secret
comes from `.env` (local) or the host's secret store (HF Spaces) — never a
hardcoded string. Import `settings` elsewhere instead of calling os.getenv.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env once at import time. In hosted environments (HF Spaces) there is no
# .env file and real env vars are used instead; load_dotenv is a no-op then.
load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _get(name: str, default: str | None = None) -> str | None:
    """Return an env var, treating empty/whitespace-only values as unset."""
    value = os.getenv(name, default)
    if value is not None:
        value = value.strip()
    return value or default


def _get_int(name: str, default: int) -> int:
    """Return an int env var, falling back to the default on missing/invalid input."""
    raw = _get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int for %s=%r; using default %d", name, raw, default)
        return default


@dataclass(frozen=True)
class Settings:
    """Typed view of all runtime configuration."""

    # --- Secrets (no default; required for the features that use them) ---
    groq_api_key: str | None = field(default_factory=lambda: _get("GROQ_API_KEY"))

    # --- Tunables (safe defaults) ---
    groq_model: str = field(
        default_factory=lambda: _get("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    db_path: str = field(default_factory=lambda: _get("DB_PATH", "data/sales.db"))
    api_base: str = field(
        default_factory=lambda: _get("ASKDB_API_BASE", "http://localhost:8000")
    )

    # --- Guardrail / execution limits ---
    max_limit: int = field(default_factory=lambda: _get_int("MAX_LIMIT", 100))
    agent_max_retries: int = field(
        default_factory=lambda: _get_int("AGENT_MAX_RETRIES", 2)
    )
    query_timeout_s: int = field(default_factory=lambda: _get_int("QUERY_TIMEOUT_S", 5))

    def require(self, *names: str) -> None:
        """Raise ConfigError if any named secret attribute is unset.

        Call this at a feature's entry point so failures are explicit and early,
        with a clear message, instead of an opaque downstream API error.
        """
        missing = [n for n in names if not getattr(self, n, None)]
        if missing:
            env_names = {"groq_api_key": "GROQ_API_KEY"}
            pretty = ", ".join(env_names.get(n, n) for n in missing)
            raise ConfigError(
                f"Missing required configuration: {pretty}. "
                "Set it in your .env file (local) or host secrets."
            )


# Singleton imported across the app.
settings = Settings()

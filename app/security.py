"""HTTP API protection: optional API-key auth and per-IP rate limiting.

Both are **opt-in** so the default (open) API keeps local development and the
Hugging Face single-service deployment working with zero configuration:

  * API-key auth is inert unless ``ASKDB_API_KEY`` is set — then callers must
    send a matching ``X-API-Key`` header, or get 401.
  * Rate limiting is a fixed one-minute window per client IP, inert unless
    ``ASKDB_RATE_LIMIT_PER_MIN`` is a positive number — then the (N+1)th request
    in a window gets 429 with a ``Retry-After`` header.

Wired onto the endpoints that cost compute or accept uploads (/query, /run-sql,
/explain, /upload); liveness and introspection routes stay open.
"""

from __future__ import annotations

import hmac
import logging
import threading
import time

from fastapi import Header, HTTPException, Request

from app.config import settings

logger = logging.getLogger(__name__)

# Prune the per-IP table once it grows past this many distinct clients, dropping
# entries from elapsed windows so the map can't grow without bound.
_MAX_TRACKED = 1024


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reject with 401 unless a configured API key matches the request header.

    A no-op when ``ASKDB_API_KEY`` is unset, so the API is open by default.
    """
    expected = settings.api_key
    if not expected:
        return
    # Constant-time compare so a wrong key can't be probed via response timing.
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


class _FixedWindowLimiter:
    """A thread-safe per-key fixed-window counter (N requests per 60s window)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits: dict[str, tuple[int, int]] = {}  # key -> (window, count)

    def hit(self, key: str, limit: int) -> None:
        """Count one request for `key`; raise 429 if it exceeds `limit`."""
        if limit <= 0:
            return  # limiting disabled
        window = int(time.time() // 60)
        with self._lock:
            stored_window, count = self._hits.get(key, (window, 0))
            count = count + 1 if stored_window == window else 1
            self._hits[key] = (window, count)
            if len(self._hits) > _MAX_TRACKED:
                self._hits = {
                    k: v for k, v in self._hits.items() if v[0] == window
                }
        if count > limit:
            retry_after = 60 - int(time.time() % 60)
            logger.info("Rate limit hit for %s (%d > %d)", key, count, limit)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please slow down and retry shortly.",
                headers={"Retry-After": str(retry_after)},
            )

    def reset(self) -> None:
        """Clear all counters (used between tests)."""
        with self._lock:
            self._hits.clear()


_limiter = _FixedWindowLimiter()


def enforce_rate_limit(request: Request) -> None:
    """Reject with 429 when a client IP exceeds the per-minute request cap."""
    client = request.client.host if request.client else "unknown"
    _limiter.hit(client, settings.rate_limit_per_min)

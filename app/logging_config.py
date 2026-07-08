"""Shared logging setup. Call configure_logging() once at each entry point."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Initialise root logging with a consistent format (idempotent).

    Never logs secret values — call sites must avoid passing secrets into
    log messages.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quiet noisy third-party loggers.
    for noisy in ("httpx", "urllib3", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True

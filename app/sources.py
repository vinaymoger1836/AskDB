"""In-memory registry mapping opaque source IDs to ingested upload databases.

The API never accepts a filesystem path from a client. A client references an
uploaded data source only by an **opaque, server-generated ID**, which this
registry resolves to a `db_path` the server itself created inside a directory it
controls. That closes path injection: there is no way for a request to point the
query engine at an arbitrary file — only at a database the server built from a
validated upload.

The registry is process-local (a single FastAPI worker); it is not shared across
replicas and does not survive a restart, which is the right scope for
session-style uploads.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path
from uuid import uuid4

from app.ingest import DataSource, ingest_upload

logger = logging.getLogger(__name__)


class SourceRegistry:
    """Ingests uploads into a server-owned directory and tracks them by opaque ID."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        """Create the registry, backed by `base_dir` (a fresh temp dir by default)."""
        self._dir = (
            Path(base_dir)
            if base_dir is not None
            else Path(tempfile.mkdtemp(prefix="askdb_api_uploads_"))
        )
        self._dir.mkdir(parents=True, exist_ok=True)
        self._sources: dict[str, DataSource] = {}
        self._lock = threading.Lock()

    def add_upload(self, filename: str, data: bytes) -> tuple[str, DataSource]:
        """Ingest an uploaded file and register it under a new opaque source ID.

        Raises IngestError (from `ingest_upload`) if the file can't be read.
        """
        source = ingest_upload(filename, data, self._dir)
        source_id = uuid4().hex
        with self._lock:
            self._sources[source_id] = source
        logger.info("Registered upload %r as source %s", filename, source_id)
        return source_id, source

    def get(self, source_id: str) -> DataSource | None:
        """Return the DataSource for an ID, or None if it isn't registered."""
        with self._lock:
            return self._sources.get(source_id)

    def path(self, source_id: str) -> str | None:
        """Return the db_path for a registered source ID, or None if unknown."""
        source = self.get(source_id)
        return str(source.db_path) if source else None

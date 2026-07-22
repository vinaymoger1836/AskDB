"""Tests for the uploaded-source registry (opaque IDs, no client paths)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingest import IngestError
from app.sources import SourceRegistry

_CSV = b"name,revenue\nWidget,10\nGadget,20\n"


def test_add_upload_registers_and_resolves(tmp_path: Path) -> None:
    reg = SourceRegistry(tmp_path)
    source_id, source = reg.add_upload("products.csv", _CSV)

    assert source_id
    assert reg.get(source_id) is source
    assert reg.path(source_id) == str(source.db_path)
    assert "products" in source.tables
    # The DB file lives under the registry's own server-owned directory.
    assert Path(source.db_path).parent == tmp_path


def test_unknown_id_resolves_to_none(tmp_path: Path) -> None:
    reg = SourceRegistry(tmp_path)
    assert reg.get("does-not-exist") is None
    assert reg.path("does-not-exist") is None


def test_bad_file_raises_ingest_error(tmp_path: Path) -> None:
    reg = SourceRegistry(tmp_path)
    with pytest.raises(IngestError):
        reg.add_upload("notes.txt", b"not a spreadsheet")

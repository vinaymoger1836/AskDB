"""Ingest uploaded CSV/Excel files into a session-scoped read-only SQLite DB.

An uploaded file is read with pandas and written to its own SQLite database in a
temporary directory. From that point on the agent treats it exactly like the
demo database: same schema introspection, same guardrails, same read-only
execution. Nothing in the query path changes — only the source of the tables.

A CSV becomes one table (named from the file); an Excel workbook becomes one
table per sheet, so questions can join across sheets. Column and table names are
sanitised into safe SQL identifiers, and identifiers are de-duplicated.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd

logger = logging.getLogger(__name__)

# Guard against pathologically large uploads: cap rows written per table.
MAX_ROWS_PER_TABLE = 100_000

_CSV_SUFFIXES = {".csv"}
_EXCEL_SUFFIXES = {".xlsx", ".xls"}
SUPPORTED_SUFFIXES = _CSV_SUFFIXES | _EXCEL_SUFFIXES


class IngestError(RuntimeError):
    """Raised when an uploaded file can't be read or holds no usable data."""


@dataclass
class DataSource:
    """A queryable data source backed by a SQLite file.

    `db_path` can be handed straight to `db.get_schema` / `db.run_query`, so the
    agent needs no knowledge that this came from an upload rather than the seed.
    """

    name: str
    db_path: Path
    tables: list[str] = field(default_factory=list)


def _sanitize_identifier(raw: object, fallback: str) -> str:
    """Turn an arbitrary column/table label into a safe lowercase SQL identifier."""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(raw).strip())
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned.lower()


def _dedupe(names: list[str]) -> list[str]:
    """Make identifiers unique, suffixing collisions with _2, _3, … ."""
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        candidate = name
        i = 2
        while candidate in seen:
            candidate = f"{name}_{i}"
            i += 1
        seen.add(candidate)
        out.append(candidate)
    return out


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Sanitise column names and cap row count before writing to SQLite."""
    columns = [
        _sanitize_identifier(col, f"col_{i}")
        for i, col in enumerate(frame.columns, start=1)
    ]
    prepared = frame.copy()
    prepared.columns = _dedupe(columns)
    if len(prepared) > MAX_ROWS_PER_TABLE:
        logger.info("Truncating upload from %d to %d rows", len(prepared), MAX_ROWS_PER_TABLE)
        prepared = prepared.head(MAX_ROWS_PER_TABLE)
    return prepared


def _read_tables(filename: str, data: bytes) -> dict[str, pd.DataFrame]:
    """Return {table_name: frame} for one upload (one table per Excel sheet)."""
    suffix = Path(filename).suffix.lower()
    stem = _sanitize_identifier(Path(filename).stem, "data")
    try:
        if suffix in _CSV_SUFFIXES:
            return {stem: pd.read_csv(BytesIO(data))}
        if suffix in _EXCEL_SUFFIXES:
            sheets = pd.read_excel(BytesIO(data), sheet_name=None)
            return {
                _sanitize_identifier(sheet, stem): frame
                for sheet, frame in sheets.items()
            }
    except Exception as exc:  # pandas/parser errors → a clean user-facing message
        raise IngestError(f"Could not read '{filename}': {exc}") from exc
    raise IngestError(
        f"Unsupported file type '{suffix or 'unknown'}'. Upload a CSV or Excel file."
    )


def ingest_upload(filename: str, data: bytes, dest_dir: str | Path) -> DataSource:
    """Load one uploaded CSV/Excel file into a new read-only SQLite database.

    Returns a DataSource whose `db_path` the agent can query like any other DB.
    Raises IngestError if the file is unreadable or contains no usable table.
    """
    if not filename:
        raise IngestError("The uploaded file has no name.")
    if not data:
        raise IngestError(f"'{filename}' is empty.")

    frames = _read_tables(filename, data)
    raw_names = list(frames.keys())
    unique_names = _dedupe(raw_names)

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    stem = _sanitize_identifier(Path(filename).stem, "data")
    db_path = dest / f"{stem}_{uuid4().hex[:8]}.db"

    written: list[str] = []
    conn = sqlite3.connect(db_path)
    try:
        for raw, table in zip(raw_names, unique_names, strict=True):
            frame = frames[raw]
            if frame.empty or len(frame.columns) == 0:
                logger.info("Skipping empty sheet/table %r in %s", raw, filename)
                continue
            _prepare_frame(frame).to_sql(table, conn, index=False, if_exists="replace")
            written.append(table)
        conn.commit()
    except Exception as exc:  # writing failed — don't leave a half-built DB around
        conn.close()
        db_path.unlink(missing_ok=True)
        raise IngestError(f"Could not load '{filename}': {exc}") from exc
    finally:
        conn.close()

    if not written:
        db_path.unlink(missing_ok=True)
        raise IngestError(f"'{filename}' contains no rows to load.")

    logger.info("Ingested %s -> %s (tables: %s)", filename, db_path, written)
    return DataSource(name=filename, db_path=db_path, tables=written)

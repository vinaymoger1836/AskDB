"""Tests for uploaded CSV/Excel ingestion into a session-scoped SQLite DB.

Files are ingested into a pytest tmp_path, then queried through the same
`app.db` helpers the agent uses — so these also cover the "uploaded source is
just another database" contract end to end.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from app import agent, ingest
from app.db import run_query
from app.ingest import IngestError, ingest_upload


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()


def test_csv_creates_table_and_is_queryable(tmp_path: Path) -> None:
    frame = pd.DataFrame({"name": ["a", "b"], "value": [10, 20]})
    source = ingest_upload("sales.csv", _csv_bytes(frame), tmp_path)

    assert source.tables == ["sales"]
    assert source.db_path.exists()
    columns, rows = run_query("SELECT SUM(value) AS total FROM sales", source.db_path)
    assert columns == ["total"]
    assert rows == [(30,)]


def test_column_and_table_names_are_sanitised(tmp_path: Path) -> None:
    frame = pd.DataFrame({"Order Date": ["2023-01-01"], "Total $": [5.0]})
    source = ingest_upload("Q1 Report.csv", _csv_bytes(frame), tmp_path)

    columns, _ = run_query(f"SELECT * FROM {source.tables[0]}", source.db_path)
    assert source.tables == ["q1_report"]  # "Q1 Report" → safe identifier
    assert columns == ["order_date", "total"]  # spaces/"$" stripped


def test_duplicate_columns_are_deduped(tmp_path: Path) -> None:
    # Two headers that sanitise to the same identifier must not collide.
    csv = b"Value,value\n1,2\n"
    source = ingest_upload("dups.csv", csv, tmp_path)
    columns, rows = run_query(f"SELECT * FROM {source.tables[0]}", source.db_path)
    assert columns == ["value", "value_2"]
    assert rows == [(1, 2)]


def test_excel_each_sheet_becomes_a_table(tmp_path: Path) -> None:
    sheets = {
        "customers": pd.DataFrame({"id": [1], "name": ["Ada"]}),
        "orders": pd.DataFrame({"id": [1], "customer_id": [1]}),
    }
    source = ingest_upload("shop.xlsx", _excel_bytes(sheets), tmp_path)

    assert set(source.tables) == {"customers", "orders"}
    _, rows = run_query(
        "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id",
        source.db_path,
    )
    assert rows == [("Ada",)]


def test_empty_file_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestError):
        ingest_upload("empty.csv", b"", tmp_path)


def test_unsupported_type_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestError):
        ingest_upload("notes.txt", b"hello", tmp_path)


def test_unreadable_csv_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise ValueError("bad parse")

    monkeypatch.setattr(ingest.pd, "read_csv", _boom)
    with pytest.raises(IngestError):
        ingest_upload("broken.csv", b"a,b\n1,2\n", tmp_path)


def test_agent_queries_uploaded_source(tmp_path: Path) -> None:
    frame = pd.DataFrame({"city": ["Paris", "Rome"], "sales": [100, 250]})
    source = ingest_upload("regions.csv", _csv_bytes(frame), tmp_path)

    class FakeLLM:
        def __call__(self, messages: list[dict[str, str]]) -> str:
            if "sentence" in messages[0]["content"].lower():
                return "Rome leads on sales."
            return "SELECT city, sales FROM regions ORDER BY sales DESC"

    result = agent.answer(
        "which city sold most?", llm=FakeLLM(), db_path=source.db_path, max_retries=0
    )
    assert result.ok
    assert result.rows[0] == ("Rome", 250)

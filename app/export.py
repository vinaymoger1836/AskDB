"""Serialize a query result set to downloadable CSV/Excel bytes.

Kept separate from the UI so the serialisation is unit-testable without Streamlit.
Both helpers take the agent's raw output (column names + row tuples) and reuse
`charts.to_dataframe` so the on-screen table and the download always agree.
"""

from __future__ import annotations

import io

import pandas as pd

from app.charts import to_dataframe


def to_csv_bytes(columns: list[str], rows: list[tuple]) -> bytes:
    """Return the result set as UTF-8 CSV bytes (ready for a download button)."""
    return to_dataframe(columns, rows).to_csv(index=False).encode("utf-8")


def to_excel_bytes(columns: list[str], rows: list[tuple]) -> bytes:
    """Return the result set as a single-sheet .xlsx workbook in bytes."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        to_dataframe(columns, rows).to_excel(writer, index=False, sheet_name="Results")
    return buffer.getvalue()

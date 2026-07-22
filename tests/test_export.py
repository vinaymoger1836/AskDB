"""Tests for serialising result sets to CSV/Excel download bytes."""

from __future__ import annotations

import io

import pandas as pd

from app.export import to_csv_bytes, to_excel_bytes

_COLUMNS = ["product", "revenue"]
_ROWS = [("Widget", 1200.5), ("Gadget", 980.0)]


def test_csv_roundtrips_to_same_rows() -> None:
    data = to_csv_bytes(_COLUMNS, _ROWS)
    df = pd.read_csv(io.BytesIO(data))
    assert list(df.columns) == _COLUMNS
    assert df.shape == (2, 2)
    assert df.iloc[0]["product"] == "Widget"


def test_excel_roundtrips_to_same_rows() -> None:
    data = to_excel_bytes(_COLUMNS, _ROWS)
    df = pd.read_excel(io.BytesIO(data))
    assert list(df.columns) == _COLUMNS
    assert df.shape == (2, 2)
    assert df.iloc[1]["revenue"] == 980.0


def test_empty_result_still_produces_a_header() -> None:
    data = to_csv_bytes(_COLUMNS, [])
    df = pd.read_csv(io.BytesIO(data))
    assert list(df.columns) == _COLUMNS
    assert df.empty

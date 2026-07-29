"""Tests for the grounded insight strip (pure computation — no DB, no LLM)."""

from __future__ import annotations

from app.insights import compute_insights


def _joined(columns: list[str], rows: list[tuple]) -> str:
    """Compute insights and join them for easy substring assertions."""
    return " | ".join(compute_insights(columns, rows))


def test_reports_sum_top_and_spread_for_a_grouped_result() -> None:
    columns = ["product", "revenue"]
    rows = [("A", 100.0), ("B", 250.0), ("C", 150.0)]
    text = _joined(columns, rows)

    assert "Sum of revenue: 500" in text
    assert "Top product: B — 250" in text
    assert "50% of the total" in text  # 250 / 500
    assert "Spread: 100 low to 250 high" in text


def test_single_row_yields_no_insights() -> None:
    # A one-row result is rendered as a KPI metric, not a strip.
    assert compute_insights(["total"], [(42,)]) == []


def test_no_numeric_measure_yields_no_insights() -> None:
    rows = [("Widget",), ("Gadget",)]
    assert compute_insights(["name"], rows) == []


def test_measure_is_the_last_numeric_column() -> None:
    # [year, revenue] → revenue is the measure; year is the label.
    columns = ["year", "revenue"]
    rows = [(2021, 10), (2022, 30), (2023, 20)]
    text = _joined(columns, rows)
    assert "Sum of revenue: 60" in text
    assert "Top year: 2022 — 30" in text


def test_negative_values_suppress_sum_and_share() -> None:
    # Mixed signs make a sum/share meaningless — only top and spread are stated.
    columns = ["account", "balance"]
    rows = [("X", -50), ("Y", 30), ("Z", 10)]
    text = _joined(columns, rows)
    assert "Sum of" not in text
    assert "of the total" not in text
    assert "Top account: Y — 30" in text
    assert "Spread: -50 low to 30 high" in text


def test_date_ordered_result_reports_trend_direction() -> None:
    columns = ["month", "revenue"]
    rows = [("2023-03", 300), ("2023-01", 100), ("2023-02", 200)]
    # Rows are deliberately out of order — the trend sorts by date first.
    text = _joined(columns, rows)
    assert "moved up from 100 to 300" in text


def test_descending_dates_report_a_downward_trend() -> None:
    columns = ["month", "revenue"]
    rows = [("2023-01", 500), ("2023-02", 400), ("2023-03", 100)]
    assert "moved down from 500 to 100" in _joined(columns, rows)


def test_non_temporal_label_has_no_trend_line() -> None:
    columns = ["product", "revenue"]
    rows = [("A", 100), ("B", 200), ("C", 300)]
    assert "Trend" not in _joined(columns, rows)


def test_integers_format_without_trailing_decimals() -> None:
    columns = ["region", "orders"]
    rows = [("N", 1200), ("S", 800)]
    text = _joined(columns, rows)
    assert "2,000" in text  # grouped, no ".0"
    assert "2000.0" not in text

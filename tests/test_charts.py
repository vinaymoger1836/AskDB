"""Tests for the auto-chart heuristic."""

from __future__ import annotations

from app.charts import (
    GROUPED_BAR,
    LINE,
    PIE,
    available_charts,
    build_chart,
    choose_chart,
    single_value,
)


def test_bar_for_category_and_number() -> None:
    fig = choose_chart(["name", "revenue"], [("A", 10.0), ("B", 25.0), ("C", 5.0)])
    assert fig is not None
    assert fig.data[0].type == "bar"


def test_line_for_time_series() -> None:
    fig = choose_chart("month revenue".split(), [("2023-01", 100), ("2023-02", 150)])
    assert fig is not None
    assert fig.data[0].type == "scatter"  # px.line renders as a scatter trace


def test_none_when_not_two_columns() -> None:
    assert choose_chart(["a", "b", "c"], [(1, 2, 3)]) is None


def test_none_when_empty() -> None:
    assert choose_chart(["a", "b"], []) is None


def test_none_when_value_not_numeric() -> None:
    assert choose_chart(["a", "b"], [("x", "y"), ("p", "q")]) is None


def test_single_value_detected_for_scalar_result() -> None:
    assert single_value(["total_revenue"], [(1204500,)]) == ("total_revenue", 1204500)


def test_single_value_none_for_multiple_rows_or_non_numeric() -> None:
    assert single_value(["n"], [(1,), (2,)]) is None
    assert single_value(["label"], [("hello",)]) is None


def test_available_charts_offers_pie_for_small_positive_sets() -> None:
    kinds = available_charts(["category", "revenue"], [("A", 10), ("B", 20)])
    assert PIE in kinds
    # A forced pie renders as a pie trace.
    fig = build_chart(["category", "revenue"], [("A", 10), ("B", 20)], PIE)
    assert fig is not None and fig.data[0].type == "pie"


def test_forced_line_overrides_the_auto_bar_pick() -> None:
    fig = build_chart(["name", "revenue"], [("A", 10), ("B", 25)], LINE)
    assert fig is not None
    assert fig.data[0].type == "scatter"  # px.line renders as a scatter trace


def test_grouped_bar_for_two_dimensions_and_one_measure() -> None:
    rows = [("2023", "Books", 100), ("2023", "Toys", 60), ("2024", "Books", 120)]
    assert GROUPED_BAR in available_charts(["year", "category", "revenue"], rows)
    fig = build_chart(["year", "category", "revenue"], rows, GROUPED_BAR)
    assert fig is not None
    assert fig.data[0].type == "bar"
    # Grouped bar splits into one trace per category value.
    assert len(fig.data) == 2

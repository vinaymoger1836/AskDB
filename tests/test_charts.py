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

"""Compute grounded, factual insights from a query result set (no LLM).

These are descriptive statistics read straight off the returned rows — the sum
of the measure, the top contributor and its share, the high/low spread, and
(for a date-ordered result) the direction of change. Because every number is
computed from the data the query already returned, the insight strip can never
contradict the table or hallucinate a figure.

The measure is taken to be the last purely-numeric column (usually the aggregate
in a "GROUP BY … SELECT label, value" result); the label is the first other
column. Everything is best-effort: an unclear shape yields no insights.
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Minimum rows before a strip is worth showing (a single row is a KPI metric).
_MIN_ROWS = 2
# Date formats accepted when deciding whether the label axis is a time dimension.
_DATE_FORMATS = ("%Y-%m-%d", "%Y-%m", "%Y/%m/%d", "%Y")


def _is_number(value: object) -> bool:
    """True for real numbers (bool is excluded — it's an int subclass in Python)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fmt(value: object) -> str:
    """Format a number with thousands separators; drop a trailing .0 on integers."""
    try:
        num = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    return f"{int(num):,}" if num == int(num) else f"{num:,.2f}"


def _column_is_numeric(rows: list[tuple], index: int) -> bool:
    """True when every non-null value in the column is a real number."""
    saw_value = False
    for row in rows:
        value = row[index]
        if value is None:
            continue
        if not _is_number(value):
            return False
        saw_value = True
    return saw_value


def _temporal_key(value: object) -> datetime | None:
    """Parse a label into a date for trend ordering, or None if it isn't one."""
    if value is None:
        return None
    if _is_number(value) and 1900 <= float(value) <= 2100:  # a bare year
        return datetime(int(value), 1, 1)
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _trend_line(
    columns: list[str], rows: list[tuple], label: int, measure: int
) -> str | None:
    """Return a trend insight when the label axis is a full date dimension."""
    pairs: list[tuple[datetime, float]] = []
    for row in rows:
        key = _temporal_key(row[label])
        if key is None or not _is_number(row[measure]):
            return None  # not a clean time series — don't guess a trend
        pairs.append((key, float(row[measure])))
    if len(pairs) < _MIN_ROWS:
        return None
    pairs.sort(key=lambda p: p[0])  # order by date, not by row order
    first, last = pairs[0][1], pairs[-1][1]
    if first == last:
        return None
    direction = "up" if last > first else "down"
    return f"Trend: {columns[measure]} moved {direction} from {_fmt(first)} to {_fmt(last)}."


def compute_insights(columns: list[str], rows: list[tuple]) -> list[str]:
    """Return up to four factual one-line insights about the result set.

    Facts are computed from the rows themselves (sum, top contributor + share,
    spread, date trend), so they are always consistent with the table shown.
    Returns an empty list when the result has no numeric measure to describe.
    """
    if len(rows) < _MIN_ROWS or not columns:
        return []

    measure = next(
        (i for i in range(len(columns) - 1, -1, -1) if _column_is_numeric(rows, i)),
        None,
    )
    if measure is None:
        return []
    values = [row[measure] for row in rows if _is_number(row[measure])]
    if len(values) < _MIN_ROWS:
        return []

    label = next((i for i in range(len(columns)) if i != measure), None)
    measure_name = columns[measure]
    total = sum(values)
    non_negative = all(v >= 0 for v in values)
    insights: list[str] = []

    # Sum is only stated when every value is non-negative — summing a column that
    # mixes signs (or is a rate/price) would read as meaningful when it isn't.
    if non_negative:
        insights.append(f"Sum of {measure_name}: {_fmt(total)} across {len(values)} rows.")

    if label is not None:
        top = max(rows, key=lambda r: r[measure] if _is_number(r[measure]) else float("-inf"))
        share = ""
        if non_negative and total > 0:
            share = f" ({top[measure] / total:.0%} of the total)"
        insights.append(
            f"Top {columns[label]}: {top[label]} — {_fmt(top[measure])}{share}."
        )

    low, high = min(values), max(values)
    if low != high:
        insights.append(f"Spread: {_fmt(low)} low to {_fmt(high)} high.")

    if label is not None:
        trend = _trend_line(columns, rows, label, measure)
        if trend is not None:
            insights.append(trend)

    return insights

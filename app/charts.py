"""Auto-visualisation: pick a sensible chart from the shape of a result set.

Heuristic, intentionally simple:
  - one numeric scalar (1 row, 1 column)            -> KPI metric (see single_value)
  - 2 columns, one label-like + one numeric         -> bar (or line for a time axis;
                                                        pie is offered for small sets)
  - 3 columns, two labels + one numeric             -> grouped bar
  - anything else                                    -> None (show the table only)

`available_charts` lists the kinds that fit a result (to drive a UI picker) and
`build_chart` renders a chosen kind; `choose_chart` is the auto-pick shortcut.
"""

from __future__ import annotations

import logging
import re

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

logger = logging.getLogger(__name__)

_MAX_BARS = 30
_MAX_PIE_SLICES = 12
_DATE_HINT_RE = re.compile(r"(date|month|year|day|period|week|quarter)", re.IGNORECASE)

# Chart-kind labels. AUTO means "let the heuristic decide"; the rest are the
# explicit types a user can force from the UI picker.
AUTO = "Auto"
BAR = "Bar"
LINE = "Line"
PIE = "Pie"
GROUPED_BAR = "Grouped bar"


def to_dataframe(columns: list[str], rows: list[tuple]) -> pd.DataFrame:
    """Build a DataFrame from agent output (columns + row tuples)."""
    return pd.DataFrame(rows, columns=columns)


def _numeric(series: pd.Series) -> pd.Series:
    """Coerce a column to numbers (non-numeric cells become NaN)."""
    return pd.to_numeric(series, errors="coerce")


def _is_numeric_col(df: pd.DataFrame, col: str) -> bool:
    """True if a column has at least one value that parses as a number."""
    return not _numeric(df[col]).isna().all()


def _looks_like_period(name: str, series: pd.Series) -> bool:
    """True if a column reads like a time axis (by name or parseable as dates)."""
    if _DATE_HINT_RE.search(name):
        return True
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.8


def _style(fig: Figure) -> Figure:
    """Apply the shared compact layout to a figure."""
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=380)
    return fig


def single_value(columns: list[str], rows: list[tuple]) -> tuple[str, object] | None:
    """Return (label, value) when the result is a single numeric scalar, else None.

    Lets the UI show a headline metric (e.g. "Total revenue: 1,204,500") instead
    of a one-cell table.
    """
    if len(columns) == 1 and len(rows) == 1:
        value = rows[0][0]
        if pd.notna(_numeric(pd.Series([value])).iloc[0]):
            return columns[0], value
    return None


def available_charts(columns: list[str], rows: list[tuple]) -> list[str]:
    """List the chart kinds that fit this result (empty if a chart won't help).

    The first entry is always AUTO, so callers can default to the heuristic pick.
    """
    if not rows:
        return []

    if len(columns) == 2:
        df = to_dataframe(columns, rows)
        values = _numeric(df[columns[1]])
        if values.isna().all():
            return []
        kinds = [AUTO, BAR, LINE]
        # Pie only makes sense for a few non-negative slices.
        if len(df) <= _MAX_PIE_SLICES and bool((values.dropna() >= 0).all()):
            kinds.append(PIE)
        return kinds

    if len(columns) == 3:
        df = to_dataframe(columns, rows)
        # Convention: the last column is the measure; the first two are
        # dimensions. Require at least one non-numeric dimension so a purely
        # numeric 3-column matrix (no natural categories) yields no chart.
        *dims, value_col = columns
        if _is_numeric_col(df, value_col) and any(
            not _is_numeric_col(df, c) for c in dims
        ):
            return [AUTO, GROUPED_BAR]

    return []


def build_chart(
    columns: list[str], rows: list[tuple], kind: str = AUTO
) -> Figure | None:
    """Build the requested chart kind for a result, or None if it can't be drawn."""
    if not rows:
        return None
    if len(columns) == 2:
        return _two_column_chart(columns, rows, kind)
    if len(columns) == 3:
        return _three_column_chart(columns, rows, kind)
    return None


def _two_column_chart(columns: list[str], rows: list[tuple], kind: str) -> Figure | None:
    """Render a bar/line/pie for a (label, value) result."""
    df = to_dataframe(columns, rows)
    label_col, value_col = columns
    values = _numeric(df[value_col])
    if values.isna().all():
        return None
    df[value_col] = values

    if kind == AUTO:
        kind = LINE if _looks_like_period(label_col, df[label_col]) else BAR

    try:
        if kind == LINE:
            df = df.sort_values(label_col)
            fig = px.line(df, x=label_col, y=value_col, markers=True)
        elif kind == PIE:
            df = df.sort_values(value_col, ascending=False).head(_MAX_PIE_SLICES)
            fig = px.pie(df, names=label_col, values=value_col)
        else:  # BAR (and any unexpected kind falls back to a bar)
            df = df.sort_values(value_col, ascending=False).head(_MAX_BARS)
            fig = px.bar(df, x=label_col, y=value_col)
    except (ValueError, TypeError) as exc:
        logger.debug("Chart build skipped: %s", exc)
        return None
    return _style(fig)


def _three_column_chart(
    columns: list[str], rows: list[tuple], kind: str
) -> Figure | None:
    """Render a grouped bar for a (label, category, value) result."""
    if kind not in (AUTO, GROUPED_BAR):
        return None
    df = to_dataframe(columns, rows)
    x_col, color_col, value_col = columns
    if not _is_numeric_col(df, value_col) or (
        _is_numeric_col(df, x_col) and _is_numeric_col(df, color_col)
    ):
        return None
    df[value_col] = _numeric(df[value_col])

    try:
        fig = px.bar(df, x=x_col, y=value_col, color=color_col, barmode="group")
    except (ValueError, TypeError) as exc:
        logger.debug("Chart build skipped: %s", exc)
        return None
    return _style(fig)


def choose_chart(columns: list[str], rows: list[tuple]) -> Figure | None:
    """Return the auto-selected Plotly figure for a result, or None."""
    return build_chart(columns, rows, AUTO)

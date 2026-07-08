"""Auto-visualisation: pick a sensible chart from the shape of a result set.

Heuristic, intentionally simple:
  - exactly 2 columns, one label-like + one numeric  -> bar (or line if the
    label looks like a date/period)
  - anything else                                    -> None (show the table only)
"""

from __future__ import annotations

import logging
import re

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

logger = logging.getLogger(__name__)

_MAX_BARS = 30
_DATE_HINT_RE = re.compile(r"(date|month|year|day|period|week|quarter)", re.IGNORECASE)


def to_dataframe(columns: list[str], rows: list[tuple]) -> pd.DataFrame:
    """Build a DataFrame from agent output (columns + row tuples)."""
    return pd.DataFrame(rows, columns=columns)


def _looks_like_period(name: str, series: pd.Series) -> bool:
    """True if a column reads like a time axis (by name or parseable as dates)."""
    if _DATE_HINT_RE.search(name):
        return True
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.8


def choose_chart(columns: list[str], rows: list[tuple]) -> Figure | None:
    """Return a Plotly figure for the result, or None if a chart won't help."""
    if not rows or len(columns) != 2:
        return None

    df = to_dataframe(columns, rows)
    label_col, value_col = columns[0], columns[1]

    # The second column must be numeric to plot.
    values = pd.to_numeric(df[value_col], errors="coerce")
    if values.isna().all():
        return None
    df[value_col] = values

    try:
        if _looks_like_period(label_col, df[label_col]):
            df = df.sort_values(label_col)
            fig = px.line(df, x=label_col, y=value_col, markers=True)
        else:
            df = df.sort_values(value_col, ascending=False).head(_MAX_BARS)
            fig = px.bar(df, x=label_col, y=value_col)
    except (ValueError, TypeError) as exc:
        logger.debug("Chart build skipped: %s", exc)
        return None

    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=380)
    return fig

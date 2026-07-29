"""Suggest real column values when a valid query returns zero rows.

A syntactically valid query can still return nothing because a filter names a
value that isn't in the data ("Northe" instead of "North"). Rather than show a
blank table, we parse the query's string filters, look up the actual distinct
values in the database, and offer the closest real matches — a grounded
"did you mean?" drawn from the data itself, never guessed by the LLM.

Everything here is best-effort: any parse/lookup problem yields no suggestions
(the caller just shows the plain "no rows" message) and nothing raises.
"""

from __future__ import annotations

import difflib
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from app.db import DatabaseError, get_connection

logger = logging.getLogger(__name__)

_DIALECT = "sqlite"
# Cap the distinct values scanned per column so suggesting stays cheap even on
# wide categorical columns; the closest matches are almost always near the top.
_MAX_DISTINCT = 500
# difflib closeness threshold (0-1) and how many near-misses to offer per filter.
_CUTOFF = 0.6
_MAX_CANDIDATES = 3
# Cap total suggestions so a many-filter query can't flood the UI.
_MAX_SUGGESTIONS = 3


@dataclass(frozen=True)
class ValueSuggestion:
    """A filter that matched nothing, paired with the closest real values."""

    column: str
    given: str
    candidates: list[str]

    def to_dict(self) -> dict:
        """Serialise for the API/UI layer (what, not how)."""
        return {
            "column": self.column,
            "given": self.given,
            "candidates": list(self.candidates),
        }


# A predicate of interest: column, table qualifier (alias/name or None), the
# string literal it was compared against, and whether it was a LIKE match.
_Filter = tuple[str, str | None, str, bool]


def _string_filters(statement: exp.Expression) -> list[_Filter]:
    """Collect `column = 'text'` / `column LIKE 'text'` predicates from the tree."""
    filters: list[_Filter] = []
    for node in statement.find_all(exp.EQ, exp.Like, exp.ILike):
        col, lit = node.this, node.expression
        if not (isinstance(col, exp.Column) and isinstance(lit, exp.Literal)):
            col, lit = node.expression, node.this  # operands may be reversed
        if not (isinstance(col, exp.Column) and isinstance(lit, exp.Literal)):
            continue
        if not lit.is_string:
            continue  # numeric filters have no useful "did you mean" list
        is_like = isinstance(node, (exp.Like, exp.ILike))
        filters.append((col.name, col.table or None, lit.this, is_like))
    return filters


def _alias_map(statement: exp.Expression) -> dict[str, str]:
    """Map every table alias/name used in the query to its real table name."""
    mapping: dict[str, str] = {}
    for table in statement.find_all(exp.Table):
        mapping[table.name] = table.name
        if table.alias:
            mapping[table.alias] = table.name
    return mapping


def _table_columns(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """Return {table_name: {lowercased column names}} for the whole database."""
    tables = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    columns: dict[str, set[str]] = {}
    for (name,) in ((row[0],) for row in tables):
        info = conn.execute(f"PRAGMA table_info('{name}')").fetchall()
        columns[name] = {row[1].lower() for row in info}
    return columns


def _resolve_table(
    column: str,
    qualifier: str | None,
    alias_map: dict[str, str],
    table_columns: dict[str, set[str]],
) -> str | None:
    """Find the real table a filtered column belongs to, or None if unresolvable."""
    if qualifier:
        return alias_map.get(qualifier)
    # Unqualified: the first query table that actually has this column wins.
    for real in dict.fromkeys(alias_map.values()):
        if column.lower() in table_columns.get(real, set()):
            return real
    return None


def _distinct_values(conn: sqlite3.Connection, table: str, column: str) -> list[str]:
    """Return up to `_MAX_DISTINCT` non-null distinct values of table.column."""
    safe_table = table.replace('"', '""')
    safe_column = column.replace('"', '""')
    sql = (
        f'SELECT DISTINCT "{safe_column}" FROM "{safe_table}" '
        f'WHERE "{safe_column}" IS NOT NULL LIMIT {_MAX_DISTINCT}'
    )
    try:
        rows = conn.execute(sql).fetchall()
    except sqlite3.Error as exc:
        logger.debug("Could not scan %s.%s for suggestions: %s", table, column, exc)
        return []
    return [str(row[0]) for row in rows if row[0] is not None]


def _closest(given: str, values: list[str], is_like: bool) -> list[str]:
    """Return the closest real values to `given` (case-insensitive), excluding it."""
    needle = (given.strip("%_ ") if is_like else given).lower()
    if not needle:
        return []
    by_lower: dict[str, str] = {}
    for value in values:
        by_lower.setdefault(value.lower(), value)
    matches = difflib.get_close_matches(
        needle, list(by_lower), n=_MAX_CANDIDATES, cutoff=_CUTOFF
    )
    return [by_lower[m] for m in matches if by_lower[m] != given]


def suggest_values(
    sql: str, db_path: str | Path | None = None
) -> list[ValueSuggestion]:
    """Suggest real values for the string filters in a zero-row query.

    Returns the closest actual values for any filter literal that matched
    nothing. Best-effort — returns an empty list rather than raising.
    """
    try:
        statement = sqlglot.parse_one(sql, read=_DIALECT)
    except ParseError:
        return []
    filters = _string_filters(statement)
    if not filters:
        return []

    alias_map = _alias_map(statement)
    try:
        conn = get_connection(db_path, read_only=True)
    except DatabaseError:
        return []
    suggestions: list[ValueSuggestion] = []
    seen: set[tuple[str, str]] = set()
    try:
        table_columns = _table_columns(conn)
        for column, qualifier, value, is_like in filters:
            key = (column.lower(), value.lower())
            if key in seen:
                continue
            table = _resolve_table(column, qualifier, alias_map, table_columns)
            if table is None:
                continue
            candidates = _closest(value, _distinct_values(conn, table, column), is_like)
            if candidates:
                suggestions.append(ValueSuggestion(column, value, candidates))
                seen.add(key)
            if len(suggestions) >= _MAX_SUGGESTIONS:
                break
    finally:
        conn.close()
    return suggestions


def apply_suggestion(sql: str, column: str, given: str, new_value: str) -> str:
    """Return `sql` with the filter `column = given` swapped to `new_value`.

    Powers the UI's one-click "did you mean" re-run. Preserves any LIKE
    wildcards around the value. Returns the SQL unchanged when no matching
    filter is found, so the caller can always run the result safely.
    """
    try:
        statement = sqlglot.parse_one(sql, read=_DIALECT)
    except ParseError:
        return sql
    target = column.lower()
    for node in statement.find_all(exp.EQ, exp.Like, exp.ILike):
        col, lit = node.this, node.expression
        if not (isinstance(col, exp.Column) and isinstance(lit, exp.Literal)):
            col, lit = node.expression, node.this
        if not (isinstance(col, exp.Column) and isinstance(lit, exp.Literal)):
            continue
        if not lit.is_string or col.name.lower() != target or lit.this != given:
            continue
        replacement = new_value
        if isinstance(node, (exp.Like, exp.ILike)):
            prefix = "%" if given.startswith("%") else ""
            suffix = "%" if given.endswith("%") else ""
            replacement = f"{prefix}{new_value}{suffix}"
        lit.replace(exp.Literal.string(replacement))
        return statement.sql(dialect=_DIALECT)
    return sql

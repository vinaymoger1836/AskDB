"""SQL guardrails — treat the LLM's generated SQL as untrusted input.

`validate_and_prepare` is the single checkpoint every generated query passes
through before it touches the database. It parses the SQL with `sqlglot` (a real
SQL parser, not a regex) and enforces:

  1. Exactly one statement (no stacked queries).
  2. The statement is a read-only query (SELECT / WITH ... SELECT / UNION).
  3. No write or DDL node anywhere in the tree
     (INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/PRAGMA/ATTACH/...).
  4. A LIMIT is present and no larger than `max_limit` (injected or clamped).

Defense in depth: even after this passes, execution uses a read-only SQLite
connection (see app/db.py).
"""

from __future__ import annotations

import logging

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

logger = logging.getLogger(__name__)

_DIALECT = "sqlite"

# Any of these node types appearing anywhere in the parsed tree means the query
# is not read-only. Built dynamically so it stays valid across sqlglot versions
# (some names exist only in newer releases).
_WRITE_TYPE_NAMES = [
    "Insert", "Update", "Delete", "Merge", "Drop", "Alter", "AlterTable",
    "Create", "TruncateTable", "Command", "Pragma", "Set", "Attach", "Detach",
]
_WRITE_TYPES = tuple(
    getattr(exp, name) for name in _WRITE_TYPE_NAMES if hasattr(exp, name)
)

# The top-level statement must be one of these read-only query shapes.
_ALLOWED_ROOT_TYPES = (exp.Select, exp.Union, exp.Subquery)


class GuardrailError(ValueError):
    """Raised when generated SQL is rejected. The message explains why.

    The agent catches this to feed the reason back to the LLM for correction.
    """


def _current_limit(statement: exp.Expression) -> int | None:
    """Return the integer LIMIT on a statement, or None if absent/non-literal."""
    limit_node = statement.args.get("limit")
    if limit_node is None:
        return None
    value = limit_node.expression
    if isinstance(value, exp.Literal) and value.is_int:
        return int(value.name)
    return None


def validate_and_prepare(sql: str, max_limit: int = 100) -> str:
    """Validate generated SQL and return a safe, LIMIT-bounded query string.

    Raises GuardrailError with a human-readable reason if the SQL is not a single
    read-only SELECT query.
    """
    text = (sql or "").strip()
    if not text:
        raise GuardrailError("Empty query.")

    try:
        statements = [s for s in sqlglot.parse(text, read=_DIALECT) if s is not None]
    except ParseError as exc:
        raise GuardrailError(f"Could not parse SQL: {exc}") from exc

    if len(statements) != 1:
        raise GuardrailError(
            f"Expected exactly one statement, found {len(statements)}. "
            "Stacked or multiple statements are not allowed."
        )

    statement = statements[0]

    # Reject any write/DDL/command node anywhere in the tree.
    for node in statement.walk():
        expr = node[0] if isinstance(node, tuple) else node
        if isinstance(expr, _WRITE_TYPES):
            raise GuardrailError(
                f"Only read-only SELECT queries are allowed; found "
                f"{type(expr).__name__.upper()}."
            )

    # The top-level statement must itself be a read-only query shape.
    if not isinstance(statement, _ALLOWED_ROOT_TYPES):
        raise GuardrailError(
            f"Only SELECT queries are allowed; got {type(statement).__name__.upper()}."
        )

    # `SELECT ... INTO t` writes a table — reject it.
    if statement.args.get("into") is not None:
        raise GuardrailError("SELECT ... INTO is not allowed (it writes a table).")

    # Inject or clamp the LIMIT.
    current = _current_limit(statement)
    if current is None or current > max_limit:
        statement = statement.limit(max_limit)

    return statement.sql(dialect=_DIALECT)

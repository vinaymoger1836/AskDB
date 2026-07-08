"""Prompt builders for SQL generation and answer summarisation.

Kept separate from the agent so the wording is easy to read, tweak, and test.
"""

from __future__ import annotations

_SQL_SYSTEM = (
    "You are a careful data analyst that writes SQL for a read-only SQLite database. "
    "Given a database schema and a question, respond with a SINGLE valid SQLite "
    "SELECT query that answers it.\n"
    "Rules:\n"
    "- Output ONLY the SQL query. No prose, no explanation, no markdown code fences.\n"
    "- Use a single SELECT statement (a leading WITH ... SELECT is allowed).\n"
    "- Never write to the database: no INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/PRAGMA.\n"
    "- Only reference tables and columns that appear in the schema.\n"
    "- Prefer explicit column lists and readable aliases over SELECT *.\n"
    "- When aggregating money, round to 2 decimals.\n"
    "- Dates are stored as ISO text (YYYY-MM-DD); filter a year with "
    "strftime('%Y', <col>) = '2023' or <col> LIKE '2023%'."
)


def build_sql_prompt(
    schema: str, question: str, prior_sql: str | None = None, prior_error: str | None = None
) -> list[dict[str, str]]:
    """Build the chat messages that ask the LLM for a SQL query.

    When `prior_error` is provided, the previous (failed) SQL and its error are
    included so the model can correct itself.
    """
    user = f"Database schema:\n{schema}\n\nQuestion: {question}\n\nSQL query:"
    if prior_error:
        user = (
            f"Database schema:\n{schema}\n\n"
            f"Question: {question}\n\n"
            f"Your previous query failed:\n{prior_sql}\n\n"
            f"Error: {prior_error}\n\n"
            "Return a corrected single SELECT query. SQL query:"
        )
    return [
        {"role": "system", "content": _SQL_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_summary_prompt(
    question: str, columns: list[str], rows: list[tuple], max_rows: int = 20
) -> list[dict[str, str]]:
    """Build the chat messages that ask the LLM for a one-line NL answer."""
    preview = [dict(zip(columns, r, strict=False)) for r in rows[:max_rows]]
    system = (
        "You summarise query results for a business user in ONE concise sentence. "
        "State the answer directly using the data. Do not mention SQL or tables. "
        "If the result is empty, say that no matching data was found."
    )
    user = (
        f"Question: {question}\n"
        f"Columns: {columns}\n"
        f"Rows (up to {max_rows} shown): {preview}\n\n"
        "One-sentence answer:"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

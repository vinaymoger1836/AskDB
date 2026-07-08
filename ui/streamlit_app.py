"""Streamlit chat frontend for AskDB: ask questions, get SQL + table + chart.

A ChatGPT/Claude-style conversation: a pinned chat input, message bubbles, and a
running history. Each answer renders as an assistant turn with a one-line
summary, a "Show SQL" toggle, a results table, and an auto-chart.

Backend strategy: call the FastAPI `/query` endpoint when it is reachable
(local two-process setup); otherwise fall back to calling the agent in-process
so the app also runs as a single Streamlit service on Hugging Face Spaces.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import requests
import streamlit as st

# Allow `import app...` when Streamlit runs this file as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.charts import choose_chart  # noqa: E402
from app.config import ConfigError, settings  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from data.seed import ensure_database  # noqa: E402

configure_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AskDB", page_icon="🧮", layout="centered")

SAMPLE_QUESTIONS = [
    "What were the top 5 products by revenue in 2023?",
    "Show monthly revenue in 2023",
    "Which 5 customers spent the most overall?",
    "How many orders were placed per country?",
    "What is the average order value by product category?",
]

_USER_AVATAR = "🧑‍💻"
_ASSISTANT_AVATAR = "🧮"

# ChatGPT/Claude-style bubbles: constrain width, round corners, tint by role.
_CHAT_CSS = """
<style>
[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 0.3rem 1rem;
    margin-bottom: 0.5rem;
    width: fit-content;
    max-width: 92%;
}
/* User turns → right-aligned, subtly tinted. */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(59, 130, 246, 0.12);
    margin-left: auto;
    flex-direction: row-reverse;
}
/* Assistant turns → left-aligned, neutral. */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: rgba(128, 128, 128, 0.10);
    margin-right: auto;
}
</style>
"""


def _query_via_api(question: str, history: list[dict]) -> dict | None:
    """Try the FastAPI backend. Return the parsed response, or None if unreachable."""
    try:
        resp = requests.post(
            f"{settings.api_base}/query",
            json={"question": question, "history": history},
            timeout=60,
        )
    except requests.RequestException:
        return None
    if resp.status_code >= 500:
        # Surface server-side config/db problems (e.g. missing key) to the user.
        detail = resp.json().get("detail", resp.text)
        return {"error": detail, "question": question}
    resp.raise_for_status()
    return resp.json()


def _query_in_process(question: str, history: list[dict]) -> dict:
    """Answer the question by calling the agent directly (HF Spaces fallback)."""
    from app import agent  # local import so the API path doesn't load it needlessly

    try:
        result = agent.answer(question, history=history)
    except ConfigError as exc:
        return {"error": str(exc), "question": question}
    return {
        "question": result.question,
        "sql": result.sql,
        "columns": result.columns,
        "rows": [list(r) for r in result.rows],
        "summary": result.summary,
        "attempts": result.attempts,
        "error": result.error,
    }


def run_query(question: str, history: list[dict] | None = None) -> dict:
    """Answer a question via the API when available, else in-process."""
    history = history or []
    result = _query_via_api(question, history)
    if result is None:
        logger.info("API unreachable at %s; answering in-process.", settings.api_base)
        result = _query_in_process(question, history)
    return result


def _conversation_history(messages: list[dict]) -> list[dict]:
    """Extract prior successful {question, sql} turns for follow-up context."""
    turns: list[dict] = []
    pending_question: str | None = None
    for message in messages:
        if message["role"] == "user":
            pending_question = message["content"]
        elif message["role"] == "assistant" and pending_question is not None:
            result = message.get("result", {})
            if not result.get("error") and result.get("sql"):
                turns.append({"question": pending_question, "sql": result["sql"]})
            pending_question = None
    return turns


def _render_answer(result: dict) -> None:
    """Render an assistant answer: summary, SQL, table, and an auto-chart."""
    if result.get("error"):
        st.error(result["error"])
        if result.get("sql"):
            with st.expander("Last SQL attempt"):
                st.code(result["sql"], language="sql")
        return

    if result.get("summary"):
        st.markdown(result["summary"])

    columns = result.get("columns", [])
    rows = result.get("rows", [])

    attempts = result.get("attempts", 1)
    if attempts and attempts > 1:
        st.caption(f"↻ Self-corrected after {attempts - 1} retry(s).")

    with st.expander("Show SQL"):
        st.code(result.get("sql") or "", language="sql")

    if not rows:
        st.info("The query ran but returned no rows.")
        return

    st.dataframe(
        [dict(zip(columns, r, strict=False)) for r in rows],
        use_container_width=True,
    )

    fig = choose_chart(columns, [tuple(r) for r in rows])
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


def _handle_question(question: str) -> None:
    """Append the user turn, run the query, and append the assistant turn."""
    # Capture prior turns for follow-up context before adding the current one.
    history = _conversation_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=_USER_AVATAR):
        st.markdown(question)
    with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
        with st.spinner("Writing and running SQL…"):
            result = run_query(question, history)
        _render_answer(result)
    st.session_state.messages.append({"role": "assistant", "result": result})


def _sidebar() -> None:
    """Render the sidebar: about, schema, and a clear-chat control."""
    with st.sidebar:
        st.subheader("About")
        st.write(
            "AskDB turns natural language into safe SQL over a seeded e-commerce "
            "database. Generated SQL is treated as untrusted input: SELECT-only, "
            "no writes, an enforced row limit, and a read-only connection."
        )
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        with st.expander("Database schema"):
            from app.db import get_schema

            st.code(get_schema(), language="sql")


def main() -> None:
    """Render the AskDB chat page."""
    ensure_database()
    st.markdown(_CHAT_CSS, unsafe_allow_html=True)

    st.title("🧮 AskDB")
    st.caption(
        "Ask a question in plain English — the agent writes a **read-only** SQL "
        "query, validates it, runs it, and answers with a table and chart."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    _sidebar()

    # Replay the conversation so far.
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar=_USER_AVATAR):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
                _render_answer(message["result"])

    # On an empty conversation, offer sample questions as clickable starters.
    pending: str | None = None
    if not st.session_state.messages:
        st.write("**Try one:**")
        for sample in SAMPLE_QUESTIONS:
            if st.button(sample, use_container_width=True):
                pending = sample

    # Pinned chat input at the bottom.
    typed = st.chat_input("Ask about the sales data…")
    if typed and typed.strip():
        pending = typed.strip()

    if pending:
        _handle_question(pending)
        st.rerun()


# Streamlit runs this file as a script top-to-bottom on every rerun.
main()

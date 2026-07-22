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
import tempfile
from pathlib import Path

import requests
import streamlit as st

# Allow `import app...` when Streamlit runs this file as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import ingest  # noqa: E402
from app.charts import available_charts, build_chart, single_value  # noqa: E402
from app.config import ConfigError, settings  # noqa: E402
from app.export import to_csv_bytes, to_excel_bytes  # noqa: E402
from app.ingest import IngestError, ingest_upload  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from data.seed import ensure_database  # noqa: E402

configure_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AskDB", page_icon="🧮", layout="centered")

# The always-present built-in source. Uploaded CSV/Excel files add more.
DEMO_SOURCE = "Demo e-commerce DB"

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


def _query_in_process(
    question: str, history: list[dict], db_path: str | None
) -> dict:
    """Answer the question by calling the agent directly (HF Spaces fallback).

    Also the only path for uploaded sources: their SQLite file lives in this
    session's temp dir, which the separate FastAPI process can't see.
    """
    from app import agent  # local import so the API path doesn't load it needlessly

    try:
        result = agent.answer(question, history=history, db_path=db_path)
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


def run_query(
    question: str, history: list[dict] | None = None, db_path: str | None = None
) -> dict:
    """Answer a question against the active source.

    The demo DB tries the FastAPI backend first (local two-process setup) and
    falls back to in-process. Uploaded sources always run in-process, since the
    API process has no access to this session's uploaded database file.
    """
    history = history or []
    if db_path is None:
        result = _query_via_api(question, history)
        if result is not None:
            return result
        logger.info("API unreachable at %s; answering in-process.", settings.api_base)
    return _query_in_process(question, history, db_path)


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


def _explain_query(sql: str) -> str:
    """Explain a query via the FastAPI backend when reachable, else in-process."""
    try:
        resp = requests.post(
            f"{settings.api_base}/explain", json={"sql": sql}, timeout=60
        )
        if resp.status_code < 500:
            resp.raise_for_status()
            return resp.json()["explanation"]
        return resp.json().get("detail", resp.text)  # surface a server-side problem
    except requests.RequestException:
        pass  # API unreachable — fall back to answering in-process.

    from app import agent

    try:
        return agent.explain_sql(sql)
    except (ConfigError, RuntimeError) as exc:
        return f"Couldn't explain the query: {exc}"


def _render_explain(sql: str, key_prefix: str) -> None:
    """Offer an on-demand plain-English explanation of the query, cached per turn."""
    store = st.session_state.setdefault("explanations", {})
    if st.button(
        "💡 Explain this query", key=f"{key_prefix}_explain", use_container_width=True
    ):
        with st.spinner("Explaining…"):
            store[key_prefix] = _explain_query(sql)
    if key_prefix in store:
        st.info(store[key_prefix])


def _render_downloads(columns: list[str], rows: list, key_prefix: str) -> None:
    """Offer the current result set as a CSV or Excel download."""
    tuples = [tuple(r) for r in rows]
    col_csv, col_xlsx = st.columns(2)
    with col_csv:
        st.download_button(
            "⬇️ CSV",
            data=to_csv_bytes(columns, tuples),
            file_name="askdb_results.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{key_prefix}_csv",
        )
    with col_xlsx:
        st.download_button(
            "⬇️ Excel",
            data=to_excel_bytes(columns, tuples),
            file_name="askdb_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"{key_prefix}_xlsx",
        )


def _render_answer(
    result: dict, show_chart: bool = True, key_prefix: str = "live"
) -> None:
    """Render an assistant answer: summary, SQL, table, downloads, and a chart.

    `show_chart` is captured per answer when it's produced, so toggling charts
    only affects future queries — it never re-renders past turns. `key_prefix`
    keeps each turn's download-button keys unique when the thread is replayed.
    """
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

    sql = result.get("sql") or ""
    with st.expander("Show SQL"):
        st.code(sql, language="sql")
        if sql:
            _render_explain(sql, key_prefix)

    if not rows:
        st.info("The query ran but returned no rows.")
        return

    st.dataframe(
        [dict(zip(columns, r, strict=False)) for r in rows],
        use_container_width=True,
    )

    _render_downloads(columns, rows, key_prefix)

    # Charts are opt-out: rendering Plotly can be slow, so skip the work entirely
    # (figure build + client render) when charts were off for this answer.
    if show_chart:
        fig = choose_chart(columns, [tuple(r) for r in rows])
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)


def _active_db_path() -> str | None:
    """Return the SQLite path for the selected source (None ⇒ demo DB)."""
    name = st.session_state.get("active_source", DEMO_SOURCE)
    if name == DEMO_SOURCE:
        return None
    source = st.session_state.sources.get(name)
    return str(source.db_path) if source else None


def _handle_question(question: str) -> None:
    """Append the user turn, run the query, and append the assistant turn."""
    # Capture prior turns for follow-up context before adding the current one.
    history = _conversation_history(st.session_state.messages)
    db_path = _active_db_path()
    # Freeze the current charts preference onto this answer so later toggling
    # only changes future queries, not this one.
    show_chart = st.session_state.get("show_charts", True)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=_USER_AVATAR):
        st.markdown(question)
    with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
        with st.spinner("Writing and running SQL…"):
            result = run_query(question, history, db_path)
        _render_answer(result, show_chart)
    st.session_state.messages.append(
        {"role": "assistant", "result": result, "show_chart": show_chart}
    )


def _ingest_uploads(uploaded_files: list) -> None:
    """Load any newly uploaded CSV/Excel files into session-scoped sources."""
    for upload in uploaded_files:
        # file_id is stable across reruns, so each file is ingested only once.
        if upload.file_id in st.session_state.processed_upload_ids:
            continue
        st.session_state.processed_upload_ids.add(upload.file_id)
        try:
            source = ingest_upload(
                upload.name, upload.getvalue(), st.session_state.upload_dir
            )
        except IngestError as exc:
            st.error(f"Couldn't load {upload.name}: {exc}")
            continue
        # Disambiguate if a file of the same name was uploaded before.
        name = source.name
        suffix = 2
        while name in st.session_state.sources:
            name = f"{source.name} ({suffix})"
            suffix += 1
        source.name = name
        st.session_state.sources[name] = source
        st.session_state.active_source = name
        st.toast(f"Loaded {name}: {', '.join(source.tables)}", icon="📄")
        if source.truncated:
            st.warning(
                f"{name} is very large and was capped at "
                f"{ingest.MAX_ROWS_PER_TABLE:,} rows per table — answers cover "
                "only that many rows.",
                icon="⚠️",
            )


def _select_source() -> None:
    """Render the data-source picker; switching sources starts a fresh chat."""
    names = [DEMO_SOURCE, *st.session_state.sources.keys()]
    active = st.session_state.active_source
    if active not in names:
        active = DEMO_SOURCE
    choice = st.radio("Data source", names, index=names.index(active))
    if choice != st.session_state.active_source:
        # Prior turns reference the old schema; clear them to avoid confusion.
        st.session_state.active_source = choice
        st.session_state.messages = []
        st.rerun()


def _sidebar() -> None:
    """Render the sidebar: data sources, upload, schema, and clear-chat."""
    from app.db import get_schema

    with st.sidebar:
        st.subheader("Data source")
        st.caption("Query the demo database, or upload your own CSV/Excel.")

        uploaded = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            help="Each file becomes a queryable source. Excel sheets become tables.",
        )
        if uploaded:
            _ingest_uploads(uploaded)

        _select_source()

        st.toggle(
            "📊 Show charts",
            key="show_charts",
            help="Turn off to skip chart rendering for faster answers.",
        )

        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        with st.expander("Schema"):
            try:
                st.code(get_schema(_active_db_path()), language="sql")
            except Exception as exc:  # a bad source shouldn't break the sidebar
                st.warning(f"Couldn't read schema: {exc}")

        st.subheader("About")
        st.write(
            "AskDB turns natural language into safe SQL. Generated SQL is treated "
            "as untrusted input: SELECT-only, no writes, an enforced row limit, "
            "and a read-only connection — the same guardrails for uploads as for "
            "the demo data."
        )


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
    # Session-scoped upload state: a temp dir plus the sources built from it.
    if "upload_dir" not in st.session_state:
        st.session_state.upload_dir = tempfile.mkdtemp(prefix="askdb_uploads_")
    if "sources" not in st.session_state:
        st.session_state.sources = {}
    if "processed_upload_ids" not in st.session_state:
        st.session_state.processed_upload_ids = set()
    if "active_source" not in st.session_state:
        st.session_state.active_source = DEMO_SOURCE
    if "show_charts" not in st.session_state:
        st.session_state.show_charts = True

    _sidebar()

    on_demo = st.session_state.active_source == DEMO_SOURCE

    # Replay the conversation so far.
    for index, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            with st.chat_message("user", avatar=_USER_AVATAR):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
                _render_answer(
                    message["result"],
                    message.get("show_chart", True),
                    key_prefix=f"msg{index}",
                )

    # On an empty conversation, offer sample questions as clickable starters.
    # Samples only fit the demo schema; uploaded sources get a plain prompt.
    pending: str | None = None
    if not st.session_state.messages:
        if on_demo:
            st.write("**Try one:**")
            for sample in SAMPLE_QUESTIONS:
                if st.button(sample, use_container_width=True):
                    pending = sample
        else:
            st.info(
                f"Ask a question about **{st.session_state.active_source}** — "
                "the agent reads its columns and writes the SQL for you."
            )

    # Pinned chat input at the bottom.
    placeholder = (
        "Ask about the sales data…"
        if on_demo
        else f"Ask about {st.session_state.active_source}…"
    )
    typed = st.chat_input(placeholder)
    if typed and typed.strip():
        pending = typed.strip()

    if pending:
        _handle_question(pending)
        st.rerun()


# Streamlit runs this file as a script top-to-bottom on every rerun.
main()

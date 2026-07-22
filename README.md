---
title: AskDB
emoji: 🧮
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Ask questions in plain English, get safe SQL + a chart
---

# AskDB — Text-to-SQL Analytics Agent

> The block above is Hugging Face Spaces configuration. It is ignored by GitHub
> rendering and required by HF to launch the Streamlit app.

Ask a question in plain English → an LLM writes a **read-only** SQL query grounded in the
database schema → the query is validated and run safely → you get the SQL, a results table,
an auto-chart, and a one-line answer. Generated SQL is treated as **untrusted input**.

## What it does

1. You type a question, e.g. *"What were the top 5 products by revenue in 2023?"*
2. The agent reads the DB schema and prompts Llama 3.3 70B (via Groq) for a single SELECT query.
3. A `sqlglot` **guardrail** validates it: SELECT-only, no writes/DDL/PRAGMA, one statement, and
   an enforced `LIMIT`.
4. It runs on a **read-only** SQLite connection.
5. On a SQL error, the error is fed back to the model and the query is retried (self-correction).
6. You get the SQL, a table, an auto-chart, and a natural-language summary.

## Features

- **Chat UI with multi-turn context** — a ChatGPT/Claude-style thread; follow-ups like
  *"now break that down by month"* resolve against the previous turn's question and SQL.
- **Bring your own data** — upload a CSV or Excel file and query it under the same guardrails
  as the demo database (each file becomes a source; Excel sheets become tables).
- **Explain this query** — one click sends the validated SQL back to the LLM for a plain-English
  description of what it does (transparency without exposing anything unsafe).
- **Auto-charts** — a heuristic picks a bar, line, pie, or grouped bar from the result shape;
  a single numeric answer renders as a headline metric. A **chart-type picker** overrides the pick.
- **Download results** — export any result table as CSV or Excel.
- **Result caching** — repeat questions (same context and data) are served from an LRU cache,
  skipping both LLM calls.

## Architecture

```
Streamlit UI (chat thread, source picker, uploads, table, charts, downloads, "Explain")
      │  calls FastAPI /query when reachable, else runs the agent in-process
      ▼
FastAPI  /query  ── app/agent.answer()
      │   1. get_schema()          → table/column DDL text
      │   2. build prompt(schema, question, history)
      │   3. LLM (Groq)            → candidate SQL
      │   4. guardrails.validate_and_prepare()  → SELECT-only, LIMIT   [safety]
      │   5. db.run_query()  on read-only SQLite → rows
      │        └─ on error → feed error back → retry (≤2)
      │   6. LLM → one-line summary
      ▼
SQLite  (data/sales.db demo dataset, or a session-scoped DB built from an upload)

FastAPI  /explain  → LLM plain-English description of an already-validated query
FastAPI  /schema, /health  → DDL text and a liveness probe
```

## The safety layer (`app/guardrails.py`)

Before any generated SQL runs, it is parsed with `sqlglot` and rejected unless it is a single
read-only query. Writes/DDL/commands (`INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/PRAGMA/ATTACH`)
are blocked at the AST level, a `LIMIT` is injected or clamped, and execution uses a read-only
connection with a statement timeout. User text is never interpolated into SQL — only the
model's validated query runs.

## Tech stack

Groq (Llama 3.3 70B) · `sqlglot` · SQLite · FastAPI + Pydantic · Streamlit · Plotly ·
pandas / openpyxl (CSV & Excel ingest and export) · pytest · ruff.

## Run locally

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # then add your GROQ_API_KEY

python -m data.seed                 # build the demo database
uvicorn app.main:app --reload       # backend  (terminal 1)
streamlit run ui/streamlit_app.py   # UI       (terminal 2)
```

Tests need no API key (the LLM is mocked):

```bash
python -m pytest -q
ruff check .
```

## Configuration

All secrets are read only from the environment (`.env` locally, Space secrets when deployed) —
never hardcoded. See `.env.example`. Required: `GROQ_API_KEY`. Optional tunables: `GROQ_MODEL`,
`DB_PATH`, `MAX_LIMIT`, `AGENT_MAX_RETRIES`, `QUERY_TIMEOUT_S`, `ASKDB_API_BASE`.

## Deploy (Hugging Face Spaces)

The Space is a **Docker** Space (`sdk: docker`, `app_port: 7860`): the `Dockerfile` installs
the deps, bakes the seeded database into the image, and launches Streamlit on port 7860
(`streamlit run ui/streamlit_app.py --server.port=7860`), which answers in-process — no separate
API host is needed. Set `GROQ_API_KEY` in **Settings → Variables and secrets** as a secret —
never commit it.

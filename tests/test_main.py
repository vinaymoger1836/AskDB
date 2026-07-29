"""Tests for the FastAPI endpoints, including the /upload → /query source flow.

The LLM is never called: /query is exercised by monkeypatching `agent.answer`,
so these tests verify the HTTP wiring and the source-ID resolution (not Groq).
"""

from __future__ import annotations

import types

from fastapi.testclient import TestClient

from app.agent import AgentResult
from app.main import app

client = TestClient(app)

_CSV = b"name,revenue\nWidget,10\nGadget,20\n"


def _upload(name: str = "products.csv", data: bytes = _CSV) -> dict:
    resp = client.post("/upload", files={"file": (name, data, "text/csv")})
    return resp


def test_health_ok() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_csv_returns_source_id_and_tables() -> None:
    resp = _upload()
    assert resp.status_code == 200
    body = resp.json()
    assert body["source_id"]
    assert "products" in body["tables"]
    assert body["truncated"] is False


def test_upload_rejects_unsupported_type() -> None:
    resp = client.post("/upload", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert resp.status_code == 400


def test_upload_rejects_oversize_file(monkeypatch) -> None:
    # A 0 MB cap makes any non-empty upload too large — exercises the 413 path.
    monkeypatch.setattr("app.main.settings", types.SimpleNamespace(max_upload_mb=0))
    resp = _upload()
    assert resp.status_code == 413


def test_query_with_unknown_source_id_is_404() -> None:
    resp = client.post(
        "/query", json={"question": "how many?", "source_id": "nope"}
    )
    assert resp.status_code == 404


def test_schema_with_unknown_source_id_is_404() -> None:
    assert client.get("/schema", params={"source_id": "nope"}).status_code == 404


def test_run_sql_executes_edited_query() -> None:
    resp = client.post("/run-sql", json={"sql": "SELECT name FROM products"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] is None
    assert "name" in body["columns"]
    assert body["rows"]


def test_run_sql_rejects_a_write_via_guardrail() -> None:
    # The guardrail runs server-side, so a write is a clean 200 with an error
    # message — not a crash — matching how the UI renders it.
    resp = client.post("/run-sql", json={"sql": "DROP TABLE products"})
    assert resp.status_code == 200
    assert resp.json()["error"] is not None


def test_run_sql_with_unknown_source_id_is_404() -> None:
    resp = client.post("/run-sql", json={"sql": "SELECT 1", "source_id": "nope"})
    assert resp.status_code == 404


def test_audit_endpoint_records_a_blocked_query() -> None:
    from app import audit

    audit.clear()
    client.post("/run-sql", json={"sql": "DROP TABLE products"})

    resp = client.get("/audit")
    assert resp.status_code == 200
    events = resp.json()["events"]
    assert events and events[0]["source"] == "edited"
    assert "DROP" in events[0]["sql"].upper()


def test_run_sql_second_call_is_served_from_cache() -> None:
    from app import agent

    agent.clear_cache()
    body = {"sql": "SELECT name FROM products"}
    assert client.post("/run-sql", json=body).json()["cached"] is False
    assert client.post("/run-sql", json=body).json()["cached"] is True


def test_saved_query_save_list_delete_flow() -> None:
    save = client.post(
        "/saved-queries",
        json={"name": "Top products", "question": "What are the top products?"},
    )
    assert save.status_code == 200
    assert save.json()["name"] == "Top products"

    listed = client.get("/saved-queries").json()["queries"]
    assert [q["name"] for q in listed] == ["Top products"]

    deleted = client.request("DELETE", "/saved-queries", params={"name": "Top products"})
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True}
    assert client.get("/saved-queries").json()["queries"] == []


def test_saved_query_scoped_by_source_id() -> None:
    client.post(
        "/saved-queries",
        json={"name": "u", "question": "q", "source_id": "src-1"},
    )
    assert client.get("/saved-queries").json()["queries"] == []
    scoped = client.get("/saved-queries", params={"source_id": "src-1"}).json()
    assert [q["name"] for q in scoped["queries"]] == ["u"]


def test_delete_unknown_saved_query_is_404() -> None:
    resp = client.request("DELETE", "/saved-queries", params={"name": "missing"})
    assert resp.status_code == 404


def test_query_returns_clarification_options(monkeypatch) -> None:
    def fake_answer(question, *, history=None, db_path=None):
        return AgentResult(
            question=question, clarification=["Interpretation A", "Interpretation B"]
        )

    monkeypatch.setattr("app.main.agent.answer", fake_answer)

    body = client.post("/query", json={"question": "vague thing"}).json()
    assert body["clarification"] == ["Interpretation A", "Interpretation B"]
    assert body["sql"] is None
    assert body["error"] is None


def test_run_sql_empty_result_returns_suggestions() -> None:
    from app import agent

    agent.clear_cache()
    body = {"sql": "SELECT name FROM customers WHERE country = 'Germny'"}
    resp = client.post("/run-sql", json=body).json()
    assert resp["rows"] == []
    assert resp["suggestions"]
    assert resp["suggestions"][0]["column"] == "country"
    assert "Germany" in resp["suggestions"][0]["candidates"]


def test_query_resolves_uploaded_source_to_its_db_path(monkeypatch) -> None:
    source_id = _upload("sales.csv", b"item,qty\nx,1\n").json()["source_id"]

    captured: dict[str, object] = {}

    def fake_answer(question, *, history=None, db_path=None):
        captured["db_path"] = db_path
        return AgentResult(
            question=question, sql="SELECT 1", columns=["n"], rows=[(1,)], summary="ok"
        )

    monkeypatch.setattr("app.main.agent.answer", fake_answer)

    resp = client.post(
        "/query", json={"question": "how many?", "source_id": source_id}
    )
    assert resp.status_code == 200
    # The opaque ID resolved to a server-owned .db path — never a client string.
    assert isinstance(captured["db_path"], str)
    assert captured["db_path"].endswith(".db")

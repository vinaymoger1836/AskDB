"""Tests for the persistent saved-queries store."""

from __future__ import annotations

import pytest

from app import saved_queries


def test_save_and_list_roundtrip() -> None:
    saved_queries.save("Top products", "What are the top 5 products?")
    rows = saved_queries.list_for()
    assert len(rows) == 1
    assert rows[0].name == "Top products"
    assert rows[0].question == "What are the top 5 products?"
    assert rows[0].source_id == "demo"  # None normalises to the demo key


def test_resaving_a_name_updates_the_question() -> None:
    saved_queries.save("q", "first question")
    saved_queries.save("q", "second question")
    rows = saved_queries.list_for()
    assert len(rows) == 1
    assert rows[0].question == "second question"


def test_sources_are_isolated() -> None:
    saved_queries.save("demo one", "demo question")
    saved_queries.save("upload one", "upload question", source_id="upload-123")
    assert [r.name for r in saved_queries.list_for()] == ["demo one"]
    assert [r.name for r in saved_queries.list_for("upload-123")] == ["upload one"]


def test_delete_returns_true_then_false() -> None:
    saved_queries.save("q", "question")
    assert saved_queries.delete("q") is True
    assert saved_queries.delete("q") is False
    assert saved_queries.list_for() == []


def test_list_is_newest_first() -> None:
    saved_queries.save("first", "q1")
    saved_queries.save("second", "q2")
    assert [r.name for r in saved_queries.list_for()] == ["second", "first"]


def test_blank_name_or_question_is_rejected() -> None:
    with pytest.raises(ValueError):
        saved_queries.save("   ", "a question")
    with pytest.raises(ValueError):
        saved_queries.save("a name", "   ")


def test_name_and_question_are_trimmed() -> None:
    saved_queries.save("  spaced  ", "  a question  ")
    row = saved_queries.list_for()[0]
    assert row.name == "spaced"
    assert row.question == "a question"

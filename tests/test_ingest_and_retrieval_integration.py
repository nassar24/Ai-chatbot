"""Integration test against a real MySQL-compatible database.

Uses FakeEmbeddingProvider (see tests/fakes.py) instead of a live Voyage
API call — this test is verifying pipeline mechanics (chunking -> hashing
-> storage -> cosine ranking -> re-ingest idempotency), not real semantic
embedding quality, which requires a live API key at integration time.

Requires env vars DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME pointing at
a disposable test database (schema.sql is applied fresh in setup).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from app.kb.ingest import ingest_knowledge_base
from app.kb.retrieval import retrieve_relevant_chunks
from tests.fakes import FakeEmbeddingProvider

KB_PATH = Path(__file__).resolve().parent.parent / "knowledge_base.md"

# db_connection fixture lives in tests/conftest.py (shared with
# test_rag_pipeline.py) — it also guards against running against a
# non-test database. See conftest.py for why.


@pytest.fixture()
def kb_text() -> str:
    return KB_PATH.read_text(encoding="utf-8")


@pytest.fixture()
def provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


def test_ingest_populates_all_chunks(db_connection, kb_text, provider):
    report = ingest_knowledge_base(kb_text, provider, db_connection)

    assert report.embedded == report.total_chunks
    assert report.unchanged == 0
    assert report.removed == 0

    with db_connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM kb_chunks")
        (count,) = cursor.fetchone()
    assert count == report.total_chunks


def test_reingest_with_no_changes_embeds_nothing(db_connection, kb_text, provider):
    ingest_knowledge_base(kb_text, provider, db_connection)
    second_report = ingest_knowledge_base(kb_text, provider, db_connection)

    assert second_report.embedded == 0
    assert second_report.unchanged == second_report.total_chunks


def test_reingest_after_edit_only_reembeds_changed_chunk(db_connection, kb_text, provider):
    ingest_knowledge_base(kb_text, provider, db_connection)

    edited_text = kb_text.replace(
        "A deposit is required before starting any project",
        "A 50% deposit is required before starting any project",
    )
    assert edited_text != kb_text  # guard: replacement actually matched something

    report = ingest_knowledge_base(edited_text, provider, db_connection)
    assert report.embedded == 1
    assert report.unchanged == report.total_chunks - 1


def test_switching_embedding_provider_forces_full_reembed_even_with_unchanged_content(
    db_connection, kb_text, provider
):
    # Regression test: content_hash alone is not enough to decide staleness.
    # If the KB text is byte-identical but a different embedding model is
    # now in use, every chunk must be re-embedded — otherwise stale vectors
    # from the old provider silently stay in place and get compared against
    # query vectors from a different vector space entirely.
    ingest_knowledge_base(kb_text, provider, db_connection)

    class OtherFakeProvider(FakeEmbeddingProvider):
        @property
        def model_name(self) -> str:
            return "fake-bow-v2-different-model"

    second_report = ingest_knowledge_base(kb_text, OtherFakeProvider(), db_connection)
    assert second_report.embedded == second_report.total_chunks
    assert second_report.unchanged == 0

    with db_connection.cursor() as cursor:
        cursor.execute("SELECT DISTINCT embedding_model FROM kb_chunks")
        models_in_use = {row[0] for row in cursor.fetchall()}
    assert models_in_use == {"fake-bow-v2-different-model"}


def test_removed_section_is_deleted_on_reingest(db_connection, kb_text, provider):
    ingest_knowledge_base(kb_text, provider, db_connection)

    # Simulate a section being deleted from the source doc entirely.
    lines = kb_text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "## Refund Policy")
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("## "))
    trimmed_text = "\n".join(lines[:start] + lines[end:])

    with db_connection.cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*) FROM kb_chunks WHERE section_title = %s",
            ("Company Policies — Refund Policy",),
        )
        (before,) = cursor.fetchone()
    assert before == 1

    report = ingest_knowledge_base(trimmed_text, provider, db_connection)
    assert report.removed == 1

    with db_connection.cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*) FROM kb_chunks WHERE section_title = %s",
            ("Company Policies — Refund Policy",),
        )
        (after,) = cursor.fetchone()
    assert after == 0


@pytest.mark.parametrize(
    "query,expected_section_title",
    [
        ("What is your refund policy?", "Company Policies — Refund Policy"),
        ("How do I pay, is a deposit required?", "Company Policies — Payment Policy"),
        (
            "Who leads software development, technical planning and system architecture?",
            "Our Team — Dakota Martinez — Software Team Lead",
        ),
        (
            "Do you offer website development and web applications?",
            "Our Services — 01. Software & Web Development",
        ),
        (
            "What are your working hours, Sunday through Thursday?",
            "About Apex Creative — Contact & Location",
        ),
    ],
)
def test_retrieval_surfaces_the_right_chunk_first(
    db_connection, kb_text, provider, query, expected_section_title
):
    ingest_knowledge_base(kb_text, provider, db_connection)

    results = retrieve_relevant_chunks(query, provider, db_connection, top_k=3, min_score=0.0)

    assert results, f"No results for query: {query!r}"
    top_titles = [r.section_title for r in results]
    assert expected_section_title in top_titles[:3], (
        f"Expected {expected_section_title!r} near the top for query {query!r}, "
        f"got: {top_titles}"
    )


def test_retrieval_rejects_empty_query(db_connection, provider):
    with pytest.raises(ValueError):
        retrieve_relevant_chunks("   ", provider, db_connection)


def test_retrieval_on_empty_kb_returns_empty_list(db_connection, provider):
    results = retrieve_relevant_chunks("anything", provider, db_connection)
    assert results == []


def test_retrieval_name_only_query(db_connection, kb_text, provider):
    """A bare proper-noun query has almost no semantic signal — this is
    the case the keyword-matching half of hybrid retrieval exists for,
    and the one that regressed when the vector path was used alone."""
    ingest_knowledge_base(kb_text, provider, db_connection)
    results = retrieve_relevant_chunks("Who is Dakota Martinez?", provider, db_connection, top_k=4, min_score=0.0)
    top_titles = [r.section_title for r in results]
    assert any("Dakota Martinez" in title for title in top_titles), (
        f"Expected section title containing 'Dakota Martinez' in top-4 results, got: {top_titles}"
    )


def test_dimension_mismatch_scores_zero_instead_of_garbage(
    db_connection, kb_text, provider, caplog
):
    """Regression, found while running the stack end-to-end.

    Stored vectors embedded by one model and a query vector from another
    have different lengths. `zip` truncates to the shorter one, so the
    cosine computation happily returns a meaningless number instead of
    failing — retrieval silently degrades to keyword-only matching and
    the bot answers from effectively random chunks, with nothing in the
    logs to say why. This is what a partial ingest or a changed
    GEMINI_EMBEDDING_DIMENSION looks like in production.
    """
    ingest_knowledge_base(kb_text, provider, db_connection)

    wrong_length_vector = [0.1] * 7  # nothing like the ingested dimension

    with caplog.at_level(logging.ERROR):
        results = retrieve_relevant_chunks(
            "what services do you offer",
            provider,
            db_connection,
            top_k=5,
            min_score=0.0,
            query_vector=wrong_length_vector,
        )

    assert results, "expected chunks back (keyword path still applies)"
    assert all(r.vector_score == 0.0 for r in results), (
        "a mismatched vector must contribute nothing, not a truncated score"
    )
    assert any("dimension mismatch" in record.message.lower() for record in caplog.records), (
        "the mismatch must be logged loudly, not swallowed"
    )


def test_querying_with_a_different_model_than_built_the_index_is_loud(
    db_connection, kb_text, provider, caplog
):
    """Regression, found while running a jailbreak test on a live server.

    The dimension check alone is not the right invariant. Two different
    models can share a dimension, and one silently did: aligning the test
    fake to 768 for the vector(768) column removed the accidental
    protection the dimension check had been giving, so a pytest run left
    the index full of bag-of-words fixture vectors and the server served
    garbage scores with nothing in the logs. Every question returned "I
    don't have that information" and it looked like a model refusal.

    The real invariant is that the index must have been BUILT by the model
    now querying it.
    """
    ingest_knowledge_base(kb_text, provider, db_connection)

    class DifferentModelProvider(FakeEmbeddingProvider):
        @property
        def model_name(self) -> str:
            return "some-other-embedding-model"

    with caplog.at_level(logging.ERROR):
        retrieve_relevant_chunks(
            "what services do you offer", DifferentModelProvider(), db_connection,
            top_k=3, min_score=0.0,
        )

    assert any("model mismatch" in r.message.lower() for r in caplog.records), (
        "querying an index built by a different embedding model must be loud"
    )


def test_matching_model_logs_no_mismatch(db_connection, kb_text, provider, caplog):
    """The check must not cry wolf on the normal path."""
    ingest_knowledge_base(kb_text, provider, db_connection)
    with caplog.at_level(logging.ERROR):
        retrieve_relevant_chunks(
            "what services do you offer", provider, db_connection, top_k=3, min_score=0.0
        )
    assert not [r for r in caplog.records if "mismatch" in r.message.lower()]

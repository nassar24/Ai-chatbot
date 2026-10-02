"""Cache-augmented generation.

The savings are the easy part. What these tests mostly cover is the
opposite: everything that must NOT be reused. A shared cache in front of
a per-visitor conversation is a correctness and privacy hazard, so each
gate in `app/rag/qa_cache.py` gets an explicit test here, exercised
through the real pipeline rather than against the cache in isolation —
the gates live at the pipeline seam, so testing the cache alone would
prove nothing about whether they're actually applied.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.kb.ingest import ingest_knowledge_base
from app.llm.base import ChatMessage
from app.rag.pipeline import answer_query
from app.rag.qa_cache import (
    QuestionAnswerCache,
    contains_contact_details,
    is_cacheable_question,
    normalize_question,
)
from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider

KB_PATH = Path(__file__).resolve().parent.parent / "knowledge_base.md"


@pytest.fixture()
def embedding_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture()
def ingested_kb(db_connection, embedding_provider):
    ingest_knowledge_base(KB_PATH.read_text(encoding="utf-8"), embedding_provider, db_connection)
    return db_connection


@pytest.fixture()
def cache(embedding_provider):
    return QuestionAnswerCache(
        kb_signature="",
        embedding_model=embedding_provider.model_name,
        llm_model="fake-llm",
    )


class CountingLLM(FakeLLMProvider):
    """FakeLLMProvider that reports how many times it was actually
    called — the whole point of the cache is that this stops going up."""

    def __init__(self, response: str):
        super().__init__(response=response)
        self.calls = 0

    def generate(self, system_prompt, messages, max_tokens=600):
        self.calls += 1
        return super().generate(system_prompt, messages, max_tokens)


# --- Pure helpers ----------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("What services do you offer?", "what services do you offer"),
        ("  WHAT   SERVICES do you offer ?  ", "what services do you offer"),
        ("What services do you offer!!", "what services do you offer!"),
    ],
)
def test_normalization_ignores_case_spacing_and_trailing_punctuation(a, b):
    assert normalize_question(a) == normalize_question(b)


@pytest.mark.parametrize(
    "text",
    [
        "my email is ahmed@example.com",
        "call me on +20 100 111 2222",
        "reach me at 01001112222",
    ],
)
def test_contact_details_are_detected(text):
    assert contains_contact_details(text) is True


@pytest.mark.parametrize(
    "text",
    ["what services do you offer", "do you work with clients outside Egypt"],
)
def test_ordinary_questions_are_not_flagged_as_contact_details(text):
    assert contains_contact_details(text) is False


def test_questions_with_contact_details_are_not_cacheable():
    assert is_cacheable_question("I'm Ahmed, my number is +20 100 111 2222, what do you do?") is False


def test_overly_long_questions_are_not_cacheable():
    assert is_cacheable_question("x" * 400) is False


# --- Through the pipeline: the savings -------------------------------------


def test_repeat_question_skips_the_llm_entirely(ingested_kb, embedding_provider, cache):
    llm = CountingLLM(response="We offer branding, web development and marketing.")

    first = answer_query(
        "What services do you offer?", embedding_provider, llm, ingested_kb,
        min_score=0.0, qa_cache=cache,
    )
    assert llm.calls == 1
    assert first.cache_hit == ""

    second = answer_query(
        "What services do you offer?", embedding_provider, llm, ingested_kb,
        min_score=0.0, qa_cache=cache,
    )
    assert llm.calls == 1, "the cached turn still called the LLM"
    assert second.cache_hit == "exact"
    assert second.answer == first.answer


def test_case_and_punctuation_variants_hit_the_same_entry(ingested_kb, embedding_provider, cache):
    llm = CountingLLM(response="We offer branding, web development and marketing.")

    answer_query("What services do you offer?", embedding_provider, llm, ingested_kb,
                 min_score=0.0, qa_cache=cache)
    repeat = answer_query("  what services do you OFFER  ", embedding_provider, llm, ingested_kb,
                          min_score=0.0, qa_cache=cache)

    assert llm.calls == 1
    assert repeat.cache_hit == "exact"


def test_a_different_question_does_not_hit_the_cache(ingested_kb, embedding_provider, cache):
    llm = CountingLLM(response="We offer branding, web development and marketing.")

    answer_query("What services do you offer?", embedding_provider, llm, ingested_kb,
                 min_score=0.0, qa_cache=cache)
    other = answer_query("What is your payment policy?", embedding_provider, llm, ingested_kb,
                         min_score=0.0, qa_cache=cache)

    assert llm.calls == 2
    assert other.cache_hit == ""


def test_re_ingesting_the_kb_retires_cached_answers(ingested_kb, embedding_provider, cache):
    """A changed knowledge base must invalidate prior answers on the very
    next turn — otherwise an edit made for a business reason keeps being
    contradicted by the cache."""
    llm = CountingLLM(response="We offer branding, web development and marketing.")

    answer_query("What services do you offer?", embedding_provider, llm, ingested_kb,
                 min_score=0.0, qa_cache=cache)
    assert llm.calls == 1

    edited = KB_PATH.read_text(encoding="utf-8") + "\n\n# Extra Section\nSome new fact.\n"
    ingest_knowledge_base(edited, embedding_provider, ingested_kb)

    after = answer_query("What services do you offer?", embedding_provider, llm, ingested_kb,
                         min_score=0.0, qa_cache=cache)

    assert after.cache_hit == "", "served an answer cached against the old KB"
    assert llm.calls == 2


# --- Through the pipeline: what must never be reused -----------------------


def test_follow_up_questions_are_never_cached(ingested_kb, embedding_provider, cache):
    """"What about their pricing?" is meaningless without the turns
    before it. Caching its answer would replay one visitor's context to
    another."""
    llm = CountingLLM(response="Branding covers strategy, logo design and guidelines.")
    history = [
        ChatMessage(role="user", content="Tell me about your branding service."),
        ChatMessage(role="assistant", content="Branding covers strategy and identity."),
    ]

    answer_query("what about that?", embedding_provider, llm, ingested_kb,
                 conversation_history=history, min_score=0.0, qa_cache=cache)
    second = answer_query("what about that?", embedding_provider, llm, ingested_kb,
                          conversation_history=history, min_score=0.0, qa_cache=cache)

    assert second.cache_hit == ""
    assert llm.calls == 2


def test_answers_echoing_visitor_supplied_numbers_are_never_cached(
    ingested_kb, embedding_provider, cache
):
    """The privacy gate. A phone number the visitor typed counts as
    grounded during their own conversation, so the answer passes
    guardrails live — but it must never be served to anyone else."""
    llm = CountingLLM(response="Got it — I've noted your number as 01234567890.")
    history = [
        ChatMessage(role="user", content="My number is 01234567890"),
        ChatMessage(role="assistant", content="Thanks."),
    ]

    first = answer_query(
        "Can you confirm what number you have for me on file?",
        embedding_provider, llm, ingested_kb,
        conversation_history=history, min_score=0.0, qa_cache=cache,
    )
    # The live turn is allowed to reflect it back...
    assert "01234567890" in first.answer

    # ...but nothing containing it may survive into the shared cache.
    with ingested_kb.cursor() as cursor:
        cursor.execute("SELECT answer FROM qa_cache")
        cached_answers = [row[0] for row in cursor.fetchall()]
    assert not any("01234567890" in answer for answer in cached_answers), (
        f"visitor PII leaked into the shared QA cache: {cached_answers}"
    )


def test_guardrail_blocked_answers_are_never_cached(ingested_kb, embedding_provider, cache):
    """A sanitized answer is a failure, not a result worth replaying."""
    llm = CountingLLM(response="Our special rate is 99,999 EGP for that.")

    result = answer_query(
        "What does a branding project cost?", embedding_provider, llm, ingested_kb,
        min_score=0.0, qa_cache=cache,
    )
    assert result.guardrail_violations, "expected the ungrounded number to be caught"

    with ingested_kb.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM qa_cache")
        assert cursor.fetchone()[0] == 0


def test_no_match_answers_are_never_cached(ingested_kb, embedding_provider, cache):
    """The no-match path never calls the LLM, so there is nothing to
    save — and caching it would freeze a KB gap in place."""
    llm = CountingLLM(response="unused")

    result = answer_query(
        "Who won the world cup in 1998?", embedding_provider, llm, ingested_kb,
        min_score=0.99, qa_cache=cache,
    )
    assert result.grounded is False

    with ingested_kb.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM qa_cache")
        assert cursor.fetchone()[0] == 0


def test_questions_containing_contact_details_are_never_cached(
    ingested_kb, embedding_provider, cache
):
    llm = CountingLLM(response="Thanks, our team will follow up.")

    answer_query(
        "I'm interested in branding, reach me at ahmed@example.com",
        embedding_provider, llm, ingested_kb, min_score=0.0, qa_cache=cache,
    )

    with ingested_kb.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM qa_cache")
        assert cursor.fetchone()[0] == 0


# --- The semantic threshold ------------------------------------------------
#
# Exercised with hand-built vectors rather than through the fake
# embedding provider: what's under test is the threshold logic itself,
# and real match quality is a property of the live embedding model that
# only `calibrate_threshold.py` against real traffic can tell you about.


# Must match the vector(768) column. An 8-wide vector is now rejected at
# INSERT rather than stored, and because Postgres aborts the transaction
# on error, that failure used to cascade into every later query in the
# same test.
_VECTOR_DIM = 768


def _unit_vector(dim: int = _VECTOR_DIM, tilt: float = 0.0) -> list[float]:
    """A vector that drifts smoothly away from [1, 0, 0, ...] as `tilt`
    grows, so cosine similarity can be dialed precisely."""
    vector = [0.0] * _VECTOR_DIM
    vector[0] = 1.0
    vector[1] = tilt
    return vector


def test_semantic_lookup_hits_a_near_identical_question(db_connection, cache):
    dim = _VECTOR_DIM
    cache.store(db_connection, "what services do you offer", _unit_vector(dim, 0.0),
                "We offer branding and web development.", grounded=True)

    hit = cache.lookup_semantic(db_connection, _unit_vector(dim, 0.05))

    assert hit is not None
    assert hit.kind == "semantic"
    assert hit.similarity >= cache.similarity_threshold
    assert hit.answer == "We offer branding and web development."


def test_semantic_lookup_misses_a_merely_related_question(db_connection, cache):
    """The failure that matters: a confidently-worded answer to a
    question the visitor didn't ask."""
    dim = _VECTOR_DIM
    cache.store(db_connection, "what services do you offer", _unit_vector(dim, 0.0),
                "We offer branding and web development.", grounded=True)

    assert cache.lookup_semantic(db_connection, _unit_vector(dim, 0.9)) is None


def test_semantic_lookup_can_be_disabled_leaving_exact_matching_intact(db_connection, cache):
    dim = _VECTOR_DIM
    cache.semantic_enabled = False
    cache.store(db_connection, "what services do you offer", _unit_vector(dim, 0.0),
                "We offer branding and web development.", grounded=True)

    assert cache.lookup_semantic(db_connection, _unit_vector(dim, 0.01)) is None
    # Exact matching is unaffected by the semantic switch.
    assert cache.lookup_exact(db_connection, "What services do you offer?") is not None


def test_entries_from_a_different_kb_signature_are_invisible(db_connection, cache):
    dim = _VECTOR_DIM
    cache.store(db_connection, "what services do you offer", _unit_vector(dim, 0.0),
                "We offer branding and web development.", grounded=True)

    cache.kb_signature = "a-different-kb-version"

    assert cache.lookup_exact(db_connection, "what services do you offer") is None
    assert cache.lookup_semantic(db_connection, _unit_vector(dim, 0.0)) is None


# --- Housekeeping ----------------------------------------------------------


def test_purge_removes_entries_from_a_previous_model(ingested_kb, embedding_provider, cache):
    llm = CountingLLM(response="We offer branding, web development and marketing.")
    answer_query("What services do you offer?", embedding_provider, llm, ingested_kb,
                 min_score=0.0, qa_cache=cache)

    with ingested_kb.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM qa_cache")
        assert cursor.fetchone()[0] == 1

    cache.llm_model = "some-newer-model"
    removed = cache.purge_stale(ingested_kb)

    assert removed == 1
    with ingested_kb.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM qa_cache")
        assert cursor.fetchone()[0] == 0


def test_cache_disabled_by_default_when_no_cache_is_passed(ingested_kb, embedding_provider):
    """Every existing caller passes no cache and must be unaffected."""
    llm = CountingLLM(response="We offer branding, web development and marketing.")

    answer_query("What services do you offer?", embedding_provider, llm, ingested_kb, min_score=0.0)
    answer_query("What services do you offer?", embedding_provider, llm, ingested_kb, min_score=0.0)

    assert llm.calls == 2
    with ingested_kb.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM qa_cache")
        assert cursor.fetchone()[0] == 0


# --- Personalized answers must never enter the shared cache ---------------
#
# Regression block. All three of these were found by running the stack
# end-to-end against the live model, not by unit tests: the reply
# "Hi Omar! It's great to meet you..." was written to the shared cache
# and would have greeted the next visitor by that name.


@pytest.mark.parametrize(
    "message",
    [
        "Hi, I'm Omar and I run a small cafe. I'm interested in branding.",
        "My name is Sara, what services do you offer?",
        "this is Ahmed — do you build websites?",
        "call me Youssef, I need branding",
    ],
)
def test_self_introductions_are_not_cacheable(message):
    assert is_cacheable_question(message) is False


@pytest.mark.parametrize(
    "message",
    [
        "What services do you offer?",
        "Do you build websites?",
        "I'm interested in branding for a cafe",  # no name -> still cacheable
    ],
)
def test_ordinary_questions_remain_cacheable(message):
    assert is_cacheable_question(message) is True


def test_answer_greeting_the_visitor_by_name_is_not_cached(
    ingested_kb, embedding_provider, cache
):
    """The exact end-to-end failure: a first-turn self-introduction whose
    answer opens with the visitor's first name."""
    llm = CountingLLM(response="Hi Omar! Branding is a great step for a cafe.")

    answer_query(
        "Hi, I'm Omar and I run a small cafe. I'm interested in branding.",
        embedding_provider, llm, ingested_kb, min_score=0.0, qa_cache=cache,
    )

    with ingested_kb.cursor() as cursor:
        cursor.execute("SELECT answer FROM qa_cache")
        cached = [row[0] for row in cursor.fetchall()]
    assert not any("Omar" in answer for answer in cached), (
        f"a visitor's name reached the shared cache: {cached}"
    )


def test_clean_mid_conversation_answers_are_still_cacheable(
    ingested_kb, embedding_provider, cache
):
    """Coverage matters too. An earlier version refused to cache anything
    with history behind it, which was safe but meant only the opening
    question of a conversation was ever stored — most of the cache's
    value, gone. A context-free question whose answer names nobody is
    fine to reuse regardless of what turn it arrived on."""
    llm = CountingLLM(response="Branding covers strategy, identity and guidelines.")
    history = [
        ChatMessage(role="user", content="Hi there."),
        ChatMessage(role="assistant", content="Hello, how can I help?"),
    ]

    answer_query(
        "What does your branding service include?",
        embedding_provider, llm, ingested_kb,
        conversation_history=history, min_score=0.0, qa_cache=cache,
    )

    with ingested_kb.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM qa_cache")
        assert cursor.fetchone()[0] == 1


def test_mid_conversation_answer_naming_the_visitor_is_not_cached(
    ingested_kb, embedding_provider, cache
):
    """The hole the first-turn rule used to cover, now closed by checking
    the answer's content instead: a lone name mid-sentence, which the
    run-based proper-noun rule never matches."""
    llm = CountingLLM(response="Thanks, Omar - I'll pass your details to the team.")
    history = [
        ChatMessage(role="user", content="Hi, I'm Omar."),
        ChatMessage(role="assistant", content="Hello, how can I help?"),
    ]

    answer_query(
        "What does your branding service include?",
        embedding_provider, llm, ingested_kb,
        conversation_history=history, min_score=0.0, qa_cache=cache,
    )

    with ingested_kb.cursor() as cursor:
        cursor.execute("SELECT answer FROM qa_cache")
        cached = [r[0] for r in cursor.fetchall()]
    assert not any("Omar" in a for a in cached), f"name reached the cache: {cached}"

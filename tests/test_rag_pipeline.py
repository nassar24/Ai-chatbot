"""Tests the standalone RAG loop (retrieve -> ground prompt -> generate)
against a real MySQL-compatible database, using fake embedding/LLM
providers so no live API keys are needed. Verifies pipeline wiring, not
real answer quality (that needs live Voyage + OpenRouter keys).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.kb.ingest import ingest_knowledge_base
from app.llm.base import ChatMessage
from app.rag.pipeline import NO_MATCH_RESPONSE, answer_query
from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider

KB_PATH = Path(__file__).resolve().parent.parent / "knowledge_base.md"

# db_connection fixture lives in tests/conftest.py (shared with
# test_ingest_and_retrieval_integration.py) — it also guards against
# running against a non-test database. See conftest.py for why.


@pytest.fixture()
def embedding_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture()
def llm_provider() -> FakeLLMProvider:
    return FakeLLMProvider(response="This is the grounded answer.")


@pytest.fixture()
def ingested_kb(db_connection, embedding_provider):
    kb_text = KB_PATH.read_text(encoding="utf-8")
    ingest_knowledge_base(kb_text, embedding_provider, db_connection)
    return db_connection


def test_answer_query_rejects_empty_query(ingested_kb, embedding_provider, llm_provider):
    with pytest.raises(ValueError):
        answer_query("   ", embedding_provider, llm_provider, ingested_kb)


def test_no_relevant_chunks_returns_fallback_without_calling_llm(
    ingested_kb, embedding_provider, llm_provider
):
    # Nonsense tokens absent from the whole KB vocabulary -> zero cosine
    # similarity against every chunk under the fake bag-of-words provider.
    result = answer_query(
        "zqxjklm vbnwplk fjhqzxo", embedding_provider, llm_provider, ingested_kb
    )

    assert result.grounded is False
    assert result.answer == NO_MATCH_RESPONSE
    assert result.retrieved_chunks == []
    assert llm_provider.last_system_prompt is None  # LLM must not be called with no context


def test_relevant_query_grounds_prompt_in_retrieved_content(
    ingested_kb, embedding_provider, llm_provider
):
    result = answer_query(
        "What is your refund policy?",
        embedding_provider,
        llm_provider,
        ingested_kb,
        min_score=0.0,  # the fake embedding's cosine scores run lower than a real model's
    )

    assert result.grounded is True
    assert result.answer == "This is the grounded answer."
    assert any(c.section_title == "Company Policies — Refund Policy" for c in result.retrieved_chunks)

    # The actual KB content (not a paraphrase) must appear in what was sent to the LLM.
    assert llm_provider.last_system_prompt is not None
    assert "non-refundable" in llm_provider.last_system_prompt
    assert llm_provider.last_messages == [
        ChatMessage(role="user", content="What is your refund policy?")
    ]


def test_system_prompt_includes_number_grounding_rule(
    ingested_kb, embedding_provider, llm_provider
):
    answer_query(
        "What is your refund policy?",
        embedding_provider,
        llm_provider,
        ingested_kb,
        min_score=0.0,
    )

    assert "NUMBERS ARE THE HIGHEST-RISK CATEGORY" in llm_provider.last_system_prompt


def test_system_prompt_includes_real_identity_section(
    ingested_kb, embedding_provider, llm_provider
):
    answer_query(
        "What is your refund policy?",
        embedding_provider,
        llm_provider,
        ingested_kb,
        min_score=0.0,
    )
    assert "Apex Creative AI Assistant" in llm_provider.last_system_prompt


def test_pipeline_catches_ungrounded_number_from_llm(ingested_kb, embedding_provider):
    # Simulates the model hallucinating a price that isn't in the KB.
    bad_llm = FakeLLMProvider(response="Our Starter Package costs 999,999 EGP.")
    result = answer_query(
        "How much does the starter package cost?",
        embedding_provider,
        bad_llm,
        ingested_kb,
        min_score=0.0,
    )
    assert result.grounded is True
    assert "999,999" not in result.answer
    assert any("ungrounded_numbers" in v for v in result.guardrail_violations)


def test_pipeline_catches_refund_percentage_from_llm(ingested_kb, embedding_provider):
    bad_llm = FakeLLMProvider(response="We can refund 50% of your payment.")
    result = answer_query(
        "What is your refund policy?",
        embedding_provider,
        bad_llm,
        ingested_kb,
        min_score=0.0,
    )
    assert result.grounded is True
    assert "50%" not in result.answer
    assert "refund_percentage_stated" in result.guardrail_violations


def test_pipeline_redacts_internal_email_from_llm(ingested_kb, embedding_provider):
    bad_llm = FakeLLMProvider(
        response="I'll notify our team at hr@apexcreative.example about your request."
    )
    result = answer_query(
        "I'd like to file a complaint.",
        embedding_provider,
        bad_llm,
        ingested_kb,
        min_score=0.0,
    )
    assert result.grounded is True
    assert "hr@apexcreative.example" not in result.answer
    assert any("internal_email_leak" in v for v in result.guardrail_violations)


# --- Phase 1: chat history -------------------------------------------------


def test_history_reaches_llm_provider_correctly(ingested_kb, embedding_provider, llm_provider):
    history = [
        ChatMessage(role="user", content="What is your refund policy?"),
        ChatMessage(role="assistant", content="Our packages are non-refundable once work begins."),
    ]
    answer_query(
        "Can you say more about that?",
        embedding_provider,
        llm_provider,
        ingested_kb,
        conversation_history=history,
        min_score=0.0,
    )

    # The prior turns must be threaded in ahead of the current message,
    # unmodified, in order.
    assert llm_provider.last_messages == [
        *history,
        ChatMessage(role="user", content="Can you say more about that?"),
    ]


def test_empty_history_behaves_identically_to_no_history_arg(
    ingested_kb, embedding_provider, llm_provider
):
    # Regression guard: passing conversation_history=[] (or omitting it)
    # must produce the exact same messages list sent to the LLM as before
    # Phase 1 existed.
    answer_query(
        "What is your refund policy?",
        embedding_provider,
        llm_provider,
        ingested_kb,
        min_score=0.0,
    )
    without_arg = llm_provider.last_messages

    answer_query(
        "What is your refund policy?",
        embedding_provider,
        llm_provider,
        ingested_kb,
        conversation_history=[],
        min_score=0.0,
    )
    with_empty_list = llm_provider.last_messages

    assert without_arg == with_empty_list == [
        ChatMessage(role="user", content="What is your refund policy?")
    ]


def test_retrieval_augmentation_resolves_pronoun_only_follow_up(
    ingested_kb, embedding_provider, llm_provider
):
    # "what about their pricing?" carries almost no content words of its
    # own — without folding in the prior turn's topic, hybrid retrieval
    # (vector + fuzzy title match) has nothing to anchor on. With the
    # prior turn folded in, both the vector similarity and the keyword
    # match against "01. Software & Web Development" (whose own content
    # includes a **Pricing:** line) should win out over generic
    # pricing-adjacent chunks like "FAQ: Pricing & Cost" or "Payment Policy".
    history = [
        ChatMessage(
            role="user",
            content="Tell me about your software and web development services",
        ),
    ]
    result = answer_query(
        "what about their pricing?",
        embedding_provider,
        llm_provider,
        ingested_kb,
        conversation_history=history,
        min_score=0.0,
    )

    assert result.retrieved_chunks, "expected at least one retrieved chunk"
    assert result.retrieved_chunks[0].section_title == "Our Services — 01. Software & Web Development"


def test_retrieval_augmentation_skipped_for_content_rich_follow_up(
    ingested_kb, embedding_provider, llm_provider
):
    # Regression guard for the exact risk the plan called out: folding
    # history into retrieval could reintroduce the "about"-style false-
    # positive keyword match, just via history text instead of a single
    # query. Here the prior turn incidentally mentions a team member
    # ("Hossam") who has nothing to do with the follow-up. The follow-up
    # itself is long and content-word-heavy (>=3 non-stopword tokens), so
    # it's already self-sufficient for retrieval and history must NOT be
    # folded in — otherwise "Hossam" gets an 0.85 keyword-match bonus
    # against "Our Team — Hossam — Cybersecurity Specialist" for a query
    # that has nothing to do with him.
    history = [
        ChatMessage(
            role="user",
            content="My colleague Hossam recommended I check you out.",
        ),
    ]
    result = answer_query(
        "How long does a typical website development project take?",
        embedding_provider,
        llm_provider,
        ingested_kb,
        conversation_history=history,
        min_score=0.0,
    )

    assert result.retrieved_chunks, "expected at least one retrieved chunk"
    returned_titles = [c.section_title for c in result.retrieved_chunks]
    assert not any("Hossam" in title for title in returned_titles)


def test_build_retrieval_query_unit_pronoun_vs_content_rich():
    # Pure-function check (no DB needed) isolating the gating logic itself,
    # independent of the fake embedder's cosine-similarity behavior.
    from app.rag.pipeline import _build_retrieval_query

    history = [ChatMessage(role="user", content="Tell me about your web development services")]

    # Thin, pronoun-only follow-up -> gets folded.
    folded = _build_retrieval_query("what about their pricing?", history)
    assert folded == "Tell me about your web development services what about their pricing?"

    # Long, content-word-heavy follow-up -> left untouched.
    standalone_query = "How long does a typical website development project usually take?"
    unfolded = _build_retrieval_query(standalone_query, history)
    assert unfolded == standalone_query

    # No history at all -> always untouched, regardless of query shape.
    assert _build_retrieval_query("what about their pricing?", []) == "what about their pricing?"


def test_history_token_budget_caps_by_tokens_not_message_count():
    # A 6-message cap would treat six one-word messages the same as six
    # paragraphs. This checks the actual behavior: a long-enough message
    # should get dropped even if it's within a small message count, and a
    # single very long message should never be dropped down to nothing.
    from app.rag.pipeline import _cap_history_to_token_budget

    short_history = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
    ]
    # Budget is generous relative to a couple of one-word messages -> both kept.
    assert _cap_history_to_token_budget(short_history, token_budget=50) == short_history

    long_message = ChatMessage(role="user", content="word " * 400)  # ~500 estimated tokens
    history_with_one_huge_turn = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
        long_message,
    ]
    capped = _cap_history_to_token_budget(history_with_one_huge_turn, token_budget=50)
    # The huge message alone exceeds the budget, but must still be kept
    # (most recent turn is never dropped to empty), while the older,
    # now-unaffordable turns get dropped.
    assert capped == [long_message]

    # Small cap with a single message that fits should NOT return empty.
    assert _cap_history_to_token_budget([ChatMessage(role="user", content="hi")], token_budget=1) == [
        ChatMessage(role="user", content="hi")
    ]

# --- Retrieval ordering: plain question first --------------------------


def _count_retrievals(monkeypatch, plain_top_score, augmented_top_score):
    """Runs answer_query with retrieval stubbed, returning the queries it
    was actually asked to retrieve on."""
    from app.kb.retrieval import RetrievedChunk
    import app.rag.pipeline as pipeline

    seen = []

    def fake_retrieve(query, provider, conn, **kwargs):
        seen.append(query)
        score = plain_top_score if len(seen) == 1 else augmented_top_score
        return [
            RetrievedChunk(
                id=1, section_title="Our Team — Team Overview",
                content="The team is ten specialists.", score=score,
            )
        ]

    monkeypatch.setattr(pipeline, "retrieve_relevant_chunks", fake_retrieve)
    return seen


def test_confident_plain_question_skips_the_history_retrieval(monkeypatch):
    """A short but self-sufficient question ("Who is on your team?" is one
    content word) is judged thin and has history folded in. When it
    retrieves confidently on its own, the second call is wasted."""
    seen = _count_retrievals(monkeypatch, plain_top_score=0.80, augmented_top_score=0.40)
    history = [
        ChatMessage(role="user", content="What services do you offer?"),
        ChatMessage(role="assistant", content="Eleven services."),
    ]
    answer_query(
        "Who is on your team?", FakeEmbeddingProvider(), FakeLLMProvider(), None,
        conversation_history=history,
    )
    assert seen == ["Who is on your team?"], (
        "a confident plain question must not pay for a second retrieval"
    )


def test_weak_plain_question_still_falls_back_to_history(monkeypatch):
    """The pronoun follow-up case augmentation exists for: the plain query
    carries no topic, so history has to be consulted."""
    seen = _count_retrievals(monkeypatch, plain_top_score=0.38, augmented_top_score=0.72)
    history = [
        ChatMessage(role="user", content="Do you do branding?"),
        ChatMessage(role="assistant", content="Yes."),
    ]
    answer_query(
        "How long does it take?", FakeEmbeddingProvider(), FakeLLMProvider(), None,
        conversation_history=history,
    )
    assert len(seen) == 2
    assert seen[0] == "How long does it take?"
    assert "Do you do branding?" in seen[1]


# --- Clarifying a follow-up when retrieval finds nothing ---------------
#
# From a real transcript: an Arabic pricing conversation, then three turns
# in a row answered with the canned "I don't have that information
# available right now". Two of them ("يعنى ايه", "ايه المشكلة") were asking
# what the bot had just said — questions no knowledge base section can
# answer, so retrieval was always going to come back empty.


def _no_chunks(monkeypatch):
    import app.rag.pipeline as pipeline

    monkeypatch.setattr(pipeline, "retrieve_relevant_chunks", lambda *a, **k: [])


def test_contentless_follow_up_is_clarified_from_the_conversation(monkeypatch):
    _no_chunks(monkeypatch)
    llm = FakeLLMProvider()
    history = [
        ChatMessage(role="user", content="what does a website cost?"),
        ChatMessage(role="assistant", content="Pricing depends on the scope of work."),
    ]
    result = answer_query(
        "what do you mean?", FakeEmbeddingProvider(), llm, None,
        conversation_history=history,
    )
    assert result.answer != NO_MATCH_RESPONSE
    assert "NO KNOWLEDGE BASE CONTENT" in llm.last_system_prompt


def test_clarification_walks_back_past_a_canned_reply(monkeypatch):
    """The transcript's actual shape: the message immediately before the
    follow-up was itself a canned no-match, so clarifying it would be
    circular. The turn worth clarifying is further back."""
    from app.rag.pipeline import _last_substantive_assistant_turn

    history = [
        ChatMessage(role="user", content="ايه الاسعار"),
        ChatMessage(role="assistant", content="اسعار المواقع بتعتمد على نطاق الشغل"),
        ChatMessage(role="user", content="مش عايز اشارك بياناتى"),
        ChatMessage(role="assistant", content=NO_MATCH_RESPONSE),
    ]
    assert _last_substantive_assistant_turn(history) == "اسعار المواقع بتعتمد على نطاق الشغل"


def test_no_clarification_when_the_assistant_has_said_nothing_real(monkeypatch):
    """Every assistant turn so far is canned, so there is genuinely
    nothing to clarify and the canned reply is the correct answer."""
    _no_chunks(monkeypatch)
    history = [
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content=NO_MATCH_RESPONSE),
    ]
    result = answer_query(
        "what do you mean?", FakeEmbeddingProvider(), FakeLLMProvider(), None,
        conversation_history=history,
    )
    assert result.answer == NO_MATCH_RESPONSE
    assert result.grounded is False


def test_a_question_with_its_own_content_still_gets_the_hard_stop(monkeypatch):
    """The exception is only for turns with no topic. A real question that
    retrieval could not answer must NOT be answered from memory."""
    _no_chunks(monkeypatch)
    history = [
        ChatMessage(role="user", content="what does a website cost?"),
        ChatMessage(role="assistant", content="Pricing depends on the scope of work."),
    ]
    result = answer_query(
        "do you offer drone videography for weddings?",
        FakeEmbeddingProvider(), FakeLLMProvider(), None,
        conversation_history=history,
    )
    assert result.answer == NO_MATCH_RESPONSE


def test_no_history_means_no_clarification(monkeypatch):
    _no_chunks(monkeypatch)
    result = answer_query(
        "what do you mean?", FakeEmbeddingProvider(), FakeLLMProvider(), None,
    )
    assert result.answer == NO_MATCH_RESPONSE

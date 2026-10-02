"""FallbackEmbeddingProvider — switching to a local model when the primary
provider's daily quota runs out, and ONLY then.

The distinction these tests pin down is what counts as "out of quota".
Switching on any error would quietly degrade retrieval quality for a
single slow request; not switching on a quota error leaves the bot
answering nothing for hours, which is what actually happened.
"""

from __future__ import annotations

import pytest

from app.embeddings.fallback import FallbackEmbeddingProvider


class _StubProvider:
    def __init__(self, name, dim, error=None):
        self._name = name
        self._dim = dim
        self.error = error
        self.calls = 0

    @property
    def model_name(self):
        return self._name

    @property
    def dimension(self):
        return self._dim

    def embed_query(self, text):
        self.calls += 1
        if self.error:
            raise self.error
        return [0.1] * self._dim

    def embed_documents(self, texts):
        self.calls += 1
        if self.error:
            raise self.error
        return [[0.1] * self._dim for _ in texts]


QUOTA_ERROR = RuntimeError(
    "429 RESOURCE_EXHAUSTED. You exceeded your current quota. "
    "quotaId: EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier"
)


def _pair(primary_error=None, cooldown=3600):
    primary = _StubProvider("gemini-embedding-001", 768, primary_error)
    fallback = _StubProvider("BAAI/bge-m3", 1024)
    return primary, fallback, FallbackEmbeddingProvider(primary, fallback, cooldown)


def test_primary_is_used_when_it_works():
    primary, fallback, provider = _pair()
    assert len(provider.embed_query("hi")) == 768
    assert provider.model_name == "gemini-embedding-001"
    assert provider.using_fallback is False
    assert fallback.calls == 0


def test_quota_error_switches_to_the_fallback():
    primary, fallback, provider = _pair(QUOTA_ERROR)
    assert len(provider.embed_query("hi")) == 1024
    assert provider.using_fallback is True
    assert fallback.calls == 1


def test_model_name_reports_who_actually_embedded():
    """Load-bearing: retrieval reads this to choose which stored vector set
    to score against. Reporting the primary's name while the fallback did
    the work would score bge-m3 vectors against Gemini chunks — the exact
    silent-nonsense failure this whole mechanism exists to prevent."""
    primary, fallback, provider = _pair(QUOTA_ERROR)
    provider.embed_query("hi")
    assert provider.model_name == "BAAI/bge-m3"
    assert provider.dimension == 1024


def test_non_quota_errors_are_raised_not_swallowed():
    """A timeout or a 500 is transient and the caller already retries.
    Silently dropping to a weaker model for one blip would trade a moment
    of latency for worse answers."""
    primary, fallback, provider = _pair(TimeoutError("connection timed out"))
    with pytest.raises(TimeoutError):
        provider.embed_query("hi")
    assert fallback.calls == 0
    assert provider.using_fallback is False


def test_primary_is_not_retried_during_the_cooldown():
    """The triggering quota is a DAILY one, so retrying every request would
    spend the turn budget on failures and add the primary's latency to
    every answer."""
    primary, fallback, provider = _pair(QUOTA_ERROR)
    for _ in range(5):
        provider.embed_query("hi")
    assert primary.calls == 1
    assert fallback.calls == 5


def test_primary_is_reclaimed_once_the_cooldown_expires():
    """Recovery has to be automatic — nobody should need to restart the app
    when the quota resets at midnight."""
    primary, fallback, provider = _pair(QUOTA_ERROR, cooldown=0)
    provider.embed_query("hi")
    assert provider.using_fallback is True

    primary.error = None  # quota reset
    assert len(provider.embed_query("hi")) == 768
    assert provider.using_fallback is False
    assert provider.model_name == "gemini-embedding-001"


def test_documents_take_the_same_path_as_queries():
    primary, fallback, provider = _pair(QUOTA_ERROR)
    vectors = provider.embed_documents(["a", "b"])
    assert len(vectors) == 2
    assert all(len(v) == 1024 for v in vectors)


@pytest.mark.parametrize(
    "message",
    [
        "429 RESOURCE_EXHAUSTED",
        "You exceeded your current quota, please check your plan",
        "Quota exceeded for metric: embed_content_free_tier_requests",
        "rate limit reached",
    ],
)
def test_quota_wording_variants_are_all_recognised(message):
    """Matched on text because each provider SDK raises its own type."""
    primary, fallback, provider = _pair(RuntimeError(message))
    provider.embed_query("hi")
    assert provider.using_fallback is True


# --- The pipeline path that a fallback actually takes ------------------


def test_fallback_turns_still_use_the_cache_scoped_to_their_own_model(
    db_connection,
):
    """Two regressions in one path.

    First: this branch once carried a NameError, and nothing covered it —
    the fallback tests drove the provider directly without going through
    the pipeline, and the pipeline tests never used a provider whose model
    name differed from the cache's. It reached a running server and
    returned 502 to every visitor asking a fresh question.

    Second: the fix for THAT disabled caching whenever the fallback was
    answering, which is backwards. A quota outage lasts hours and is
    exactly when the cache matters most. Entries are now scoped to the
    model that produced them, so the two providers keep separate sets
    instead of one poisoning the other.
    """
    from pathlib import Path
    from unittest.mock import MagicMock

    from app.kb.ingest import ingest_knowledge_base
    from app.rag.pipeline import answer_query
    from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider

    class _FallbackNamed(FakeEmbeddingProvider):
        """Reports a model name the cache was NOT built with, the way the
        real fallback provider does once it has taken over."""

        @property
        def model_name(self):
            return "BAAI/bge-m3"

    kb = Path(__file__).resolve().parent.parent / "knowledge_base.md"
    provider = _FallbackNamed()
    ingest_knowledge_base(kb.read_text(encoding="utf-8"), provider, db_connection)

    qa_cache = MagicMock()
    qa_cache.embedding_model = "gemini-embedding-001"
    qa_cache.lookup_exact.return_value = None
    qa_cache.lookup_semantic.return_value = None

    result = answer_query(
        "What services do you offer?",
        provider,
        FakeLLMProvider(),
        db_connection,
        qa_cache=qa_cache,
    )

    assert result.answer

    # Consulted, not skipped — and told which model is asking.
    assert qa_cache.lookup_semantic.call_args.kwargs["embedding_model"] == "BAAI/bge-m3"
    assert qa_cache.lookup_exact.call_args.kwargs["embedding_model"] == "BAAI/bge-m3"

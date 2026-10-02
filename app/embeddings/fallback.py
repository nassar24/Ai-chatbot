"""An embedding provider that answers from a local model when the primary
provider's quota is exhausted — and only then.

WHY
---
The Gemini free tier allows 1,000 embed requests per DAY
(EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier). Every chat
turn spends at least one, so once it runs out the bot stops answering
anything at all: retrieval cannot embed the question, and /api/chat
returns 502 for every visitor until the quota resets at midnight Pacific.
That happened, and the site was dark for the rest of the day.

WHAT THIS IS NOT
----------------
This is NOT a general "try local if Gemini is slow or erroring" wrapper.
It switches on quota exhaustion specifically, because that is the failure
that lasts for hours and that a retry cannot fix. A timeout, a 500, or a
network blip is transient and is left to the caller's own retry — quietly
degrading retrieval quality because one request was slow would be a worse
trade than a moment's latency.

THE THING THAT MAKES THIS SAFE
------------------------------
Swapping the model swaps the VECTOR SPACE. Chunks embedded by Gemini and a
query embedded by bge-m3 are not comparable, and comparing them does not
fail loudly — it returns confidently-ranked nonsense. This project has
already been burned by exactly that once, when a test fake was aligned to
768 dimensions and became indistinguishable from the real model, leaving
the index full of garbage vectors with nothing in the logs.

So the fallback only works because the knowledge base is embedded TWICE
(see migrate_fallback_vectors.py and scripts/build_fallback_index.py), and
because `model_name` below always reports the model that produced the most
recent vector. Retrieval reads that name to decide which stored vector set
to score against. If the fallback index has not been built, retrieval
finds no chunks in that space and says so loudly rather than guessing.

MEASURED COST OF DEGRADED MODE
------------------------------
bge-m3 against gemini-embedding-001 on the same 69-query eval: hit@1 74.6%
vs 78.0%, MRR 0.803 vs 0.848, Arabic 10/12 vs 11/12, off-topic rejection
7/10 vs 9/10. Noticeably worse, clearly usable, and far better than the
alternative of answering nothing. See eval/embedding_comparison.md.
"""

from __future__ import annotations

import logging
import os
import threading
import time

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)

# How long to stay on the fallback before testing the primary again.
#
# The quota that triggers this is a DAILY one, so retrying every request
# would spend the whole turn budget on failing calls and add the primary's
# latency to every answer. An hour keeps the recovery automatic — nobody
# has to restart anything when the quota resets — while making the retries
# themselves negligible.
_DEFAULT_COOLDOWN_SECONDS = 3600

# Substrings that mean "quota", not "something went wrong". Matched against
# the exception text because the provider SDKs raise their own error types
# and this wrapper should not have to import each one.
_QUOTA_MARKERS = (
    "resource_exhausted",
    "exceeded your current quota",
    "quota exceeded",
    "rate limit",
    "429",
)


def _is_quota_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


class FallbackEmbeddingProvider(EmbeddingProvider):
    """Delegates to `primary`, switching to `fallback` on quota errors.

    `model_name` reflects whichever provider served the most recent call,
    so retrieval can pick the matching stored vectors. Callers should embed
    first and read `model_name` after, which is the order the RAG pipeline
    already uses.
    """

    def __init__(
        self,
        primary: EmbeddingProvider,
        fallback: EmbeddingProvider,
        cooldown_seconds: float | None = None,
    ):
        self._primary = primary
        self._fallback = fallback
        self._cooldown = (
            cooldown_seconds
            if cooldown_seconds is not None
            else float(os.environ.get("EMBEDDING_FALLBACK_COOLDOWN_SECONDS",
                                      _DEFAULT_COOLDOWN_SECONDS))
        )
        self._lock = threading.Lock()
        self._primary_blocked_until = 0.0
        self._active = primary

    # --- state ---------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._active.model_name

    @property
    def dimension(self) -> int:
        """The dimension of the provider currently serving requests.

        Deliberately not a fixed number: the two providers differ (768 vs
        1024), and reporting the primary's while the fallback is answering
        would misdescribe the vectors actually being produced.
        """
        return self._active.dimension

    @property
    def using_fallback(self) -> bool:
        return self._active is self._fallback

    def _primary_available(self) -> bool:
        with self._lock:
            return time.monotonic() >= self._primary_blocked_until

    def _block_primary(self, exc: Exception) -> None:
        with self._lock:
            first_time = time.monotonic() >= self._primary_blocked_until
            self._primary_blocked_until = time.monotonic() + self._cooldown
        if first_time:
            # Warning, not error: the bot is still answering. But it must be
            # visible, because answers are measurably worse until the quota
            # returns and nothing else would say why.
            logger.warning(
                "Primary embedding provider %r is out of quota (%s). Falling "
                "back to %r for the next %.0f seconds — retrieval quality is "
                "degraded until then (see eval/embedding_comparison.md).",
                self._primary.model_name, exc, self._fallback.model_name,
                self._cooldown,
            )

    # --- delegation ----------------------------------------------------

    def _call(self, method: str, *args):
        if self._primary_available():
            try:
                result = getattr(self._primary, method)(*args)
                with self._lock:
                    self._active = self._primary
                return result
            except Exception as exc:  # noqa: BLE001 - re-raised unless quota
                if not _is_quota_error(exc):
                    raise
                self._block_primary(exc)
        else:
            logger.debug("Primary embedding provider still in cooldown.")

        result = getattr(self._fallback, method)(*args)
        with self._lock:
            self._active = self._fallback
        return result

    def embed_query(self, text: str) -> list[float]:
        return self._call("embed_query", text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._call("embed_documents", texts)

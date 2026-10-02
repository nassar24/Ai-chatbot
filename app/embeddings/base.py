"""Embedding provider contract.

Nothing in app.kb (chunker, ingest, retrieval) depends on a specific
embedding vendor — only on this interface. That's what makes swapping
Voyage for OpenAI (or anything else) a one-file change instead of a
rewrite.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier stored alongside each embedding, e.g. 'voyage-3-lite'.

        Used to detect when a model change means existing embeddings are
        stale and need a full re-embed rather than an incremental one.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector length this provider produces, for storage validation."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds knowledge-base content for storage."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embeds a visitor's message for retrieval.

        Kept separate from embed_documents because some providers (Voyage
        included) use different encodings for queries vs. documents to
        improve retrieval quality — collapsing the two into one method
        would silently degrade relevance.
        """

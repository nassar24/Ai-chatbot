"""Voyage AI embedding provider.

Default choice for this project (see build-plan discussion) because it's
a plain HTTPS API call — no compiled binaries, no GPU, nothing that would
be awkward on Hostinger shared hosting — and Anthropic recommends Voyage
embeddings for Claude-based RAG.

Model default: voyage-4-lite. The voyage-4 family (lite/base/large) is
the current generation and the one that qualifies for Voyage's 200M
free-token tier; the older voyage-3.x models no longer get that
allocation. All voyage-4 models default to 1024 dimensions (Matryoshka:
256/512/1024/2048 also available, not used here).
"""

from __future__ import annotations

import os

from .base import EmbeddingProvider

_MODEL_DIMENSIONS = {
    "voyage-4-lite": 1024,
    "voyage-4": 1024,
    "voyage-4-large": 1024,
    # Legacy models — still callable, but no longer covered by the free tier.
    "voyage-3-lite": 512,
    "voyage-3": 1024,
    "voyage-3-large": 1024,
}

_DEFAULT_MODEL = "voyage-4-lite"


class VoyageEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        resolved_model = model or os.environ.get("VOYAGE_MODEL", _DEFAULT_MODEL)
        if resolved_model not in _MODEL_DIMENSIONS:
            raise ValueError(
                f"Unknown Voyage model '{resolved_model}'. Supported: "
                f"{', '.join(_MODEL_DIMENSIONS)}"
            )

        resolved_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "VOYAGE_API_KEY is not set. Provide it as an environment "
                "variable — never hardcode API keys in source."
            )

        try:
            import voyageai
        except ImportError as exc:
            raise RuntimeError(
                "The 'voyageai' package is not installed. Add it to "
                "requirements.txt (pip install voyageai)."
            ) from exc

        self._model = resolved_model
        self._client = voyageai.Client(api_key=resolved_key)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return _MODEL_DIMENSIONS[self._model]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = self._client.embed(texts, model=self._model, input_type="document")
        return result.embeddings

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Query text must be non-empty.")
        result = self._client.embed([text], model=self._model, input_type="query")
        return result.embeddings[0]

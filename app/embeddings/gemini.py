"""Google Gemini embedding provider (gemini-embedding-001).

Alternative to Voyage, chosen specifically because Google AI Studio's
free tier (1,500 requests/day) needs no credit card or billing account
at all — unlike Voyage, which throttles to 3 RPM until a card is on
file, even within the free-token allowance.

Trade-offs vs Voyage, worth knowing before switching:
- Different vector space entirely. Embeddings from this provider and
  Voyage are not interchangeable — switching means re-ingesting the
  whole KB, not a config change.
- gemini-embedding-001 only accepts one input text per request (no
  batch embedding call), so embed_documents() loops one call per chunk.
  Fine at this KB's scale (44 chunks); would need batching logic to
  scale much further.
"""

from __future__ import annotations

import os

from .base import EmbeddingProvider

_DEFAULT_MODEL = "gemini-embedding-001"
# Matryoshka-reduced from the model's 3072-dim default — plenty of
# resolution for a KB this size, smaller MySQL LONGBLOB storage.
_DEFAULT_DIMENSION = 768


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        output_dimension: int | None = None,
    ):
        resolved_model = model or os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL)
        resolved_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not resolved_key:
            raise RuntimeError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set. Provide it "
                "as an environment variable — never hardcode API keys in "
                "source. Get a free key (no billing required) at "
                "aistudio.google.com/apikey."
            )

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "The 'google-genai' package is not installed. Add it to "
                "requirements.txt (pip install google-genai)."
            ) from exc

        self._model = resolved_model
        self._dimension = output_dimension or int(
            os.environ.get("GEMINI_EMBEDDING_DIMENSION", str(_DEFAULT_DIMENSION))
        )
        self._types = types
        self._client = genai.Client(api_key=resolved_key)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [self._embed_one(text, task_type="RETRIEVAL_DOCUMENT") for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Query text must be non-empty.")
        return self._embed_one(text, task_type="RETRIEVAL_QUERY")

    def _embed_one(self, text: str, task_type: str) -> list[float]:
        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=self._types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self._dimension,
            ),
        )
        return result.embeddings[0].values

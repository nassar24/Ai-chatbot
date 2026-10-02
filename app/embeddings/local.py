"""Local sentence-transformer embeddings, run in-process on CPU.

Same `EmbeddingProvider` interface as the Gemini and Voyage providers, so
swapping is a one-line change at the call site and nothing in app/kb/
changes.

WHY THIS EXISTS - and what it is NOT for
----------------------------------------
Not for Arabic quality. That was the assumption going in, and measuring
it killed the assumption: gemini-embedding-001 scores 11/12 hit@1 on the
Arabic eval set, the best of any category and better than English
paraphrase. Arabic retrieval was never the weak part.

The real reasons are operational:

  - QUOTA. The free tier allows 100 embed requests per minute. A single
    re-ingest of this KB spends 52 of them, and that ceiling was hit
    twice during development. It is a hard cap on how many visitors can
    ask a question in a minute.
  - LATENCY. A network round trip to Gemini is 300ms-3s. Local CPU
    inference on a KB this size is tens of milliseconds.
  - COST and privacy. No per-request billing, and visitor questions stop
    leaving the machine.

WHAT TO WATCH
-------------
A local model has to EARN the swap. The bar it must clear is the measured
Gemini baseline, not a vibe:

    hit@1 78.0%   hit@3 91.5%   MRR 0.848
    arabic 11/12   arabizi 1/4   english paraphrase 12/17

Run eval_retrieval.py before and after. If the numbers drop, the quota
and latency wins are not worth it - the whole point of building that
harness was to make this decision on evidence.

DIMENSIONS ARE A MIGRATION, NOT A SETTING
-----------------------------------------
Different models emit different vector lengths, and the stored vectors
must match the query vectors. Ingestion re-embeds automatically when
`embedding_model` changes, and retrieval logs loudly on a dimension
mismatch - but a re-ingest is mandatory after switching, and until it
finishes retrieval returns meaningless results. Decide the model BEFORE
creating a pgvector column, which fixes its dimension at creation.
"""

from __future__ import annotations

import os
import threading

from .base import EmbeddingProvider

# Multilingual by default, because visitors write Arabic. Notably NOT
# all-MiniLM-L6-v2, the usual suggestion: it is English-only and 384-dim,
# and would throw away the Arabic performance the current setup already
# has.
_DEFAULT_MODEL = "intfloat/multilingual-e5-base"

# e5 models are trained with these prefixes and degrade noticeably without
# them - asymmetric retrieval, so the question and the document are
# marked differently. Harmless for models that ignore them.
_QUERY_PREFIX = "query: "
_DOCUMENT_PREFIX = "passage: "


class LocalEmbeddingProvider(EmbeddingProvider):
    """Loads the model once per process and embeds in-process.

    Model loading takes seconds and holds the weights in memory, so it is
    done lazily on first use and shared - constructing this per request
    would reload the weights every time.
    """

    _lock = threading.Lock()

    def __init__(
        self,
        model: str | None = None,
        use_e5_prefixes: bool | None = None,
        truncate_dim: int | None = None,
    ):
        self._model_name = model or os.environ.get("LOCAL_EMBEDDING_MODEL", _DEFAULT_MODEL)
        # Cut the output vector to this many dimensions.
        #
        # Exists so the fallback can match the dimension the rest of the
        # system is built around (768), rather than every table that stores
        # a vector needing a second column at the fallback's native width.
        # Whether a given model SURVIVES truncation is an empirical
        # question - models trained with Matryoshka representation keep
        # their quality, others degrade - so this is measured with
        # eval_retrieval.py before being switched on, never assumed.
        env_dim = os.environ.get("LOCAL_EMBEDDING_TRUNCATE_DIM", "").strip()
        self._truncate_dim = (
            truncate_dim if truncate_dim is not None
            else (int(env_dim) if env_dim else None)
        )
        self._use_prefixes = (
            use_e5_prefixes
            if use_e5_prefixes is not None
            else "e5" in self._model_name.lower()
        )
        self._model = None
        self._dimension: int | None = None

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RuntimeError(
                        "sentence-transformers is not installed. Add it to "
                        "requirements.txt (pip install sentence-transformers)."
                    ) from exc
                model = SentenceTransformer(
                    self._model_name, device="cpu",
                    truncate_dim=self._truncate_dim,
                )
                # Renamed in sentence-transformers 6; keep both so the
                # provider works either side of that release.
                getter = getattr(model, "get_embedding_dimension", None) or                     model.get_sentence_embedding_dimension
                self._dimension = int(getter())
                self._model = model
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return int(self._dimension or 0)

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        model = self._ensure_loaded()
        prepared = [prefix + t for t in texts] if self._use_prefixes else texts
        # normalize_embeddings=False on purpose: retrieval normalises when
        # it builds its matrix, and doing it twice is wasted work that
        # also hides which layer owns the invariant.
        vectors = model.encode(
            prepared, batch_size=16, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=False,
        )
        return [[float(x) for x in row] for row in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts, _DOCUMENT_PREFIX)

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Query text must be non-empty.")
        return self._encode([text], _QUERY_PREFIX)[0]

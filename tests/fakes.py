"""Test doubles. FakeEmbeddingProvider never calls a real API — it's for
exercising ingestion/retrieval *mechanics* (chunking, storage, cosine
ranking), not for judging real semantic embedding quality, which depends
on Voyage at integration time.
"""

from __future__ import annotations

import hashlib
import math
import re

from app.embeddings.base import EmbeddingProvider
from app.llm.base import ChatMessage, LLMProvider

# Must equal the production embedding dimension. The Postgres schema
# declares `vector(768)`, so a 4096-wide fake is now REJECTED AT INSERT
# with "expected 768 dimensions, not 4096" rather than being stored and
# discovered later during scoring. That is the pgvector type doing what
# the old LONGBLOB + embedding_dim pair could only document, and it is
# why this constant is no longer free to pick.
#
# Hash collisions rise as the space narrows, but this fake exists to
# exercise pipeline MECHANICS - chunk stored, right chunk ranks first for
# an on-topic query - not semantic quality, and 768 buckets is ample for
# a 52-section corpus.
_VOCAB_SIZE = 768


def _bag_of_words_vector(text: str) -> list[float]:
    """Crude deterministic embedding: hashes each word into a fixed-size
    vector and counts occurrences. Similar wording -> similar vector, so
    cosine similarity over these behaves sensibly for pipeline tests
    (chunk got stored, right chunk ranks first for an on-topic query)
    without needing network access or an API key.
    """
    vector = [0.0] * _VOCAB_SIZE
    words = re.findall(r"[a-z0-9]+", text.lower())
    for word in words:
        index = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % _VOCAB_SIZE
        vector[index] += 1.0
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return vector
    return [component / norm for component in vector]


class FakeEmbeddingProvider(EmbeddingProvider):
    @property
    def model_name(self) -> str:
        return "fake-bow-v1"

    @property
    def dimension(self) -> int:
        return _VOCAB_SIZE

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_bag_of_words_vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return _bag_of_words_vector(text)


class FakeLLMProvider(LLMProvider):
    """Records the last call it received and returns a canned response, so
    tests can assert on prompt assembly (e.g. "did the retrieved chunk's
    content actually make it into the system prompt") without a live
    OpenRouter API key.
    """

    def __init__(self, response: str = "FAKE_LLM_RESPONSE"):
        self._response = response
        self.last_system_prompt: str | None = None
        self.last_messages: list[ChatMessage] | None = None

    @property
    def model_name(self) -> str:
        return "fake-llm-v1"

    def generate(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        max_tokens: int = 600,
    ) -> str:
        self.last_system_prompt = system_prompt
        self.last_messages = messages
        return self._response

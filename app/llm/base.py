"""Generation LLM provider contract.

Mirrors app.embeddings.base: the RAG pipeline only depends on this
interface, never on a specific vendor, so swapping providers later is a
one-file change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

_VALID_ROLES = ("user", "assistant")


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self):
        if self.role not in _VALID_ROLES:
            raise ValueError(f"Invalid role '{self.role}'; must be one of {_VALID_ROLES}.")
        if not self.content or not self.content.strip():
            raise ValueError("ChatMessage content must be non-empty.")


class LLMProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier for logging/debugging, e.g. 'qwen/qwen3.7-plus'."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        max_tokens: int = 600,
    ) -> str:
        """Returns the assistant's reply text."""

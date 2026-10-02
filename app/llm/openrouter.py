"""Generation via OpenRouter, defaulting to Alibaba's Qwen3.7 Plus.

Chosen originally because OpenRouter gives an OpenAI-compatible schema
and lets the underlying model be swapped with one config value
(OPENROUTER_MODEL) without touching this class — and the default model
itself is an Alibaba (Qwen) model. See app/llm/alibaba.py for a direct
Alibaba DashScope provider (same LLMProvider interface, same model
naming minus the "qwen/" prefix) used as a stand-in until an OpenRouter
key is available.
"""

from __future__ import annotations

import os

from app.llm.openai_compatible import OpenAICompatibleLLMProvider

_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterLLMProvider(OpenAICompatibleLLMProvider):
    DEFAULT_MODEL = "qwen/qwen3.7-plus"
    API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
    MODEL_ENV_VAR = "OPENROUTER_MODEL"
    PROVIDER_LABEL = "OpenRouter"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int = 60,
    ):
        super().__init__(
            api_url=_API_URL,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )

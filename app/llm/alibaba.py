"""Direct Alibaba Cloud Model Studio (DashScope) access, via its
OpenAI-compatible endpoint.

Temporary stand-in for OpenRouterLLMProvider until an OpenRouter key is
available — both implement LLMProvider, so app/rag/pipeline.py doesn't
care which one is passed in. Swapping later is a one-line change at the
call site (construct OpenRouterLLMProvider() instead of
AlibabaLLMProvider()), nothing else in the codebase changes.

Default model: deepseek-v4-flash, chosen by measurement against the real
question set rather than inherited. Medians were qwen3.7-plus 18.8s,
deepseek-v4-flash 3.5s, qwen-flash 1.3s. qwen-flash is fastest but
answers a mixed message with a generic redirect that lets the lead walk,
and writes emoji; deepseek-v4-flash accepts the business, declines the
off-topic part and asks for the WhatsApp number in one reply. All three
repelled 22 of 22 adversarial attempts, so this was a quality call.

Two operational properties. It is a reasoning model, so it spends tokens
before emitting visible text — too small a max_tokens returns an EMPTY
completion rather than a short answer, which generate() surfaces as an
explicit provider error. And the id is exactly "deepseek-v4-flash" on the
international endpoint; "deepseek-v4", "deepseek-flash", "deepseek-chat"
and "deepseek-v3" all 404. Override with DASHSCOPE_MODEL.

Region note: defaults to Alibaba's international endpoint
(dashscope-intl.aliyuncs.com). If your API key was issued on the China
(Beijing) console instead, set DASHSCOPE_BASE_URL to
https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
(a "your API key is not valid" or region-mismatch error from Alibaba
usually means this needs changing).
"""

from __future__ import annotations

import os

from app.llm.openai_compatible import OpenAICompatibleLLMProvider

_DEFAULT_INTL_API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"


class AlibabaLLMProvider(OpenAICompatibleLLMProvider):
    DEFAULT_MODEL = "deepseek-v4-flash"
    API_KEY_ENV_VAR = "DASHSCOPE_API_KEY"
    MODEL_ENV_VAR = "DASHSCOPE_MODEL"
    PROVIDER_LABEL = "Alibaba DashScope"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 60,
    ):
        resolved_url = (
            base_url or os.environ.get("DASHSCOPE_BASE_URL") or _DEFAULT_INTL_API_URL
        )
        super().__init__(
            api_url=resolved_url,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.llm.base import ChatMessage
from app.llm.openrouter import OpenRouterLLMProvider


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterLLMProvider()


def test_defaults_to_qwen_model_when_unset(monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    provider = OpenRouterLLMProvider(api_key="test-key")
    assert provider.model_name == "qwen/qwen3.7-plus"


def test_model_overridable_via_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "qwen/qwen3.7-flash")
    provider = OpenRouterLLMProvider(api_key="test-key")
    assert provider.model_name == "qwen/qwen3.7-flash"


def test_generate_sends_expected_payload_and_parses_response():
    provider = OpenRouterLLMProvider(api_key="test-key", model="qwen/qwen3.7-plus")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Here is the answer."}}]
    }

    with patch("app.llm.openai_compatible.requests.post", return_value=mock_response) as mock_post:
        result = provider.generate(
            system_prompt="You are grounded in this context.",
            messages=[ChatMessage(role="user", content="Hello")],
        )

    assert result == "Here is the answer."

    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["json"]["model"] == "qwen/qwen3.7-plus"
    assert kwargs["json"]["messages"][0] == {
        "role": "system",
        "content": "You are grounded in this context.",
    }
    assert kwargs["json"]["messages"][1] == {"role": "user", "content": "Hello"}


def test_generate_raises_on_malformed_response():
    provider = OpenRouterLLMProvider(api_key="test-key")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"unexpected": "shape"}

    with patch("app.llm.openai_compatible.requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Unexpected OpenRouter response shape"):
            provider.generate("system", [ChatMessage(role="user", content="hi")])


def test_generate_rejects_empty_system_prompt():
    provider = OpenRouterLLMProvider(api_key="test-key")
    with pytest.raises(ValueError):
        provider.generate("   ", [ChatMessage(role="user", content="hi")])


def test_generate_rejects_empty_messages():
    provider = OpenRouterLLMProvider(api_key="test-key")
    with pytest.raises(ValueError):
        provider.generate("system prompt", [])

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.llm.alibaba import AlibabaLLMProvider
from app.llm.base import ChatMessage
from app.llm.openai_compatible import EmptyCompletionError


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        AlibabaLLMProvider()


def test_defaults_to_configured_model_and_intl_endpoint(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_MODEL", raising=False)
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    provider = AlibabaLLMProvider(api_key="test-key")
    assert provider.model_name == "deepseek-v4-flash"
    assert provider._api_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"


def test_model_overridable_via_env(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_MODEL", "qwen3.7-flash")
    provider = AlibabaLLMProvider(api_key="test-key")
    assert provider.model_name == "qwen3.7-flash"


def test_base_url_overridable_via_env_for_china_region(monkeypatch):
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    )
    provider = AlibabaLLMProvider(api_key="test-key")
    assert provider._api_url == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def test_base_url_overridable_via_constructor_arg():
    provider = AlibabaLLMProvider(api_key="test-key", base_url="https://custom.example.com/v1/chat/completions")
    assert provider._api_url == "https://custom.example.com/v1/chat/completions"


def test_generate_sends_expected_payload_and_parses_response():
    provider = AlibabaLLMProvider(api_key="test-key", model="qwen3.7-plus")

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

    args, kwargs = mock_post.call_args
    assert args[0] == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["json"]["model"] == "qwen3.7-plus"


def test_generate_raises_on_malformed_response():
    provider = AlibabaLLMProvider(api_key="test-key")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"unexpected": "shape"}

    with patch("app.llm.openai_compatible.requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Unexpected Alibaba DashScope response shape"):
            provider.generate("system", [ChatMessage(role="user", content="hi")])


# --- Empty completions ------------------------------------------------
#
# The provider intermittently returns 200 OK with a null `content` —
# measured at roughly 3% of turns in live testing, each one surfacing to
# the visitor as a 502. Unlike a malformed response, it is a sampling
# outcome, so the same request usually succeeds on a second attempt.


def _response(content):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}]
    }
    return mock_response


def test_empty_completion_is_retried_and_the_retry_is_returned():
    provider = AlibabaLLMProvider(api_key="test-key")

    with patch(
        "app.llm.openai_compatible.requests.post",
        side_effect=[_response(None), _response("a real answer")],
    ) as mock_post, patch("app.llm.openai_compatible.time.sleep"):
        answer = provider.generate("system", [ChatMessage(role="user", content="hi")])

    assert answer == "a real answer"
    assert mock_post.call_count == 2


def test_whitespace_only_completion_counts_as_empty():
    provider = AlibabaLLMProvider(api_key="test-key")

    with patch(
        "app.llm.openai_compatible.requests.post",
        side_effect=[_response("   \n  "), _response("a real answer")],
    ), patch("app.llm.openai_compatible.time.sleep"):
        answer = provider.generate("system", [ChatMessage(role="user", content="hi")])

    assert answer == "a real answer"


def test_two_empty_completions_in_a_row_still_surface_as_an_error():
    """Retrying must not turn a deterministic emptiness — a content
    filter tripping on this specific input — into an infinite wait or a
    silently blank answer. It fails exactly as before, one retry later."""
    provider = AlibabaLLMProvider(api_key="test-key")

    with patch(
        "app.llm.openai_compatible.requests.post",
        side_effect=[_response(None), _response(None)],
    ) as mock_post, patch("app.llm.openai_compatible.time.sleep"):
        with pytest.raises(EmptyCompletionError, match="empty completion"):
            provider.generate("system", [ChatMessage(role="user", content="hi")])

    assert mock_post.call_count == 2


def test_malformed_response_is_still_not_retried():
    """Guards the distinction the exception class exists to make: a
    broken response shape is a bug, and retrying it just spends the
    timeout twice for the same guaranteed failure."""
    provider = AlibabaLLMProvider(api_key="test-key")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"unexpected": "shape"}

    with patch(
        "app.llm.openai_compatible.requests.post", return_value=mock_response
    ) as mock_post:
        with pytest.raises(RuntimeError):
            provider.generate("system", [ChatMessage(role="user", content="hi")])

    assert mock_post.call_count == 1


# --- Truncated completions --------------------------------------------
#
# Found by the faithfulness eval: "Do you offer ongoing content
# creation?" was answered with the complete string "Yes, PixNo". Not
# empty, so the emptiness guard passed it, and no guardrail asks whether
# an answer is FINISHED — it went to the visitor as-is.


def test_truncated_completion_is_retried():
    provider = AlibabaLLMProvider(api_key="test-key")

    with patch(
        "app.llm.openai_compatible.requests.post",
        side_effect=[_response("Yes, PixNo"), _response("Yes, we offer that monthly.")],
    ) as mock_post, patch("app.llm.openai_compatible.time.sleep"):
        answer = provider.generate("system", [ChatMessage(role="user", content="hi")])

    assert answer == "Yes, we offer that monthly."
    assert mock_post.call_count == 2


def test_a_still_truncated_retry_returns_the_longer_answer_not_an_error():
    """The heuristic is allowed to be wrong, so it must never cost the
    visitor an answer that exists. Worst case is the better of the two
    attempts — never a 502."""
    provider = AlibabaLLMProvider(api_key="test-key")

    with patch(
        "app.llm.openai_compatible.requests.post",
        side_effect=[_response("Yes, PixNo"), _response("Yes")],
    ), patch("app.llm.openai_compatible.time.sleep"):
        answer = provider.generate("system", [ChatMessage(role="user", content="hi")])

    assert answer == "Yes, PixNo"  # the longer of the two, not an exception


def test_a_short_but_finished_answer_is_not_retried():
    """Terminal punctuation is what separates "short" from "cut off"."""
    provider = AlibabaLLMProvider(api_key="test-key")

    with patch(
        "app.llm.openai_compatible.requests.post", return_value=_response("Yes, we do.")
    ) as mock_post:
        answer = provider.generate("system", [ChatMessage(role="user", content="hi")])

    assert answer == "Yes, we do."
    assert mock_post.call_count == 1


def test_a_long_answer_ending_without_punctuation_is_not_retried():
    """Length is the other half of the condition: a real answer may end
    on a bullet or a URL, and must not be second-guessed for it."""
    provider = AlibabaLLMProvider(api_key="test-key")
    long_answer = "We offer the following services and more, see " + "x" * 100

    with patch(
        "app.llm.openai_compatible.requests.post", return_value=_response(long_answer)
    ) as mock_post:
        assert provider.generate("system", [ChatMessage(role="user", content="hi")]) == long_answer

    assert mock_post.call_count == 1


def test_a_truncated_answer_survives_a_network_failure_on_the_retry():
    """A partial answer beats a 502 — the earlier attempt is not thrown
    away just because the retry failed to connect."""
    provider = AlibabaLLMProvider(api_key="test-key")

    with patch(
        "app.llm.openai_compatible.requests.post",
        side_effect=[_response("Yes, PixNo"), requests.exceptions.ConnectionError("boom")],
    ), patch("app.llm.openai_compatible.time.sleep"):
        answer = provider.generate("system", [ChatMessage(role="user", content="hi")])

    assert answer == "Yes, PixNo"


def test_arabic_answer_ending_in_an_arabic_question_mark_is_not_retried():
    provider = AlibabaLLMProvider(api_key="test-key")
    arabic = "نعم، نقدم هذه الخدمة؟"

    with patch(
        "app.llm.openai_compatible.requests.post", return_value=_response(arabic)
    ) as mock_post:
        assert provider.generate("system", [ChatMessage(role="user", content="hi")]) == arabic

    assert mock_post.call_count == 1

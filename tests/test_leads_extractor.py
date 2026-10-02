"""Tests for app.leads.extractor — uses tests/fakes.py's FakeLLMProvider
so no live DashScope call is needed."""

from __future__ import annotations

from app.leads.extractor import extract_lead_signal
from app.llm.base import ChatMessage
from tests.fakes import FakeLLMProvider


def test_parses_well_formed_json_response():
    llm = FakeLLMProvider(
        response="""{
        "name": "Ahmed", "email": "ahmed@example.com", "phone": null,
        "whatsapp_number": null, "company": null, "business_type": null,
        "service_requested": "website", "budget": null, "timeline": null,
        "notes": null, "ready_to_submit": true, "escalate": false,
        "escalate_reason": null
        }"""
    )
    signal = extract_lead_signal(llm, [], "I'm Ahmed, ahmed@example.com, need a website", "Great!")
    assert signal.name == "Ahmed"
    assert signal.email == "ahmed@example.com"
    assert signal.service_requested == "website"
    assert signal.ready_to_submit is True
    assert signal.escalate is False


def test_tolerates_markdown_fenced_json():
    llm = FakeLLMProvider(
        response='```json\n{"name": "Sara", "ready_to_submit": false, "escalate": false}\n```'
    )
    signal = extract_lead_signal(llm, [], "hi I'm Sara", "hi Sara")
    assert signal.name == "Sara"
    assert signal.ready_to_submit is False


def test_malformed_response_yields_empty_signal_not_a_crash():
    llm = FakeLLMProvider(response="I cannot comply with that request.")
    signal = extract_lead_signal(llm, [], "hello", "hi there")
    assert signal.has_any_field() is False
    assert signal.escalate is False


def test_llm_exception_yields_empty_signal_not_a_crash():
    class ExplodingLLM(FakeLLMProvider):
        def generate(self, *args, **kwargs):
            raise RuntimeError("network down")

    signal = extract_lead_signal(ExplodingLLM(), [], "hello", "hi there")
    assert signal.has_any_field() is False


def test_escalation_signal_parsed():
    llm = FakeLLMProvider(
        response='{"escalate": true, "escalate_reason": "wants a refund"}'
    )
    signal = extract_lead_signal(llm, [], "I want my money back!", "I understand...")
    assert signal.escalate is True
    assert signal.escalate_reason == "wants a refund"


def test_null_string_literals_treated_as_none():
    llm = FakeLLMProvider(response='{"name": "null", "email": ""}')
    signal = extract_lead_signal(llm, [], "hi", "hello")
    assert signal.name is None
    assert signal.email is None


def test_history_window_is_bounded_and_includes_latest_turn():
    llm = FakeLLMProvider(response="{}")
    history = [ChatMessage(role="user", content=f"turn {i}") for i in range(10)]
    extract_lead_signal(llm, history, "final question", "final answer")
    assert "final question" in llm.last_messages[0].content
    assert "final answer" in llm.last_messages[0].content
    # only the last 6 history turns should be folded in, not all 10
    assert "turn 0" not in llm.last_messages[0].content
    assert "turn 9" in llm.last_messages[0].content

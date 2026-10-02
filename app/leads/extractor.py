"""Pulls structured lead fields and an escalation signal out of a
conversation, via a second, separate LLM call.

Why a second call instead of asking the main answer-generating call to
also return structured data: `app/llm/openai_compatible.py` is a plain
chat-completions call with no tool-calling/JSON-mode support, and the
visitor-facing answer has to stay natural prose that's already passed
through `app/rag/guardrails.py`. Mixing "produce a safe, grounded reply"
and "produce machine-parseable JSON" into one call risks both — the
model either breaks JSON to stay conversational, or breaks tone to stay
structured. Keeping them as two calls with two different system prompts
means each one only has one job.

This call is internal-only: its system prompt and output are never
shown to the visitor, so none of `app/rag/guardrails.py`'s outbound
rules apply to it — the guardrails exist to protect what a visitor
*sees*, not this backend-only extraction step.
"""

from __future__ import annotations

import json
import re

from app.leads.models import LeadSignal
from app.llm.base import ChatMessage, LLMProvider

_EXTRACTION_SYSTEM_PROMPT = """\
You are a silent data-extraction step behind a customer-facing chat \
assistant for Apex Creative, a creative/AI marketing agency. You are NOT the \
assistant the visitor is talking to — you only read a transcript and \
output structured JSON. Nothing you write is ever shown to the visitor.

Read the conversation and extract any of the following the visitor has \
clearly and explicitly provided. Never guess, infer, or fabricate a \
value — leave a field null if it wasn't actually stated.

Return ONLY a single JSON object, no prose, no markdown fences, with \
exactly these keys:
{
  "name": string or null,
  "email": string or null,
  "phone": string or null,
  "whatsapp_number": string or null,
  "company": string or null,
  "business_type": string or null,
  "service_requested": string or null,
  "budget": string or null,
  "timeline": string or null,
  "notes": string or null (any other relevant detail volunteered),
  "ready_to_submit": boolean,
  "escalate": boolean,
  "escalate_reason": string or null
}

Rules:
- "ready_to_submit" is true only if name AND (phone OR whatsapp_number — \
  NOT email alone; the team follows up over WhatsApp, so a WhatsApp/phone \
  number specifically must be present, an email address by itself does \
  not satisfy this) AND service_requested are all present (from this \
  turn or earlier in the conversation).
- "escalate" is true if the visitor explicitly asked for a human, \
  expressed a complaint/dispute/frustration about work or billing, or \
  asked something clearly outside a knowledge base that matters for a \
  real decision (legal terms, contract specifics, negotiated pricing). \
  If true, "escalate_reason" is a short (<12 word) plain description.
- If nothing extractable is present and there's no escalation signal, \
  return all fields null/false.
"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_signal(raw_text: str) -> LeadSignal:
    match = _JSON_OBJECT_RE.search(raw_text)
    if not match:
        return LeadSignal()

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return LeadSignal()

    if not isinstance(data, dict):
        return LeadSignal()

    def _clean_str(value) -> str | None:
        if isinstance(value, str) and value.strip() and value.strip().lower() != "null":
            return value.strip()
        return None

    return LeadSignal(
        name=_clean_str(data.get("name")),
        email=_clean_str(data.get("email")),
        phone=_clean_str(data.get("phone")),
        whatsapp_number=_clean_str(data.get("whatsapp_number")),
        company=_clean_str(data.get("company")),
        business_type=_clean_str(data.get("business_type")),
        service_requested=_clean_str(data.get("service_requested")),
        budget=_clean_str(data.get("budget")),
        timeline=_clean_str(data.get("timeline")),
        notes=_clean_str(data.get("notes")),
        ready_to_submit=bool(data.get("ready_to_submit", False)),
        escalate=bool(data.get("escalate", False)),
        escalate_reason=_clean_str(data.get("escalate_reason")),
    )


def extract_lead_signal(
    llm_provider: LLMProvider,
    conversation_history: list[ChatMessage],
    latest_user_message: str,
    latest_assistant_message: str,
) -> LeadSignal:
    """Runs the extraction call over the last few turns plus the turn
    that just happened, and returns whatever it could parse out.

    Never raises on a malformed/unparseable model response — extraction
    is a best-effort enhancement, not something that should ever break
    the actual chat reply the visitor already received. A parse failure
    just yields an empty `LeadSignal` (nothing captured, nothing
    escalated this turn).
    """
    # Bounded window: extraction only needs recent context to catch
    # what was just said, not the entire history — mirrors the same
    # reasoning as pipeline.py's retrieval-history folding.
    recent_turns = conversation_history[-6:]
    transcript_lines = [f"{m.role}: {m.content}" for m in recent_turns]
    transcript_lines.append(f"user: {latest_user_message}")
    transcript_lines.append(f"assistant: {latest_assistant_message}")
    transcript = "\n".join(transcript_lines)

    try:
        raw_text = llm_provider.generate(
            system_prompt=_EXTRACTION_SYSTEM_PROMPT,
            messages=[ChatMessage(role="user", content=transcript)],
            max_tokens=400,
        )
    except Exception:
        # Network/provider failure on the extraction call must never
        # take down the chat response itself — see module docstring.
        return LeadSignal()

    return _parse_signal(raw_text)
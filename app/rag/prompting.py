"""Builds the prompt sent to the LLM for a given turn.

Phase 3: this now uses the real, fixed Apex Creative system prompt (see
app/prompts/loader.py) instead of the temporary phase-2 grounding
instructions. The system prompt's own rules ("You may only state facts
that exist in the provided knowledge base...") are what govern behavior;
this function's only job is to hand the model the retrieved chunks as
"the provided knowledge base" for this turn.
"""

from __future__ import annotations

from app.kb.retrieval import RetrievedChunk
from app.prompts.loader import load_system_prompt

_CONTEXT_HEADER = (
    "\n\n---\n"
    "RETRIEVED KNOWLEDGE BASE CONTENT FOR THIS QUESTION "
    "(this is the entirety of what you may treat as knowledge-base fact "
    "for this turn — nothing outside it):\n"
)


_CLARIFICATION_HEADER = (
    "\n\n---\n"
    "NO KNOWLEDGE BASE CONTENT WAS RETRIEVED FOR THIS TURN.\n"
    "The visitor has asked something with no topic of its own — a "
    'follow-up such as "what do you mean?" or "يعني ايه" — so there is '
    "nothing to look up.\n"
    "You may ONLY restate or clarify what you have already said earlier in "
    "this conversation, in the visitor's language.\n"
    "You must NOT introduce any fact, service, price, name, timeline or "
    "detail that does not already appear earlier in this conversation. If "
    "clarifying would require something you have not already said, say you "
    "don't have it and offer to connect them with the team.\n"
    "Keep it short — two sentences at most.\n"
)


def build_system_prompt(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        raise ValueError("Cannot build a system prompt with zero context chunks.")

    context_blocks = "\n\n".join(
        f"[{chunk.section_title}]\n{chunk.content}" for chunk in chunks
    )
    return f"{load_system_prompt()}{_CONTEXT_HEADER}{context_blocks}"


def build_clarification_prompt() -> str:
    """The prompt for a turn where retrieval found nothing but the visitor
    is plainly asking about what was just said.

    Kept separate from `build_system_prompt`, which raises on zero chunks
    and should keep doing so: "no context" is normally a hard stop, and
    this is one narrow exception rather than a loosening of it.

    The exception exists because a contentless follow-up can never match a
    knowledge base section — no chunk answers "what do you mean?" — so the
    hard stop was replying to a clarification request with a canned "I
    don't have that information", three turns running in a real
    conversation. The model already has what it needs: its own earlier
    message.

    This does NOT relax grounding. The outbound guardrails still run, and
    with no chunks their grounding text is the conversation alone, so any
    number or proper noun the model has not already said is rejected. The
    instruction above is what covers plain prose, and a prompt is the
    weaker of the two layers — which is why the caller's gate is narrow.
    """
    return load_system_prompt() + _CLARIFICATION_HEADER

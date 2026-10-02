"""The RAG loop: retrieve relevant chunks, build the (real) system
prompt around them, generate an answer, then run it through outbound
guardrails before it's considered safe to show a visitor.

Phase 1 adds multi-turn chat history:
- `conversation_history` is threaded into the LLM call as prior turns,
  capped to a token budget (not a message count — see
  `_cap_history_to_token_budget`).
- A *separate*, retrieval-only augmented query folds the last 1-2 user
  turns' text in before embedding/keyword-matching, so pronoun-only
  follow-ups ("what about their pricing?") still retrieve the right
  chunk — see `_build_retrieval_query`.

Not yet included: session/lead persistence or escalation — those are
later build phases.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

from app.embeddings.base import EmbeddingProvider
from app.kb.retrieval import RetrievedChunk, retrieve_relevant_chunks
from app.llm.base import ChatMessage, LLMProvider
from app.text.normalize import has_arabic
from app.rag.guardrails import (
    GuardrailResult,
    _ungrounded_proper_nouns,
    apply_guardrails,
    ungrounded_capitalized_words,
)
from app.rag.prompting import build_clarification_prompt, build_system_prompt
from app.rag.qa_cache import is_cacheable_question

logger = logging.getLogger(__name__)

NO_MATCH_RESPONSE = (
    "I don't have that information available right now — I can connect "
    "you with our team so they can help directly."
)

# The same thing in Arabic.
#
# This is not a nicety. Retrieval returns nothing most often for SHORT
# conversational turns — "يعني ايه", "ايه المشكلة" — which is exactly the
# moment a visitor is already confused. Answering those in English told
# an Arabic speaker, mid-Arabic-conversation, that the bot had stopped
# understanding them, three turns in a row. Seen in a real transcript.
#
# Chosen by the script of the visitor's own message rather than by any
# session-level language setting: a visitor may switch languages
# mid-conversation, and the reply should follow the turn they just typed,
# the same rule retrieval already uses to pick which half of the
# knowledge base to search.
NO_MATCH_RESPONSE_AR = (
    "معنديش المعلومة دي دلوقتي — بس أقدر أوصلك بفريقنا "
    "عشان يساعدوك بشكل مباشر."
)


def no_match_response(query: str) -> str:
    """The "nothing retrieved" reply, in the language of the question."""
    return NO_MATCH_RESPONSE_AR if has_arabic(query) else NO_MATCH_RESPONSE

# Number of trailing user turns folded into the retrieval query. Kept
# small and user-turns-only on purpose: assistant turns and older user
# turns add unrelated content words that can drag retrieval toward the
# wrong chunk (the same false-positive-keyword-match failure mode the
# "about" stopword bug came from, just via history text instead of a
# single query) rather than helping resolve a pronoun-only follow-up.
_RETRIEVAL_HISTORY_TURNS = 2

# Mirrors the stopword list in app/kb/retrieval.py's keyword-match path
# (plus pronoun forms, since those are exactly what signal "this query
# needs history context" here). Kept separate rather than imported: this
# list answers a different question ("is this query thin enough to need
# history?") than retrieval's ("does this word match a title token?"),
# and conflating the two would couple unrelated concerns.
_RETRIEVAL_AUGMENTATION_STOPWORDS = {
    "who", "is", "what", "are", "the", "do", "you", "how", "much", "does",
    "can", "i", "a", "an", "in", "of", "to", "for", "on", "with", "your",
    "my", "our", "about", "and", "or", "me", "it", "that", "this",
    "their", "them", "they", "its", "there",
}

# A query with at least this many non-stopword tokens is treated as
# already self-sufficient for retrieval.
_MIN_CONTENT_TOKENS_FOR_STANDALONE_QUERY = 3

# Above this, the visitor's plain question is a clear direct hit and
# folding history in cannot plausibly beat it — so the second embedding
# call is skipped. Measured top-1 scores for self-sufficient questions:
# "What is your refund policy?" 0.82, "Who is the founder?" 0.79, "Who is
# on your team?" 0.77. Deliberately conservative: set too low, a thin
# query that matches the WRONG chunk strongly would short-circuit before
# history could correct it, so anything below a clear hit still pays for
# both retrievals and keeps whichever scored better.
_PLAIN_QUERY_CONFIDENT_SCORE = 0.7

# Default cap on how much prior conversation gets sent to the LLM,
# approximated in tokens (see `_estimate_tokens`) rather than message
# count, since cost/context size scales with tokens, not turns.
_DEFAULT_HISTORY_TOKEN_BUDGET = 2000


@dataclass(frozen=True)
class RagResult:
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    grounded: bool  # False when nothing in the KB cleared the relevance threshold
    guardrail_violations: list[str] = field(default_factory=list)
    # "" on a normal generated answer, "exact"/"semantic" when the answer
    # came from the QA cache instead of an LLM call. Useful for logging
    # hit rate without a separate metrics path.
    cache_hit: str = ""


def _blocked_emails() -> list[str]:
    configured = os.environ.get("INTERNAL_NOTIFICATION_EMAIL", "").strip()
    return [configured] if configured else []


def _query_needs_history_context(query: str) -> bool:
    """True when `query` is thin enough (mostly pronouns/stopwords, e.g.
    "what about their pricing?") that it can't stand alone for retrieval
    and benefits from prior-turn context.

    This is the gate that keeps history-folding from corrupting a query
    that doesn't need it: a long, content-word-heavy follow-up is already
    self-sufficient for retrieval, and folding unrelated history text
    into it only adds false-positive keyword-match risk (retrieval.py's
    `_compute_keyword_score` gives an 0.85 bonus to ANY single
    non-stopword token that happens to match ANY chunk title — e.g. an
    incidental team member's name mentioned earlier in the conversation)
    without adding retrieval signal the query didn't already have.
    """
    content_tokens = [
        word
        for word in re.findall(r"[a-z0-9']+", query.lower())
        if word not in _RETRIEVAL_AUGMENTATION_STOPWORDS and len(word) > 1
    ]
    return len(content_tokens) < _MIN_CONTENT_TOKENS_FOR_STANDALONE_QUERY



def _canned_replies() -> frozenset[str]:
    """Every fixed string the system can emit without the model.

    Collected rather than hand-listed so a new safe response cannot be
    added elsewhere and quietly become something the bot tries to
    "clarify".
    """
    from app.rag import guardrails
    from app.rag import input_guard

    replies = {NO_MATCH_RESPONSE, NO_MATCH_RESPONSE_AR}
    for module in (guardrails, input_guard):
        for name in dir(module):
            if "SAFE_RESPONSE" in name or "REFUSAL_RESPONSE" in name:
                value = getattr(module, name)
                if isinstance(value, str):
                    replies.add(value)
    return frozenset(r.strip() for r in replies)


def _last_substantive_assistant_turn(
    conversation_history: list[ChatMessage],
) -> str | None:
    """The most recent assistant turn that actually said something, or
    None if the assistant has not yet said anything worth clarifying.

    Walks BACK past canned replies rather than taking the previous message
    as-is. That is the case the real transcript hit: retrieval missed
    three turns in a row, so the message immediately before "يعنى ايه" was
    itself "I don't have that information available right now". Clarifying
    that is circular — it tells the visitor nothing and confirms the bot
    is stuck. The turn worth clarifying was two further back, where it had
    explained that website pricing depends on scope.

    Returns None when every assistant turn so far is canned, which is the
    honest outcome: there is genuinely nothing to clarify, and the canned
    reply is then the correct answer rather than a failure.
    """
    canned = _canned_replies()
    for message in reversed(conversation_history):
        if message.role != "assistant":
            continue
        content = (message.content or "").strip()
        if content and content not in canned:
            return content
    return None


def _build_retrieval_query(query: str, conversation_history: list[ChatMessage]) -> str:
    """Returns the string handed to retrieval (embedding + keyword
    matching) for this turn.

    Only folds in prior user turns when `query` itself needs the help
    (see `_query_needs_history_context`) — otherwise returns `query`
    unchanged. Deliberately NOT the same string sent to the LLM as
    conversational context: the full `conversation_history` (all roles,
    uncapped here) goes to `generate()` separately via
    `_cap_history_to_token_budget`. Retrieval only needs recent topical
    words, not the full exchange, and only when the current query is
    too thin to carry its own retrieval signal.
    """
    if not conversation_history or not _query_needs_history_context(query):
        return query
    # Only fold in turns written in the same script as the current
    # question. The knowledge base is stored per language, and this
    # string is what gets EMBEDDED — mixing scripts drags the vector
    # into the space between the two halves, so it matches neither well.
    # Measured on a real conversation: two Arabic turns followed by
    # "What services do you offer?" pulled the top English chunks down to
    # ~0.41 and pushed the Services Overview section out of the results
    # entirely, leaving the model to answer from four unrelated service
    # sections and the guardrails to reject what it produced.
    query_is_arabic = has_arabic(query)
    user_turns = [m.content for m in conversation_history if m.role == "user"]
    same_script = [t for t in user_turns if has_arabic(t) == query_is_arabic]

    # Prefer same-script turns, but fall back to whatever the visitor
    # actually said rather than giving up. A thin follow-up in the OTHER
    # language ("بتعملوا هوية بصرية؟" then "how long does it take?") has no
    # topic of its own, and same-script history alone would leave it with
    # nothing to retrieve on. Reaching across languages is safe here only
    # because the caller keeps whichever query retrieved better — a mixed
    # string that embeds badly simply loses to the plain question.
    prior_user_turns = (same_script or user_turns)[-_RETRIEVAL_HISTORY_TURNS:]
    if not prior_user_turns:
        return query
    return " ".join([*prior_user_turns, query])


def _estimate_tokens(text: str) -> int:
    """Coarse token estimate (chars // 4). Real tokenization varies by
    model/provider, so an exact count isn't meaningful here anyway — this
    only needs to be good enough to stop a token *budget* from behaving
    like a message *count* (see module docstring)."""
    return max(1, len(text) // 4)


def _cap_history_to_token_budget(
    conversation_history: list[ChatMessage], token_budget: int
) -> list[ChatMessage]:
    """Keeps the most recent turns that fit within `token_budget`,
    dropping older turns first. A fixed message-count cap would treat six
    one-word messages the same as six paragraphs; since the cap exists to
    control cost/context size, tokens are the thing actually being
    managed, so that's what's budgeted here instead.

    Always keeps at least the single most recent turn, even if it alone
    exceeds the budget, rather than silently dropping to empty history.
    """
    kept: list[ChatMessage] = []
    used = 0
    for message in reversed(conversation_history):
        cost = _estimate_tokens(message.content)
        if kept and used + cost > token_budget:
            break
        kept.append(message)
        used += cost
    kept.reverse()
    return kept


def answer_query(
    query: str,
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider,
    db_connection,
    conversation_history: list[ChatMessage] | None = None,
    top_k: int = 4,
    min_score: float | None = None,
    history_token_budget: int = _DEFAULT_HISTORY_TOKEN_BUDGET,
    qa_cache=None,
) -> RagResult:
    """Runs the RAG loop for one turn.

    `qa_cache` is optional and defaults to off, so every existing caller
    and test keeps the exact behavior it had. When a
    `QuestionAnswerCache` is passed (the API layer does), a previously
    answered question can short-circuit before generation — see
    app/rag/qa_cache.py for the rules governing what may be reused.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    conversation_history = conversation_history or []

    retrieval_query = _build_retrieval_query(query, conversation_history)

    # A question is only reusable across visitors if it stands on its
    # own. `retrieval_query is query` is exactly that test already
    # computed: they differ only when prior turns had to be folded in to
    # make the query retrievable, which is the same condition that makes
    # its answer conversation-specific.
    cacheable = qa_cache is not None and retrieval_query == query and is_cacheable_question(query)

    if cacheable:
        # Scopes the cache to the CURRENT knowledge base before any
        # lookup — a re-ingest must retire prior answers on the very
        # next turn, not whenever the process happens to restart.
        qa_cache.refresh_scope(db_connection)
        hit = qa_cache.lookup_exact(
            db_connection, query,
            embedding_model=getattr(embedding_provider, "model_name", None),
        )
        if hit is not None:
            # Cheapest possible turn: no embedding call, no LLM call.
            return RagResult(
                answer=hit.answer, retrieved_chunks=[], grounded=hit.grounded,
                cache_hit=hit.kind,
            )

    # Embedded here rather than inside retrieval so the same vector can
    # serve the semantic cache lookup and the retrieval scan — one
    # embedding call either way, hit or miss.
    #
    # The VISITOR'S OWN question is what gets embedded, not the augmented
    # one. When they differ the cache is off anyway (see `cacheable`), and
    # the plain question is both the better retrieval bet and the one
    # worth spending the guaranteed call on.
    query_vector = embedding_provider.embed_query(query)

    # Which provider actually served that embedding is only knowable now:
    # the fallback engages inside embed_query, so reading model_name any
    # earlier reports the PREVIOUS call's provider.
    #
    # Everything cache-related from here is scoped to it. The primary and
    # the fallback write and read SEPARATE entries rather than sharing
    # them: their vectors are different spaces, so a bge-m3 question
    # compared against a Gemini-embedded one would pick a near-neighbour
    # that is not one.
    #
    # Scoping rather than switching the cache off is the point. A quota
    # outage lasts hours, and that is exactly when caching earns its keep
    # — disabling it then makes every visitor pay full latency during the
    # one period the system is already degraded.
    active_model = getattr(embedding_provider, "model_name", None)

    if cacheable:
        hit = qa_cache.lookup_semantic(
            db_connection, query_vector, embedding_model=active_model
        )
        if hit is not None:
            return RagResult(
                answer=hit.answer, retrieved_chunks=[], grounded=hit.grounded,
                cache_hit=hit.kind,
            )

    # Retrieve on the visitor's actual question FIRST, and only reach for
    # conversation history if that came back weak.
    #
    # `_query_needs_history_context` counts content words, so a complete
    # but short question like "Who is on your team?" (one content word,
    # "team") is judged thin and has the previous turn folded into it.
    # Ask about services and then about the team, and retrieval embeds
    # "What services do you offer? Who is on your team?", lands on service
    # sections, and the bot reports having no team information — which is
    # what a visitor hit. Reproduced with no Arabic involved, so it
    # predates the bilingual index.
    #
    # Ordering it this way rather than always doing both keeps the fix and
    # drops an embedding call on exactly the case that was broken: a short
    # question that stands on its own retrieves confidently by itself, and
    # never needs the second call. Genuine pronoun follow-ups ("how long
    # does it take?") retrieve weakly alone and still pay for both.
    chunks = retrieve_relevant_chunks(
        query, embedding_provider, db_connection, top_k=top_k,
        script_query=query, min_score=min_score, query_vector=query_vector,
    )

    best_plain = chunks[0].score if chunks else 0.0
    if retrieval_query != query and best_plain < _PLAIN_QUERY_CONFIDENT_SCORE:
        augmented = retrieve_relevant_chunks(
            retrieval_query, embedding_provider, db_connection, top_k=top_k,
            # Language comes from what the visitor just asked, not from the
            # augmented query, which may carry turns in another language.
            script_query=query, min_score=min_score,
        )
        if augmented and augmented[0].score > best_plain:
            chunks = augmented
    if not chunks:
        # Normally a hard stop. The one exception is a follow-up that has
        # no topic of its own ("what do you mean?", "يعني ايه") asked after
        # the assistant has actually said something — there is no chunk
        # that can answer those, so stopping here replied to a
        # clarification request with a canned "I don't have that
        # information", three turns running in a real conversation.
        clarifies = _last_substantive_assistant_turn(conversation_history)
        if clarifies is None or not _query_needs_history_context(query):
            return RagResult(
                answer=no_match_response(query), retrieved_chunks=[], grounded=False
            )
        system_prompt = build_clarification_prompt()
    else:
        system_prompt = build_system_prompt(chunks)
    capped_history = _cap_history_to_token_budget(conversation_history, history_token_budget)
    messages = [*capped_history, ChatMessage(role="user", content=query)]
    raw_answer = llm_provider.generate(system_prompt, messages)

    # Numbers the visitor themselves already typed this conversation
    # (e.g. a phone number given two turns ago) count as grounded when
    # the assistant reflects them back — see guardrails.py rule 2.
    conversation_context = "\n".join(f"{m.role}: {m.content}" for m in messages)
    guardrail: GuardrailResult = apply_guardrails(
        raw_answer, chunks, blocked_emails=_blocked_emails(), conversation_context=conversation_context
    )

    if cacheable:
        _store_if_reusable(
            qa_cache, db_connection, query, query_vector, guardrail, chunks,
            conversation_history, embedding_model=active_model,
        )

    return RagResult(
        answer=guardrail.safe_answer,
        retrieved_chunks=chunks,
        grounded=True,
        guardrail_violations=guardrail.violations,
    )


def _store_if_reusable(
    qa_cache, db_connection, query, query_vector, guardrail, chunks,
    conversation_history, embedding_model=None,
) -> None:
    """Writes this answer to the QA cache only if it is safe to serve to
    a different visitor. Three gates, each closing a hole the previous
    one leaves open.

    1. Every capitalized word must be grounded in the retrieved chunks.
       This is the name check, and it is the one that matters: gates 2
       and 3 both miss a bare first name, and the live proper-noun rule
       deliberately ignores single words as too noisy. Caught in
       end-to-end testing, where "Hi Omar! It's great to meet you..."
       was written to the shared cache and would have greeted the next
       visitor by his name.

       This replaced a blunter rule — "first turn only" — which was
       correct but cost most of the cache's value, since in a real
       conversation only the opening question was ever stored. Checking
       the answer's actual content keeps later context-free questions
       cacheable while closing the same hole more completely: it also
       catches "Thanks, Omar" mid-sentence, which the run-based rule
       never saw.

    2. Re-run the guardrails with NO conversation context. Numbers the
       visitor supplied count as grounded during the live turn, so an
       answer reflecting one back passes in-conversation and fails here
       — exactly the answer that must never be replayed to someone else.

    3. Re-check proper-noun RUNS at `min_words=1` as well, which catches
       a name left over after a sentence-start drop ("Hi Omar!") that
       gate 1's per-word scan and the live rule both treat differently.

    `is_cacheable_question` rejects self-introductions before any of
    this runs, so the common path never gets here in the first place.
    """
    if guardrail.violations:
        return  # a sanitized answer is a failure, not a result to reuse

    kb_only = apply_guardrails(
        guardrail.safe_answer, chunks, blocked_emails=_blocked_emails(), conversation_context=""
    )
    if not kb_only.passed or kb_only.safe_answer != guardrail.safe_answer:
        return

    grounding_text = "\n".join(f"{c.section_title}\n{c.content}" for c in chunks)
    if ungrounded_capitalized_words(guardrail.safe_answer, grounding_text):
        return
    if _ungrounded_proper_nouns(guardrail.safe_answer, grounding_text, min_words=1):
        return

    qa_cache.store(
        db_connection, query, query_vector, guardrail.safe_answer, grounded=True,
        embedding_model=embedding_model,
    )
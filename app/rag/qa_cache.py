"""Cache-augmented generation: a question/answer cache in front of the
RAG pipeline, so a question the bot has already answered doesn't pay for
retrieval and generation a second time.

Two lookup paths, cheapest first:

1. EXACT — the normalized question text hashed and matched. Costs one
   indexed SELECT and NOTHING else: no embedding call, no LLM call.
2. SEMANTIC — cosine similarity between the query embedding and cached
   question embeddings. Costs no *extra* embedding call, because the
   pipeline already embeds the query for retrieval and hands the vector
   here (see `answer_query`). A hit skips the LLM call, which is the
   expensive half of a turn.

On a miss the pipeline runs normally and the result is written back.

WHAT IS DELIBERATELY NOT CACHED — this is the part that matters
---------------------------------------------------------------
A shared cache across visitors is a correctness and privacy hazard if
anything conversation-specific gets in, so entry is gated hard:

- Only context-free questions. "What about their pricing?" means
  nothing without the turns before it; caching its answer and replaying
  it for a different visitor would be actively wrong. The pipeline uses
  the same `_query_needs_history_context` test it already uses to
  decide whether to fold history into retrieval.
- Only answers that would still pass guardrails with NO conversation
  context. This is the privacy gate, and it's exact rather than
  heuristic: guardrails count numbers the visitor typed earlier (a
  phone number, a budget) as grounded, so an answer echoing them back
  passes in-conversation but FAILS when re-checked against the KB
  alone. Re-running the guardrails with an empty context is therefore a
  precise test for "is this answer made only of public KB facts" — and
  only those are safe to serve to somebody else.
- Never questions that contain contact details. Those would put visitor
  PII in a shared table and, worse, could semantically match another
  visitor's question.
- Never ungrounded answers, guardrail-sanitized answers, or the
  no-match fallback. A blocked answer is a failure, not a result worth
  replaying — and the no-match path never calls the LLM anyway, so
  there's nothing to save.

Invalidation is by fingerprint, not by TTL alone: every row records the
KB signature, embedding model, and LLM model it was produced under, and
lookups filter on all three. Re-ingesting the knowledge base changes the
signature, so every prior answer stops being visible immediately, with
no cache-clearing step to remember. A max age exists on top of that so
nothing lives forever.

THE THRESHOLD IS THE RISK. Semantic matching trades correctness for
savings: too low and a visitor gets a confidently-worded answer to a
question they didn't ask. The default is deliberately conservative and
`QA_CACHE_SIMILARITY_THRESHOLD` should be calibrated against real
traffic before being loosened — the same exercise `calibrate_threshold.py`
does for retrieval. Set `QA_CACHE_ENABLED=false` to turn the whole
thing off; exact-match-only is available via
`QA_CACHE_SEMANTIC_ENABLED=false` and carries none of this risk.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import threading
from dataclasses import dataclass

from app.kb.retrieval import kb_signature
from app.kb.vector_codec import from_pgvector, to_pgvector

logger = logging.getLogger(__name__)

# High on purpose — see the module docstring. At this threshold only
# near-paraphrases match ("what services do you offer" / "what services
# do you provide"), not merely related questions.
_DEFAULT_SIMILARITY_THRESHOLD = 0.95

_DEFAULT_MAX_AGE_DAYS = 30
_DEFAULT_MAX_ENTRIES = 500

# Questions longer than this are almost never repeated verbatim, and
# caching them just fills the table with single-use rows.
_MAX_CACHEABLE_QUESTION_CHARS = 300

_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", re.IGNORECASE)
# Any run of 7+ digits, allowing the separators people type in phone
# numbers. Deliberately broad: a false positive only means "don't cache
# this one", which costs nothing but a repeated LLM call.
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{5,}\d)")

_WHITESPACE_RE = re.compile(r"\s+")

# Self-introductions are the practical source of personalized answers.
# "Hi, I'm Omar and I run a small cafe, I'm interested in branding" is a
# perfectly cacheable-LOOKING question — long, context-free, no email or
# phone in it — whose answer opens "Hi Omar!". Caught in end-to-end
# testing, where exactly that reply was written to the shared cache.
#
# The trigger phrase is matched case-insensitively but the word AFTER it
# must be capitalised, which is what separates "I'm Omar" from "I'm
# interested in branding". The flag has to be scoped with (?i:...) rather
# than applied to the whole pattern — a global re.IGNORECASE makes [A-Z]
# match lowercase too, which flagged every ordinary "I'm interested..."
# message as an introduction and silently disabled caching for them.
_SELF_INTRODUCTION_RE = re.compile(
    r"(?:(?i:i\s*'?\s*m|i\s+am|my\s+name\s+is|this\s+is|call\s+me)\s+[A-Z])"
    r"|(?i:\bmy\s+name\b)"
    r"|\bاسمي\b"
)


@dataclass(frozen=True)
class CachedAnswer:
    answer: str
    grounded: bool
    kind: str  # "exact" | "semantic"
    similarity: float = 1.0


@dataclass(frozen=True)
class _CacheEntry:
    answer: str
    grounded: bool
    vector: tuple[float, ...]
    norm: float


def normalize_question(question: str) -> str:
    """Case, whitespace, and trailing punctuation don't change what a
    question means, so they shouldn't produce a cache miss."""
    collapsed = _WHITESPACE_RE.sub(" ", question.strip().lower())
    return collapsed.rstrip(" ?!.،")


def _question_hash(question: str) -> str:
    return hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()


def contains_contact_details(text: str) -> bool:
    return bool(_EMAIL_RE.search(text) or _PHONE_RE.search(text))


def _safe_rollback(conn) -> None:
    """Rolls back without raising.

    The caller is already in an error path and its job is to keep the
    cache from breaking the visitor's answer - a rollback that itself
    fails must not become the exception that does."""
    try:
        conn.rollback()
    except Exception:
        logger.debug("rollback after a cache error also failed", exc_info=True)


def _cosine(a: tuple[float, ...] | list[float], a_norm: float,
            b: tuple[float, ...], b_norm: float) -> float:
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (a_norm * b_norm)


class QuestionAnswerCache:
    """Owns the `qa_cache` table. Stateless with respect to the DB
    connection — every method takes one, matching the rest of this
    codebase, so it works identically on a request connection and on the
    scheduler's own.
    """

    def __init__(
        self,
        kb_signature: str,
        embedding_model: str,
        llm_model: str,
        similarity_threshold: float | None = None,
        semantic_enabled: bool | None = None,
        max_age_days: int | None = None,
        max_entries: int | None = None,
    ):
        self.kb_signature = kb_signature
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else float(
                os.environ.get(
                    "QA_CACHE_SIMILARITY_THRESHOLD", str(_DEFAULT_SIMILARITY_THRESHOLD)
                )
            )
        )
        self.semantic_enabled = (
            semantic_enabled
            if semantic_enabled is not None
            else os.environ.get("QA_CACHE_SEMANTIC_ENABLED", "true").strip().lower() != "false"
        )
        self.max_age_days = max_age_days if max_age_days is not None else int(
            os.environ.get("QA_CACHE_MAX_AGE_DAYS", str(_DEFAULT_MAX_AGE_DAYS))
        )
        self.max_entries = max_entries if max_entries is not None else int(
            os.environ.get("QA_CACHE_MAX_ENTRIES", str(_DEFAULT_MAX_ENTRIES))
        )

        # Decoded entries for the semantic scan, same pattern as the KB
        # chunk cache: the alternative is pulling every cached embedding
        # blob out of MySQL on every single request, which would spend
        # more than the cache saves.
        self._lock = threading.Lock()
        self._entries_fingerprint: tuple | None = None
        self._entries: list[_CacheEntry] = []

    # --- scope -------------------------------------------------------

    def refresh_scope(self, conn) -> None:
        """Re-reads the current KB fingerprint and, if it moved, drops
        every cached answer from view immediately.

        Called once per turn. It costs one small aggregate query, which
        is the price of the guarantee that re-ingesting the knowledge
        base can never serve a stale answer — cheap next to the LLM call
        a hit avoids, and cheap next to the cost of being wrong.
        """
        current = kb_signature(conn)
        if current == self.kb_signature:
            return
        self.kb_signature = current
        with self._lock:
            self._entries_fingerprint = None
            self._entries = []

    def _model_for(self, embedding_model: str | None) -> str:
        """Which embedding model this call is scoped to.

        Normally the one this cache was built with. The exception is the
        quota fallback: when the local provider answers, its vectors live
        in a different space, so entries it writes must be scoped to ITS
        name or a later Gemini query would match against them and get a
        near-neighbour that is not one.

        Scoping per call rather than skipping the cache outright is
        deliberate. A quota outage lasts hours, and it is exactly when the
        cache earns its keep — turning it off then means every visitor pays
        full latency during the one period the system is already degraded.
        The two models simply keep separate entries.
        """
        return embedding_model or self.embedding_model

    # --- reads -------------------------------------------------------

    def lookup_exact(
        self, conn, question: str, embedding_model: str | None = None
    ) -> CachedAnswer | None:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT answer, grounded FROM qa_cache "
                "WHERE question_hash = %s AND kb_signature = %s "
                "  AND embedding_model = %s AND llm_model = %s "
                "  AND created_at >= (NOW() - make_interval(days => %s)) "
                "LIMIT 1",
                (
                    _question_hash(question),
                    self.kb_signature,
                    self._model_for(embedding_model),
                    self.llm_model,
                    self.max_age_days,
                ),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        self._record_hit(conn, question)
        return CachedAnswer(answer=row[0], grounded=bool(row[1]), kind="exact")

    def lookup_semantic(
        self, conn, query_vector: list[float], embedding_model: str | None = None
    ) -> CachedAnswer | None:
        if not self.semantic_enabled or not query_vector:
            return None

        entries = self._get_entries(conn, embedding_model)
        if not entries:
            return None

        query_norm = math.sqrt(sum(value * value for value in query_vector))
        best: _CacheEntry | None = None
        best_score = 0.0
        for entry in entries:
            score = _cosine(query_vector, query_norm, entry.vector, entry.norm)
            if score > best_score:
                best, best_score = entry, score

        if best is None or best_score < self.similarity_threshold:
            return None
        return CachedAnswer(
            answer=best.answer, grounded=best.grounded, kind="semantic", similarity=best_score
        )

    # --- writes ------------------------------------------------------

    def store(
        self, conn, question: str, query_vector: list[float], answer: str, grounded: bool,
        embedding_model: str | None = None,
    ) -> None:
        """Best-effort. A cache write failing must never affect the
        answer the visitor is already getting, so everything here is
        caught and logged."""
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO qa_cache "
                    "(question_hash, question_text, question_embedding, embedding_dim, "
                    " answer, grounded, kb_signature, embedding_model, llm_model) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (question_hash) DO UPDATE SET "
                    "  answer = EXCLUDED.answer, "
                    "  grounded = EXCLUDED.grounded, "
                    "  question_embedding = EXCLUDED.question_embedding, "
                    "  embedding_dim = EXCLUDED.embedding_dim, "
                    "  kb_signature = EXCLUDED.kb_signature, "
                    "  embedding_model = EXCLUDED.embedding_model, "
                    "  llm_model = EXCLUDED.llm_model, "
                    "  created_at = CURRENT_TIMESTAMP",
                    (
                        _question_hash(question),
                        normalize_question(question)[:_MAX_CACHEABLE_QUESTION_CHARS],
                        to_pgvector(query_vector),
                        len(query_vector),
                        answer,
                        # MySQL took TINYINT(1); Postgres BOOLEAN
                        # rejects an int outright.
                        bool(grounded),
                        self.kb_signature,
                        self._model_for(embedding_model),
                        self.llm_model,
                    ),
                )
            conn.commit()
        except Exception:
            # Postgres aborts the whole transaction on any error, so
            # swallowing this without a rollback leaves the connection
            # unusable for the REST OF THE REQUEST - every later query
            # fails with InFailedSqlTransaction. MySQL did not behave this
            # way, which is why the original code got away without it.
            _safe_rollback(conn)
            logger.exception("Failed to write QA cache entry")

    def _record_hit(self, conn, question: str) -> None:
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE qa_cache SET hit_count = hit_count + 1, "
                    "last_hit_at = CURRENT_TIMESTAMP WHERE question_hash = %s",
                    (_question_hash(question),),
                )
            conn.commit()
        except Exception:
            _safe_rollback(conn)
            logger.exception("Failed to record QA cache hit")

    def purge_stale(self, conn) -> int:
        """Drops entries produced under a different KB/model fingerprint,
        entries past the max age, and the coldest rows once the table
        exceeds `max_entries`. Called from the scheduler job — this is
        housekeeping, not something a request should ever pay for.
        """
        removed = 0
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM qa_cache WHERE kb_signature <> %s "
                    "   OR embedding_model <> %s OR llm_model <> %s "
                    "   OR created_at < (NOW() - make_interval(days => %s))",
                    (
                        self.kb_signature,
                        self.embedding_model,
                        self.llm_model,
                        self.max_age_days,
                    ),
                )
                removed += cursor.rowcount

                # Evict coldest-first past the cap. Ordering by hit_count
                # then recency keeps the entries that actually earn their
                # place, instead of whatever happened to be inserted last.
                cursor.execute("SELECT COUNT(*) FROM qa_cache")
                total = cursor.fetchone()[0]
                if total > self.max_entries:
                    # Postgres has no DELETE ... ORDER BY ... LIMIT; the
                    # rows to evict are selected first, then deleted by id.
                    cursor.execute(
                        "DELETE FROM qa_cache WHERE id IN ("
                        "  SELECT id FROM qa_cache "
                        "  ORDER BY hit_count ASC, COALESCE(last_hit_at, created_at) ASC "
                        "  LIMIT %s)",
                        (total - self.max_entries,),
                    )
                    removed += cursor.rowcount
            conn.commit()
        except Exception:
            _safe_rollback(conn)
            logger.exception("QA cache purge failed")
        return removed

    # --- semantic index cache ----------------------------------------

    def _entries_signature(self, conn, embedding_model: str | None = None) -> tuple:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM qa_cache "
                "WHERE kb_signature = %s AND embedding_model = %s AND llm_model = %s",
                (self.kb_signature, self._model_for(embedding_model), self.llm_model),
            )
            row = cursor.fetchone()
        # Deliberately excludes last_hit_at: that changes on every hit,
        # which would invalidate the decoded index constantly and undo
        # the point of having one. Inserts and deletes both move these.
        return tuple(str(value) for value in row)

    def _get_entries(self, conn, embedding_model: str | None = None) -> list[_CacheEntry]:
        # The decoded index is per MODEL as well as per KB: a fallback turn
        # and a primary turn must never see each other's vectors.
        signature = self._entries_signature(conn, embedding_model) + (
            self._model_for(embedding_model),
        )
        with self._lock:
            if signature == self._entries_fingerprint:
                return self._entries

        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT answer, grounded, question_embedding, embedding_dim FROM qa_cache "
                "WHERE kb_signature = %s AND embedding_model = %s AND llm_model = %s "
                "  AND created_at >= (NOW() - make_interval(days => %s))",
                (
                    self.kb_signature,
                    self._model_for(embedding_model),
                    self.llm_model,
                    self.max_age_days,
                ),
            )
            rows = cursor.fetchall()

        entries: list[_CacheEntry] = []
        for answer, grounded, blob, dim in rows:
            try:
                vector = from_pgvector(blob, dim)
            except ValueError:
                # A dimension change (model switch mid-life) makes the
                # blob unreadable. Skip it; purge_stale removes it.
                continue
            entries.append(
                _CacheEntry(
                    answer=answer,
                    grounded=bool(grounded),
                    vector=tuple(vector),
                    norm=math.sqrt(sum(v * v for v in vector)),
                )
            )

        with self._lock:
            self._entries_fingerprint = signature
            self._entries = entries
        return entries


def is_cacheable_question(question: str) -> bool:
    """Gate applied before a question is ever written to a shared cache.
    See the module docstring for why each condition is here."""
    if not question or not question.strip():
        return False
    if len(question) > _MAX_CACHEABLE_QUESTION_CHARS:
        return False
    if contains_contact_details(question):
        return False
    if _SELF_INTRODUCTION_RE.search(question):
        return False
    return True

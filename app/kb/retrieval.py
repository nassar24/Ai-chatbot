"""Retrieves the top-k most relevant KB chunks for a visitor query using hybrid retrieval.

Hybrid retrieval merges:
1. Vector similarity path: cosine similarity over embedding vectors.
2. Keyword matching path: token-containment against `section_title` —
   fraction of the query's content words (stopwords excluded) that
   appear in the title, exact or close-fuzzy per word (typo tolerance).

Why token-containment and not whole-string fuzzy matching (rapidfuzz
partial_ratio/token_set_ratio): tested directly against real cases
before committing to this — "who is hossam?" against a title containing
"Hossam" verbatim scored only ~57% under partial_ratio, because
whole-string scorers measure overall edit distance, not "does a specific
term appear." An earlier version of this file patched around that with
a flat +0.85 bonus when any token matched, which worked but conflated
two different signals (fuzzy string similarity, term containment) in one
number. Token-containment gives the sharp, near-binary signal actually
wanted here directly: 1.0 when every content word in the query appears
in the title, 0.0 when none do.

Merging both paths gives proper-noun and short/sparse queries ("Who is
Hossam?") a direct, non-semantic path to the right chunk while
preserving semantic context for broader questions. It's also the fix for
Gemini's narrower relevant/irrelevant score gap vs Voyage — the merged
score doesn't inherit that mushiness, since the keyword component is
closer to binary.

PERFORMANCE — the decoded-chunk cache
-------------------------------------
Every chat turn scores the query against the WHOLE table; there's no
index that helps, because the ranking function is cosine similarity in
Python. The naive version of that re-read every row (title + full
content + a 3KB embedding blob, ~200KB per request at this KB's size),
re-ran `struct.unpack` over 51x768 floats, and recomputed every chunk's
vector norm and title tokenization — all to produce a result that only
changes when someone re-ingests the knowledge base, which happens on
the order of once a month.

So chunks are decoded once into a process-level cache, with each chunk's
vector norm and tokenized title precomputed at load time. Freshness is
checked per request with a single cheap aggregate query (row count, last
update time, and CRC sums over the title/content hashes) — if that
signature is unchanged, the cached decode is reused. An ingest run
changes the content hashes, so the very next request rebuilds the cache
on its own; nothing has to remember to restart the app or clear
anything. `clear_chunk_cache()` exists for tests, which drop and rebuild
the table faster than a timestamp can distinguish.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz

from app.embeddings.base import EmbeddingProvider
from app.kb.vector_codec import unpack_embedding
from app.text.normalize import ARABIC_STOPWORDS, has_arabic, tokenize

logger = logging.getLogger(__name__)

# Words that carry no distinguishing signal for "does this query mention
# a term found in the title" — excluded so a query like "who is hossam?"
# isn't judged mostly on "who". Broadened from an earlier version that
# only excluded a narrower set and missed "about" (a real bug: every
# "About Apex Creative —" chunk scored KW: 0.850 on any query containing the
# word "about", including totally unrelated ones like "write me a poem
# about cats").
_STOPWORDS = frozenset({
    "who", "is", "are", "was", "were", "the", "a", "an", "do", "you", "does",
    "your", "what", "how", "can", "i", "we", "of", "for", "to", "in", "on",
    "and", "or", "with", "about", "me", "my", "our", "us", "this", "that",
    "have", "has", "will", "would", "could", "any", "all", "much", "it",
})
# Tokenisation lives in app/text/normalize.py so retrieval and the
# guardrails share one script-aware implementation. It was `[a-z0-9]+`
# here, which returned an empty list for Arabic - the keyword half of
# hybrid retrieval scored 0.0 for every Arabic query, silently leaving
# only the vector half.
_ARABIC_STOPWORDS = ARABIC_STOPWORDS
_FUZZY_TOKEN_MATCH_THRESHOLD = 85  # per-word typo tolerance, rapidfuzz.fuzz.ratio scale (0-100)

# How much a body-text keyword match is worth relative to a title match.
# A title is a deliberate label for what a chunk is about; a body word
# can be incidental, so a full body match counts for less than a full
# title match. See `_keyword_score_for_tokens`.
_CONTENT_MATCH_WEIGHT = 0.7

# Minimum merged score for a chunk to be considered relevant at all.
#
# Calibrated against the real Gemini embeddings and this KB (12 genuine
# questions, 8 clearly off-topic ones — see the numbers below), NOT
# guessed. The old 0.2 let every off-topic question through: "what's
# your company's stock ticker?" scored 0.468, above several real
# questions, so the pipeline treated it as grounded and spent an LLM
# call on it.
#
# After the keyword fixes the measured spread was:
#   genuine questions   0.395 .. 0.819   (worst: "where are you located?")
#   off-topic questions 0.304 .. 0.379   (best:  "...stock ticker?")
# which separates cleanly, but by only 0.016 — far too narrow to sit a
# threshold in the middle of without over-fitting to those 20 samples.
# 0.35 instead leaves real headroom under the worst genuine question
# while still rejecting 7 of the 8 off-topic ones.
#
# Deliberately NOT set high enough to be the topicality defense on its
# own. The system prompt already declines off-topic questions correctly
# (verified live: "write me a poem about cats" retrieves chunks and
# still gets the scripted redirect), and a threshold tuned tight enough
# to catch the last off-topic case would start rejecting real questions
# that happen to word themselves poorly. Tune with RETRIEVAL_MIN_SCORE
# after watching real traffic rather than by moving this constant.
_DEFAULT_MIN_SCORE = float(os.environ.get("RETRIEVAL_MIN_SCORE", "0.35"))

# The floor when the local fallback model is answering. A threshold is a
# property of the MODEL, not of the system: bge-m3 packs its scores into a
# lower range than gemini-embedding-001, so reusing 0.35 throws away real
# matches. Swept on the same 69-query eval against the bilingual KB:
#
#   0.35  hit@1 67.8%  hit@3 71.2%  MRR 0.695  rejection 10/10
#   0.30  hit@1 72.9%  hit@3 83.1%  MRR 0.780  rejection 10/10
#   0.25  hit@1 78.0%  hit@3 91.5%  MRR 0.848  rejection  6/10
#
# 0.25 matches Gemini's quality exactly but starts answering off-topic
# questions, and a lead-capture bot that confidently discusses the capital
# of France is a worse failure than one that misses a section. 0.30 keeps
# rejection perfect and recovers most of the recall.
_DEFAULT_FALLBACK_MIN_SCORE = float(
    os.environ.get("RETRIEVAL_MIN_SCORE_FALLBACK", "0.30")
)


@dataclass(frozen=True)
class RetrievedChunk:
    id: int
    section_title: str
    content: str
    score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0


@dataclass(frozen=True)
class _DecodedChunk:
    """One kb_chunks row with everything query-independent precomputed."""

    id: int
    section_title: str
    content: str
    vector: tuple[float, ...]
    norm: float
    title_tokens: frozenset[str]
    content_tokens: frozenset[str]
    embedding_model: str
    # The same chunk embedded by the local fallback model, or empty when
    # the fallback index has not been built. Kept as a SEPARATE vector
    # rather than overwriting the primary one: the two live in different
    # vector spaces and must never be mixed in one comparison.
    fallback_vector: tuple[float, ...] = ()
    fallback_norm: float = 0.0
    fallback_model: str = ""


# Guarded by _cache_lock: the pair must be read and written together, and
# the app scores queries from request threads, per-request lead-capture
# threads, and the scheduler thread.
_cache_lock = threading.Lock()
_cached_signature: tuple | None = None
_cached_chunks: list[_DecodedChunk] = []
_cached_common_tokens: frozenset[str] = frozenset()

# Every chunk vector stacked into one (n_chunks x dim) float32 array with
# each ROW already L2-normalised, so a whole query scores as a single
# matrix-vector product instead of a Python loop over n dot products.
# None when the table is empty or the stored vectors disagree on
# dimension - the per-chunk fallback below handles that case and reports
# it loudly.
_cached_matrix = None
# The same chunks stacked in the FALLBACK model's vector space. Held
# separately because the two spaces are not comparable — mixing them is
# the failure this whole mechanism exists to avoid.
_cached_fallback_matrix = None

# A word appearing in at least this share of chunk bodies carries no
# information about which chunk to pick, so it's ignored on the content
# keyword path. Derived from the corpus rather than hand-listed — a
# hand-written list goes stale the moment the KB changes, and the words
# that need excluding ("company", "team", "project" here) are specific
# to this knowledge base, not to English.
_COMMON_TOKEN_DOC_FREQUENCY = 0.35


def clear_chunk_cache() -> None:
    """Drops the decoded-chunk cache. Production never needs this (the
    signature check below picks up an ingest on its own) — it exists for
    tests, which drop and recreate kb_chunks within the same second and
    can otherwise land on a signature collision across databases.
    """
    global _cached_signature, _cached_chunks, _cached_common_tokens, _cached_matrix
    global _cached_fallback_matrix
    with _cache_lock:
        _cached_signature = None
        _cached_chunks = []
        _cached_common_tokens = frozenset()
        _cached_matrix = None
        _cached_fallback_matrix = None


def kb_signature(db_connection) -> str:
    """Public, stable fingerprint of the knowledge base's current
    contents. Used by the QA cache to scope cached answers to the KB
    version that produced them, so a re-ingest retires them all without
    anyone having to remember a cache-clearing step.
    """
    return "|".join(_table_signature(db_connection))


def _table_signature(db_connection) -> tuple:
    """Cheap fingerprint of kb_chunks: changes whenever any row is
    added, removed, or re-ingested. CRC sums are included alongside the
    timestamp because ingestion can rewrite rows within the same second
    (and tests certainly do), which MAX(last_updated) alone would miss.
    """
    with db_connection.cursor() as cursor:
        cursor.execute(
            # Postgres has no CRC32. md5 over the concatenated, ordered
            # hashes serves the same purpose: a value that changes if any
            # row's content or title changes. string_agg needs an explicit
            # ORDER BY, or the digest varies between identical tables.
            "SELECT COUNT(*), "
            "       COALESCE(MAX(last_updated)::text, '0'), "
            "       COALESCE(md5(string_agg(content_hash, ',' ORDER BY section_title)), ''), "
            "       COALESCE(md5(string_agg(section_title, ',' ORDER BY section_title)), '') "
            "FROM kb_chunks"
        )
        row = cursor.fetchone()
    return tuple(str(value) for value in row)


def _load_chunks(db_connection) -> list[_DecodedChunk]:
    with db_connection.cursor() as cursor:
        cursor.execute(
            # LEFT JOIN: a missing fallback row simply means no fallback
            # is available for that chunk, which is not an error.
            "SELECT c.id, c.section_title, c.content, c.embedding, "
            "       c.embedding_dim, c.embedding_model, "
            "       f.embedding, f.embedding_dim, f.embedding_model "
            "FROM kb_chunks c "
            "LEFT JOIN kb_chunk_fallback_vectors f ON f.chunk_id = c.id"
        )
        rows = cursor.fetchall()

    decoded: list[_DecodedChunk] = []
    for (row_id, title, content, blob, dim, model,
         fb_blob, fb_dim, fb_model) in rows:
        vector = unpack_embedding(blob, dim)
        fallback: tuple[float, ...] = ()
        if fb_blob is not None and fb_dim:
            fallback = tuple(unpack_embedding(fb_blob, fb_dim))
        decoded.append(
            _DecodedChunk(
                id=row_id,
                section_title=title,
                content=content,
                vector=tuple(vector),
                norm=math.sqrt(sum(value * value for value in vector)),
                title_tokens=frozenset(_tokenize(title)),
                content_tokens=frozenset(_tokenize(content)),
                embedding_model=model,
                fallback_vector=fallback,
                fallback_norm=math.sqrt(sum(v * v for v in fallback)),
                fallback_model=fb_model or "",
            )
        )
    return decoded


def _build_matrix(chunks: list[_DecodedChunk], attribute: str = "vector"):
    """Stacks chunk vectors into one normalised float32 array.

    Returns None when there is nothing to stack or the stored vectors
    have inconsistent lengths (a partial re-ingest, or a changed
    embedding dimension). Callers fall back to the per-chunk path, which
    logs the mismatch rather than silently scoring garbage.
    """
    if not chunks:
        return None
    vectors = [getattr(c, attribute) for c in chunks]
    if any(len(v) == 0 for v in vectors):
        return None  # fallback index missing or only partially built
    dims = {len(v) for v in vectors}
    if len(dims) != 1:
        return None
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero-length vector would divide to NaN and poison the ranking.
    np.divide(matrix, np.where(norms == 0.0, 1.0, norms), out=matrix)
    return matrix


def _compute_common_tokens(chunks: list[_DecodedChunk]) -> frozenset[str]:
    """Words present in at least `_COMMON_TOKEN_DOC_FREQUENCY` of chunk
    bodies — this KB's own non-discriminative vocabulary."""
    if not chunks:
        return frozenset()
    counts: dict[str, int] = {}
    for chunk in chunks:
        for token in chunk.content_tokens:
            counts[token] = counts.get(token, 0) + 1
    cutoff = len(chunks) * _COMMON_TOKEN_DOC_FREQUENCY
    return frozenset(token for token, count in counts.items() if count >= cutoff)


def _get_chunks(db_connection):
    global _cached_signature, _cached_chunks, _cached_common_tokens, _cached_matrix
    global _cached_fallback_matrix

    signature = _table_signature(db_connection)
    with _cache_lock:
        if signature == _cached_signature:
            return (_cached_chunks, _cached_common_tokens, _cached_matrix,
                    _cached_fallback_matrix)

    # Loaded outside the lock — decoding is the slow part and holding the
    # lock across it would serialize every concurrent request behind the
    # first one. A rare duplicate load under a race is cheaper than that,
    # and both loads produce identical data.
    chunks = _load_chunks(db_connection)
    common_tokens = _compute_common_tokens(chunks)
    matrix = _build_matrix(chunks, "vector")
    fallback_matrix = _build_matrix(chunks, "fallback_vector")
    with _cache_lock:
        _cached_signature = signature
        _cached_chunks = chunks
        _cached_common_tokens = common_tokens
        _cached_matrix = matrix
        _cached_fallback_matrix = fallback_matrix
    return chunks, common_tokens, matrix, fallback_matrix


def _tokenize(text: str) -> list[str]:
    """Content words for keyword matching, Arabic and Latin alike."""
    return tokenize(text, stopwords=_STOPWORDS | _ARABIC_STOPWORDS, min_length=2)


def _title_match_fraction(
    query_tokens: list[str], title_tokens: frozenset[str]
) -> float:
    """Fraction of the query's content words that appear in the chunk's
    section title — exact match, or a close fuzzy match per word (typo
    tolerance) via rapidfuzz. 1.0 for a full match (e.g. every content
    word in "who is hossam?" -> "hossam" -> found), 0.0 for a genuinely
    unrelated title.

    Takes pre-tokenized inputs: the query is tokenized once per request
    rather than once per chunk (it used to be re-tokenized inside this
    function for all 51 chunks), and titles are tokenized once at cache
    load. The fuzzy scan only runs for tokens that miss exactly, so an
    exact hit costs a set lookup instead of a rapidfuzz sweep.
    """
    if not query_tokens or not title_tokens:
        return 0.0

    matches = 0
    for token in query_tokens:
        if token in title_tokens:
            matches += 1
            continue
        if any(
            fuzz.ratio(token, title_token) >= _FUZZY_TOKEN_MATCH_THRESHOLD
            for title_token in title_tokens
        ):
            matches += 1
    return matches / len(query_tokens)


def _content_match_fraction(
    query_tokens: list[str],
    content_tokens: frozenset[str],
    common_tokens: frozenset[str] = frozenset(),
) -> float:
    """Same fraction, measured against the chunk BODY instead of its
    title. Exact matches only — a fuzzy sweep over every word of every
    chunk body would cost far more than the signal is worth, and typo
    tolerance already applies on the title path.

    `common_tokens` are words that appear in most chunks of this KB and
    therefore say nothing about which chunk to pick. They're excluded
    because a body match on one is pure noise: "what's your company's
    stock ticker?" matched a third of its words against a policy chunk
    purely on "company", scoring higher than several genuine questions.
    The list is derived from the corpus at load time rather than
    hand-written, so it stays correct as the KB changes.
    """
    if not query_tokens or not content_tokens:
        return 0.0
    discriminative = [token for token in query_tokens if token not in common_tokens]
    if not discriminative:
        return 0.0
    matched = sum(1 for token in discriminative if token in content_tokens)
    # Still divided by the FULL query length: a question whose only
    # matching word was a throwaway shouldn't score as a complete match.
    return matched / len(query_tokens)


def _keyword_score_for_tokens(
    query_tokens: list[str],
    title_tokens: frozenset[str],
    content_tokens: frozenset[str] = frozenset(),
    common_tokens: frozenset[str] = frozenset(),
) -> float:
    """Keyword signal for one chunk: the stronger of a title match and a
    discounted body match.

    Title-only matching left a real gap. Plenty of facts are stated in a
    chunk body under a title that shares none of the question's words —
    working hours and the address both live under "About Apex Creative —
    Contact & Location", so "what are your working hours?" scored 0.0 on
    the keyword path and had to survive on Gemini's vector scores alone,
    which put the wrong chunk first. Measured against real embeddings,
    not assumed.

    Body matches are discounted rather than treated as equal: a title is
    a deliberate label for what a chunk is about, while a body word can
    be incidental. `max` rather than a weighted sum keeps a strong title
    match from being diluted by a weak body one.

    Irrelevant questions gain nothing here, which is the point — "stock
    ticker", "weather", "poem about cats" contain words that appear
    nowhere in the KB, so both fractions stay 0 and the separation
    between relevant and irrelevant widens rather than blurs.
    """
    title_fraction = _title_match_fraction(query_tokens, title_tokens)
    content_fraction = _content_match_fraction(query_tokens, content_tokens, common_tokens)
    best = max(title_fraction, _CONTENT_MATCH_WEIGHT * content_fraction)
    # Squared so the signal stays near-binary, which is what this path is
    # for (see module docstring). A full match is unchanged at 1.0, while
    # a lone incidental word matching is heavily discounted: "what's your
    # company's stock ticker?" matched exactly one of three words against
    # "Company Policies — Project Timeline" and collected a flat 0.333,
    # which pushed a completely off-topic question above several genuine
    # ones. Squaring takes that to 0.111 and leaves real matches alone.
    return best * best


def _compute_keyword_score(query: str, section_title: str, content: str = "") -> float:
    """String-in/string-out wrapper, kept for callers and tests that
    score a single chunk ad hoc. The retrieval loop uses the
    pre-tokenized path above instead."""
    return _keyword_score_for_tokens(
        _tokenize(query),
        frozenset(_tokenize(section_title)),
        frozenset(_tokenize(content)),
    )


def retrieve_relevant_chunks(
    query: str,
    embedding_provider: EmbeddingProvider,
    db_connection,
    top_k: int = 4,
    min_score: float | None = None,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
    use_hybrid: bool = True,
    query_vector: list[float] | None = None,
    script_query: str | None = None,
) -> list[RetrievedChunk]:
    """Returns up to `top_k` chunks scoring at least `min_score`.

    When `use_hybrid=True` (default), scores are a weighted combination of
    vector cosine similarity and fuzzy title matching. When `use_hybrid=False`,
    only vector cosine similarity is used.

    `script_query` decides WHICH LANGUAGE half of the index to answer
    from, and defaults to `query`. It exists because `query` is not always
    the visitor's actual message: the pipeline folds the previous turn or
    two into a retrieval-only query so a pronoun follow-up still finds its
    chunk. Detecting the script from that merged string reads the language
    of the CONVERSATION rather than of the QUESTION, so one English
    question after two Arabic turns was answered from Arabic chunks — the
    model then replied in English about Arabic source text, and guardrail
    rule 4 replaced the whole answer because none of its proper nouns
    appeared in the grounding text. Callers that augment the query should
    pass the visitor's raw message here.

    `query_vector` lets a caller that has already embedded this exact
    query pass the vector in rather than paying for a second embedding
    call. The QA cache needs the vector before retrieval runs (to check
    for a semantic hit), so without this the cache would cost an extra
    metered embedding request on every miss — spending on the hot path
    to save on it.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")
    # Threshold resolution is deferred until the active vector space is
    # known — see below. `None` here means "use the default for whichever
    # model ends up answering".

    if query_vector is None:
        query_vector = embedding_provider.embed_query(query)
    chunks, common_tokens, matrix, fallback_matrix = _get_chunks(db_connection)

    # WHICH VECTOR SPACE ANSWERS THIS QUERY.
    #
    # The knowledge base is embedded twice — by the primary provider and
    # by the local fallback that takes over when the primary's daily quota
    # runs out. Those vectors are not comparable, so the set scored
    # against has to be the set built by the model that just embedded this
    # query. Deciding on MODEL NAME rather than dimension is deliberate:
    # two models can share a dimension, and one already did here, which
    # produced confident nonsense with nothing in the logs.
    current_model = getattr(embedding_provider, "model_name", None)
    primary_models = {c.embedding_model for c in chunks}
    fallback_models = {c.fallback_model for c in chunks if c.fallback_model}
    using_fallback = (
        current_model is not None
        and current_model not in primary_models
        and current_model in fallback_models
    )
    if min_score is None:
        min_score = (
            _DEFAULT_FALLBACK_MIN_SCORE if using_fallback else _DEFAULT_MIN_SCORE
        )

    if using_fallback:
        chunk_vectors = [c.fallback_vector for c in chunks]
        chunk_norms = [c.fallback_norm for c in chunks]
        matrix = fallback_matrix
        logger.info(
            "Retrieving with the fallback embedding model %r (%s chunks indexed).",
            current_model, len(fallback_models) and len(chunks),
        )
    else:
        chunk_vectors = [c.vector for c in chunks]
        chunk_norms = [c.norm for c in chunks]

    # Cosine similarity is dot(a,b) / (|a|*|b|). Chunk rows are stored
    # pre-normalised at cache load and the query is normalised once here,
    # so the whole scan reduces to one matrix-vector product.
    query_array = np.asarray(query_vector, dtype=np.float32)
    query_norm = float(np.linalg.norm(query_array))
    query_tokens = _tokenize(query) if use_hybrid else []

    total_weight = vector_weight + keyword_weight if use_hybrid else 1.0
    norm_vector_weight = (vector_weight / total_weight) if use_hybrid and total_weight > 0 else 1.0
    norm_keyword_weight = (keyword_weight / total_weight) if use_hybrid and total_weight > 0 else 0.0

    # A stored vector whose length doesn't match the query's means the
    # KB was embedded by a different model (or a different
    # GEMINI_EMBEDDING_DIMENSION) than the one answering now. `zip` would
    # silently truncate to the shorter of the two and produce a
    # meaningless similarity — retrieval then degrades to keyword-only
    # matching and the bot quietly answers from the wrong chunks or falls
    # back to "I don't have that information", with nothing in the logs.
    # Ingestion re-embeds on a model change, so in normal operation this
    # cannot happen; it shows up after a partial ingest or a dimension
    # change, and it must be loud when it does.
    # The index must have been BUILT by the model now querying it.
    # Dimension alone is not enough: two different models can share a
    # dimension, and one silently did. Aligning the test fake to 768 for
    # the vector(768) column removed the accidental protection the
    # dimension check had been providing, and a pytest run left the index
    # full of bag-of-words fixture vectors that scored as garbage with
    # nothing in the logs. Comparing model NAMES catches that, and catches
    # a half-finished model swap too.
    stored_models = primary_models | fallback_models
    if current_model and stored_models and current_model not in stored_models:
        logger.error(
            "Embedding model mismatch: querying with %r but the index was built "
            "by %s. Retrieval results are meaningless until the knowledge base "
            "is re-ingested (scripts/ingest_kb.py).",
            current_model,
            ", ".join(sorted(repr(m) for m in stored_models)),
        )

    mismatched = [
        c for c, v in zip(chunks, chunk_vectors) if len(v) != len(query_vector)
    ]
    if mismatched:
        logger.error(
            "Embedding dimension mismatch: query is %s-dim but %s of %s stored "
            "chunk(s) are not (e.g. %r at %s-dim). The knowledge base needs "
            "re-ingesting with the current embedding model — retrieval results "
            "until then are meaningless.",
            len(query_vector),
            len(mismatched),
            len(chunks),
            mismatched[0].section_title,
            len(mismatched[0].vector),
        )

    # Fast path: one matvec over the pre-normalised matrix. Falls back to
    # the per-chunk loop when the matrix could not be built (empty table,
    # or stored vectors of differing length) or when the query dimension
    # does not match it - the cases the mismatch error above describes.
    vector_scores = None
    if matrix is not None and query_norm > 0.0 and matrix.shape[1] == query_array.shape[0]:
        vector_scores = np.maximum(matrix @ (query_array / query_norm), 0.0)

    scored: list[RetrievedChunk] = []
    for index, chunk in enumerate(chunks):
        if vector_scores is not None:
            vec_score = float(vector_scores[index])
        elif (
            query_norm == 0.0
            or chunk_norms[index] == 0.0
            or len(chunk_vectors[index]) != len(query_vector)
        ):
            vec_score = 0.0
        else:
            dot = sum(x * y for x, y in zip(query_vector, chunk_vectors[index]))
            vec_score = max(0.0, dot / (query_norm * chunk_norms[index]))

        if use_hybrid:
            kw_score = _keyword_score_for_tokens(
                query_tokens, chunk.title_tokens, chunk.content_tokens, common_tokens
            )
            merged_score = (vec_score * norm_vector_weight) + (kw_score * norm_keyword_weight)
        else:
            kw_score = 0.0
            merged_score = vec_score

        scored.append(
            RetrievedChunk(
                id=chunk.id,
                section_title=chunk.section_title,
                content=chunk.content,
                score=merged_score,
                vector_score=vec_score,
                keyword_score=kw_score,
            )
        )

    scored.sort(key=lambda chunk: chunk.score, reverse=True)
    return _prefer_matching_script(script_query or query, scored, top_k, min_score)


def _prefer_matching_script(
    query: str, scored: list[RetrievedChunk], top_k: int, min_score: float
) -> list[RetrievedChunk]:
    """Answers an Arabic question from the Arabic half of the index, and
    an English one from the English half.

    The knowledge base is stored twice, once per language, because an
    Arabic question embedded against English text scored ~0.40 where the
    same question in English scored ~0.63 — barely clear of the 0.35
    floor, so ranking among Arabic results was close to random.

    Storing both halves in one index then created the opposite problem.
    The Arabic team headings carry the Latin name too ("أحمد نصار (Ahmed
    Nassar)"), so "Who is Nassar?" started ranking the ARABIC chunk above
    the English one — an English visitor being answered from Arabic
    source text. Three name lookups regressed that way, which is what
    this split is really for: the two halves say the same thing, so
    whichever one matches the visitor's own language is the right one,
    and letting them compete on raw score only decides it by accident.

    Falls back to the whole index rather than returning nothing, so a
    topic that exists in only one language is still answerable — and so
    Arabizi, which is Latin script but not English, keeps working the way
    it did before.
    """
    query_is_arabic = has_arabic(query)
    same_script = [
        chunk
        for chunk in scored
        if has_arabic(chunk.section_title + " " + chunk.content) == query_is_arabic
    ]

    preferred = [chunk for chunk in same_script[:top_k] if chunk.score >= min_score]
    if preferred:
        return preferred

    return [chunk for chunk in scored[:top_k] if chunk.score >= min_score]

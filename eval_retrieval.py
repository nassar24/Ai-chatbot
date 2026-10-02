"""Retrieval quality evaluation against REAL embeddings.

Answers one question with a number: when a visitor asks something, does
the right knowledge base section come back?

Run it after touching anything that moves retrieval - chunking, the
embedding model, the keyword scorer, the stopword list, or
RETRIEVAL_MIN_SCORE. It turns "I calibrated it and eyeballed the output"
into a figure that can be compared against the previous figure.

    python eval_retrieval.py                 # summary + every miss
    python eval_retrieval.py --verbose       # every query, hit or miss
    python eval_retrieval.py --min-score 0.5 # try a different threshold

WHY REAL EMBEDDINGS, NOT THE TEST FAKE
--------------------------------------
tests/ uses FakeEmbeddingProvider, a deterministic bag-of-words vector.
That proves the ranking code executes; it cannot fail the way real
retrieval fails. "Where are you located?" returns the wrong section under
real Gemini embeddings and would look fine under a fake that shares the
literal word "location". Every number this script prints costs real
embedding API calls, on purpose.

METRICS
-------
hit@1 / hit@3   does the expected section come back first / in the top 3
MRR             mean reciprocal rank - 1.0 if first, 0.5 if second, and
                so on, 0 if absent. One number that rewards ranking
                rather than just presence.

Off-topic queries are scored INVERTED: they are labeled with a null
expectation and count as correct only when retrieval returns nothing
above the threshold. Folding them into the same hit rate would let a
system that answers everything look good, which is the failure mode the
threshold exists to prevent. They are reported separately as a
correct-rejection rate.

Pronoun-only follow-ups are run through the same query augmentation the
pipeline uses (app.rag.pipeline._build_retrieval_query), so they measure
the path that actually serves them.
"""

from __future__ import annotations

import sys as _sys
# Arabic in the miss report crashes a cp1252 console otherwise.
_sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.db import get_connection  # noqa: E402
from app.embeddings.gemini import GeminiEmbeddingProvider  # noqa: E402
from app.kb.retrieval import retrieve_relevant_chunks  # noqa: E402
from app.llm.base import ChatMessage  # noqa: E402
from app.rag.pipeline import _build_retrieval_query  # noqa: E402

SET_PATH = Path(__file__).parent / "eval" / "retrieval_set.json"
TOP_K = 5  # deeper than the pipeline's 4, so a near-miss at rank 5 is visible


def load_set() -> dict:
    return json.loads(SET_PATH.read_text(encoding="utf-8"))


def validate_labels(queries, conn) -> list[str]:
    """Every expected section must exist verbatim in kb_chunks.

    A typo or a renamed heading would otherwise show up as a retrieval
    regression, sending someone to debug the search when the label is
    what moved. Fail loudly instead.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT section_title FROM kb_chunks")
        titles = {row[0] for row in cur.fetchall()}
    if not titles:
        return ["kb_chunks is empty - run scripts/ingest_kb.py first"]
    # The pytest fixtures drop and rebuild kb_chunks with a bag-of-words
    # fake provider. Scoring against that produces a meaningless ~8% and
    # looks exactly like a catastrophic regression, which has now wasted
    # three runs. Refuse instead of reporting a number.
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT embedding_model FROM kb_chunks")
        models = {row[0] for row in cur.fetchall()}
    fake = {m for m in models if "fake" in m.lower()}
    if fake:
        return [f"kb_chunks holds TEST FIXTURE embeddings {sorted(fake)} - "
                "a pytest run overwrote the index. Re-run scripts/ingest_kb.py."]
    return [
        f"label not found in kb_chunks: {q['expected']!r}  (query: {q['query']!r})"
        for q in queries
        if q.get("expected") and q["expected"] not in titles
    ]


class PacedProvider:
    """Wraps the embedding provider with backoff on quota errors.

    Gemini's free tier allows 100 embed requests per minute. A single
    re-ingest spends 52 of them, so a full eval run immediately after one
    hits the wall. This is a batch tool, so it waits rather than failing -
    but note the APP has no such retry: a 429 during real traffic
    surfaces to the visitor as an error. Worth fixing separately.
    """

    def __init__(self, inner, pause: float = 0.65):
        self._inner = inner
        self._pause = pause

    @property
    def model_name(self):
        return self._inner.model_name

    def embed_query(self, text):
        for attempt in range(6):
            try:
                value = self._inner.embed_query(text)
                time.sleep(self._pause)  # stay under 100/min
                return value
            except Exception as exc:
                if "RESOURCE_EXHAUSTED" not in str(exc) and "429" not in str(exc):
                    raise
                wait = min(60, 20 * (attempt + 1))
                print(f"    [quota] waiting {wait}s ...", flush=True)
                time.sleep(wait)
        raise RuntimeError("embedding quota still exhausted after 6 retries")


def rank_of(expected: str, results) -> int | None:
    for i, chunk in enumerate(results, start=1):
        if chunk.section_title == expected:
            return i
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="print every query, not just misses")
    parser.add_argument("--min-score", type=float, default=None,
                        help="override the relevance floor to test a different threshold")
    parser.add_argument("--provider", choices=("gemini", "local"), default="gemini",
                        help="which embedding provider to score (must match the "
                             "provider the target database was ingested with)")
    parser.add_argument("--model", default=None,
                        help="model id, for --provider local")
    args = parser.parse_args()

    data = load_set()
    queries = data["queries"]
    if args.provider == "local":
        from app.embeddings.local import LocalEmbeddingProvider
        # No quota to pace around - that is the point of running locally.
        provider = LocalEmbeddingProvider(model=args.model)
    else:
        provider = PacedProvider(GeminiEmbeddingProvider())
    conn = get_connection()

    problems = validate_labels(queries, conn)
    if problems:
        print("LABEL VALIDATION FAILED - refusing to score against a stale set:\n")
        for p in problems:
            print("  " + p)
        conn.close()
        return 2

    on_topic = [q for q in queries if q.get("expected")]
    off_topic = [q for q in queries if not q.get("expected")]

    ranks: dict[str, int | None] = {}
    per_category: dict[str, list] = defaultdict(list)
    misses = []

    for q in on_topic:
        history = [ChatMessage(role="user", content=h) for h in q.get("history", [])]
        # Follow-ups go through the same augmentation the pipeline applies.
        retrieval_query = _build_retrieval_query(q["query"], history)
        results = retrieve_relevant_chunks(
            retrieval_query, provider, conn, top_k=TOP_K, min_score=args.min_score
        )
        rank = rank_of(q["expected"], results)
        ranks[q["query"]] = rank
        per_category[q["category"]].append(rank)
        got = results[0].section_title if results else "(nothing returned)"
        if rank != 1:
            misses.append((q, rank, got, results))
        if args.verbose:
            mark = "hit@1" if rank == 1 else (f"rank {rank}" if rank else "MISS ")
            print(f"  [{mark:>7}] {q['query'][:62]:64s} -> {got[:44]}")

    # Off-topic: correct means nothing cleared the threshold.
    rejected, leaked = 0, []
    for q in off_topic:
        results = retrieve_relevant_chunks(
            q["query"], provider, conn, top_k=TOP_K, min_score=args.min_score
        )
        if results:
            leaked.append((q, results[0]))
        else:
            rejected += 1

    total = len(on_topic)
    hit1 = sum(1 for r in ranks.values() if r == 1)
    hit3 = sum(1 for r in ranks.values() if r and r <= 3)
    mrr = sum(1 / r for r in ranks.values() if r) / total if total else 0.0

    print("\n" + "=" * 78)
    print(f"RETRIEVAL EVAL - {total} on-topic queries, {len(off_topic)} off-topic")
    print(f"threshold: {args.min_score if args.min_score is not None else 'default (RETRIEVAL_MIN_SCORE)'}")
    print("=" * 78)
    print(f"  hit@1                {hit1}/{total}   {hit1/total*100:5.1f}%")
    print(f"  hit@3                {hit3}/{total}   {hit3/total*100:5.1f}%")
    print(f"  MRR                  {mrr:.3f}")
    print(f"  correct rejection    {rejected}/{len(off_topic)}   "
          f"{rejected/len(off_topic)*100:5.1f}%   (off-topic returning nothing)")

    print("\n  by category (hit@1 / hit@3 / n):")
    for cat in sorted(per_category):
        rs = per_category[cat]
        h1 = sum(1 for r in rs if r == 1)
        h3 = sum(1 for r in rs if r and r <= 3)
        print(f"    {cat:18s} {h1:2d} / {h3:2d} / {len(rs):2d}")

    if misses:
        print(f"\n  {len(misses)} queries not ranked first:")
        for q, rank, got, _ in misses:
            where = f"rank {rank}" if rank else "NOT IN TOP 5"
            print(f"\n    [{where}] {q['query']!r}  ({q['category']})")
            print(f"      expected: {q['expected']}")
            print(f"      got 1st : {got}")
            print(f"      why here: {q['note']}")

    if leaked:
        print(f"\n  {len(leaked)} off-topic queries that returned something:")
        for q, top in leaked:
            print(f"    {q['query']!r} -> {top.section_title} (score {top.score:.3f})")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

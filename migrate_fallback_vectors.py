"""One-off migration: creates `kb_chunk_fallback_vectors` on an EXISTING
PostgreSQL database.

WHY A SECOND SET OF VECTORS EXISTS
----------------------------------
The Gemini free tier allows 1,000 embed requests per DAY
(EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier). Every
chat turn spends one, so the bot stops answering entirely once that runs
out — which it did. The fallback provider keeps it answering from a local
model until the quota resets.

That only works if the knowledge base is embedded TWICE. Gemini vectors
and local vectors are different spaces, and scoring a query from one
against chunks from the other returns confident nonsense rather than an
error.

Both are stored at 768 dimensions — the one width this system uses
everywhere, including the QA cache — so the two spaces are NOT
distinguishable by size. That is deliberate but it removes a safety net,
and it is exactly the shape of a bug this project already hit once: a
test fake aligned to 768 became indistinguishable from the real model and
retrieval silently served garbage with nothing in the logs. Retrieval
therefore selects the vector set by MODEL NAME, never by dimension, and
that check is the only thing standing between a fallback query and the
wrong index.

bge-m3 is natively 1024 and is truncated to fit. Measured on the same
69-query eval, truncation costs nothing: hit@1 76.3% / hit@3 86.4% /
MRR 0.805 at 768 versus 72.9 / 83.1 / 0.780 at 1024, both rejecting
10/10 off-topic questions.

A separate table rather than extra columns on kb_chunks, for three
reasons: the ingest path stays completely untouched (the primary index
is the one that must never break), ON DELETE CASCADE keeps the two in
step when a section is removed from the KB, and a missing table or row
simply means "no fallback available" rather than a half-populated column.

`content_hash` is copied from kb_chunks at build time so staleness is
detectable: if the KB is re-ingested and the fallback index is not
rebuilt, the hashes diverge and `scripts/build_fallback_index.py` can
re-embed only what actually changed.

Safe to re-run.

Run: python migrate_fallback_vectors.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from app.db import get_connection  # noqa: E402

# 1024 to match BAAI/bge-m3. Measured against gemini-embedding-001 on the
# same 69-query eval: hit@1 74.6% vs 78.0%, MRR 0.803 vs 0.848, Arabic
# 10/12 vs 11/12, and — the reason it was chosen over the lighter
# multilingual-e5-base — it still rejects 7/10 off-topic questions where
# e5 rejects 0/10 at any threshold. Degraded mode has to stay safe, not
# just usable. See eval/embedding_comparison.md.
_CREATE = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS kb_chunk_fallback_vectors (
    chunk_id        INTEGER      PRIMARY KEY
                    REFERENCES kb_chunks(id) ON DELETE CASCADE,
    embedding       vector(768)  NOT NULL,
    embedding_dim   SMALLINT     NOT NULL,
    embedding_model VARCHAR(100) NOT NULL,
    content_hash    CHAR(64)     NOT NULL,
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_kb_fallback_model
    ON kb_chunk_fallback_vectors (embedding_model);
"""

_EXPECTED_COLUMNS = (
    "chunk_id", "embedding", "embedding_dim", "embedding_model",
    "content_hash", "updated_at",
)


def main() -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(_CREATE)
        conn.commit()
        print("Applied: CREATE TABLE IF NOT EXISTS kb_chunk_fallback_vectors")
    finally:
        conn.close()

    # Verify rather than trusting the CREATE reported success.
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'kb_chunk_fallback_vectors'"
            )
            columns = [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()

    print("\nkb_chunk_fallback_vectors columns now:", columns)
    missing = [c for c in _EXPECTED_COLUMNS if c not in columns]
    if missing:
        print(f"\nSTILL MISSING: {missing} — something went wrong above.")
    else:
        print("\nAll expected columns confirmed present.")
        print("Next: python scripts/build_fallback_index.py")


if __name__ == "__main__":
    main()

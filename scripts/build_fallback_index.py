"""Builds (or refreshes) the local fallback vectors for every KB chunk.

Run this AFTER any `scripts/ingest_kb.py`, and whenever the fallback
model changes:

    python scripts/build_fallback_index.py

Costs no Gemini quota — the whole point is that it runs when Gemini is
unavailable. It reads the chunk text that ingest already stored and
embeds it locally, so it works fine while the primary provider is capped.

Only re-embeds what actually needs it. A chunk is skipped when its stored
fallback vector was built from the same `content_hash` by the same model,
which makes this cheap to run on a schedule and safe to run twice.

Chunks whose text changed since their fallback vector was written are
re-embedded; the `content_hash` copied into the fallback table is what
makes that detectable. Rows for deleted chunks disappear on their own —
the table cascades from kb_chunks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.db import get_connection  # noqa: E402
from app.embeddings.local import LocalEmbeddingProvider  # noqa: E402
from app.kb.vector_codec import to_pgvector  # noqa: E402

_BATCH = 16


def main() -> int:
    model_name = os.environ.get("FALLBACK_EMBEDDING_MODEL", "BAAI/bge-m3")
    # Must match EMBEDDING_FALLBACK_DIM in app/api/app.py: the index and
    # the queries against it have to be the same width, and 768 is the one
    # width the whole system uses.
    dim = int(os.environ.get("EMBEDDING_FALLBACK_DIM", "768"))
    provider = LocalEmbeddingProvider(model_name, truncate_dim=dim)

    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            # LEFT JOIN so chunks with no fallback vector yet come back too.
            cursor.execute(
                "SELECT c.id, c.section_title, c.content, c.content_hash, "
                "       f.content_hash, f.embedding_model "
                "FROM kb_chunks c "
                "LEFT JOIN kb_chunk_fallback_vectors f ON f.chunk_id = c.id "
                "ORDER BY c.id"
            )
            rows = cursor.fetchall()

        stale = [
            (cid, title, content, chash)
            for cid, title, content, chash, fhash, fmodel in rows
            if fhash != chash or fmodel != model_name
        ]
        print(f"chunks: {len(rows)}   need embedding: {len(stale)}")
        if not stale:
            print("Fallback index already current — nothing to do.")
            return 0

        print(f"Loading {model_name} (first run downloads the weights)...")
        written = 0
        for start in range(0, len(stale), _BATCH):
            batch = stale[start:start + _BATCH]
            # Same "title + body" text the primary index embeds, so the two
            # halves rank on the same information.
            texts = [f"{title}\n{content}" for _, title, content, _ in batch]
            vectors = provider.embed_documents(texts)

            with connection.cursor() as cursor:
                for (cid, _, _, chash), vector in zip(batch, vectors):
                    cursor.execute(
                        "INSERT INTO kb_chunk_fallback_vectors "
                        "  (chunk_id, embedding, embedding_dim, embedding_model, "
                        "   content_hash, updated_at) "
                        "VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP) "
                        "ON CONFLICT (chunk_id) DO UPDATE SET "
                        "  embedding = EXCLUDED.embedding, "
                        "  embedding_dim = EXCLUDED.embedding_dim, "
                        "  embedding_model = EXCLUDED.embedding_model, "
                        "  content_hash = EXCLUDED.content_hash, "
                        "  updated_at = CURRENT_TIMESTAMP",
                        (cid, to_pgvector(vector), len(vector), model_name, chash),
                    )
            connection.commit()
            written += len(batch)
            print(f"  embedded {written}/{len(stale)}")

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT embedding_model, count(*), min(vector_dims(embedding)) "
                "FROM kb_chunk_fallback_vectors GROUP BY 1"
            )
            print("\nfallback index now:", cursor.fetchall())
    finally:
        connection.close()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

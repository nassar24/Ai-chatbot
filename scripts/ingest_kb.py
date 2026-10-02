"""CLI entry point: ingest the knowledge base into PostgreSQL.

Usage:
    python scripts/ingest_kb.py knowledge_base.md knowledge_base_ar.md

Pass EVERY knowledge-base file in one command. `ingest_knowledge_base`
deletes any stored chunk that is not present in the text it is given —
that is what keeps a removed section from lingering in the index — so
ingesting the English file alone would delete every Arabic chunk, and
vice versa. Running it twice, once per file, leaves you with whichever
language you ingested last.

Both files are ingested as one document. Each one's YAML frontmatter is
stripped first: only the first file's block would sit at the start of the
concatenation, and the others would be indexed as though their metadata
were knowledge.

Requires environment variables: DB_HOST, DB_NAME, DB_USER, DB_PASSWORD,
GOOGLE_API_KEY (see .env.example).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.db import get_connection

from app.embeddings.gemini import GeminiEmbeddingProvider
from app.kb.chunker import strip_frontmatter
from app.kb.ingest import ingest_knowledge_base


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/ingest_kb.py <knowledge-base.md> [more.md ...]\n"
            "Pass every KB file at once — ingesting one alone deletes the "
            "others' chunks as stale.",
            file=sys.stderr,
        )
        return 1

    kb_paths = [Path(arg) for arg in sys.argv[1:]]
    missing = [p for p in kb_paths if not p.is_file()]
    if missing:
        for p in missing:
            print(f"Error: file not found: {p}", file=sys.stderr)
        return 1

    documents = [strip_frontmatter(p.read_text(encoding="utf-8")) for p in kb_paths]
    markdown_text = "\n\n".join(documents)

    for path, doc in zip(kb_paths, documents):
        print(f"Reading {path} ({len(doc):,} chars)")

    provider = GeminiEmbeddingProvider()

    connection = get_connection()
    try:
        report = ingest_knowledge_base(markdown_text, provider, connection)
    finally:
        connection.close()

    print(
        f"Ingestion complete — total: {report.total_chunks}, "
        f"embedded (new/changed): {report.embedded}, "
        f"unchanged (skipped): {report.unchanged}, "
        f"removed (stale): {report.removed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
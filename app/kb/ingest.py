"""Ingests the knowledge-base markdown into kb_chunks.

Re-embedding is targeted, not a full rebuild: a chunk is only sent to
the embedding provider if its content changed since the last run, OR if
it was embedded with a different model than the one currently in use
(e.g. switching from Voyage to Gemini) — comparing content hash alone
would otherwise silently leave stale, incompatible vectors in place,
since the content itself hasn't changed even though the vector space
has. Chunks removed from the source document are deleted from the
table so stale content can't be retrieved.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.embeddings.base import EmbeddingProvider
from app.kb.chunker import RawChunk, chunk_markdown
from app.kb.vector_codec import to_pgvector


@dataclass(frozen=True)
class IngestReport:
    total_chunks: int
    embedded: int
    unchanged: int
    removed: int


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _load_existing_state(db_connection) -> dict[str, tuple[str, str]]:
    """Returns {section_title: (content_hash, embedding_model)} for every
    stored chunk, so ingestion can detect either kind of staleness.
    """
    with db_connection.cursor() as cursor:
        cursor.execute("SELECT section_title, content_hash, embedding_model FROM kb_chunks")
        return {
            title: (content_hash, embedding_model)
            for title, content_hash, embedding_model in cursor.fetchall()
        }


def _load_existing_titles(db_connection) -> set[str]:
    with db_connection.cursor() as cursor:
        cursor.execute("SELECT section_title FROM kb_chunks")
        return {row[0] for row in cursor.fetchall()}


def ingest_knowledge_base(
    markdown_text: str,
    embedding_provider: EmbeddingProvider,
    db_connection,
) -> IngestReport:
    """Chunks `markdown_text`, embeds new/changed/stale-model chunks, and
    syncs kb_chunks.

    `db_connection` is a DB-API 2.0 connection (PyMySQL). Caller owns its
    lifecycle (opening/closing); this function commits its own writes.
    """
    raw_chunks: list[RawChunk] = chunk_markdown(markdown_text)

    existing_state = _load_existing_state(db_connection)
    hashes_by_title = {chunk.section_title: _content_hash(chunk.content) for chunk in raw_chunks}
    current_model = embedding_provider.model_name

    def _needs_embedding(chunk: RawChunk) -> bool:
        stored = existing_state.get(chunk.section_title)
        if stored is None:
            return True  # new chunk
        stored_hash, stored_model = stored
        return stored_hash != hashes_by_title[chunk.section_title] or stored_model != current_model

    changed_chunks = [chunk for chunk in raw_chunks if _needs_embedding(chunk)]

    new_vectors = (
        embedding_provider.embed_documents([chunk.content for chunk in changed_chunks])
        if changed_chunks
        else []
    )
    vectors_by_title = {
        chunk.section_title: vector for chunk, vector in zip(changed_chunks, new_vectors)
    }

    upsert_sql = """
        INSERT INTO kb_chunks
            (section_title, content, content_hash, embedding, embedding_dim, embedding_model)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (section_title) DO UPDATE SET
            content = EXCLUDED.content,
            content_hash = EXCLUDED.content_hash,
            embedding = EXCLUDED.embedding,
            embedding_dim = EXCLUDED.embedding_dim,
            embedding_model = EXCLUDED.embedding_model,
            last_updated = CURRENT_TIMESTAMP
    """
    with db_connection.cursor() as cursor:
        for chunk in changed_chunks:
            vector = vectors_by_title[chunk.section_title]
            cursor.execute(
                upsert_sql,
                (
                    chunk.section_title,
                    chunk.content,
                    hashes_by_title[chunk.section_title],
                    to_pgvector(vector),
                    len(vector),
                    current_model,
                ),
            )
    db_connection.commit()

    current_titles = {chunk.section_title for chunk in raw_chunks}
    stale_titles = _load_existing_titles(db_connection) - current_titles
    if stale_titles:
        with db_connection.cursor() as cursor:
            cursor.executemany(
                "DELETE FROM kb_chunks WHERE section_title = %s",
                [(title,) for title in stale_titles],
            )
        db_connection.commit()

    return IngestReport(
        total_chunks=len(raw_chunks),
        embedded=len(changed_chunks),
        unchanged=len(raw_chunks) - len(changed_chunks),
        removed=len(stale_titles),
    )


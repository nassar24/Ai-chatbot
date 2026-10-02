"""Converts embedding vectors to and from pgvector's wire format.

Under MySQL this module packed float32s into a LONGBLOB with struct, and
carried `embedding_dim` alongside so a length mismatch could be caught at
read time. pgvector has a real `vector(768)` type, so the database now
enforces what the blob-plus-dimension pair only documented: a wrong-length
vector is rejected on INSERT rather than discovered later during scoring.

`embedding_dim` is deliberately KEPT as a column even though the type
constrains it. It is what the retrieval mismatch check reads to produce a
loud error rather than silently meaningless scores, and it stays useful
during a model change, when the column type and the stored data disagree
for exactly as long as the re-ingest takes.
"""

from __future__ import annotations


def to_pgvector(vector: list[float]) -> str:
    """Formats a vector as the literal pgvector accepts: '[1,2,3]'.

    Passed as a bound parameter, so this is not string interpolation into
    SQL - psycopg sends it as a value and Postgres casts it to `vector`.
    """
    if not vector:
        raise ValueError("Cannot store an empty vector.")
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


def from_pgvector(value, dimension: int | None = None) -> list[float]:
    """Reads a vector column back into a plain list of floats.

    psycopg returns it as the string '[1,2,3]' unless the pgvector adapter
    is registered, and as a list/array when it is. Both are handled, since
    which one applies depends on connection setup that callers should not
    have to know about.

    `dimension`, when given, is validated. Ingestion re-embeds on a model
    change so this should never fire in normal operation - it exists
    because a partial re-ingest silently produced meaningless similarity
    scores once already.
    """
    if isinstance(value, str):
        parsed = [float(part) for part in value.strip().strip("[]").split(",") if part]
    else:
        parsed = [float(v) for v in value]

    if dimension is not None and len(parsed) != dimension:
        raise ValueError(
            f"Embedding length ({len(parsed)}) does not match declared "
            f"dimension {dimension}. The stored embedding_dim may be wrong, "
            "or the row was written by a different embedding model."
        )
    return parsed


# Kept under the old names so nothing outside this module had to change
# when the storage format did.
pack_embedding = to_pgvector
unpack_embedding = from_pgvector

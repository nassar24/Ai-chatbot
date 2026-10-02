"""Standalone Gemini embedding sanity check — bypasses the whole app.

Run this locally (it needs real network access to Google's API, which I
can't reach from my sandbox). It calls embed_content directly, with no
retrieval.py, no vector_codec.py, no MySQL — just the raw API — so if
this ALSO shows near-zero similarity for obviously-related sentences,
the bug is upstream of your app code (API key / quota / model / the
output_dimensionality truncation itself). If this script shows healthy,
well-separated scores, the bug is somewhere in the app's ingest/storage/
retrieval path instead.

Usage:
    pip install google-genai python-dotenv --break-system-packages
    python diagnose_gemini_embeddings.py
"""

from __future__ import annotations

import math
import os

from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def embed(client, model, dim, text, task_type):
    result = client.models.embed_content(
        model=model,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=dim,
        ),
    )
    vec = result.embeddings[0].values
    norm = math.sqrt(sum(v * v for v in vec))
    return vec, norm


def main() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL", "gemini-embedding-001")
    dim = int(os.environ.get("GEMINI_EMBEDDING_DIMENSION", "768"))

    if not api_key:
        print("ERROR: GOOGLE_API_KEY / GEMINI_API_KEY not set in environment.")
        return

    print(f"Model: {model}  |  output_dimensionality: {dim}")
    print(f"API key present: yes (len={len(api_key)}, prefix={api_key[:6]}...)")
    print()

    client = genai.Client(api_key=api_key)

    # Also test at FULL dimension (3072, no truncation) as a control —
    # if truncated (768) scores are broken but full-dim scores are fine,
    # that isolates the bug to the Matryoshka truncation step specifically.
    for test_dim, label in [(dim, f"output_dimensionality={dim}"), (3072, "output_dimensionality=3072 (full, no truncation)")]:
        print(f"=== {label} ===")

        doc_vec, doc_norm = embed(
                    client, model, test_dim,
                    "Hossam is a Cybersecurity Specialist on the Apex Creative team responsible for "
                    "protecting digital systems, applications, and data against security threats.",
                    "RETRIEVAL_DOCUMENT",
                )
        unrelated_doc_vec, unrelated_norm = embed(
            client, model, test_dim,
            "The weather today is sunny with a light breeze and mild temperatures.",
            "RETRIEVAL_DOCUMENT",
        )
        query_vec, query_norm = embed(
            client, model, test_dim,
            "Who is Hossam?",
            "RETRIEVAL_QUERY",
        )

        print(f"  doc_norm={doc_norm:.4f}  unrelated_doc_norm={unrelated_norm:.4f}  query_norm={query_norm:.4f}")
        print(f"  cosine(query, RELATED doc)   = {cosine(query_vec, doc_vec):.4f}   <-- should be clearly higher")
        print(f"  cosine(query, UNRELATED doc) = {cosine(query_vec, unrelated_doc_vec):.4f}   <-- should be clearly lower")
        print()

    print("How to read this:")
    print("- Norms near 0 => the API is returning empty/degenerate vectors (key/quota/model issue).")
    print("- Norms look normal (~1.0 at full dim, possibly not 1.0 at truncated dim) but the two")
    print("  cosine scores above are close together / both near 0 => the embedding call itself isn't")
    print("  producing separable vectors here — try task_type='SEMANTIC_SIMILARITY' for both texts")
    print("  as a second control, and double check the API key belongs to a project with the")
    print("  Generative Language API enabled.")
    print("- If truncated (768) is broken but full (3072) is fine => the bug is specific to")
    print("  output_dimensionality truncation; store full 3072-dim vectors instead, or manually")
    print("  L2-renormalize after truncating client-side.")
    print("- If BOTH dims look healthy here => the bug is not the API call; it's in ingest.py /")
    print("  vector_codec.py / retrieval.py or in what's actually stored in kb_chunks right now.")


if __name__ == "__main__":
    main()

import os

from dotenv import load_dotenv
load_dotenv()

from app.db import get_connection
from app.embeddings.gemini import GeminiEmbeddingProvider
from app.kb.retrieval import retrieve_relevant_chunks

# Printed up front on purpose: this script reads whatever's currently in
# kb_chunks for DB_NAME. If DB_NAME points at the same database the
# pytest integration suite just ran against (tests/conftest.py drops and
# rebuilds kb_chunks with fake fixture data), these results will reflect
# stale/fixture rows, not real KB content. See tests/conftest.py's
# _require_test_database for the guard on the test side of this.
print(f"Calibrating against DB_NAME={os.environ.get('DB_NAME')!r} on DB_HOST={os.environ.get('DB_HOST')!r}")

conn = get_connection()
embedder = GeminiEmbeddingProvider()

queries = [
    ("What is your refund policy?", "clearly relevant"),
    ("Do you offer website development?", "clearly relevant"),
    ("Who is Hossam?", "clearly relevant"),
    ("What's your company's stock ticker?", "clearly IRRELEVANT"),
    ("What's the weather like today?", "clearly IRRELEVANT"),
    ("Can you write me a poem about cats?", "clearly IRRELEVANT"),
]

for query, label in queries:
    print(f"\n--- {label}: {query!r} ---")
    results = retrieve_relevant_chunks(query, embedder, conn, top_k=4, min_score=0.0)
    for r in results:
        print(f"  Merged: {r.score:.3f} | Vector: {r.vector_score:.3f} | KW: {r.keyword_score:.3f}  ->  {r.section_title}")

conn.close()
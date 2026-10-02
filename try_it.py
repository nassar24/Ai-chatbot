from dotenv import load_dotenv
load_dotenv()

from app.db import get_connection
from app.embeddings.gemini import GeminiEmbeddingProvider
from app.llm.alibaba import AlibabaLLMProvider
from app.llm.base import ChatMessage
from app.rag.pipeline import answer_query

embedder = GeminiEmbeddingProvider()
llm = AlibabaLLMProvider()
conn = get_connection()


def _print_result(result) -> None:
    print("\n--- RETRIEVED CHUNKS ---")
    for c in result.retrieved_chunks:
        print(f"  Score: {c.score:.3f} | Vector: {c.vector_score:.3f} | KW: {c.keyword_score:.3f} -> {c.section_title}")

    print("\nANSWER:", result.answer)
    print("GROUNDED:", result.grounded)
    print("VIOLATIONS:", result.guardrail_violations)


try:
    # result = answer_query("Who is Hossam?", embedder, llm, conn)
    # result = answer_query("What's the weather like today?", embedder, llm, conn)
    # result = answer_query("Can you write me a poem about cats?", embedder, llm, conn)

    # Multi-turn demo (Phase 1): a pronoun-only follow-up that only makes
    # sense in light of the first turn. `answer_query` folds the last 1-2
    # user turns into the retrieval query when the current query is too
    # thin to stand on its own (see _query_needs_history_context in
    # app/rag/pipeline.py) — but always threads the *full* history
    # through to the LLM as conversational context, capped by token
    # budget rather than message count.
    history: list[ChatMessage] = []

    print("=" * 20, "TURN 1", "=" * 20)
    turn_1_query = "TI'm Mohammed, mohammed@modev.com, I need a website built, for my business, I need it to be fast and responsive, and I want it to be built using modern web technologies, its called modev and its an automation company in egypt"
    result_1 = answer_query(turn_1_query, embedder, llm, conn, conversation_history=history)
    _print_result(result_1)
    history.append(ChatMessage(role="user", content=turn_1_query))
    history.append(ChatMessage(role="assistant", content=result_1.answer))

    print("\n" + "=" * 20, "TURN 2 (pronoun follow-up)", "=" * 20)
    turn_2_query = "0123456789"
    result_2 = answer_query(turn_2_query, embedder, llm, conn, conversation_history=history)
    _print_result(result_2)
finally:
    conn.close()
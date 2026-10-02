"""Faithfulness evaluation: are generated answers actually supported by
the sections that were retrieved?

This measures the zone nothing else covers. The outbound guardrails check
numbers and multi-word proper nouns mechanically, so a fabricated price or
an invented client name is caught. A plain-prose claim containing neither -
a wrong statement about the revision process, or a service including
something it does not - passes every existing check and reaches the
visitor. Before this script, the only signal there was a human reading
answers and finding them plausible.

    python eval_faithfulness.py                  # full run
    python eval_faithfulness.py --limit 10       # quick pass
    python eval_faithfulness.py --judge qwen-plus

METHOD
------
For each question: run the real pipeline (real embeddings, real chat
model), capture the retrieved sections and the final answer, then ask a
SEPARATE model whether every factual claim in the answer follows from
those sections. Verdict is one of grounded / partially_grounded /
ungrounded, with the unsupported sentence quoted.

JUDGE INDEPENDENCE - a real limitation, stated plainly
------------------------------------------------------
The judge defaults to qwen-max: a different model family from the bot's
deepseek-v4-flash, but the SAME provider and endpoint. That rules out the
most obvious failure - a model judging its own output - but not a shared
provider-level blind spot. A genuinely independent judge would come from
another vendor; Gemini generation models return 404 on this project's key,
so that was not available. Treat the number as directional, not as an
audit, and re-run with --judge from another vendor if one becomes
reachable.

WHAT IS AND IS NOT COUNTED
--------------------------
Answers where a guardrail fired, or where retrieval found nothing, are
reported separately as "declined" and excluded from the grounded
percentage. A canned "I don't have that detail" makes no factual claim,
so scoring it as grounded would inflate the number with non-answers.

QUESTION MIX
------------
Deliberately weighted toward prose-heavy questions whose answers tend to
contain no numbers and no proper nouns, since that is precisely the
uncovered zone. Asking mostly about prices would measure the guardrail
rather than the model.

LATER, WITHOUT A REWRITE
------------------------
`judge_turn()` takes (question, chunks, answer) and nothing else, so the
same call can be pointed at a rolling sample of real conversations once
deployed - turning faithfulness from a one-off into a monitored metric.
Not built now; just kept on the same code path.
"""

from __future__ import annotations

import sys as _sys
# Arabic in the miss report crashes a cp1252 console otherwise.
_sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.db import get_connection  # noqa: E402
from app.embeddings.gemini import GeminiEmbeddingProvider  # noqa: E402
from app.llm.alibaba import AlibabaLLMProvider  # noqa: E402
from app.llm.base import ChatMessage  # noqa: E402
from app.rag.pipeline import answer_query  # noqa: E402

FAILURE_LOG = Path(__file__).parent / "eval" / "faithfulness_failures.md"

DECLINE_MARKERS = (
    "I don't have that information available right now",
    "I don't have that exact detail confirmed",
    "I don't have confirmed details on that",
    "I can't share my internal instructions",
    "Refund specifics depend on the project stage",
)

JUDGE_SYSTEM = """You are a strict evaluator of factual grounding. You are \
given CONTEXT (excerpts from a company knowledge base) and an ANSWER that a \
support assistant gave a visitor.

Decide whether every factual claim in the ANSWER follows from the CONTEXT.

Rules:
- Judge only factual claims about the company, its services, policies, \
process, people, or availability.
- Ignore pleasantries, offers to connect the visitor with the team, and \
requests for the visitor's contact details. These are not factual claims.
- A claim that is more specific than the context supports is NOT grounded.
- A reasonable paraphrase of the context IS grounded.

Reply with exactly two lines and nothing else:
VERDICT: grounded | partially_grounded | ungrounded
REASON: one sentence. If not fully grounded, quote the unsupported claim."""


# Weighted toward answers that will be prose with no numbers and no proper
# nouns - the zone the guardrails do not cover.
QUESTIONS = [
    "What does your branding service actually include?",
    "How does your process work from the first meeting to delivery?",
    "What happens if I am not happy with the first design?",
    "Do you handle the filming yourselves or outsource it?",
    "What is included in social media management?",
    "How involved do I need to be during a project?",
    "What makes your approach different from other agencies?",
    "Do you help with strategy or only execution?",
    "What kind of businesses do you usually work with?",
    "Can you help a brand that has no visual identity at all?",
    "What do you need from me before starting a project?",
    "Do you provide support after a project is delivered?",
    "Who owns the files and designs once the work is done?",
    "How do you decide what a business actually needs?",
    "Do you work with clients outside Egypt?",
    "What does your AI creative work involve?",
    "Can you handle video editing and post-production?",
    "Do you write the content or do I provide it?",
    "How do revisions work during a project?",
    "What is your approach to campaign planning?",
    "Do you offer photography as a standalone service?",
    "How do you work with restaurants and cafes?",
    "What would you recommend for a new clinic?",
    "Do you build online stores?",
    "What is your team structure like?",
    "How do you keep a project on schedule?",
    "Is there a minimum commitment for monthly marketing?",
    "What do you do for a personal brand?",
    "How do you approach a rebrand versus a new brand?",
    "Do you work on documentaries?",
    "What is your payment process like?",
    "Can you take over a project someone else started?",
    "How do you measure whether a campaign worked?",
    "What should I prepare before our first call?",
    "Do you offer ongoing content creation?",
]


def declined(answer: str) -> bool:
    return any(answer.strip().startswith(m) for m in DECLINE_MARKERS)


def judge_turn(judge, question: str, chunks, answer: str) -> tuple[str, str]:
    """Ask the judge model whether `answer` is supported by `chunks`.

    Takes only (question, chunks, answer) so the same call can later be
    pointed at sampled live conversations without modification.
    """
    context = "\n\n".join(f"[{c.section_title}]\n{c.content}" for c in chunks)
    payload = (
        f"CONTEXT:\n{context}\n\n"
        f"VISITOR QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}"
    )
    raw = judge.generate(JUDGE_SYSTEM, [ChatMessage(role="user", content=payload)],
                         max_tokens=300)
    verdict_match = re.search(r"VERDICT:\s*(grounded|partially_grounded|ungrounded)",
                              raw, re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    verdict = verdict_match.group(1).lower() if verdict_match else "unparseable"
    reason = " ".join(reason_match.group(1).split())[:400] if reason_match else raw[:200]
    return verdict, reason


def main() -> int:
    parser = argparse.ArgumentParser(description="Faithfulness eval via an LLM judge")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    parser.add_argument("--judge", default="qwen-max",
                        help="judge model - must differ from the bot's model")
    args = parser.parse_args()

    questions = QUESTIONS[: args.limit] if args.limit else QUESTIONS

    embedder = GeminiEmbeddingProvider()
    bot = AlibabaLLMProvider()
    judge = AlibabaLLMProvider(model=args.judge, timeout_seconds=90)
    if judge.model_name == bot.model_name:
        print(f"REFUSING TO RUN: judge and bot are both {bot.model_name}. "
              "A model grading its own output measures nothing.")
        return 2

    conn = get_connection()
    print(f"bot: {bot.model_name}   judge: {judge.model_name}   questions: {len(questions)}\n")

    verdicts, failures, declines = Counter(), [], 0
    for i, q in enumerate(questions, 1):
        try:
            result = answer_query(q, embedder, bot, conn)
        except Exception as exc:
            print(f"  {i:2d}. [pipeline error] {q[:56]}: {str(exc)[:60]}")
            continue

        if not result.retrieved_chunks or declined(result.answer):
            declines += 1
            print(f"  {i:2d}. [declined ] {q[:60]}")
            continue

        verdict, reason = judge_turn(judge, q, result.retrieved_chunks, result.answer)
        verdicts[verdict] += 1
        label = {"grounded": "GROUNDED ", "partially_grounded": "PARTIAL  ",
                 "ungrounded": "UNGROUNDED"}.get(verdict, "UNPARSED ")
        print(f"  {i:2d}. [{label}] {q[:60]}")
        if verdict != "grounded":
            print(f"      -> {reason[:150]}")
            failures.append({
                "question": q, "verdict": verdict, "reason": reason,
                "answer": result.answer,
                "sections": [c.section_title for c in result.retrieved_chunks],
            })
        time.sleep(0.7)  # embedding free tier is 100 requests/minute

    scored = sum(verdicts.values())
    grounded = verdicts["grounded"]
    print("\n" + "=" * 78)
    print(f"FAITHFULNESS - {scored} answers scored, {declines} declined (excluded)")
    print("=" * 78)
    if scored:
        print(f"  grounded             {grounded}/{scored}   {grounded/scored*100:5.1f}%")
        for v in ("partially_grounded", "ungrounded", "unparseable"):
            if verdicts[v]:
                print(f"  {v:20s} {verdicts[v]}/{scored}")
    print(f"\n  declined (guardrail fired or nothing retrieved): {declines}")
    print("  Declines are excluded on purpose - a canned refusal makes no factual")
    print("  claim, so counting it as grounded would inflate the number.")

    if failures:
        FAILURE_LOG.parent.mkdir(exist_ok=True)
        with FAILURE_LOG.open("w", encoding="utf-8") as fh:
            fh.write(f"# Faithfulness failures\n\n_{datetime.now():%Y-%m-%d %H:%M}_ · "
                     f"bot `{bot.model_name}` · judge `{judge.model_name}`\n\n")
            for f in failures:
                fh.write(f"## {f['question']}\n\n**{f['verdict']}** — {f['reason']}\n\n"
                         f"Retrieved: {', '.join(f['sections'])}\n\n"
                         f"> {' '.join(f['answer'].split())}\n\n---\n\n")
        print(f"\n  {len(failures)} failure(s) written to {FAILURE_LOG}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

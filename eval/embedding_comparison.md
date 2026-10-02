# Embedding model comparison

Scored with `eval_retrieval.py` against the same 69 labeled queries
(59 on-topic, 10 off-topic), the same knowledge base and the same hybrid
scoring. Each candidate was ingested into its own database, so the live
index was never disturbed.

Each local model was threshold-swept before comparison. Comparing a tuned
model against an untuned one would have been meaningless — the two local
models needed opposite corrections, e5 upward and bge-m3 downward.

| | gemini-embedding-001 | multilingual-e5-base | BAAI/bge-m3 |
|---|---|---|---|
| dimension | 768 | 768 | 1024 |
| best min_score | 0.35 | *(none works)* | 0.25 |
| **hit@1** | **78.0%** | 69.5% | 74.6% |
| **hit@3** | **91.5%** | 81.4% | 84.7% |
| **MRR** | **0.848** | 0.760 | 0.803 |
| correct rejection | **9/10** | 0/10 | 7/10 |
| arabic | **11/12** | 8/12 | 10/12 |
| arabizi | 1/4 | **2/4** | 0/4 |
| english paraphrase | **12/17** | 11/17 | 10/17 |
| query latency | 300ms–3s | **64ms** | 215ms |
| first load | — | 248s | 923s |
| full re-ingest | 19.5s | **5.3s** | 16.8s |
| quota | 100 req/min | none | none |
| disk / RAM | — | ~1GB | ~2.2GB |

## The finding that mattered most

Arabic was the entire reason for considering a local multilingual model.
Measuring killed that reason: **gemini-embedding-001 scores 11/12 on the
Arabic set — the best of any category**, ahead of English paraphrase.
Arabic retrieval was never the weak part.

The genuinely weak category is **Arabizi** (1/4), and no candidate fixed
it: e5 reached 2/4, bge-m3 scored 0/4. Latin-script Egyptian Arabic has no
standard orthography, so it is a poor fit for any embedding model. If it
matters commercially, the fix is transliteration at the query layer, not a
different model.

## Why e5 could not be rescued by tuning

At 0.35 it returns something for **every** off-topic query. Sweeping does
not help:

| min_score | hit@1 | correct rejection |
|---|---|---|
| 0.35 | 69.5% | 0/10 |
| 0.60 | 30.5% | 10/10 |
| 0.70 | 10.2% | 10/10 |

There is no setting where e5 both finds the right section and rejects
off-topic questions — its relevant and irrelevant score distributions
overlap. Gemini's separate, narrowly, which is what the 0.35 floor rests
on.

bge-m3 behaves the opposite way (scores run low, needs 0.25) and is the
closer contender, but still loses on every quality axis.

## Decision

**Gemini stays.** The operational wins from going local are real — no
quota ceiling, faster queries, no per-request cost, and visitor questions
never leaving the machine — but they do not buy back 3.4 points of hit@1,
6.8 of hit@3, and two of ten off-topic rejections.

`app/embeddings/local.py` stays in the tree as a tested, working option
behind the same interface. Revisit when:

- the 100 req/min ceiling becomes a practical limit (that is ~100 visitor
  questions per minute, and one re-ingest spends 52), or
- a stronger multilingual model appears — re-run
  `eval_retrieval.py --provider local --model X` and compare against the
  table above.

Reproduce any row with:

    EVAL_DB=apexcreative_x_eval python eval_local.py <model>
    DB_NAME=apexcreative_x_eval python eval_retrieval.py --provider local --model <model>

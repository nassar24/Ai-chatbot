# Apex Creative AI Chat — Backend

## Status: Phase 5 complete (sessions, lead capture, escalation emails, API layer, frontend widget)

Phase 1 (chunking, embedding, MySQL storage, vector retrieval), Phase 2
(a standalone RAG loop), Phase 3 (the real system prompt + outbound
guardrails), Phase 4 (session/history persistence), and Phase 5 (lead
capture, escalation, email notifications, the Flask API layer, and the
React frontend widget) are done. Not yet built: an admin view over
captured leads, rate limiting, and closing the confirmation-language
loop noted in the Phase 5 section below.

## What's been verified

- Ran against a real MySQL-compatible server (MariaDB), not mocks.
- `knowledge_base.md` chunks into **44 sections** matching the
  KB's actual heading structure (11 services, 4 packages, 6 policies,
  11 team members, etc. — see the chunk list printed by the ingest
  script, or `tests/test_chunker.py` for the structural assertions).
- Ingestion is idempotent and incremental: re-running with no changes
  re-embeds nothing; editing one section only re-embeds that section;
  deleting a section from the source file removes it from `kb_chunks`.
- Retrieval does real cosine similarity over embedding vectors (not
  keyword matching) — validated with word-overlap-friendly test queries
  against a **fake** embedding provider (see below), since a live
  embedding API key isn't available in this environment.

## Phase 2: retrieval + generation core

`app/rag/pipeline.py`'s `answer_query()` retrieves top-k chunks, and if
none clear the relevance threshold, returns a fixed "I don't have that
information" fallback **without calling the LLM at all** (verified by
test — the fake LLM provider's `last_system_prompt` stays `None` in that
case). If chunks are found, it builds a system prompt around them and
generates a reply.

### Generation provider: Alibaba DashScope, defaulting to deepseek-v4-flash

`app/llm/alibaba.py` calls DashScope's OpenAI-compatible endpoint,
defaulting to `deepseek-v4-flash`. That default was chosen by measuring
candidates against the real question set, not inherited: medians of
qwen3.7-plus 18.8s, deepseek-v4-flash 3.5s, qwen-flash 1.3s. qwen-flash
is the fastest but answers a mixed message ("I want a website, but first
tell me how to build a Python app") with a generic redirect that lets the
lead walk, and writes emoji. deepseek-v4-flash accepts the business,
declines the off-topic half and asks for the WhatsApp number in one
reply. All three repelled 22 of 22 adversarial attempts, so this was a
quality and latency call rather than a safety one. Override with
`DASHSCOPE_MODEL` without touching code.

`app/llm/openrouter.py` remains as an alternative provider behind the
same interface, still defaulting to `qwen/qwen3.7-plus` — OpenRouter uses
its own model-slug namespace, so a DashScope id is not valid there.

`app/rag/pipeline.py` only depends on the `LLMProvider` interface in
`app/llm/base.py`, so switching vendors later doesn't touch the pipeline.

OpenRouter's API isn't reachable from this build/test environment, so
`OpenRouterLLMProvider` is unit-tested with a mocked HTTP layer
(`tests/test_openrouter_provider.py` — verifies request payload shape
and response parsing) rather than a live call. Worth a real smoke test
against a live `OPENROUTER_API_KEY` once you have one.

## Phase 3: system prompt + outbound guardrails

`app/prompts/system_prompt.md` is a verbatim, untouched copy of
the system prompt you supplied — `app/prompts/loader.py` is the only
place that reads it, and nothing in the codebase edits its rules.
`app/rag/prompting.py` now builds the actual prompt sent to the model:
the full system prompt, followed by the retrieved chunks presented as
"the provided knowledge base" for that turn (matching the system
prompt's own framing). This replaces the phase-2 temporary grounding
prompt entirely.

**Outbound guardrails** (`app/rag/guardrails.py`) run on every generated
answer before it's considered safe to show a visitor — a code-level
second layer of defense behind the system prompt's own instructions, per
the build plan's explicit requirement for this:

1. **Refund percentages** — if a retrieved chunk is about refunds and the
   answer contains a `%` figure, the *entire* answer is replaced with the
   system prompt's own deferral line, regardless of whether that
   percentage is technically present in the KB. Permanent policy, not a
   gap.
2. **Ungrounded numbers** — every number in the answer must appear
   verbatim in the retrieved content (exact-string match, so even
   reformatting like comma placement fails this check on purpose — the
   system prompt's own rule says "verbatim"). A failure swaps the whole
   answer for a safe "let me connect you with the team" response.
3. **Internal email leakage** — `hr@apexcreative.example` is always redacted
   from output regardless of configuration (it's named explicitly in the
   build plan as never-to-leak); whatever's configured in
   `INTERNAL_NOTIFICATION_EMAIL` is redacted too, once that's set. This
   is an in-place redaction (→ "our team"), not a full-answer swap, since
   the rest of the reply is usually still fine. The public contact
   address (`info@apexcreative.example`) is never touched.

All three are unit-tested directly (`tests/test_guardrails.py`) and
tested end-to-end through the pipeline with a fake LLM deliberately
returning bad output, to prove the pipeline actually calls the guardrail
and swaps/redacts the answer before it comes back (`tests/test_rag_pipeline.py`).

`RagResult` now carries `guardrail_violations: list[str]` — empty when
clean, otherwise a short machine-readable trail (e.g.
`["ungrounded_numbers:999,999"]`) worth logging in production so
violations (which should be rare, since the model is already instructed
not to do this) are visible for review rather than silently corrected
and forgotten.

## Two decisions made to unblock phase 1 (flagged, not silently assumed)

1. **Embedding vendor: Voyage AI** (`app/embeddings/voyage.py`). The
   build plan specifies an "external LLM API" for generation but never
   names an embeddings vendor — Anthropic's own API doesn't do
   embeddings. Voyage is Anthropic's recommended pairing for Claude-based
   RAG and needs nothing beyond an HTTPS call, so it fits Hostinger
   shared hosting cleanly. Everything else in `app/kb/` only depends on
   the `EmbeddingProvider` interface in `app/embeddings/base.py` — swap
   in an OpenAI (or other) provider by adding one file, nothing else
   changes.
2. **No ANN/vector-index library** (FAISS, etc.). At ~44 chunks, an
   indexed search gives no real speed benefit over a full similarity
   scan (microseconds either way) and would add a compiled-binary
   dependency that's a worse fit for shared hosting. Retrieval is genuine
   embedding cosine similarity, just unindexed — this only becomes a
   real tradeoff at a KB scale Hostinger shared hosting couldn't serve
   anyway.

## Why tests use a fake embedding provider

`tests/fakes.py` has a `FakeEmbeddingProvider` — a deterministic
bag-of-words vector, no network calls. It exists to test **pipeline
mechanics** (chunk → hash → store → rank → re-ingest idempotency) without
a live Voyage API key. It is *not* a stand-in for real semantic quality:
its test queries were deliberately written with literal word overlap
with the target chunk (e.g. "cybersecurity, security monitoring and
vulnerability assessment" for the Cybersecurity Specialist chunk),
because a bag-of-words fake can't do the semantic matching a real
embedding model does. Real retrieval quality needs to be checked again
once `VOYAGE_API_KEY` is live — the ad-hoc queries printed during manual
testing (e.g. "Tell me about your website development services")
ranked oddly under the fake, which is the fake's limitation, not a bug
in the retrieval code itself (confirmed by the targeted tests passing
and by the chunk/hash/delete mechanics all working against real MySQL).

## Setup

Always install into a virtual environment — never into the system or
user Python. Hostinger's Passenger setup gives the app its own venv too,
so matching that locally is what keeps `requirements.txt` honest about
what the deploy actually needs.

```bash
python -m venv venv
source venv/Scripts/activate       # Windows (Git Bash); use venv/bin/activate on Linux/macOS
pip install -r requirements.txt
cp .env.example .env   # fill in DB credentials, VOYAGE_API_KEY/GOOGLE_API_KEY,
                        # DASHSCOPE_API_KEY, SMTP_*, INTERNAL_NOTIFICATION_EMAIL
```

`venv/` is gitignored. Every command below assumes it's active — or
call the interpreter directly, e.g. `./venv/Scripts/python.exe -m pytest`.

Apply the schema:
```bash
mysql -u <user> -p <database> < schema.sql
```

Ingest the knowledge base:
```bash
python scripts/ingest_kb.py /path/to/knowledge_base.md
```

Run the API locally:
```bash
python wsgi.py   # serves on http://localhost:5000
```

Run tests (needs a disposable MySQL/MariaDB test database — the
integration tests drop and recreate `kb_chunks`/`sessions`/`messages`/
`leads` on every run, so never point `DB_NAME` at production):
```bash
export DB_HOST=localhost DB_PORT=3306 DB_USER=... DB_PASSWORD=... DB_NAME=apexcreative_test
pytest tests/ -v
```
The session/leads/email/API tests (`test_sessions_service.py`,
`test_leads_extractor.py`, `test_leads_service.py`,
`test_email_notifications.py`, `test_api_chat.py`) run against mocked
DB/SMTP/LLM and don't need a live database at all.

## Phase 4: session + history persistence

`app/sessions/service.py` — the client (chat widget) generates a UUID
on first load and sends it with every request; that's the join key
(`sessions.session_id`), not IP. `get_or_create_session()` is called on
every request and is idempotent. `load_history()`/`save_turn()` read
and write `messages`, so a page refresh replays the same session
instead of starting over. The full transcript is always stored
untrimmed; only a token-budget-capped slice (existing
`_cap_history_to_token_budget` in `app/rag/pipeline.py`, unchanged) is
what actually gets replayed to the LLM each turn — storage and replay
are two separate concerns on purpose (see the module docstring).

## Phase 5: lead capture + escalation + email notifications

Two new pieces, wired together in `app/api/app.py::_run_lead_capture`,
run after every chat turn (best-effort — failures are logged, never
surfaced to the visitor or allowed to affect the answer already sent):

1. **`app/leads/extractor.py`** — a second, separate LLM call (not the
   main conversational one) reads the last few turns and returns
   strict JSON: any lead fields the visitor has actually stated, plus
   independent `ready_to_submit` and `escalate` flags. Kept as a
   separate call because the current LLM provider has no tool-calling/
   JSON-mode support, and mixing "stay in character, guardrail-safe
   prose" with "produce parseable JSON" in one call risks both.
2. **`app/leads/service.py`** — persists whatever was extracted to
   `leads`, merging new non-null fields onto whatever's already there
   (so a name given in turn 2 and an email given in turn 5 both stick).
   A lead row is created from even a partial capture, so a visitor who
   drops off mid-conversation still leaves something for the sales
   team. Two independent, each-fires-once notification triggers:
   - **Ready** — first time name + a contact method + service are all
     present. Dedup via `leads.notified_at` (a small, flagged schema
     addition — see `schema.sql`).
   - **Escalation** — first time the extractor flags `escalate` (human
     requested, complaint, out-of-KB decision-relevant question).
     Dedup via `leads.status` flipping to `'escalated'`.
3. **`app/notifications/email.py`** — plain `smtplib` against
   Hostinger's SMTP, sends to `INTERNAL_NOTIFICATION_EMAIL`. Missing
   config or a send failure returns `False` and logs rather than
   raising, so it can never take down a chat response.

**Known limitation, flagged not silent:** the visitor-facing answer
text isn't currently updated to reflect whether a lead was *actually*
captured this turn (e.g. confirming "got it, the team will reach out")
— the system prompt already tells the model not to fabricate a
confirmation, but nothing today feeds "yes, this was really saved"
back into the model's context to let it confirm accurately. Closing
that loop needs either a second pipeline pass after extraction or real
function-calling support in the LLM provider — worth doing before this
goes live if accurate confirmation language matters.

## API layer

`app/api/app.py` — Flask app factory, one real endpoint:

```
POST /api/chat
  { "session_id": "<uuid>", "message": "<visitor text>" }
  -> { "session_id": "<uuid>", "answer": "<text>", "grounded": true|false }

GET /api/health -> { "status": "ok" }
```

CORS is restricted to `FRONTEND_ORIGIN` (env var, defaults to `*` for
local dev — set it to the real site origin before going live).
`wsgi.py` is the entry point for both local dev (`python wsgi.py`) and
Hostinger's Passenger/WSGI "Setup Python App" (exposes both `app` and
`application`).

## Frontend

`frontend/ChatWidget.jsx` — a self-contained React + Tailwind
floating chat widget (matches the site's confirmed palette: `#010131`
background, `#EC61C4` pink, `#6650F2` gradient). Talks only to
`POST /api/chat` above. See `frontend/README.md` for how to mount it.

## Next phase

Not yet built: authenticated admin/dashboard views over `leads` (the
data's there, nothing reads it back out yet beyond the raw table), and
closing the confirmation-language loop noted above. Also worth adding
before production traffic: per-session/IP rate limiting on `/api/chat`
(currently unlimited) and a retry-with-backoff around the DashScope
call (see the earlier `ReadTimeoutError` — an occasional slow/failed
upstream call currently just fails the whole turn).

## Project layout

```
app/
  db.py                     MySQL connection factory (env-var only, no hardcoded creds)
  embeddings/
    base.py                 EmbeddingProvider interface
    voyage.py                Voyage AI implementation
  llm/
    base.py                 LLMProvider interface + ChatMessage
    openai_compatible.py     Shared OpenAI-compatible-endpoint request/response handling
    openrouter.py             OpenRouter implementation (defaults to Qwen)
    alibaba.py                 Direct Alibaba DashScope implementation
  kb/
    chunker.py               Markdown -> chunk list
    vector_codec.py          Pack/unpack embeddings to MySQL LONGBLOB
    ingest.py                Chunk -> hash -> embed (changed only) -> upsert -> prune stale
    retrieval.py              Cosine-similarity top-k search
  prompts/
    system_prompt.md  Verbatim copy of the supplied system prompt — never edited in code
    loader.py                   The only place that reads it
  rag/
    prompting.py               Real system prompt + retrieved context -> final prompt
    guardrails.py               Outbound checks: refund %, ungrounded numbers, internal email
    pipeline.py                 answer_query(): retrieve -> prompt -> generate -> guardrail
  sessions/
    service.py                  Phase 4: session get-or-create, message read/write, transcript
  leads/
    models.py                   LeadSignal dataclass
    extractor.py                 Phase 5: LLM-based structured lead/escalation extraction
    service.py                    Lead upsert + once-only notification-trigger logic
  notifications/
    email.py                    Phase 5: SMTP lead-capture / escalation emails
  api/
    app.py                       Flask app factory, POST /api/chat, GET /api/health
scripts/
  ingest_kb.py                CLI entry point
frontend/
  ChatWidget.jsx      React + Tailwind floating chat widget
  README.md                    Frontend integration instructions
tests/
  fakes.py                    Fake embedding + LLM providers for tests
  test_chunker.py
  test_vector_codec.py
  test_ingest_and_retrieval_integration.py   Real MySQL, real KB file
  test_rag_pipeline.py                        Real MySQL, fake embedding/LLM, guardrail interception
  test_guardrails.py                           Guardrail rules in isolation
  test_openrouter_provider.py                  Mocked HTTP layer
  test_alibaba_provider.py                     Mocked HTTP layer
  test_sessions_service.py                     Mocked DB — Phase 4
  test_leads_extractor.py                      Fake LLM — Phase 5
  test_leads_service.py                        Mocked DB — Phase 5
  test_email_notifications.py                  Mocked smtplib — Phase 5
  test_api_chat.py                             Mocked pipeline/DB — API layer
schema.sql                    Full data model (kb_chunks, sessions, messages, leads)
wsgi.py                      Flask entry point (local + Hostinger Passenger)
requirements.txt
.env.example
```

## Next phase

Phase 4: session + history handling — client-generated session ID as
the join key (not IP), MySQL-backed message read/write, conversation
surviving a page refresh, and trimming what's replayed into the model's
context window separately from the full stored transcript.
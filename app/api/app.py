"""Flask API layer — the HTTP surface the frontend widget talks to.

One real endpoint: POST /api/chat. Per request it:
1. Rate-limits by IP (the real defense — see note below) and by
   session (a secondary, looser throttle).
2. Validates the session ID and message from the request body.
3. Screens the message for prompt injection (app.rag.input_guard) and
   short-circuits with a canned refusal if it's a clear attempt —
   before spending an embedding call or an LLM call on it.
4. Gets-or-creates the session row, loads its stored history.
5. Runs the existing RAG pipeline (app.rag.pipeline.answer_query) —
   nothing about retrieval/generation/guardrails changes here, this
   layer is purely plumbing around it.
6. Persists the new turn.
7. Returns the answer immediately (never the internal retrieval/
   guardrail detail — that stayed in `try_it.py`'s debug printing, a
   visitor doesn't need to see chunk scores).

Lead extraction used to run HERE, synchronously, before the response
went out — meaning every chat turn paid for TWO sequential LLM calls
before the visitor saw anything. It's now handed to a bounded background
executor instead. See `_submit_lead_capture`.

Separately, a scheduled background job (`_send_debounced_lead_emails`,
started once at app startup) periodically sends the actual "ready lead"
notification email once a lead's session has gone quiet — see
app/leads/service.py's module docstring.

IMPORTANT — single-process assumption: both the rate limiter's default
in-memory storage AND the scheduler assume ONE worker process. If this
ever gets deployed behind multiple worker processes (e.g. a future
gunicorn --workers > 1 setup), each worker gets its own independent
rate-limit counters (limits effectively multiply by worker count) and
its own scheduler (duplicate emails). Fine for a single Hostinger
Passenger process; would need Redis-backed rate-limit storage and an
external cron/single-owner lock for the scheduler job before scaling to
multiple workers. Flagged, not silently assumed.

Provider instances (embedding, LLM) are constructed once at app
startup, not per-request — they're stateless HTTP clients, so
recreating them per request would just add setup overhead for nothing.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, g, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

from app.db import get_connection
from app.embeddings.gemini import GeminiEmbeddingProvider
from app.leads import service as lead_service
from app.leads.extractor import extract_lead_signal
from app.llm.alibaba import AlibabaLLMProvider
from app.llm.base import ChatMessage
from app.notifications.email import (
    send_escalation_notification,
    send_lead_notification,
    send_lead_update_notification,
)
from app.rag.input_guard import screen_user_message
from app.rag.pipeline import answer_query
from app.rag.qa_cache import QuestionAnswerCache
from app.sessions import service as session_service

logger = logging.getLogger(__name__)

_MAX_MESSAGE_LENGTH = 4000  # visitor input sanity cap, well above any real question

_DEBOUNCE_CHECK_INTERVAL_SECONDS = 60

_scheduler: BackgroundScheduler | None = None

# Set by create_app so the scheduler job (which runs outside any request
# and has no handle on the app) can do the cache's housekeeping.
_qa_cache: QuestionAnswerCache | None = None

_GENERIC_ERROR_RESPONSE = (
    "Something went wrong on our side just then — please try again, or "
    "reach us directly at info@apexcreative.example."
)

# The real, hard defense against abuse and runaway API cost — every
# chat turn calls a metered embedding API plus one-or-two metered LLM
# calls, so an unthrottled /api/chat is a direct cost/availability risk.
# Two windows (a tight per-minute cap and a looser per-hour cap) catch
# both a burst and a slow sustained hammering. Configurable via env var
# since the right number depends on real traffic patterns you don't
# have data on yet — tune once you see actual usage.
_DEFAULT_IP_RATE_LIMIT = "20 per minute;300 per hour"

# A looser, secondary throttle keyed on session_id. NOT a security
# boundary by itself — session_id is client-generated and trivially
# spoofable (a bad actor can mint a fresh UUID every request to reset
# this), so the IP limit above is what actually protects the API. This
# exists to catch a legitimate single conversation somehow firing
# messages unrealistically fast (a buggy frontend retry loop, etc.)
# without waiting for the IP-wide limit to catch it.
_DEFAULT_SESSION_RATE_LIMIT = "15 per minute"

# Background lead-capture concurrency. Deliberately well under
# DB_POOL_MAX_CONNECTIONS (default 10): each running task borrows a
# pooled connection and holds it for a full extraction LLM call, and the
# pool blocks when empty, so an unbounded pool of background work can
# starve the request path it was moved off in the first place.
_DEFAULT_LEAD_CAPTURE_WORKERS = 4

# Hard ceiling on queued-but-not-started lead captures. Past this, new
# ones are dropped with a warning rather than queued forever: lead
# extraction is a best-effort enhancement, and a backlog that deep means
# the provider is down or rate-limiting us, where every queued task is
# going to fail anyway.
_MAX_PENDING_LEAD_CAPTURES = 50


def _rate_limit_key_by_ip() -> str:
    return get_remote_address()


def _rate_limit_key_by_session() -> str:
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    if session_service.is_valid_session_id(session_id):
        return session_id
    # No valid session_id yet (e.g. malformed body) — fall back to IP
    # for this limiter too rather than a shared bucket that would let
    # every malformed request count against every other one.
    return get_remote_address()


def _send_debounced_lead_emails() -> None:
    """Scheduled job body: finds ready-but-unsent, now-quiet leads and
    sends their notification email. Opens and closes its own DB
    connection — this runs on the scheduler's own thread, entirely
    outside any Flask request/app context, so there's no `g` to use.
    Never raises — a failure here should be logged and retried on the
    next poll, not crash the scheduler thread permanently.
    """
    conn = None
    try:
        conn = get_connection()
        if _qa_cache is not None:
            # Housekeeping rides along on the existing poll rather than
            # adding a second scheduled job and a second connection: it
            # runs off the request path either way, which is the point.
            removed = _qa_cache.purge_stale(conn)
            if removed:
                logger.info("Purged %s stale QA cache entries", removed)
        ready_leads = lead_service.find_leads_ready_for_debounced_notification(conn)
        for lead in ready_leads:
            session_id = lead["session_id"]
            chat_summary = session_service.recent_transcript_text(conn, session_id)
            sent = send_lead_notification(lead, session_id, chat_summary)
            if sent:
                lead_service.mark_lead_notified(conn, lead["id"])
            else:
                logger.warning(
                    "Debounced lead email failed to send for lead_id=%s "
                    "(session %s) — will retry on next poll since "
                    "notified_at was not set.",
                    lead["id"],
                    session_id,
                )
    except Exception:
        logger.exception("Debounced lead-notification job failed")
    finally:
        if conn is not None:
            conn.close()


def _start_scheduler_once() -> None:
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        return
    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        _send_debounced_lead_emails,
        "interval",
        seconds=_DEBOUNCE_CHECK_INTERVAL_SECONDS,
        id="send_debounced_lead_emails",
        replace_existing=True,
    )
    _scheduler.start()


def _build_embedding_provider():
    """The primary embedding provider, wrapped so a quota outage does not
    take the bot down.

    The Gemini free tier allows 1,000 embed requests per DAY. Every chat
    turn spends one, so when it runs out /api/chat returns 502 for every
    visitor until midnight Pacific — which happened. With a fallback the
    site keeps answering from a local model instead, at measurably lower
    retrieval quality (see eval/embedding_comparison.md).

    Opt-out rather than opt-in: set EMBEDDING_FALLBACK_MODEL empty to run
    Gemini alone. The fallback is skipped automatically if
    sentence-transformers is not installed, so a deployment that never
    installed the heavy dependency behaves exactly as it did before rather
    than failing at startup.

    Requires `scripts/build_fallback_index.py` to have been run — the
    fallback model has its own vector space and can only score against
    chunks embedded by itself.
    """
    primary = GeminiEmbeddingProvider()

    model = os.environ.get("EMBEDDING_FALLBACK_MODEL", "BAAI/bge-m3").strip()
    if not model:
        logger.info("Embedding fallback disabled (EMBEDDING_FALLBACK_MODEL is empty).")
        return primary

    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        logger.warning(
            "Embedding fallback requested (%s) but sentence-transformers is "
            "not installed — running on the primary provider alone. A quota "
            "outage will take the bot down until it resets.",
            model,
        )
        return primary

    from app.embeddings.fallback import FallbackEmbeddingProvider
    from app.embeddings.local import LocalEmbeddingProvider

    # Deliberately does not read primary.model_name here: this runs at
    # startup, and the provider is a bare mock in the API tests.
    logger.info(
        "Embedding fallback armed: %s, used only on a primary quota error.",
        model,
    )
    # Truncated to the system's one embedding width. bge-m3 is natively
    # 1024, but every vector column here is 768, and measuring showed
    # truncation costs nothing: hit@1 76.3% / hit@3 86.4% / MRR 0.805 at
    # 768 against 72.9 / 83.1 / 0.780 at the native 1024, both rejecting
    # 10/10 off-topic. One width means the QA cache, the chunk index and
    # every future vector column stay interchangeable instead of each
    # needing a second column at the fallback's size.
    dim = int(os.environ.get("EMBEDDING_FALLBACK_DIM", "768"))

    # NOT loaded here. The weights are ~2GB and most deployments never hit
    # the quota, so the model loads lazily on the first call that needs it.
    return FallbackEmbeddingProvider(
        primary, LocalEmbeddingProvider(model, truncate_dim=dim)
    )


def _build_qa_cache(embedding_provider, llm_provider) -> QuestionAnswerCache | None:
    """Constructs the QA cache, or returns None to run without one.

    Fails closed. A cached answer is only valid for the exact embedding
    model and LLM that produced it, so those names are part of the cache
    key — if either can't be read (a stubbed provider in tests, or a
    provider whose interface changed), the cache is disabled rather than
    keyed on a placeholder that would let answers leak across a model
    change. Losing the cache costs money; a bad key costs correctness.

    The KB signature starts empty and is filled in per turn by
    `refresh_scope`, since there's no DB connection at startup.
    """
    if os.environ.get("QA_CACHE_ENABLED", "true").strip().lower() == "false":
        logger.info("QA cache disabled via QA_CACHE_ENABLED=false")
        return None

    try:
        embedding_model = embedding_provider.model_name
        llm_model = llm_provider.model_name
    except Exception:
        logger.warning(
            "Could not read provider model names — running without the QA "
            "cache rather than risking answers cached under an unknown model."
        )
        return None

    if not embedding_model or not llm_model:
        return None

    cache = QuestionAnswerCache(
        kb_signature="",
        embedding_model=embedding_model,
        llm_model=llm_model,
    )
    logger.info(
        "QA cache enabled (embedding=%s, llm=%s, semantic=%s, threshold=%.2f)",
        embedding_model,
        llm_model,
        cache.semantic_enabled,
        cache.similarity_threshold,
    )
    return cache


def _configure_proxy_awareness(app: Flask) -> None:
    """Makes `request.remote_addr` reflect the real visitor IP when the
    app runs behind a reverse proxy (Passenger/nginx on Hostinger).

    This is load-bearing for rate limiting, not cosmetic. Without it
    every request arrives from the proxy's own address, so the per-IP
    limiter — documented as the one defense a bad actor can't sidestep —
    collapses into a single site-wide bucket: 20 requests per minute
    shared by every visitor, and one abuser locks out the whole site.

    TRUSTED_PROXY_COUNT is the number of proxy hops in front of the app,
    and it must match reality in both directions. Too low and the real
    client IP stays hidden. Too high — or any value above 0 when there
    is NO proxy — and a client can forge an X-Forwarded-For header to
    mint a fresh rate-limit identity per request, which is strictly
    worse than the bug this fixes. Set it to 0 when running the app
    directly with no proxy in front (e.g. local `flask run`).
    """
    proxy_count = int(os.environ.get("TRUSTED_PROXY_COUNT", "1"))
    if proxy_count <= 0:
        logger.info(
            "TRUSTED_PROXY_COUNT=0 — trusting the raw socket address for "
            "rate limiting. Correct only if nothing proxies this app."
        )
        return
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=proxy_count, x_proto=proxy_count)
    logger.info(
        "ProxyFix enabled for %s proxy hop(s) — rate limiting keys on the "
        "client IP from X-Forwarded-For. Set TRUSTED_PROXY_COUNT=0 if this "
        "app is not actually behind a proxy.",
        proxy_count,
    )


def create_app() -> Flask:
    app = Flask(__name__)

    _configure_proxy_awareness(app)

    try:
        from flask_cors import CORS

        # FRONTEND_ORIGIN supports a comma-separated list (e.g. local
        # dev + production domain at once) — flask-cors wants an actual
        # list for that, a raw comma-joined string would be treated as
        # one literal origin that never matches any real Origin header.
        origin_setting = os.environ.get("FRONTEND_ORIGIN", "*")
        allowed_origins = (
            [origin.strip() for origin in origin_setting.split(",") if origin.strip()]
            if origin_setting != "*"
            else "*"
        )
        CORS(app, resources={r"/api/*": {"origins": allowed_origins}})
    except ImportError:
        logger.warning(
            "flask-cors not installed — /api/* will not send CORS headers, "
            "the widget will fail to call this API from a browser on "
            "another origin. Add flask-cors to requirements.txt."
        )

    # RATE_LIMIT_STORAGE_URI defaults to in-process memory — fine for a
    # single worker process (see module docstring), would need
    # something like redis://localhost:6379 before ever running multiple
    # worker processes.
    limiter = Limiter(
        key_func=_rate_limit_key_by_ip,
        app=app,
        storage_uri=os.environ.get("RATE_LIMIT_STORAGE_URI", "memory://"),
        default_limits=[],  # no blanket default — applied explicitly per-route below
    )

    @app.errorhandler(429)
    def _rate_limit_exceeded(_exc):
        response = jsonify(
            {"error": "Too many messages, please slow down and try again shortly."}
        )
        # Retry-After lets a well-behaved client back off for exactly as
        # long as needed instead of guessing or hammering. Flask-Limiter
        # normally attaches this itself, but a custom error handler
        # replaces the response it would have decorated, so the window's
        # reset time is read off the limiter and re-applied here.
        current = getattr(limiter, "current_limit", None)
        if current is not None and getattr(current, "reset_at", None):
            seconds = math.ceil(current.reset_at - time.time())
            response.headers["Retry-After"] = str(max(1, seconds))
        return response, 429

    embedding_provider = _build_embedding_provider()
    llm_provider = AlibabaLLMProvider()

    global _qa_cache
    _qa_cache = _build_qa_cache(embedding_provider, llm_provider)

    lead_capture_pool = ThreadPoolExecutor(
        max_workers=int(
            os.environ.get("LEAD_CAPTURE_WORKERS", str(_DEFAULT_LEAD_CAPTURE_WORKERS))
        ),
        thread_name_prefix="lead-capture",
    )
    pending_lead_captures = _PendingCounter()

    _start_scheduler_once()

    def _get_db():
        if "db" not in g:
            g.db = get_connection()
        return g.db

    @app.teardown_appcontext
    def _close_db(_exc):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/chat")
    @limiter.limit(
        os.environ.get("CHAT_RATE_LIMIT_PER_IP", _DEFAULT_IP_RATE_LIMIT),
        key_func=_rate_limit_key_by_ip,
    )
    @limiter.limit(
        os.environ.get("CHAT_RATE_LIMIT_PER_SESSION", _DEFAULT_SESSION_RATE_LIMIT),
        key_func=_rate_limit_key_by_session,
    )
    def chat():
        payload = request.get_json(silent=True) or {}
        session_id = payload.get("session_id")
        message = payload.get("message")

        if not session_service.is_valid_session_id(session_id):
            return jsonify({"error": "session_id must be a valid UUID."}), 400
        if not isinstance(message, str) or not message.strip():
            return jsonify({"error": "message must be a non-empty string."}), 400
        if len(message) > _MAX_MESSAGE_LENGTH:
            return jsonify({"error": f"message exceeds {_MAX_MESSAGE_LENGTH} characters."}), 400

        conn = _get_db()
        session_service.get_or_create_session(conn, session_id)

        # Inbound injection screen, before any paid API call. A blocked
        # turn is still persisted so the transcript stays coherent, but
        # skips lead extraction — there's nothing to extract from an
        # attack, and running it would spend the LLM call the screen
        # just saved.
        screen = screen_user_message(message)
        if screen.blocked:
            logger.warning(
                "Blocked prompt-injection attempt (%s) on session %s",
                screen.category,
                session_id,
            )
            session_service.save_turn(conn, session_id, message, screen.response)
            return jsonify(
                {"session_id": session_id, "answer": screen.response, "grounded": False}
            )

        history: list[ChatMessage] = session_service.load_history(conn, session_id)

        try:
            result = answer_query(
                query=message,
                embedding_provider=embedding_provider,
                llm_provider=llm_provider,
                db_connection=conn,
                conversation_history=history,
                qa_cache=_qa_cache,
            )
        except Exception:
            # Provider outage, empty completion, embedding failure — the
            # visitor gets a plain apology instead of a raw 500 page, and
            # the turn is deliberately NOT persisted, so a retry replays
            # cleanly rather than leaving a user message with no reply
            # stranded in the history the next turn replays to the model.
            logger.exception("Chat pipeline failed for session %s", session_id)
            return jsonify({"error": _GENERIC_ERROR_RESPONSE}), 502

        if result.cache_hit:
            logger.info(
                "QA cache %s hit for session %s — skipped generation",
                result.cache_hit,
                session_id,
            )

        session_service.save_turn(conn, session_id, message, result.answer)

        _submit_lead_capture(
            lead_capture_pool,
            pending_lead_captures,
            session_id,
            history,
            message,
            result.answer,
            llm_provider,
        )

        return jsonify(
            {
                "session_id": session_id,
                "answer": result.answer,
                "grounded": result.grounded,
            }
        )

    return app


class _PendingCounter:
    """Thread-safe counter of queued-or-running background tasks, used
    to apply backpressure (see _MAX_PENDING_LEAD_CAPTURES)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0

    def try_acquire(self, limit: int) -> bool:
        with self._lock:
            if self._count >= limit:
                return False
            self._count += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._count -= 1


def _submit_lead_capture(
    pool: ThreadPoolExecutor,
    pending: _PendingCounter,
    session_id: str,
    prior_history: list[ChatMessage],
    user_message: str,
    answer: str,
    llm_provider,
) -> None:
    """Hands lead capture to the background pool so the response returns
    as soon as the RAG answer is ready — the visitor never waits on the
    extraction LLM call.

    Bounded on purpose. The previous version started a fresh unbounded
    `threading.Thread` per chat turn, each of which borrows its own
    pooled DB connection and holds it for up to a full extraction call
    (60s timeout plus a retry). Enough concurrent turns and every pooled
    connection is held by background work while request threads block
    waiting for one — the request path stalls on exactly the work it was
    restructured to stop waiting for.
    """
    if not pending.try_acquire(_MAX_PENDING_LEAD_CAPTURES):
        logger.warning(
            "Lead-capture backlog at capacity (%s) — skipping extraction for "
            "session %s. The lead row keeps whatever earlier turns captured.",
            _MAX_PENDING_LEAD_CAPTURES,
            session_id,
        )
        return

    def _task() -> None:
        try:
            _run_lead_capture_background(
                session_id, prior_history, user_message, answer, llm_provider
            )
        finally:
            pending.release()

    try:
        pool.submit(_task)
    except RuntimeError:
        # Pool already shut down (process is exiting) — nothing to do.
        pending.release()


def _run_lead_capture_background(
    session_id: str,
    prior_history: list[ChatMessage],
    user_message: str,
    answer: str,
    llm_provider,
) -> None:
    """Task entry point — opens its own DB connection (it can't reuse
    the request's `g.db`, which `_close_db` closes the moment the
    request context tears down, racing with this still running),
    delegates to `_run_lead_capture`, always closes the connection."""
    conn = None
    try:
        conn = get_connection()
        _run_lead_capture(conn, session_id, prior_history, user_message, answer, llm_provider)
    except Exception:
        logger.exception("Background lead capture failed for session %s", session_id)
    finally:
        if conn is not None:
            conn.close()


def _run_lead_capture(
    conn,
    session_id: str,
    prior_history: list[ChatMessage],
    user_message: str,
    answer: str,
    llm_provider,
) -> None:
    """Best-effort: extraction/notification failures are logged, never
    surfaced to the visitor and never allowed to affect the response
    that already went out in `chat()` above.

    Note what this does NOT do anymore: send the initial "ready" lead
    email. That's handled entirely by the scheduled
    `_send_debounced_lead_emails` job now — see app/leads/service.py's
    module docstring. This function only ever sends an escalation email
    (immediate, never delayed) or an update email (new info after an
    already-sent ready email).
    """
    try:
        signal = extract_lead_signal(
            llm_provider=llm_provider,
            conversation_history=prior_history,
            latest_user_message=user_message,
            latest_assistant_message=answer,
        )
        result = lead_service.upsert_lead_for_session(conn, session_id, signal)

        if not (result.should_send_escalation_notification or result.should_send_update_notification):
            return

        lead = lead_service.get_lead(conn, result.lead_id) if result.lead_id else None
        chat_summary = session_service.recent_transcript_text(conn, session_id)

        if result.should_send_update_notification and lead:
            send_lead_update_notification(lead, session_id, result.updated_fields, chat_summary)
        if result.should_send_escalation_notification:
            send_escalation_notification(
                session_id, signal.escalate_reason or "", lead, chat_summary
            )
    except Exception:
        logger.exception("Lead capture step failed for session %s", session_id)

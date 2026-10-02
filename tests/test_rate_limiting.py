"""Rate limiting on POST /api/chat.

Every chat turn costs a metered embedding call plus one or two metered
LLM calls, so an unthrottled endpoint is a direct cost and availability
risk. Two limiters guard it, and they are NOT equally trustworthy:

- The per-IP limit is the real defense. A caller cannot choose their own
  source address.
- The per-session limit is a convenience throttle. `session_id` is
  client-generated, so anyone can mint a fresh UUID per request to reset
  it — `test_rotating_session_ids_still_hit_ip_limit` is the test that
  pins down that this evasion still runs into the IP limit.

The proxy tests matter just as much as the limit tests: behind a reverse
proxy with no X-Forwarded-For handling, every visitor shares the proxy's
address, which silently collapses the per-IP limit into one site-wide
bucket. A limiter that counts the wrong identity is worse than no
limiter, because it looks like it's working.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("GOOGLE_API_KEY", "fake")
os.environ.setdefault("DASHSCOPE_API_KEY", "fake")
os.environ.setdefault("DB_HOST", "x")
os.environ.setdefault("DB_NAME", "x")
os.environ.setdefault("DB_USER", "x")
os.environ.setdefault("DB_PASSWORD", "x")

from app.rag.pipeline import RagResult  # noqa: E402

VALID_SESSION_ID = "550e8400-e29b-41d4-a716-446655440000"
OTHER_SESSION_ID = "660e8400-e29b-41d4-a716-446655440001"


def _make_client(monkeypatch, *, ip_limit="3 per minute", session_limit="100 per minute",
                 trusted_proxy_count="0"):
    """Builds an app with the limits under test. Limits are read from the
    environment when routes are registered, so they must be set before
    `create_app()` runs — hence building the client per test rather than
    sharing a fixture."""
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_IP", ip_limit)
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_SESSION", session_limit)
    monkeypatch.setenv("TRUSTED_PROXY_COUNT", trusted_proxy_count)

    with patch("app.embeddings.gemini.GeminiEmbeddingProvider.__init__", return_value=None), \
            patch("app.llm.alibaba.AlibabaLLMProvider.__init__", return_value=None):
        from app.api.app import create_app

        app = create_app()
        app.config["TESTING"] = True
        return app.test_client()


def _stubbed_pipeline():
    """Patches out everything below the API layer so these tests measure
    only the limiter, never a real DB or provider call."""
    fake_result = RagResult(answer="ok", retrieved_chunks=[], grounded=True)
    return (
        patch("app.api.app.session_service.get_or_create_session"),
        patch("app.api.app.session_service.load_history", return_value=[]),
        patch("app.api.app.session_service.save_turn"),
        patch("app.api.app.answer_query", return_value=fake_result),
        patch("app.api.app._run_lead_capture"),
        patch("app.api.app.get_connection", return_value=MagicMock()),
    )


def _post(client, session_id=VALID_SESSION_ID, ip="9.9.9.9", forwarded_for=None):
    headers = {"X-Forwarded-For": forwarded_for} if forwarded_for else {}
    return client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "hello"},
        environ_base={"REMOTE_ADDR": ip},
        headers=headers,
    )


@pytest.fixture()
def stub_pipeline():
    patches = _stubbed_pipeline()
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


def test_ip_limit_blocks_after_threshold(monkeypatch, stub_pipeline):
    client = _make_client(monkeypatch, ip_limit="3 per minute")

    for _ in range(3):
        assert _post(client).status_code == 200
    assert _post(client).status_code == 429


def test_rotating_session_ids_still_hit_ip_limit(monkeypatch, stub_pipeline):
    """The evasion the per-session limit cannot stop: a caller minting a
    fresh session UUID for every request. The IP limit is what has to
    catch it, which is precisely why it's the one that matters."""
    client = _make_client(monkeypatch, ip_limit="3 per minute", session_limit="100 per minute")

    session_ids = [f"550e8400-e29b-41d4-a716-44665544{i:04d}" for i in range(10)]

    statuses = [_post(client, session_id=sid).status_code for sid in session_ids]

    assert statuses[:3] == [200, 200, 200]
    assert 429 in statuses[3:], "rotating session IDs evaded the IP limit entirely"


def test_session_limit_blocks_a_single_runaway_conversation(monkeypatch, stub_pipeline):
    client = _make_client(monkeypatch, ip_limit="100 per minute", session_limit="2 per minute")

    assert _post(client).status_code == 200
    assert _post(client).status_code == 200
    assert _post(client).status_code == 429
    # A different session from the same IP is unaffected by the session bucket.
    assert _post(client, session_id=OTHER_SESSION_ID).status_code == 200


def test_separate_ips_get_separate_budgets(monkeypatch, stub_pipeline):
    client = _make_client(monkeypatch, ip_limit="2 per minute")

    assert _post(client, ip="1.1.1.1").status_code == 200
    assert _post(client, ip="1.1.1.1").status_code == 200
    assert _post(client, ip="1.1.1.1").status_code == 429
    # A different visitor must not inherit the first one's exhausted budget.
    assert _post(client, ip="2.2.2.2").status_code == 200


def test_rate_limited_response_is_json_with_retry_after(monkeypatch, stub_pipeline):
    client = _make_client(monkeypatch, ip_limit="1 per minute")

    assert _post(client).status_code == 200
    blocked = _post(client)

    assert blocked.status_code == 429
    assert "error" in blocked.get_json()
    assert blocked.headers.get("Retry-After") is not None


# --- Proxy awareness -------------------------------------------------------


def test_behind_a_proxy_each_forwarded_client_gets_its_own_budget(monkeypatch, stub_pipeline):
    """With TRUSTED_PROXY_COUNT=1, the limiter must key on the forwarded
    client IP. Without this, every visitor behind the proxy shares one
    bucket and a handful of concurrent visitors lock out the whole site.
    """
    client = _make_client(monkeypatch, ip_limit="2 per minute", trusted_proxy_count="1")

    # All requests arrive from the proxy's own address.
    proxy = "127.0.0.1"
    assert _post(client, ip=proxy, forwarded_for="203.0.113.10").status_code == 200
    assert _post(client, ip=proxy, forwarded_for="203.0.113.10").status_code == 200
    assert _post(client, ip=proxy, forwarded_for="203.0.113.10").status_code == 429

    # A different real visitor, same proxy — must still be served.
    assert _post(client, ip=proxy, forwarded_for="203.0.113.99").status_code == 200


def test_without_a_proxy_forwarded_headers_are_ignored(monkeypatch, stub_pipeline):
    """The inverse failure: if the app is NOT behind a proxy, trusting
    X-Forwarded-For would let a client forge a fresh identity per request
    and bypass the limit entirely. TRUSTED_PROXY_COUNT=0 must ignore it.
    """
    client = _make_client(monkeypatch, ip_limit="2 per minute", trusted_proxy_count="0")

    assert _post(client, ip="5.5.5.5", forwarded_for="203.0.113.1").status_code == 200
    assert _post(client, ip="5.5.5.5", forwarded_for="203.0.113.2").status_code == 200
    assert _post(client, ip="5.5.5.5", forwarded_for="203.0.113.3").status_code == 429

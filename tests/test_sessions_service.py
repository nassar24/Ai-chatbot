"""Session-service tests against a mocked PyMySQL connection/cursor —
no real database needed, since this module's job is "does it issue the
right SQL with the right params," not "does MySQL work."
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.sessions import service


def _mock_conn(fetchall_return=None, fetchone_return=None):
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = fetchall_return or []
    cursor.fetchone.return_value = fetchone_return
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn, cursor


def test_is_valid_session_id():
    assert service.is_valid_session_id("550e8400-e29b-41d4-a716-446655440000")
    assert not service.is_valid_session_id("not-a-uuid")
    assert not service.is_valid_session_id("")
    assert not service.is_valid_session_id(None)


def test_get_or_create_session_rejects_bad_id():
    conn, _ = _mock_conn()
    with pytest.raises(ValueError):
        service.get_or_create_session(conn, "bad-id")
    conn.cursor.assert_not_called()


def test_get_or_create_session_upserts_and_commits():
    conn, cursor = _mock_conn()
    session_id = "550e8400-e29b-41d4-a716-446655440000"
    service.get_or_create_session(conn, session_id)
    cursor.execute.assert_called_once()
    sql = cursor.execute.call_args[0][0]
    assert "INSERT INTO sessions" in sql
    assert "ON CONFLICT (session_id) DO UPDATE" in sql
    conn.commit.assert_called_once()


def test_load_history_reverses_to_oldest_first():
    conn, cursor = _mock_conn(
        fetchall_return=[("assistant", "second"), ("user", "first")]
    )
    history = service.load_history(conn, "550e8400-e29b-41d4-a716-446655440000")
    assert [m.content for m in history] == ["first", "second"]
    assert history[0].role == "user"
    assert history[1].role == "assistant"


def test_save_turn_writes_both_roles_in_one_transaction():
    conn, cursor = _mock_conn()
    service.save_turn(conn, "550e8400-e29b-41d4-a716-446655440000", "hi", "hello")
    assert cursor.execute.call_count == 2
    roles = [call.args[1][1] for call in cursor.execute.call_args_list]
    assert roles == ["user", "assistant"]
    conn.commit.assert_called_once()

"""app.leads.service tests against a mocked psycopg connection — checks
the merge/notification-trigger logic, which is the part actually worth
testing (SQL correctness), not real MySQL behavior.

Notification timing is the thing these tests pin down, and it is NOT
uniform across notification kinds (see the module docstring in
app/leads/service.py):

- READY   — never decided here. `upsert_lead_for_session` only merges
            fields; the email goes out later, from the scheduled job,
            once the conversation has gone quiet. Covered by the
            `find_leads_ready_for_debounced_notification` tests below.
- ESCALATION — decided here, fires immediately, exactly once per lead.
- UPDATE  — decided here, fires only after the ready email already went
            out (`notified_at` set) and only for blank->filled fields.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.leads.models import LeadSignal
from app.leads.service import (
    find_leads_ready_for_debounced_notification,
    mark_lead_notified,
    upsert_lead_for_session,
)


def _mock_conn(session_lead_id=None, existing_lead_row=None):
    """`existing_lead_row` should be a tuple matching
    (id, *_LEAD_COLUMNS, status, notified_at) if a lead already exists."""
    conn = MagicMock()
    cursor = MagicMock()

    def fetchone_side_effect():
        # First query in the flow is always "SELECT lead_id FROM sessions",
        # second (if it happens) is "SELECT ... FROM leads WHERE id = %s".
        calls = cursor.execute.call_args_list
        last_sql = calls[-1].args[0]
        if "FROM sessions" in last_sql:
            return (session_lead_id,) if session_lead_id is not None else None
        if "FROM leads" in last_sql:
            return existing_lead_row
        # The Postgres port replaced lastrowid with INSERT ... RETURNING id,
        # so the new id now arrives through fetchone() like any other row.
        if "INSERT INTO leads" in last_sql and "RETURNING id" in last_sql:
            return (42,)
        return None

    cursor.fetchone.side_effect = fetchone_side_effect

    conn.cursor.return_value.__enter__.return_value = cursor
    return conn, cursor


SESSION_ID = "550e8400-e29b-41d4-a716-446655440000"

NOT_YET_NOTIFIED = None
ALREADY_NOTIFIED = "2026-01-01 00:00:00"


def test_empty_signal_is_a_noop():
    conn, cursor = _mock_conn()
    result = upsert_lead_for_session(conn, SESSION_ID, LeadSignal())
    assert result.lead_id is None
    assert result.should_send_escalation_notification is False
    assert result.should_send_update_notification is False
    conn.cursor.assert_not_called()


def test_new_partial_lead_creates_row_without_notifying():
    conn, cursor = _mock_conn(session_lead_id=None)
    signal = LeadSignal(name="Ahmed", ready_to_submit=False)
    result = upsert_lead_for_session(conn, SESSION_ID, signal)

    assert result.lead_id == 42
    assert result.should_send_escalation_notification is False
    assert result.should_send_update_notification is False

    insert_call = [c for c in cursor.execute.call_args_list if "INSERT INTO leads" in c.args[0]]
    assert len(insert_call) == 1
    conn.commit.assert_called_once()


def test_ready_lead_does_not_notify_from_the_request_path():
    """A lead crossing the "ready" bar must NOT trigger an email here —
    that's the whole point of debouncing. The row is written so nothing
    is lost; the scheduled job decides when to send."""
    conn, cursor = _mock_conn(session_lead_id=None)
    signal = LeadSignal(
        name="Ahmed", phone="+20 100 111 2222", service_requested="website",
        ready_to_submit=True,
    )
    result = upsert_lead_for_session(conn, SESSION_ID, signal)

    assert result.should_send_escalation_notification is False
    assert result.should_send_update_notification is False

    insert_sql = [
        c for c in cursor.execute.call_args_list if "INSERT INTO leads" in c.args[0]
    ][0].args[0]
    # notified_at is owned exclusively by mark_lead_notified, after the
    # debounced send — writing it here would suppress the email entirely.
    assert "notified_at" not in insert_sql


def test_new_info_before_ready_email_does_not_send_an_update():
    """Fields filled in while the ready email is still pending need no
    update email — the pending send will already include them."""
    existing_row = (
        7, "Ahmed", None, None, None, None, None, "website", None, None, None,
        "new", NOT_YET_NOTIFIED,
    )
    conn, cursor = _mock_conn(session_lead_id=7, existing_lead_row=existing_row)
    signal = LeadSignal(phone="+20 100 111 2222")
    result = upsert_lead_for_session(conn, SESSION_ID, signal)

    assert result.lead_id == 7
    assert result.should_send_update_notification is False


def test_new_info_after_ready_email_sends_an_update():
    existing_row = (
        7, "Ahmed", None, None, None, None, None, "website", None, None, None,
        "new", ALREADY_NOTIFIED,
    )
    conn, cursor = _mock_conn(session_lead_id=7, existing_lead_row=existing_row)
    signal = LeadSignal(whatsapp_number="+20 100 111 2222", budget="30,000 EGP")
    result = upsert_lead_for_session(conn, SESSION_ID, signal)

    assert result.should_send_update_notification is True
    assert result.updated_fields == {
        "whatsapp_number": "+20 100 111 2222",
        "budget": "30,000 EGP",
    }


def test_unchanged_fields_after_ready_email_send_nothing():
    existing_row = (
        7, "Ahmed", "a@x.com", None, None, None, None, "website", None, None, None,
        "new", ALREADY_NOTIFIED,
    )
    conn, cursor = _mock_conn(session_lead_id=7, existing_lead_row=existing_row)
    signal = LeadSignal(name="Ahmed", email="a@x.com")  # nothing new
    result = upsert_lead_for_session(conn, SESSION_ID, signal)

    assert result.should_send_update_notification is False
    assert result.updated_fields == {}


def test_escalation_fires_immediately_and_independently_of_ready_state():
    existing_row = (
        7, "Ahmed", "a@x.com", None, None, None, None, "website", None, None, None,
        "new", ALREADY_NOTIFIED,
    )
    conn, cursor = _mock_conn(session_lead_id=7, existing_lead_row=existing_row)
    signal = LeadSignal(escalate=True, escalate_reason="angry about refund")
    result = upsert_lead_for_session(conn, SESSION_ID, signal)

    assert result.should_send_escalation_notification is True


def test_already_escalated_lead_does_not_renotify_escalation():
    existing_row = (
        7, "Ahmed", None, None, None, None, None, None, None, None, None,
        "escalated", NOT_YET_NOTIFIED,
    )
    conn, cursor = _mock_conn(session_lead_id=7, existing_lead_row=existing_row)
    signal = LeadSignal(escalate=True, escalate_reason="still upset")
    result = upsert_lead_for_session(conn, SESSION_ID, signal)

    assert result.should_send_escalation_notification is False


def test_merge_keeps_existing_fields_new_values_fill_gaps():
    existing_row = (
        7, "Ahmed", None, None, None, None, None, None, None, None, None,
        "new", NOT_YET_NOTIFIED,
    )
    conn, cursor = _mock_conn(session_lead_id=7, existing_lead_row=existing_row)
    signal = LeadSignal(email="ahmed@x.com")  # name omitted this turn
    upsert_lead_for_session(conn, SESSION_ID, signal)

    update_call = [c for c in cursor.execute.call_args_list if "UPDATE leads" in c.args[0]][0]
    values = update_call.args[1]
    # name should still be "Ahmed" (carried over), email should be the new value
    assert "Ahmed" in values
    assert "ahmed@x.com" in values


# --- The debounced-send query (the other half of the timing model) ---------


def test_debounce_query_requires_name_phone_service_and_quiet_and_unnotified():
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    conn.cursor.return_value.__enter__.return_value = cursor

    find_leads_ready_for_debounced_notification(conn, quiet_minutes=3)

    sql, params = cursor.execute.call_args.args
    normalized = " ".join(sql.split())
    assert "leads.notified_at IS NULL" in normalized
    assert "leads.name IS NOT NULL" in normalized
    assert "leads.phone IS NOT NULL" in normalized
    assert "leads.whatsapp_number IS NOT NULL" in normalized
    assert "leads.service_requested IS NOT NULL" in normalized
    # "Quiet" is a property of the conversation, not the lead row.
    assert "sessions.last_active_at <= (NOW() - make_interval(mins => %s))" in normalized
    assert params == (3,)


def test_debounce_query_maps_rows_onto_expected_keys():
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        (
            7, "Ahmed", "a@x.com", "+20 100 111 2222", None, None, None,
            "website", None, None, None, "new", SESSION_ID,
        )
    ]
    conn.cursor.return_value.__enter__.return_value = cursor

    leads = find_leads_ready_for_debounced_notification(conn)

    assert len(leads) == 1
    assert leads[0]["id"] == 7
    assert leads[0]["name"] == "Ahmed"
    assert leads[0]["session_id"] == SESSION_ID


def test_mark_lead_notified_sets_notified_at_and_commits():
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor

    mark_lead_notified(conn, 7)

    sql, params = cursor.execute.call_args.args
    assert "notified_at = CURRENT_TIMESTAMP" in sql
    assert params == (7,)
    conn.commit.assert_called_once()

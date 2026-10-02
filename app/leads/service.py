"""Turns a `LeadSignal` (see extractor.py) into durable `leads` rows and
decides when a notification email should fire.

Two DIFFERENT timing models, on purpose:

- ESCALATION notification: fires immediately, the first time `escalate`
  comes back true for this session — tracked via `status` flipping to
  'escalated' (idempotent: once it's 'escalated', it stays that way).
  Never delayed — if someone's asking for a human or is visibly
  frustrated, waiting defeats the point.

- READY notification (a lead has name + a real contact method +
  service_requested): NOT sent immediately from here anymore. Every
  turn keeps merging fresh fields into the `leads` row as before, but
  the actual email only goes out once the conversation has gone quiet
  for a few minutes — see `find_leads_ready_for_debounced_notification`
  and the scheduled job in app/api/app.py that calls it. This exists
  because sending the instant a lead crosses the "ready" bar meant the
  sales team got emailed with whatever happened to be filled in at that
  exact moment — often missing a WhatsApp number, a budget, or anything
  else the visitor was about to type next. Waiting for quiet means the
  email that eventually goes out reflects the most complete, settled
  state of the conversation instead of a snapshot mid-typing.

- UPDATE notification: fires whenever a field that was previously blank
  gets filled in AFTER the ready notification already went out (i.e.
  after the debounced job actually sent it, or if the visitor comes
  back well after the fact). Not tracked via a separate "already sent"
  column — each field can only transition from blank to filled once, so
  this is naturally self-limiting without needing dedup state. NOTE:
  this only catches blank->filled transitions, not corrections to an
  already-filled field (e.g. a budget change from 6,000 to 7,000) —
  that's a known gap, flagged rather than silently accepted.

A lead row is created as soon as ANY field or an escalation is captured
— even partial/incomplete — so a visitor who drops off mid-conversation
still leaves something for the sales team to work with, rather than all
the data being silently discarded because the visitor never finished.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.leads.models import LeadSignal

_LEAD_COLUMNS = (
    "name",
    "email",
    "phone",
    "whatsapp_number",
    "company",
    "business_type",
    "service_requested",
    "budget",
    "timeline",
    "notes",
)

# How many minutes of no new message in a session before a ready-but-
# unsent lead's notification email actually goes out. Chosen deliberately
# short enough that the team still hears about a lead promptly, long
# enough that a visitor mid-conversation isn't cut off mid-typing.
DEFAULT_DEBOUNCE_MINUTES = 3


@dataclass(frozen=True)
class LeadUpdateResult:
    lead_id: int | None
    should_send_escalation_notification: bool
    should_send_update_notification: bool = False
    updated_fields: dict = field(default_factory=dict)


def _get_session_lead_id(conn, session_id: str) -> int | None:
    with conn.cursor() as cursor:
        cursor.execute("SELECT lead_id FROM sessions WHERE session_id = %s", (session_id,))
        row = cursor.fetchone()
    return row[0] if row and row[0] is not None else None


def _get_lead_row(conn, lead_id: int) -> dict | None:
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT id, {', '.join(_LEAD_COLUMNS)}, status, notified_at "
            "FROM leads WHERE id = %s",
            (lead_id,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    keys = ["id", *_LEAD_COLUMNS, "status", "notified_at"]
    return dict(zip(keys, row))


def _merged_fields(existing: dict | None, signal: LeadSignal) -> dict:
    """New non-null values win; otherwise keep whatever was already
    captured. This is what lets lead capture accumulate across turns —
    a visitor giving their name in turn 2 and their email in turn 5
    ends up with both, instead of the later extraction overwriting the
    earlier one with nulls."""
    incoming = signal.as_lead_columns()
    if existing is None:
        return incoming
    return {
        field: incoming[field] if incoming[field] is not None else existing.get(field)
        for field in _LEAD_COLUMNS
    }


def _newly_filled_fields(existing: dict | None, merged: dict) -> dict:
    """Fields that went from blank to filled this call — used to decide
    whether an UPDATE notification is warranted (see module docstring)."""
    if existing is None:
        return {}
    return {
        field: merged[field]
        for field in _LEAD_COLUMNS
        if not existing.get(field) and merged.get(field)
    }


def _link_session_to_lead(conn, session_id: str, lead_id: int) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            "UPDATE sessions SET lead_id = %s WHERE session_id = %s",
            (lead_id, session_id),
        )


def upsert_lead_for_session(conn, session_id: str, signal: LeadSignal) -> LeadUpdateResult:
    """Applies `signal` to whatever lead exists for `session_id` (or
    creates one), and reports which notification(s) the caller should
    send RIGHT NOW — which is only ever escalation or an update-after-
    already-notified. The initial "ready" notification is deliberately
    NOT decided here anymore; see module docstring. Commits its own
    transaction.

    A signal with nothing captured and no escalation is a no-op — most
    turns won't extract anything new, and this avoids writing a lead
    row (or touching `updated_at`) on every single message.
    """
    if not signal.has_any_field() and not signal.escalate:
        return LeadUpdateResult(None, False)

    existing_lead_id = _get_session_lead_id(conn, session_id)
    existing = _get_lead_row(conn, existing_lead_id) if existing_lead_id else None
    merged = _merged_fields(existing, signal)

    already_notified_ready = bool(existing and existing["notified_at"] is not None)
    already_escalated = bool(existing and existing["status"] == "escalated")

    should_send_escalation = signal.escalate and not already_escalated

    # Only relevant once the ready email already went out (via the
    # debounced job, in a prior call to that job) — if it hasn't sent
    # yet, whatever gets filled in now will just be picked up whenever
    # the debounced send eventually happens.
    newly_filled = _newly_filled_fields(existing, merged) if already_notified_ready else {}
    should_send_update = already_notified_ready and bool(newly_filled)

    new_status = "escalated" if (signal.escalate or already_escalated) else (existing["status"] if existing else "new")

    with conn.cursor() as cursor:
        if existing_lead_id is None:
            columns = [*_LEAD_COLUMNS, "status"]
            placeholders = ", ".join(["%s"] * len(columns))
            values = [merged[field] for field in _LEAD_COLUMNS] + [new_status]
            # psycopg has no lastrowid - RETURNING is the Postgres way,
            # and it is safer besides: lastrowid is per-connection state,
            # while RETURNING comes back with the statement itself.
            cursor.execute(
                f"INSERT INTO leads ({', '.join(columns)}) VALUES ({placeholders}) "
                "RETURNING id",
                values,
            )
            lead_id = cursor.fetchone()[0]
            _link_session_to_lead(conn, session_id, lead_id)
        else:
            lead_id = existing_lead_id
            set_clauses = [f"{field} = %s" for field in _LEAD_COLUMNS]
            values = [merged[field] for field in _LEAD_COLUMNS]
            set_clauses.append("status = %s")
            values.append(new_status)
            values.append(lead_id)
            cursor.execute(
                f"UPDATE leads SET {', '.join(set_clauses)} WHERE id = %s",
                values,
            )

    conn.commit()

    return LeadUpdateResult(
        lead_id=lead_id,
        should_send_escalation_notification=should_send_escalation,
        should_send_update_notification=should_send_update,
        updated_fields=newly_filled,
    )


def get_lead(conn, lead_id: int) -> dict | None:
    """Public read accessor — e.g. for building a notification email
    body from the freshest row after `upsert_lead_for_session`."""
    return _get_lead_row(conn, lead_id)


def find_leads_ready_for_debounced_notification(
    conn, quiet_minutes: int = DEFAULT_DEBOUNCE_MINUTES
) -> list[dict]:
    """Returns leads that satisfy the "ready" bar (name + phone/WhatsApp
    + service_requested) and haven't been notified yet, whose session
    has had no new message for at least `quiet_minutes`. Called by the
    scheduled job in app/api/app.py, not from the per-request path.

    Joins to `sessions` (not just `leads`) because "has it gone quiet"
    is a property of the conversation, not the lead row itself —
    `sessions.last_active_at` is what actually tracks recent activity.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT leads.id, {', '.join(f'leads.{c}' for c in _LEAD_COLUMNS)},
                   leads.status, sessions.session_id
            FROM leads
            JOIN sessions ON sessions.lead_id = leads.id
            WHERE leads.notified_at IS NULL
              AND leads.name IS NOT NULL AND leads.name != ''
              AND (
                    (leads.phone IS NOT NULL AND leads.phone != '')
                 OR (leads.whatsapp_number IS NOT NULL AND leads.whatsapp_number != '')
              )
              AND leads.service_requested IS NOT NULL AND leads.service_requested != ''
              AND sessions.last_active_at <= (NOW() - make_interval(mins => %s))
            """,
            (quiet_minutes,),
        )
        rows = list(cursor.fetchall())

    keys = ["id", *_LEAD_COLUMNS, "status", "session_id"]
    return [dict(zip(keys, row)) for row in rows]


def mark_lead_notified(conn, lead_id: int) -> None:
    """Sets `notified_at` after the debounced job has actually sent the
    ready email — this is the ONLY place notified_at gets set now
    (escalation uses `status`, not `notified_at`, as its own marker)."""
    with conn.cursor() as cursor:
        cursor.execute(
            "UPDATE leads SET notified_at = CURRENT_TIMESTAMP WHERE id = %s",
            (lead_id,),
        )
    conn.commit()
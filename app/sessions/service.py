"""Phase 4: session + message persistence.

The client (chat widget) generates a UUID on first load and sends it
with every request — that UUID is the join key (`sessions.session_id`),
not IP, since IP breaks on mobile networks/NAT and doesn't survive a
device switch (see README's Phase 4 note).

This module owns exactly two things:
1. Getting-or-creating a session row and bumping `last_active_at`.
2. Reading/writing `messages` for a session.

It deliberately does NOT own token-budget trimming for what gets
replayed to the LLM — that logic already exists in
`app/rag/pipeline.py::_cap_history_to_token_budget` and stays there.
The split is: this module persists the *full* untrimmed transcript
(needed later for `leads.chat_summary` / human review), the caller
loads it back and lets the pipeline decide how much of it actually goes
to the model. Storage is complete; replay is bounded — two different
concerns, kept in two different places on purpose.
"""

from __future__ import annotations

import uuid

from app.llm.base import ChatMessage

# Hard ceiling on how many past messages get pulled back per request.
#
# This is a different cap from the LLM replay budget in pipeline.py,
# which is token-based. Measured on real traffic, visitor turns run about
# 20 characters and answers about 420, so the 2000-token budget swallowed
# roughly 40 messages — meaning THIS number, not the budget, is what
# decides how far back the bot can see, and 60 was further than any
# conversation needs.
#
# 20 is about ten exchanges: enough for a visitor to establish a topic,
# get an answer, and follow up several times, which is the span anything
# in this pipeline actually reads back over (`_RETRIEVAL_HISTORY_TURNS`
# looks at 2, and the clarification walk-back stops at the first real
# assistant turn). Beyond that it is cost and context noise.
_MAX_MESSAGES_LOADED = 20


def is_valid_session_id(session_id: str) -> bool:
    """True when `session_id` is a well-formed UUID string.

    Session IDs are client-generated, so they're untrusted input at the
    API boundary — validated here rather than trusting the caller to
    have already checked.
    """
    if not session_id or not isinstance(session_id, str):
        return False
    try:
        uuid.UUID(session_id)
        return True
    except ValueError:
        return False


def get_or_create_session(conn, session_id: str) -> None:
    """Ensures a `sessions` row exists for `session_id` and bumps
    `last_active_at`. Idempotent — safe to call on every request.

    Raises ValueError if `session_id` isn't a valid UUID, so a malformed
    or spoofed ID never reaches the database.
    """
    if not is_valid_session_id(session_id):
        raise ValueError(f"Invalid session_id: {session_id!r} is not a UUID.")

    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO sessions (session_id)
            VALUES (%s)
            ON CONFLICT (session_id) DO UPDATE SET
                last_active_at = CURRENT_TIMESTAMP
            """,
            (session_id,),
        )
    conn.commit()


def load_history(conn, session_id: str, limit: int = _MAX_MESSAGES_LOADED) -> list[ChatMessage]:
    """Returns this session's stored turns, oldest first, ready to hand
    to `app.rag.pipeline.answer_query`'s `conversation_history` param.

    Pulls the most recent `limit` rows then reverses, rather than
    ordering ascending with an offset — cheaper for a session that's
    grown large, since MySQL only has to scan back `limit` rows instead
    of the whole table.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT role, content FROM messages
            WHERE session_id = %s
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """,
            (session_id, limit),
        )
        rows = cursor.fetchall()

    # fetchall()'s return type isn't guaranteed to be a mutable list
    # (depends on cursor class) — wrap explicitly rather than relying
    # on in-place .reverse() working on whatever it returns.
    rows = list(rows)
    rows.reverse()
    return [ChatMessage(role=row[0], content=row[1]) for row in rows]


def save_turn(conn, session_id: str, user_message: str, assistant_message: str) -> None:
    """Persists one user turn + one assistant turn for `session_id`.

    Both rows are written in the same transaction so a request can never
    leave a session with a dangling user message and no reply (or vice
    versa) if something fails mid-way.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
            (session_id, "user", user_message),
        )
        cursor.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
            (session_id, "assistant", assistant_message),
        )
    conn.commit()


def full_transcript_text(conn, session_id: str) -> str:
    """Returns the entire stored conversation for `session_id` as plain
    text (`role: content` per line), oldest first. Kept as a general
    utility (e.g. a future admin view over a lead's full history) — but
    NOT what gets put in lead-notification emails; see
    `recent_transcript_text` for that.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT role, content FROM messages
            WHERE session_id = %s
            ORDER BY created_at ASC, id ASC
            """,
            (session_id,),
        )
        rows = cursor.fetchall()
    return "\n".join(f"{role}: {content}" for role, content in rows)


# How many trailing messages go into a lead/escalation notification
# email. Deliberately short: the email already has the extracted
# structured fields (name, contact, service, etc.) up top — the
# transcript underneath is just enough for a salesperson to get the gist
# of how the conversation got there, not a full read of every back-
# and-forth. 8 messages = last ~4 exchanges.
_EMAIL_SUMMARY_MAX_MESSAGES = 8


def recent_transcript_text(conn, session_id: str, max_messages: int = _EMAIL_SUMMARY_MAX_MESSAGES) -> str:
    """Returns just the last `max_messages` messages for `session_id`,
    oldest-first — used for lead/escalation notification emails, which
    want enough context to be useful without dumping the entire
    conversation on whoever reads the email.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT role, content FROM messages
            WHERE session_id = %s
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """,
            (session_id, max_messages),
        )
        rows = list(cursor.fetchall())
    rows.reverse()
    return "\n".join(f"{role}: {content}" for role, content in rows)

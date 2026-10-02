"""Outbound email notifications to the Apex Creative team — new lead
captured, or a conversation escalated to a human.

Plain smtplib against SMTP (matches the SMTP notification
pattern already used elsewhere in the company's workflows) rather
than a transactional-email SDK — one more HTTPS/SMTP dependency isn't
worth it for "send a plaintext email to one internal address."

Failure handling: a notification send failing must never break the
chat response the visitor already got, or lose the lead data itself
(the lead row is already committed to PostgreSQL before this is called) —
so every public function here catches and logs rather than raising.
Losing a notification email is recoverable (the lead is still in the
`leads` table); crashing the request over it is not.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

logger = logging.getLogger(__name__)

_REQUIRED_ENV_VARS = ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD")


class EmailConfigError(RuntimeError):
    pass


def _load_config() -> dict:
    missing = [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise EmailConfigError(
            f"Missing required SMTP environment variable(s): {', '.join(missing)}"
        )

    recipient = os.environ.get("INTERNAL_NOTIFICATION_EMAIL", "").strip()
    if not recipient:
        raise EmailConfigError(
            "INTERNAL_NOTIFICATION_EMAIL is not set — nowhere to send "
            "lead/escalation notifications."
        )

    return {
        "host": os.environ["SMTP_HOST"],
        "port": int(os.environ.get("SMTP_PORT", "587")),
        "user": os.environ["SMTP_USER"],
        "password": os.environ["SMTP_PASSWORD"],
        "use_ssl": os.environ.get("SMTP_USE_SSL", "false").strip().lower() == "true",
        "from_addr": os.environ.get("SMTP_FROM_EMAIL", os.environ["SMTP_USER"]),
        "from_name": os.environ.get("SMTP_FROM_NAME", "Apex Creative AI Assistant"),
        "recipient": recipient,
    }


def _send(subject: str, body: str) -> bool:
    """Returns True on a successful send, False on any failure (config
    missing, network error, auth error) — never raises, per module
    docstring. Callers that want to know *why* it failed can check logs.
    """
    try:
        config = _load_config()
    except EmailConfigError as exc:
        logger.warning("Skipping email notification — %s", exc)
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = f"{config['from_name']} <{config['from_addr']}>"
    message["To"] = config["recipient"]
    message.set_content(body)

    try:
        if config["use_ssl"]:
            with smtplib.SMTP_SSL(config["host"], config["port"], timeout=15) as server:
                server.login(config["user"], config["password"])
                server.send_message(message)
        else:
            with smtplib.SMTP(config["host"], config["port"], timeout=15) as server:
                server.starttls()
                server.login(config["user"], config["password"])
                server.send_message(message)
        return True
    except (smtplib.SMTPException, OSError) as exc:
        logger.error("Failed to send email notification: %s", exc)
        return False


def _format_lead_fields(lead: dict) -> str:
    labels = {
        "name": "Name",
        "email": "Email",
        "phone": "Phone",
        "whatsapp_number": "WhatsApp",
        "company": "Company",
        "business_type": "Business type",
        "service_requested": "Service requested",
        "budget": "Budget",
        "timeline": "Timeline",
        "notes": "Notes",
    }
    lines = [f"{label}: {lead.get(field) or '—'}" for field, label in labels.items()]
    return "\n".join(lines)


def send_lead_notification(lead: dict, session_id: str, chat_summary: str = "") -> bool:
    subject = f"New Apex Creative lead — {lead.get('name') or 'unnamed visitor'}"
    body = (
        "A new lead was captured by the Apex Creative AI Assistant.\n\n"
        f"{_format_lead_fields(lead)}\n\n"
        f"Session: {session_id}\n"
    )
    if chat_summary:
        body += f"\n--- Conversation ---\n{chat_summary}\n"
    return _send(subject, body)


def send_lead_update_notification(
    lead: dict, session_id: str, updated_fields: dict, chat_summary: str = ""
) -> bool:
    """Sent when a lead the team already got a READY notification for
    volunteers new info afterward (e.g. a phone number given a few
    turns after email alone had already satisfied "a contact method").
    Leads with the subject and updated_fields up front so the team
    doesn't have to diff the full lead by eye against what they already
    have — see app/leads/service.py's module docstring for why this
    exists at all.
    """
    labels = {
        "name": "Name",
        "email": "Email",
        "phone": "Phone",
        "whatsapp_number": "WhatsApp",
        "company": "Company",
        "business_type": "Business type",
        "service_requested": "Service requested",
        "budget": "Budget",
        "timeline": "Timeline",
        "notes": "Notes",
    }
    updated_lines = "\n".join(
        f"{labels.get(field, field)}: {value}" for field, value in updated_fields.items()
    )
    subject = f"Lead update — {lead.get('name') or 'unnamed visitor'} added new info"
    body = (
        "A lead the team was already notified about just provided "
        "additional information.\n\n"
        f"New this update:\n{updated_lines}\n\n"
        f"Full current details:\n{_format_lead_fields(lead)}\n\n"
        f"Session: {session_id}\n"
    )
    if chat_summary:
        body += f"\n--- Recent conversation ---\n{chat_summary}\n"
    return _send(subject, body)


def send_escalation_notification(
    session_id: str, reason: str, lead: dict | None = None, chat_summary: str = ""
) -> bool:
    subject = "Apex Creative AI Assistant — conversation needs a human"
    body = (
        "A conversation was escalated by the Apex Creative AI Assistant and "
        "needs a team member to follow up.\n\n"
        f"Reason: {reason or 'Not specified'}\n"
        f"Session: {session_id}\n"
    )
    if lead:
        body += f"\n{_format_lead_fields(lead)}\n"
    if chat_summary:
        body += f"\n--- Conversation ---\n{chat_summary}\n"
    return _send(subject, body)
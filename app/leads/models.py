"""Data shapes shared between lead extraction and lead persistence.

Kept in their own module (not inline in extractor.py or service.py) so
neither one has to import the other just to know the shape of a lead —
both depend on this instead, which is the actual coupling that exists.
"""

from __future__ import annotations

from dataclasses import dataclass

# Mirrors `leads` table columns 1:1 (minus id/status/created_at, which
# service.py owns) — see schema.sql. Kept in this exact order/naming so
# `dataclasses.asdict()`-style unpacking maps directly onto an INSERT
# without a translation layer.
_LEAD_FIELDS = (
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


@dataclass(frozen=True)
class LeadSignal:
    """What the extractor produces from a conversation snapshot: any
    lead fields it could confidently pull out, plus two independent
    triggers (a lead can be ready without escalating, or escalate
    without being a complete lead yet — e.g. an angry visitor who
    hasn't given contact info).
    """

    name: str | None = None
    email: str | None = None
    phone: str | None = None
    whatsapp_number: str | None = None
    company: str | None = None
    business_type: str | None = None
    service_requested: str | None = None
    budget: str | None = None
    timeline: str | None = None
    notes: str | None = None
    ready_to_submit: bool = False
    escalate: bool = False
    escalate_reason: str | None = None

    def has_any_field(self) -> bool:
        return any(getattr(self, field) for field in _LEAD_FIELDS)

    def as_lead_columns(self) -> dict:
        """Returns just the `leads`-table-shaped fields, dropping the
        signal-only flags (ready_to_submit/escalate/escalate_reason)."""
        return {field: getattr(self, field) for field in _LEAD_FIELDS}

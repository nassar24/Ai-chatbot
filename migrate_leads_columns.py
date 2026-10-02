"""One-off migration: adds leads.notified_at and leads.updated_at to an
EXISTING leads table. apply_schema.py used CREATE TABLE IF NOT EXISTS,
which does nothing to a table that's already there — this is the actual
fix for that gap.

Safe to re-run: each ALTER is wrapped so an already-added column (error
1060, "Duplicate column name") is reported and skipped rather than
crashing the run.

Run: python migrate_leads_columns.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from app.db import get_connection  # noqa: E402

# Postgres supports IF NOT EXISTS on ADD COLUMN, so the duplicate-column
# dance the MySQL version needed is gone. `ON UPDATE CURRENT_TIMESTAMP`
# has no Postgres equivalent either - schema.sql maintains updated_at
# with the trg_leads_updated_at trigger instead.
_ALTERS = [
    "ALTER TABLE leads ADD COLUMN IF NOT EXISTS notified_at TIMESTAMPTZ NULL",
    "ALTER TABLE leads ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL "
    "DEFAULT CURRENT_TIMESTAMP",
]

_DUPLICATE_COLUMN_ERRNO = 1060


def main() -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            for statement in _ALTERS:
                try:
                    cursor.execute(statement)
                    print(f"Applied: {statement}")
                except Exception as exc:
                    errno = getattr(exc, "args", [None])[0]
                    if errno == _DUPLICATE_COLUMN_ERRNO:
                        print(f"Already present, skipped: {statement}")
                    else:
                        print(f"FAILED ({exc}): {statement}")
        conn.commit()
    finally:
        conn.close()

    # Verify, rather than just trusting the ALTER succeeded silently.
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'leads'"
            )
            columns = [row[0] for row in cursor.fetchall()]
        print("\nleads table columns now:", columns)
        missing = [c for c in ("notified_at", "updated_at") if c not in columns]
        if missing:
            print(f"\nSTILL MISSING: {missing} — something went wrong above.")
        else:
            print("\nBoth columns confirmed present.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

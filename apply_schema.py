"""Applies schema.sql using the connection config already in .env.

Avoids needing the `psql` CLI on PATH — psycopg is already installed and
already proven working, and this is the same connection the app uses.

Safe to re-run: every object in schema.sql is created with
IF NOT EXISTS, and the trigger is dropped before being recreated.

Run: python apply_schema.py
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.db import get_connection  # noqa: E402

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_EXPECTED_TABLES = ("kb_chunks", "leads", "sessions", "messages", "qa_cache")


def main() -> None:
    sql = SCHEMA_PATH.read_text(encoding="utf-8")

    conn = get_connection()
    try:
        # Executed whole rather than split on ";". The schema contains a
        # plpgsql function whose body has its own semicolons, so naive
        # splitting produces invalid fragments. Postgres accepts a
        # multi-statement script in a single execute().
        with conn.cursor() as cursor:
            cursor.execute(sql)
        conn.commit()
        print("Applied schema.sql")
    finally:
        conn.close()

    # Verify rather than trusting that the script reported success.
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
            present = {row[0] for row in cursor.fetchall()}
            cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            row = cursor.fetchone()
    finally:
        conn.close()

    print(f"pgvector: {row[0] if row else 'NOT INSTALLED'}")
    print(f"tables:   {', '.join(sorted(present & set(_EXPECTED_TABLES)))}")
    missing = [t for t in _EXPECTED_TABLES if t not in present]
    if missing:
        print(f"\nSTILL MISSING: {missing} — something went wrong above.")
    else:
        print("All expected tables confirmed present.")


if __name__ == "__main__":
    main()

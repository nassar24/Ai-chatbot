import os
import sys
from pathlib import Path

import psycopg
import pytest
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.kb.retrieval import clear_chunk_cache  # noqa: E402  (needs sys.path above)

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.sql"


def _require_test_database(db_name: str) -> None:
    """Refuses to run destructive integration-test fixtures against
    anything that isn't clearly a disposable test database.

    This exists because of a real incident (Bug 1): `conftest.py` loads
    `.env` via `load_dotenv()`, which only fills in env vars that aren't
    already set. In a fresh shell where DB_NAME wasn't exported by hand,
    that silently falls back to whatever DB_NAME is in `.env` — which is
    the same database `calibrate_threshold.py` and the live app read.
    These fixtures DROP and rebuild `kb_chunks`/`sessions`/`messages`/
    `leads` on every run, so running them against that database leaves
    calibration and dev data corrupted with fixture rows, with no error
    at the time it happens. README.md documents "export DB_NAME=..._test"
    as the convention; this makes that convention load-bearing instead of
    just advisory.
    """
    if "test" not in db_name.lower():
        raise RuntimeError(
            f"Refusing to run destructive DB test fixtures against DB_NAME={db_name!r}. "
            "This fixture drops and recreates kb_chunks/sessions/messages/leads on "
            "every run. Point DB_NAME at a disposable database whose name contains "
            "'test' (e.g. apexcreative_test), per README.md's test setup instructions."
        )


@pytest.fixture()
def db_connection():
    """Shared fixture for integration tests: connects to a fresh copy of
    the schema on a database confirmed to be a test database, and closes
    the connection afterward. Centralized here (rather than duplicated in
    each integration test module) so the safety check in
    `_require_test_database` can't be skipped by adding a new test file.
    """
    db_name = os.environ["DB_NAME"]
    _require_test_database(db_name)

    # Retrieval keeps a process-level decoded-chunk cache keyed on a
    # fingerprint of kb_chunks. These fixtures drop and rebuild that table
    # between tests faster than the fingerprint's timestamp component can
    # resolve, so the cache is dropped explicitly here rather than relying
    # on the production freshness check to notice.
    clear_chunk_cache()

    connection = psycopg.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", "5432")),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        dbname=db_name,
        autocommit=False,
    )
    with connection.cursor() as cursor:
        # CASCADE because sessions references leads; dropping in
        # dependency order by hand is one more thing to get wrong when a
        # table is added.
        cursor.execute(
            "DROP TABLE IF EXISTS messages, sessions, leads, kb_chunks, qa_cache CASCADE"
        )
    connection.commit()
    # Executed whole rather than split on ";" - the schema now contains a
    # plpgsql function whose body has its own semicolons, and naive
    # splitting cuts it in half.
    with connection.cursor() as cursor:
        cursor.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    connection.commit()

    yield connection
    connection.close()
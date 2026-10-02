"""PostgreSQL connection factory. Credentials come only from environment
variables — never hardcoded (see .env.example).

Backed by a psycopg connection pool. Every request, every background
lead-capture task, and the debounced-lead-email scheduler job all call
get_connection() — without pooling that is a fresh TCP handshake plus
auth round-trip on each one, and under concurrent traffic it risks
hitting the server's connection cap outright rather than degrading
gracefully.

PORTED FROM MYSQL
-----------------
Call sites are unchanged: `conn.cursor()`, `.commit()`, `.close()` behave
exactly as before, and `%s` placeholders are the same in psycopg as in
PyMySQL. That is deliberate — the point of this migration was to move
storage, not to rewrite every query.

The one thing needing care is `.close()`. Code throughout this codebase
does `conn = get_connection()` ... `conn.close()`, and with a pool that
must RETURN the connection rather than destroy it. psycopg_pool's native
API is a context manager, so `_PooledConnection` below adapts it to the
close()-based lifecycle the rest of the code already uses. Without that
wrapper, every `.close()` would leak a connection out of the pool until
it was exhausted.
"""

from __future__ import annotations

import os
import threading

from psycopg_pool import ConnectionPool

_REQUIRED_ENV_VARS = ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD")

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


class _PooledConnection:
    """Adapts a pooled psycopg connection to a close()-based lifecycle.

    Delegates everything to the real connection except `close()`, which
    hands it back to the pool. Idempotent: closing twice is a no-op
    rather than an error, because several call sites close in a `finally`
    after an inner path may already have done so.
    """

    __slots__ = ("_conn", "_pool", "_returned")

    def __init__(self, conn, pool):
        self._conn = conn
        self._pool = pool
        self._returned = False

    def cursor(self, *args, **kwargs):
        return self._conn.cursor(*args, **kwargs)

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def close(self):
        if self._returned:
            return
        self._returned = True
        try:
            # Roll back anything left uncommitted, so the next borrower
            # never inherits a half-finished transaction.
            self._conn.rollback()
        except Exception:
            pass
        self._pool.putconn(self._conn)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _build_pool() -> ConnectionPool:
    missing = [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Missing required DB environment variable(s): {', '.join(missing)}"
        )

    # Pool size is intentionally conservative and configurable. Sized with
    # real headroom under the server's max_connections, which is shared
    # with anything else touching the same database (n8n, psql sessions).
    max_size = int(os.environ.get("DB_POOL_MAX_CONNECTIONS", "10"))
    min_size = int(os.environ.get("DB_POOL_MIN_CACHED", "2"))

    conninfo = (
        f"host={os.environ['DB_HOST']} "
        f"port={os.environ.get('DB_PORT', '5432')} "
        f"user={os.environ['DB_USER']} "
        f"password={os.environ['DB_PASSWORD']} "
        f"dbname={os.environ['DB_NAME']}"
    )
    pool = ConnectionPool(
        conninfo=conninfo,
        min_size=min_size,
        max_size=max_size,
        # Wait for a free connection under load rather than raising
        # immediately — a request briefly queueing is a better failure
        # mode than an unhandled exception during a traffic burst.
        timeout=30.0,
        open=True,
    )
    pool.wait(timeout=30.0)
    return pool


def get_connection():
    """Returns a pooled connection with the same interface as before.

    The pool is a lazily-created module-level singleton, built on first
    call and reused for the process lifetime. Construction is locked
    because callers are concurrent by design (request threads,
    lead-capture tasks, the scheduler); an unsynchronised
    check-then-assign lets two cold-start callers each build a pool, and
    the discarded one keeps its connections open against the server's cap.
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:  # re-checked: another thread may have built it
                _pool = _build_pool()
    return _PooledConnection(_pool.getconn(), _pool)

"""WSGI entry point — what Passenger (Hostinger) or gunicorn imports.

This file's whole job is to produce a module-level `application` object.
It previously contained a byte-identical copy of `app/api/app.py`, which
defines `create_app()` and nothing else — so there was no `application`
(or `app`) attribute for a WSGI server to find, and the deploy could
not have started regardless of how hPanel was configured.

Both names are exported because hPanel's "Application entry point" field
asks for one or the other depending on the panel version, and there's no
reason to make that a guess.

Environment and logging are set up HERE rather than inside
`create_app()`: they're properties of running as a deployed process, not
of the Flask app itself, and doing them in the factory would mean tests
and CLI scripts that import `create_app` silently inherit a production
logging configuration they never asked for.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

# Loads a .env sitting next to this file, if present. load_dotenv never
# overrides variables that are already set, so real hPanel environment
# variables always win over a stale .env left in the app root.
load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

from app.api.app import create_app  # noqa: E402  (must follow load_dotenv)

application = create_app()

# Alias — some Passenger/hPanel configurations look for `app`.
app = application

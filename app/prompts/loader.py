"""Loads the Apex Creative system prompt.

This file is treated as fixed content, per instruction — nothing in this
codebase edits its rules. If the prompt needs to change, edit
system_prompt.md directly; don't string-manipulate it in code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPT_PATH = Path(__file__).resolve().parent / "system_prompt.md"


@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    if not _PROMPT_PATH.is_file():
        raise RuntimeError(f"System prompt file not found at {_PROMPT_PATH}")
    text = _PROMPT_PATH.read_text(encoding="utf-8")
    if not text.strip():
        raise RuntimeError(f"System prompt file at {_PROMPT_PATH} is empty.")
    return text

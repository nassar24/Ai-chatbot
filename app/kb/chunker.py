"""Splits the knowledge-base markdown into retrievable chunks.

Chunking rule (per the build plan: "chunked by ## header"), with one
necessary refinement: a few top-level (#) sections in the actual KB file
have no ## children at all (FAQ, Sales Guidance, Lead Collection Fields).
For those, the whole # block becomes a single chunk — otherwise that
content would be silently dropped. Where a # section *does* have ##
children (About, Services, Packages, Policies, Team), the children are
each their own chunk and the parent heading is folded into the chunk's
title for retrieval context, e.g. "Company Policies — Refund Policy".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,2})\s+(.+?)\s*$")


@dataclass(frozen=True)
class RawChunk:
    section_title: str
    content: str


def strip_frontmatter(text: str) -> str:
    """Removes a leading YAML frontmatter block (--- ... ---), if present.

    Public because the ingest CLI needs it too: the knowledge base is now
    several files (English + Arabic) that have to be ingested as ONE text,
    and only the first file's frontmatter would sit at the start of the
    concatenation. The rest would be read as body content and indexed as
    if it were knowledge.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1:])
    return text  # unterminated frontmatter — treat whole thing as content, don't crash


def chunk_markdown(markdown_text: str) -> list[RawChunk]:
    """Parses markdown into a flat list of (section_title, content) chunks.

    Raises ValueError on empty input rather than silently returning [] —
    an empty KB is a configuration error the caller needs to see, not a
    valid ingestion outcome.
    """
    if not markdown_text or not markdown_text.strip():
        raise ValueError("Cannot chunk empty markdown content.")

    body = strip_frontmatter(markdown_text)

    headings: list[dict] = []
    current: dict | None = None
    for line in body.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            current = {
                "level": len(match.group(1)),
                "title": match.group(2).strip(),
                "lines": [],
            }
            headings.append(current)
            continue
        if line.strip() == "---":
            continue  # horizontal-rule separator between # sections, not content
        if current is not None:
            current["lines"].append(line)

    chunks: list[RawChunk] = []
    parent_h1_title: str | None = None
    i, n = 0, len(headings)

    while i < n:
        heading = headings[i]

        if heading["level"] == 1:
            next_is_h2_child = (i + 1 < n) and headings[i + 1]["level"] == 2
            if next_is_h2_child:
                parent_h1_title = heading["title"]
                i += 1
                continue
            content = "\n".join(heading["lines"]).strip()
            if content:
                chunks.append(RawChunk(section_title=heading["title"], content=content))
            parent_h1_title = None
            i += 1
            continue

        # level == 2
        content = "\n".join(heading["lines"]).strip()
        if content:
            title = (
                f"{parent_h1_title} — {heading['title']}"
                if parent_h1_title
                else heading["title"]
            )
            chunks.append(RawChunk(section_title=title, content=content))
        i += 1

    if not chunks:
        raise ValueError(
            "Markdown parsed but produced zero chunks — check heading structure."
        )
    return chunks

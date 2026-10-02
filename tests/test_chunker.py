from pathlib import Path

import pytest

from app.kb.chunker import chunk_markdown

KB_PATH = Path(__file__).resolve().parent.parent / "knowledge_base.md"


@pytest.fixture(scope="module")
def kb_text() -> str:
    return KB_PATH.read_text(encoding="utf-8")


def test_raises_on_empty_input():
    with pytest.raises(ValueError):
        chunk_markdown("")


def test_frontmatter_is_stripped_not_treated_as_content(kb_text):
    chunks = chunk_markdown(kb_text)
    for chunk in chunks:
        assert "last_updated:" not in chunk.content
        assert "scope: Public-facing" not in chunk.content


def test_h2_sections_get_parent_prefixed_titles(kb_text):
    chunks = chunk_markdown(kb_text)
    titles = {chunk.section_title for chunk in chunks}
    assert "Company Policies — Refund Policy" in titles
    assert "Company Policies — Payment Policy" in titles
    assert "Our Services — 09. AI Creative Solutions" in titles
    assert "Frequently Asked Questions — FAQ: Clients Outside Egypt" in titles


def test_h1_only_sections_become_single_chunks(kb_text):
    chunks = chunk_markdown(kb_text)
    titles = {chunk.section_title for chunk in chunks}
    # These sections have no ## children in the source file.
    assert "How We Work (Workflow)" in titles
    assert "Lead Collection Fields (for chatbot to gather from prospective clients)" in titles


def test_faq_subchunks_exist(kb_text):
    chunks = chunk_markdown(kb_text)
    faq_subchunk = next(c for c in chunks if c.section_title == "Frequently Asked Questions — FAQ: Clients Outside Egypt")
    assert "Do you work with clients outside Egypt?" in faq_subchunk.content



def test_team_members_each_get_own_chunk(kb_text):
    chunks = chunk_markdown(kb_text)
    titles = {chunk.section_title for chunk in chunks}
    assert "Our Team — Dr. Alex Chen — Founder & CEO" in titles
    assert "Our Team — Dakota Martinez — Software Team Lead" in titles


def test_no_duplicate_section_titles(kb_text):
    chunks = chunk_markdown(kb_text)
    titles = [chunk.section_title for chunk in chunks]
    assert len(titles) == len(set(titles)), "Duplicate section_title breaks the UNIQUE KEY upsert"


def test_horizontal_rules_not_captured_as_content(kb_text):
    chunks = chunk_markdown(kb_text)
    for chunk in chunks:
        assert chunk.content.strip() != "---"
        lines = chunk.content.splitlines()
        assert all(line.strip() != "---" for line in lines)


def test_reasonable_total_chunk_count(kb_text):
    chunks = chunk_markdown(kb_text)
    # Sanity bound, not an exact count — protects against a parsing
    # regression that silently collapses everything into 1 chunk or
    # explodes every line into its own chunk.
    assert 25 <= len(chunks) <= 60

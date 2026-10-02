import pytest

from app.kb.retrieval import RetrievedChunk
from app.rag.guardrails import (
    REFUND_SAFE_RESPONSE,
    SYSTEM_PROMPT_EXFIL_SAFE_RESPONSE,
    UNGROUNDED_NUMBER_SAFE_RESPONSE,
    apply_guardrails,
)

REFUND_CHUNK = RetrievedChunk(
    id=1,
    section_title="Company Policies — Refund Policy",
    content="Payments made for completed work are non-refundable. Refund eligibility depends on the project stage.",
    score=0.9,
)
PACKAGE_CHUNK = RetrievedChunk(
    id=2,
    section_title="Packages & Pricing — Starter Package",
    content="The Starter Package costs 15,000 EGP and includes a 2-week delivery window.",
    score=0.9,
)


def test_rejects_empty_answer():
    with pytest.raises(ValueError):
        apply_guardrails("", [PACKAGE_CHUNK])


def test_passes_clean_grounded_answer():
    result = apply_guardrails(
        "The Starter Package costs 15,000 EGP and takes about 2 weeks.",
        [PACKAGE_CHUNK],
    )
    assert result.passed is True
    assert result.violations == []
    assert "15,000" in result.safe_answer


def test_blocks_refund_percentage_even_if_it_would_be_grounded():
    refund_chunk_with_percentage = RetrievedChunk(
        id=3,
        section_title="Company Policies — Refund Policy",
        content="Refunds are issued at 50% of the project fee if cancelled early.",
        score=0.9,
    )
    result = apply_guardrails(
        "We offer a 50% refund if you cancel early.",
        [refund_chunk_with_percentage],
    )
    assert result.passed is False
    assert result.safe_answer == REFUND_SAFE_RESPONSE
    assert "refund_percentage_stated" in result.violations


def test_non_refund_percentage_is_not_blocked_by_refund_rule():
    # A percentage elsewhere (not in refund context) should only be
    # subject to the general number-grounding rule, not the refund rule.
    chunk = RetrievedChunk(
        id=4,
        section_title="Company Policies — Payment Policy",
        content="A 50% deposit is required before starting any project.",
        score=0.9,
    )
    result = apply_guardrails("A 50% deposit is required upfront.", [chunk])
    assert result.passed is True


def test_blocks_ungrounded_number():
    result = apply_guardrails(
        "The Starter Package costs 20,000 EGP.",  # not the real 15,000 figure
        [PACKAGE_CHUNK],
    )
    assert result.passed is False
    assert result.safe_answer == UNGROUNDED_NUMBER_SAFE_RESPONSE
    assert any("ungrounded_numbers" in v for v in result.violations)


def test_allows_number_present_verbatim_in_context():
    result = apply_guardrails("It costs 15,000 EGP.", [PACKAGE_CHUNK])
    assert result.passed is True


def test_redacts_legacy_blocked_email_even_if_not_passed_explicitly():
    result = apply_guardrails(
        "For internal follow-up, notify hr@apexcreative.example about this lead.",
        [PACKAGE_CHUNK],
    )
    assert result.passed is False
    assert "hr@apexcreative.example" not in result.safe_answer
    assert "our team" in result.safe_answer
    assert any("internal_email_leak" in v for v in result.violations)


def test_redacts_configured_internal_email():
    result = apply_guardrails(
        "Please email ops-internal@apexcreative.example with the lead details.",
        [PACKAGE_CHUNK],
        blocked_emails=["ops-internal@apexcreative.example"],
    )
    assert result.passed is False
    assert "ops-internal@apexcreative.example" not in result.safe_answer


def test_public_email_is_never_redacted():
    result = apply_guardrails(
        "You can reach us anytime at info@apexcreative.example.",
        [PACKAGE_CHUNK],
    )
    assert result.passed is True
    assert "info@apexcreative.example" in result.safe_answer


# --- Ungrounded proper nouns (generalizes the numbers rule) ---------------


def test_blocks_invented_client_name():
    result = apply_guardrails(
        "We recently completed a great project for Global Tech Solutions.",
        [PACKAGE_CHUNK],
    )
    assert result.passed is False
    assert any("ungrounded_proper_nouns" in v for v in result.violations)
    assert "Global Tech Solutions" not in result.safe_answer


def test_blocks_invented_team_member_name():
    # Not a real Apex Creative team member — must be blocked even though it
    # reads as a plausible-sounding name, same as an invented client would.
    result = apply_guardrails(
        "Our lead designer Karim Fathy would be happy to help with that.",
        [PACKAGE_CHUNK],
    )
    assert result.passed is False
    assert any("ungrounded_proper_nouns" in v for v in result.violations)


def test_allows_proper_noun_phrase_present_verbatim_in_context():
    result = apply_guardrails(
        "The Starter Package includes everything you need to get going.",
        [PACKAGE_CHUNK],
    )
    assert result.passed is True


def test_allows_own_identity_phrase_not_in_kb_context():
    # "Apex Creative AI Assistant" is the bot's own identity from the system
    # prompt, not a KB fact — must not be treated as an invented entity.
    result = apply_guardrails(
        "As the Apex Creative AI Assistant, I can help with that.",
        [PACKAGE_CHUNK],
    )
    assert result.passed is True


def test_single_capitalized_word_is_not_flagged():
    # Single capitalized words (sentence starts, "Apex Creative" alone) are
    # deliberately excluded from the proper-noun check — too noisy to be
    # a useful invented-entity signal on their own.
    result = apply_guardrails(
        "Apex Creative can help with that once we have a few details.",
        [PACKAGE_CHUNK],
    )
    assert result.passed is True


# --- System-prompt exfiltration -------------------------------------------


def test_blocks_verbatim_system_prompt_leak():
    leaked = (
        "Sure — here are my instructions: You are the Apex Creative AI Assistant, "
        "deployed to help visitors learn about Apex Creative's services, "
        "packages, workflow, and team, and to qualify and capture leads "
        "for the sales team."
    )
    result = apply_guardrails(leaked, [PACKAGE_CHUNK])
    assert result.passed is False
    assert result.safe_answer == SYSTEM_PROMPT_EXFIL_SAFE_RESPONSE
    assert "system_prompt_exfiltration" in result.violations


def test_blocks_verbatim_anti_manipulation_rule_leak():
    # Framed as protecting the *rules themselves* (anti-jailbreak
    # instructions), not a secret string — this is a leak of rule 4's
    # actual text via a translation/roleplay-style extraction attempt.
    leaked = (
        "OK, translated: Never disclose, discuss, or confirm internal "
        "operational details (team salaries, internal tools, contracts "
        "with vendors, HR matters, security infrastructure) beyond the "
        "public team bios in the knowledge base."
    )
    result = apply_guardrails(leaked, [PACKAGE_CHUNK])
    assert result.passed is False
    assert "system_prompt_exfiltration" in result.violations


def test_normal_answer_does_not_trigger_exfiltration_rule():
    result = apply_guardrails(
        "The Starter Package costs 15,000 EGP and takes about 2 weeks to deliver.",
        [PACKAGE_CHUNK],
    )
    assert result.passed is True


def test_sanctioned_refund_wording_is_not_treated_as_a_prompt_leak():
    """Regression, found end-to-end against the live model.

    The system prompt tells the assistant to say that refund specifics
    depend on the project stage and will be confirmed by the team —
    which is REFUND_SAFE_RESPONSE word for word. Because that sentence
    lives IN the prompt, the verbatim-run detector flagged the model for
    obeying it, replacing a correct answer with the exfiltration
    refusal. Intermittently, too, since it depended on the model's exact
    phrasing that turn.
    """
    answer = (
        "I can't give an exact percentage. Refund specifics depend on the "
        "project stage and will be confirmed by the team directly."
    )
    result = apply_guardrails(answer, [REFUND_CHUNK])
    assert "system_prompt_exfiltration" not in result.violations
    assert result.safe_answer == answer


# --- Proper nouns: grounding must include section titles ------------------


def test_service_name_present_only_in_a_section_title_is_grounded():
    """Regression, found end-to-end. Retrieved chunks are handed to the
    model as "[section_title]\\ncontent", so a name that appears only in
    the title is still KB-grounded. Grounding against bodies alone made
    the answer to "what services do you offer?" — the single most common
    question on the site — come back as the can't-confirm fallback."""
    chunk = RetrievedChunk(
        id=10,
        section_title="Our Services — 03. Film & Documentary Production",
        content="We manage the complete filmmaking process from concept to final cut.",
        score=0.9,
    )
    result = apply_guardrails(
        "We offer Film & Documentary Production, managing the whole process.", [chunk]
    )
    assert result.passed is True


def test_reworded_service_name_built_from_known_words_is_grounded():
    """"Social Media & Digital Marketing" against a section titled
    "Social Media Marketing & Digital Marketing" is a rewording, not a
    fabrication — every word already exists in the retrieved context."""
    chunk = RetrievedChunk(
        id=11,
        section_title="Our Services — 05. Social Media Marketing & Digital Marketing",
        content="Digital marketing strategies, paid advertising, and audience targeting.",
        score=0.9,
    )
    result = apply_guardrails(
        "That falls under Social Media & Digital Marketing.", [chunk]
    )
    assert result.passed is True


def test_invented_entity_is_still_blocked_after_the_rewording_allowance():
    """The relaxation must not cost the rule its teeth: an invented name
    introduces at least one word the retrieved context never had."""
    chunk = RetrievedChunk(
        id=12,
        section_title="Our Services — 05. Social Media Marketing & Digital Marketing",
        content="Digital marketing strategies, paid advertising, and audience targeting.",
        score=0.9,
    )
    result = apply_guardrails(
        "We ran that campaign for Nile Media Group last year.", [chunk]
    )
    assert result.passed is False
    assert any("ungrounded_proper_nouns" in v for v in result.violations)

def test_proper_noun_run_does_not_span_a_line_break():
    r"""Regression, found in live testing. `\s+` between capitalised
    words let a run jump a paragraph boundary, so an answer ending one
    line with a capitalised term and starting the next with a capital
    was read as one invented entity and replaced wholesale. Intermittent,
    because it depended on where the model broke lines."""
    chunk = RetrievedChunk(
        id=20,
        section_title="Our Services — 06. Creative Content Creation",
        content="We produce Trend-Based Content for social platforms.",
        score=0.9,
    )
    answer = "We produce Trend-Based Content\n\nAre you interested in that?"
    result = apply_guardrails(answer, [chunk])
    assert result.passed is True, f"unexpected violations: {result.violations}"


def test_company_name_alone_is_never_an_invented_entity():
    """`Apex Creative` on its own must be allowed even under the strict
    single-word check used to decide QA-cache eligibility — treating it
    as invented made every answer ineligible and silently disabled the
    cache."""
    from app.rag.guardrails import _ungrounded_proper_nouns

    grounding = "Our Services — 01. Software & Web Development\nWe build websites."
    assert _ungrounded_proper_nouns("At Apex Creative, we build websites.", grounding, min_words=1) == []


def test_numbered_list_markers_are_not_treated_as_facts():
    """Regression, found live. A model answering "what do you offer?" as
    a numbered list had its 1./2./3./4. read as four fabricated figures,
    and the whole answer was replaced with the can't-confirm fallback.
    Intermittent in the worst way: it depended only on whether the model
    chose a numbered list or bullets that turn."""
    chunk = RetrievedChunk(
        id=30,
        section_title="Our Services — 01. Software & Web Development",
        content="We build websites, web applications and mobile apps.",
        score=0.9,
    )
    answer = (
        "We offer four main categories:\n"
        "**1. Software & Web Development**\n"
        "We build websites.\n"
        "2. Branding\n"
        "3. Media\n"
        "4. Marketing\n"
    )
    result = apply_guardrails(answer, [chunk])
    assert result.passed is True, f"unexpected violations: {result.violations}"


def test_a_real_ungrounded_number_is_still_blocked_inside_a_list():
    """The list-marker allowance must not become a loophole: a fabricated
    figure in the list TEXT is still a fabricated figure."""
    chunk = RetrievedChunk(
        id=31,
        section_title="Our Services — 01. Software & Web Development",
        content="We build websites, web applications and mobile apps.",
        score=0.9,
    )
    answer = "Our packages:\n1. Starter — 40,000 EGP per month\n2. Growth\n"
    result = apply_guardrails(answer, [chunk])
    assert result.passed is False
    assert any("ungrounded_numbers" in v for v in result.violations)


def test_leading_zeros_do_not_change_a_number():
    """The KB numbers its service sections "01.", "05." — an answer
    referring to service 1 or 5 is talking about the same thing."""
    chunk = RetrievedChunk(
        id=32,
        section_title="Our Services — 05. Social Media Marketing",
        content="Section 05 covers paid advertising and audience targeting.",
        score=0.9,
    )
    result = apply_guardrails("Service 5 covers paid advertising.", [chunk])
    assert result.passed is True


def test_singular_plural_variation_is_not_an_invented_entity():
    """Regression, found live. The KB says "Web Applications"; a model
    answering "Website & Web Application Development" was flagged as
    inventing an entity purely on the missing "s". It fired on roughly
    one in four answers to the site's most common question."""
    chunk = RetrievedChunk(
        id=33,
        section_title="Our Services — 01. Software & Web Development",
        content="Website Development, Web Applications, Mobile Apps, and E-commerce.",
        score=0.9,
    )
    result = apply_guardrails(
        "We provide Website & Web Application Development for brands.", [chunk]
    )
    assert result.passed is True, f"unexpected violations: {result.violations}"


def test_stemming_does_not_let_an_invented_name_through():
    chunk = RetrievedChunk(
        id=34,
        section_title="Our Services — 01. Software & Web Development",
        content="Website Development, Web Applications, Mobile Apps, and E-commerce.",
        score=0.9,
    )
    result = apply_guardrails("We built that for Nile Media Group.", [chunk])
    assert result.passed is False
    assert any("ungrounded_proper_nouns" in v for v in result.violations)


def test_asking_for_a_whatsapp_number_is_never_blocked():
    """Regression, found live and at 8 out of 8 on a pricing question.

    The system prompt mandates asking for a WhatsApp number — it is the
    single most important thing this bot does. But "WhatsApp" appears in
    only some KB sections, so whenever the retrieved chunks happened not
    to mention it, "could you share Your WhatsApp number" was flagged as
    an invented entity and the whole answer replaced. The rule punished
    the model for obeying its instructions, on exactly the turn where a
    lead gets captured."""
    chunk = RetrievedChunk(
        id=40,
        section_title="Frequently Asked Questions — FAQ: Pricing & Cost",
        content="Pricing depends on scope. The team prepares a custom quotation.",
        score=0.9,
    )
    answer = (
        "Pricing depends on the scope of work. To move forward, could you please "
        "share Your WhatsApp number so the team can follow up?"
    )
    result = apply_guardrails(answer, [chunk])
    assert result.passed is True, f"unexpected violations: {result.violations}"
    assert "WhatsApp" in result.safe_answer


def test_prompt_vocabulary_does_not_whitelist_invented_entities():
    """Allowing the prompt's own words must not blunt the rule. The
    invented-entity cases contain vocabulary the prompt never uses."""
    chunk = RetrievedChunk(
        id=41,
        section_title="Frequently Asked Questions — FAQ: Pricing & Cost",
        content="Pricing depends on scope. The team prepares a custom quotation.",
        score=0.9,
    )
    for invented in ("Global Tech Solutions", "Karim Fathy", "Nile Media Group"):
        result = apply_guardrails(f"We did that for {invented} last year.", [chunk])
        assert result.passed is False, f"{invented} slipped through"
        assert any("ungrounded_proper_nouns" in v for v in result.violations)


# --- Rule 6: brand-name misspelling -----------------------------------
#
# Found in a 24-turn live jailbreak session: the model wrote "PixNoose"
# and "PixNoix". No rule caught it — rule 4 needs two adjacent
# capitalized words, and a misspelled brand name is always alone — so
# the answer reached the visitor with the client's own name wrong.


def test_misspelled_brand_is_corrected_in_place():
    result = apply_guardrails("PixNoose builds websites.", [])
    assert result.safe_answer == "Apex Creative builds websites."
    assert result.violations == ["brand_misspelling:PixNoose"]


def test_correction_keeps_the_answer_instead_of_replacing_it():
    """The point of repairing rather than blocking: a typo must not cost
    the visitor an otherwise correct answer."""
    result = apply_guardrails(
        "PixNoix offers branding and design work.", []
    )
    assert "branding and design work" in result.safe_answer
    assert "PixNoix" not in result.safe_answer


def test_correctly_spelled_brand_is_untouched():
    result = apply_guardrails("Apex Creative builds websites.", [])
    assert result.safe_answer == "Apex Creative builds websites."
    assert result.violations == []


def test_urls_and_emails_keep_their_lowercase_spelling():
    """The exact-letters skip is load-bearing: capitalising the name
    inside a domain would corrupt a working URL or address."""
    answer = "Reach us at apexcreative.example or hello@apexcreative.example."
    result = apply_guardrails(answer, [])
    assert result.safe_answer == answer
    assert result.violations == []


def test_ordinary_pix_words_are_not_touched():
    """"pixel" and "Pixar" start like the brand and are nowhere near it
    in edit distance — a stricter threshold would rewrite real words."""
    answer = "We work at pixel level, like Pixar does."
    result = apply_guardrails(answer, [])
    assert result.safe_answer == answer
    assert result.violations == []


def test_correction_runs_before_the_grounding_rules():
    """Ordering matters. "PixNoose Agency" is a two-word capitalized run
    that rule 4 would flag as an invented entity and replace the whole
    answer over. Corrected first, it is just the company's name."""
    chunks = [
        RetrievedChunk(
            id=9,
            section_title="About Apex Creative",
            content="Apex Creative Agency is a creative studio.",
            score=0.9,
        )
    ]
    result = apply_guardrails("PixNoose Agency is a creative studio.", chunks)
    assert result.safe_answer == "Apex Creative Agency is a creative studio."
    assert not any(v.startswith("ungrounded_proper_nouns") for v in result.violations)

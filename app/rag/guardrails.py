"""Outbound response guardrails, applied after generation and before an
answer is ever shown to a visitor.

The system prompt already instructs the model not to violate these
rules. These checks exist because a hard, code-level filter doesn't rely
on the model getting it right every single time — per the build plan's
explicit requirement for these as a second layer of defense (e.g. "an
outbound-message filter as a second layer of defense against [the
internal email] ever appearing in a bot response").

Six rules. Rules 1-5 are checked in order of strictness; rule 6 runs
before all of them because it only normalizes text the others then read.
1. Refund percentages are never relayed, even if one is technically
   present in retrieved content — a permanent policy, not a KB gap.
2. System-prompt exfiltration: an 8+ word verbatim run shared with the
   system prompt never survives into an answer. This is the outbound
   half of prompt-injection defense — the inbound half lives in
   `app/rag/input_guard.py`. Deliberately a verbatim-run detector, not a
   semantic one: a model that genuinely paraphrases its rules in its own
   words slips past (a known, documented gap), but the far more common
   failure — a model pressured into quoting or "translating" its
   instructions, which reproduces long runs of the original wording —
   is caught deterministically with no second LLM call.
3. Any other number in the answer must be numerically grounded — either
   present in the retrieved KB content, or present somewhere earlier in
   this conversation (a number the visitor themselves typed, like a
   phone number, isn't a hallucination when the assistant reflects it
   back). Comma/whitespace formatting is normalized before comparing
   (e.g. "6,000" and "6000" are the same number) — formatting shouldn't
   determine groundedness, only numeric identity should.
4. Multi-word proper nouns must be grounded the same way numbers are —
   this generalizes rule 3 to the other high-risk hallucination
   category the system prompt calls out (invented client names, case
   studies, team members). Single capitalized words are deliberately
   NOT flagged: sentence starts make them far too noisy to carry signal.
5. Known internal-only email addresses are redacted from the answer,
   regardless of whether they leaked via the model or some other path.
6. The company's own name is spelled correctly. Rule 4 needs two
   adjacent capitalized words, so a lone misspelled "Apex Creativ" was
   invisible to every rule here and went out to visitors as-is.

Rules 1-4 replace the whole answer with a safe fallback; rules 5 and 6
edit in place, since the rest of that answer may still be good. An
in-place edit still counts as a violation, so `passed` is False and
`_store_if_reusable` will not put a repaired answer into the shared QA
cache — the intent being that the cache only ever holds answers the
model got right unaided. Every rule fails
toward "say less than we could" rather than "let something through" —
for a lead-gen bot, a visitor being told to talk to the team is a
recoverable outcome, a fabricated price or a leaked prompt is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from rapidfuzz.distance import Levenshtein

from app.kb.retrieval import RetrievedChunk
from app.prompts.loader import load_system_prompt
from app.text.normalize import has_arabic, normalize_digits, tokenize

# Numbers: a leading digit followed by more digits or INTERNAL separators
# only. The trailing-separator exclusion matters — an earlier version
# (`\d[\d,.]*%?`) swallowed sentence-ending punctuation, so an answer
# ending "...founded in 2015." produced the token "2015." which never
# matched the KB's "2015" and got a correct, fully grounded answer
# replaced by the fallback response. Any number ending a sentence hit it.
_NUMBER_RE = re.compile(r"\d(?:[\d,.]*\d)?%?")
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")

# Named explicitly in the build plan as an address that must never reach
# a visitor, independent of whatever the *current* configured internal
# recipient is (that value is still unset/TBD).
_LEGACY_BLOCKED_EMAIL = "hr@apexcreative.example"

REFUND_SAFE_RESPONSE = (
    "Refund specifics depend on the project stage and will be confirmed "
    "by the team directly."
)
UNGROUNDED_NUMBER_SAFE_RESPONSE = (
    "I don't have that exact detail confirmed — I can connect you with "
    "our team so they can give you precise information."
)
UNGROUNDED_PROPER_NOUN_SAFE_RESPONSE = (
    "I don't have confirmed details on that — I can connect you with our "
    "team so they can give you accurate information."
)
SYSTEM_PROMPT_EXFIL_SAFE_RESPONSE = (
    "I can't share my internal instructions or configuration — happy "
    "to help with anything about Apex Creative's services, pricing, or "
    "team though."
)

# Arabic counterparts. A guardrail replaces the model's answer wholesale,
# so without these an Arabic conversation switched to English at exactly
# the moment something went wrong — the visitor sees a language change and
# reads it as the bot breaking, which is also a tell that a rule fired.
#
# Keyed off the language of the ANSWER being replaced rather than the
# question: the answer is what the visitor was about to read, and it is
# the model's own judgement of which language this turn is in.
REFUND_SAFE_RESPONSE_AR = (
    "تفاصيل الاسترجاع بتعتمد على مرحلة المشروع، والفريق هو اللي بيأكدها "
    "معاك مباشرة."
)
UNGROUNDED_NUMBER_SAFE_RESPONSE_AR = (
    "معنديش الرقم ده مؤكد — أقدر أوصلك بفريقنا عشان يدوك معلومة دقيقة."
)
UNGROUNDED_PROPER_NOUN_SAFE_RESPONSE_AR = (
    "معنديش تفاصيل مؤكدة عن ده — أقدر أوصلك بفريقنا عشان يدوك معلومات دقيقة."
)
SYSTEM_PROMPT_EXFIL_SAFE_RESPONSE_AR = (
    "مش هقدر أشارك تعليماتي أو إعداداتي الداخلية — بس تحت أمرك في أي حاجة "
    "عن خدمات Apex Creative أو الأسعار أو الفريق."
)

_ARABIC_SAFE_RESPONSES = {
    REFUND_SAFE_RESPONSE: REFUND_SAFE_RESPONSE_AR,
    UNGROUNDED_NUMBER_SAFE_RESPONSE: UNGROUNDED_NUMBER_SAFE_RESPONSE_AR,
    UNGROUNDED_PROPER_NOUN_SAFE_RESPONSE: UNGROUNDED_PROPER_NOUN_SAFE_RESPONSE_AR,
    SYSTEM_PROMPT_EXFIL_SAFE_RESPONSE: SYSTEM_PROMPT_EXFIL_SAFE_RESPONSE_AR,
}


def _localized(safe_response: str, answer: str) -> str:
    """The safe response in the language of the answer it replaces."""
    if not has_arabic(answer):
        return safe_response
    return _ARABIC_SAFE_RESPONSES.get(safe_response, safe_response)


# Lines the assistant is sanctioned to say verbatim. They appear in the
# system prompt (that's where the model is told to say them), so they
# must be excluded from the exfiltration n-gram set — see
# `_system_prompt_ngrams`.
_SANCTIONED_OUTPUTS = (
    REFUND_SAFE_RESPONSE,
    UNGROUNDED_NUMBER_SAFE_RESPONSE,
    UNGROUNDED_PROPER_NOUN_SAFE_RESPONSE,
    SYSTEM_PROMPT_EXFIL_SAFE_RESPONSE,
)


@dataclass(frozen=True)
class GuardrailResult:
    passed: bool
    safe_answer: str
    violations: list[str] = field(default_factory=list)


# --- Shared text helpers ---------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9']+")


def _words(text: str) -> list[str]:
    """Lowercased word tokens, punctuation stripped, Arabic included.

    Was `[a-z0-9']+`, which produced an empty list for Arabic text. That
    silently disabled two rules at once: the exfiltration detector needs
    8+ tokens before it can fire, and the proper-noun check had no words
    to examine. Single characters are kept here - unlike retrieval, an
    8-word run is being reconstructed and dropping short tokens would
    shift the window.
    """
    return tokenize(text, min_length=1)


# --- Rule 2: system-prompt exfiltration ------------------------------------

_EXFIL_NGRAM_SIZE = 8

# Double-quoted spans in the system prompt are sanctioned *visitor-facing*
# example replies ("I'm just set up to help with Apex Creative's services..."),
# not internal instructions — the model is explicitly told to say them.
# Leaving them in the n-gram set would make the guardrail block the model
# for following the prompt correctly, which is the exact opposite of the
# intent. Stripped rather than special-cased at match time so the
# exclusion is visible in one place.
_QUOTED_SPAN_RE = re.compile(r'"[^"]*"', re.DOTALL)


def _strip_quoted_examples(prompt_text: str) -> str:
    return _QUOTED_SPAN_RE.sub(" ", prompt_text)


def _ngrams(text: str) -> frozenset[str]:
    words = _words(text)
    if len(words) < _EXFIL_NGRAM_SIZE:
        return frozenset()
    return frozenset(
        " ".join(words[i : i + _EXFIL_NGRAM_SIZE])
        for i in range(len(words) - _EXFIL_NGRAM_SIZE + 1)
    )


@lru_cache(maxsize=1)
def _system_prompt_ngrams() -> frozenset[str]:
    """Every 8-word run in the system prompt, normalized, MINUS the runs
    the assistant is explicitly supposed to say out loud. Cached — the
    prompt is fixed content loaded once per process, and rebuilding this
    set per request would put a few thousand string joins on the hot path
    for a value that never changes.

    Subtracting the sanctioned responses is not a nicety. The prompt
    instructs the model to tell visitors that "refund specifics depend on
    the project stage and will be confirmed by the team directly" — which
    is REFUND_SAFE_RESPONSE word for word. Without this subtraction, a
    model that follows that instruction correctly gets its answer
    replaced with the exfiltration refusal, and only sometimes, since it
    depends on the model's exact phrasing that turn. Caught end-to-end
    against the live model, not in unit tests: every canned line the
    prompt dictates is also a verbatim run of the prompt.
    """
    prompt_ngrams = _ngrams(_strip_quoted_examples(load_system_prompt()))
    sanctioned: frozenset[str] = frozenset()
    for response in _SANCTIONED_OUTPUTS:
        sanctioned |= _ngrams(response)
    return prompt_ngrams - sanctioned


def _leaks_system_prompt(answer: str) -> bool:
    words = _words(answer)
    if len(words) < _EXFIL_NGRAM_SIZE:
        return False
    prompt_ngrams = _system_prompt_ngrams()
    return any(
        " ".join(words[i : i + _EXFIL_NGRAM_SIZE]) in prompt_ngrams
        for i in range(len(words) - _EXFIL_NGRAM_SIZE + 1)
    )


# --- Rule 3: number grounding ----------------------------------------------


def _normalize_number(raw: str) -> str:
    """Strips comma thousands-separators, surrounding whitespace, and
    leading zeros so "6,000"/"6000" and "01"/"1" compare as numerically
    identical — see rule 3 in the module docstring. Only formatting is
    normalized; the digits themselves are untouched, so a genuinely
    different number (a hallucinated "60,000" vs the real "6,000") still
    doesn't match."""
    # Arabic-Indic digits fold to ASCII first: a fabricated price written
    # as ٩٩٩٩ must be compared as 9999, not treated as a different token
    # that happens to match nothing in the knowledge base.
    cleaned = normalize_digits(raw).replace(",", "").strip()
    if cleaned and cleaned[0] == "0" and any(ch.isdigit() and ch != "0" for ch in cleaned):
        cleaned = cleaned.lstrip("0")
    return cleaned


# Ordered-list markers: a number opening a line, after any markdown
# decoration (bullets, bold, blockquote, indentation), followed by "." or
# ")". These are formatting, not claims about the world.
_LIST_MARKER_RE = re.compile(r"^[ \t]*[*_>\-#]*[ \t]*\d+[.)](?=\s)", re.MULTILINE)


def _strip_list_markers(text: str) -> str:
    return _LIST_MARKER_RE.sub(" ", text)


def _numbers_in(text: str, strip_list_markers: bool = False) -> set[str]:
    """Numbers appearing in `text`, normalized.

    `strip_list_markers` drops ordered-list numbering first, and is used
    for the ANSWER side of the grounding check. Without it, a model that
    formats "what do you offer?" as a numbered list gets its 1., 2., 3.
    read as four fabricated figures and the whole answer replaced with
    the can't-confirm fallback. Caught live, and intermittent in the
    worst way — it depended purely on whether the model chose a numbered
    list or bullets that turn.

    Not applied to the grounding side: a number that only ever appears in
    the KB as list numbering shouldn't become a licence for the model to
    state it as a fact.
    """
    source = _strip_list_markers(text) if strip_list_markers else text
    # Fold Arabic-Indic digits BEFORE matching - _NUMBER_RE is ASCII-only,
    # so ٩٩٩٩ would otherwise not be found at all.
    source = normalize_digits(source)
    return {_normalize_number(match) for match in _NUMBER_RE.findall(source)}


# --- Rule 4: proper-noun grounding -----------------------------------------

# Two or more consecutive capitalized words — "Global Tech Solutions",
# "Karim Fathy". Ampersands are allowed inside a run so real KB titles
# like "Software & Web Development" match as one phrase instead of
# splitting into two unrelated runs.
#
# The separator is spaces and tabs ONLY, never a line break. With `\s+`
# a run happily jumped a paragraph boundary, so an answer ending a line
# with "...Trend-Based Content" and opening the next with "Are you
# interested..." was read as the invented entity "Trend-Based Content
# Are" and the whole answer got replaced. Intermittent by nature — it
# depended on where the model happened to break lines — and it was
# firing on ordinary answers in live testing.
_PROPER_NOUN_RUN_RE = re.compile(
    r"\b[A-Z][\w'’-]*(?:[ \t]+(?:&[ \t]+)?[A-Z][\w'’-]*)+"
)

# Phrases that are the assistant's own identity, established by the
# system prompt rather than by any KB chunk — flagging these as invented
# entities would block the bot for correctly identifying itself.
_IDENTITY_PHRASES = frozenset({
    "apex creative ai assistant",
    "apex creative ai",
    # The company's own name. Irrelevant to the live rule (a single word
    # never reaches the check there), but the stricter min_words=1 pass
    # used for cache-eligibility does see it — and flagging "Apex Creative" as
    # an invented entity made every answer ineligible for caching, which
    # silently turned the whole QA cache off.
    "apex creative",
})

_SENTENCE_START_RE = re.compile(r"(?:^|[.!?:;\n]\s*)$")


def _normalize_phrase(phrase: str) -> str:
    return " ".join(phrase.lower().replace("’", "'").split())


def _stem(word: str) -> str:
    """Crudest possible singular/plural fold: drop one trailing "s".

    Enough for the failure it exists to stop. The KB says "Web
    Applications"; a model answering "Website & Web Application
    Development" was flagged as inventing an entity purely because
    "application" is not literally "applications". That fired on roughly
    one in four answers to "what services do you offer?" — the site's
    most common question — and only sometimes, which is the worst way
    for a guardrail to be wrong.

    Deliberately not a real stemmer: this only ever decides whether a
    capitalized word is *recognized*, so over-folding costs a little
    strictness while under-folding rejects correct answers, and there is
    no dependency worth adding for the difference.
    """
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


@lru_cache(maxsize=1)
def _sanctioned_vocabulary() -> frozenset[str]:
    """Stemmed words drawn from the system prompt itself.

    The prompt is our own text, not model invention, so a term it
    establishes is as legitimate a source for a proper noun as a
    retrieved KB section is. Without this the rule punishes the model for
    following instructions: the prompt mandates asking for a "WhatsApp
    number", but "WhatsApp" appears only in some KB sections, so whenever
    the retrieved chunks happened not to mention it the answer "could you
    share Your WhatsApp number" was flagged as an invented entity and
    replaced. Measured at 8 out of 8 on a pricing question, which is
    exactly the turn where lead capture is supposed to happen.

    Safe because the prompt contains only vocabulary we wrote. Checked
    against the invented-entity cases this rule exists to catch: none of
    "Global", "Tech", "Solutions", "Karim", "Fathy", "Nile", "Media" or
    "Group" appears anywhere in it.
    """
    return frozenset(_stem(word) for word in _words(load_system_prompt()))


def _ungrounded_proper_nouns(
    answer: str, grounding_text: str, min_words: int = 2
) -> list[str]:
    """Multi-word capitalized phrases in `answer` that appear nowhere in
    `grounding_text` and aren't the bot's own identity.

    A run that begins a sentence has its first word dropped before being
        judged: "Also Apex Creative offers..." would otherwise flag the invented
        entity "Also Apex Creative", and "The Starter Package" would have to match
    the KB including a capital "The" it only has by position. After the
    drop, a run of one remaining word falls below the two-word bar and is
    ignored — which is what keeps sentence starts from generating noise.

    A phrase is grounded two ways, and it needs both to be usable:
    verbatim presence, OR every word in it appearing somewhere in the
    grounding text. The second is what allows a legitimate rewording of
    known terms — the model answering "Social Media & Digital Marketing"
    when the KB section is titled "Social Media Marketing & Digital
    Marketing" is not a fabrication. Requiring a verbatim match rejected
    that, which meant the single most common question on the site ("what
    services do you offer?") got the can't-confirm fallback. An invented
    entity still fails, because it introduces words the KB never
    contained: "Global Tech Solutions", "Karim Fathy", "Nile Media
    Group" each carry a word found nowhere in the retrieved context.

    `min_words=1` makes the check stricter by also judging what's left
    after a sentence-start drop, so "Hi Omar!" is examined as "Omar".
    That is too aggressive for live answers — greeting a visitor by the
    name they just gave is correct behavior, and the name is grounded in
    the conversation. It is exactly right for deciding what may go into
    the SHARED QA cache, where that same name must never be replayed to
    a different visitor. See `_store_if_reusable` in pipeline.py.
    """
    grounding_normalized = _normalize_phrase(grounding_text)
    grounding_stems = {_stem(word) for word in _words(grounding_text)} | _sanctioned_vocabulary()
    ungrounded: list[str] = []

    for match in _PROPER_NOUN_RUN_RE.finditer(answer):
        phrase = match.group(0)
        if _SENTENCE_START_RE.search(answer[: match.start()]):
            parts = phrase.split(None, 1)
            phrase = parts[1] if len(parts) > 1 else ""
        normalized = _normalize_phrase(phrase)
        if len(normalized.split()) < min_words:
            continue
        if normalized in _IDENTITY_PHRASES:
            continue
        if normalized in grounding_normalized:
            continue
        phrase_words = _words(phrase)
        if phrase_words and all(_stem(word) in grounding_stems for word in phrase_words):
            continue
        if normalized not in (_normalize_phrase(p) for p in ungrounded):
            ungrounded.append(phrase.strip())

    return ungrounded


# --- Rule 6 helper: brand-name misspelling ---------------------------------

_BRAND = "Apex Creative"

# Words shaped like the brand. Case-insensitive so a lowercase "apex creative"
# is caught too, and bounded at 8 trailing letters so this can only ever
# look at something brand-shaped rather than sweeping the answer.
_BRAND_CANDIDATE_RE = re.compile(r"\bApex [A-Za-z]{2,8}\b", re.IGNORECASE)

# One or two edits from "apex creative". Two is enough for every misspelling
# observed ("Apex Creativ" is 1, "Apex Creatve" is 2) and still far from any real
# word this regex can reach: "apex" is 0 edits away, "apex creativ" 2,
# "apex cretive" 3. Three would start to be a guess.
_BRAND_MAX_EDITS = 2


def _correct_brand_misspellings(answer: str) -> tuple[str, list[str]]:
    """Rewrites near-misses of the company's own name to `Apex Creative`.

    Found in live testing: the model occasionally writes "Apex Creativ" or
    "Apex Creatve". Nothing caught it. Rule 4 only judges runs of two or more
    capitalized words, so a lone misspelled brand name — which is what
    this always is — never reached it, and the answer went out with the
    client's name wrong.

    Repaired in place rather than blocked, like rule 5 and unlike rules
    1-4. The distinction is what kind of failure it is: an invented
    entity or an ungrounded number means the answer's CONTENT can't be
    trusted, so the whole thing is replaced. A misspelled brand name is
    a typo in an otherwise correct answer, and swapping a good answer
    for "I can't confirm that" would be a strictly worse outcome for the
    visitor than fixing the spelling.

    An EXACT letter match is left completely alone, whatever its case.
    That is deliberate and load-bearing: it is what keeps this away from
    "apexcreative.example" and "...@apexcreative.example", where capitalising the name
    would corrupt a URL or an email address. Only genuinely misspelled
    letters are touched.
    """
    corrected: list[str] = []

    def replace(match: re.Match) -> str:
        word = match.group(0)
        if word.lower() == _BRAND.lower():
            return word  # right letters — casing is not ours to change
        if Levenshtein.distance(word.lower(), _BRAND.lower()) > _BRAND_MAX_EDITS:
            return word  # an ordinary word that merely starts with "apex"
        corrected.append(word)
        return _BRAND

    return _BRAND_CANDIDATE_RE.sub(replace, answer), corrected


# --- Rule 1 helper ---------------------------------------------------------


_CAPITALIZED_WORD_RE = re.compile(r"\b[A-Z][a-z'’]{1,}\b")


def ungrounded_capitalized_words(answer: str, grounding_text: str) -> list[str]:
    """Every capitalized word in `answer` that appears nowhere in the
    grounding text, ignoring sentence-initial position.

    Stricter and simpler than `_ungrounded_proper_nouns`, and used for a
    different purpose: deciding whether an answer may enter the SHARED QA
    cache. The run-based rule needs two adjacent capitalized words, so a
    lone name mid-sentence ("Thanks, Omar — I'll pass this along") slips
    straight past it. For live answers that's the right call, since names
    the visitor just gave are legitimately groundable in the
    conversation; for a cache entry served to somebody else it is not.

    Sentence-initial words are skipped because position, not proper-noun
    status, is why they're capitalized.
    """
    grounding_stems = {_stem(word) for word in _words(grounding_text)}
    ungrounded: list[str] = []
    for match in _CAPITALIZED_WORD_RE.finditer(answer):
        if _SENTENCE_START_RE.search(answer[: match.start()]):
            continue
        word = match.group(0)
        normalized = _normalize_phrase(word)
        if _stem(normalized) in grounding_stems or normalized in _IDENTITY_PHRASES:
            continue
        if normalized in _COMMON_CAPITALIZED_WORDS:
            continue
        if word not in ungrounded:
            ungrounded.append(word)
    return ungrounded


# Capitalized mid-sentence in ordinary English without being an entity,
# plus the assistant's own vocabulary. Without these the check rejects
# almost every answer and the cache goes dark.
_COMMON_CAPITALIZED_WORDS = frozenset({
    "i", "i'm", "i'd", "i'll", "i've", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "january", "february",
    "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "whatsapp", "english", "arabic",
})


def _is_refund_context(retrieved_chunks: list[RetrievedChunk]) -> bool:
    return any("refund" in chunk.section_title.lower() for chunk in retrieved_chunks)


def apply_guardrails(
    answer: str,
    retrieved_chunks: list[RetrievedChunk],
    blocked_emails: list[str] | None = None,
    conversation_context: str = "",
) -> GuardrailResult:
    """Returns a GuardrailResult with a safe-to-send answer.

        `blocked_emails` should include the currently configured internal
        notification recipient once that's set (e.g. from
        INTERNAL_NOTIFICATION_EMAIL) — the legacy hr@apexcreative.example address is
        always blocked regardless.

    `conversation_context` is the prior turns of this conversation (plain
    text, any format) — numbers and names the visitor themselves already
    typed (e.g. a phone number, or their own company name, given two
    turns ago) count as grounded when the assistant reflects them back,
    same as a KB-sourced value does. Optional and defaults to empty, so
    existing single-turn callers are unaffected.
    """
    if not answer or not answer.strip():
        raise ValueError("answer must be non-empty.")

    violations: list[str] = []

    # Rule 6 — brand-name spelling. Applied FIRST, before any rule reads
        # the answer, so every later rule judges canonical text. A corrected
        # name can only make the grounding rules more accurate: "Apex Creativ
        # Agency" is an ungrounded proper-noun run that would have replaced
        # the whole answer, and as "Apex Creative Agency" it is grounded and the
        # answer survives.
    answer, misspellings = _correct_brand_misspellings(answer)
    if misspellings:
        violations.append(f"brand_misspelling:{','.join(sorted(set(misspellings)))}")

    # Rule 1 — refund percentage, permanent policy, checked first because
    # it's stricter than "grounded is enough": a refund % might genuinely
    # be present in the KB and still must never be relayed.
    if _is_refund_context(retrieved_chunks) and _PERCENT_RE.search(answer):
        return GuardrailResult(
            passed=False,
            safe_answer=_localized(REFUND_SAFE_RESPONSE, answer),
            violations=["refund_percentage_stated"],
        )

    # Rule 2 — system-prompt exfiltration. Checked before the grounding
    # rules because a leaked prompt is a security failure, not an
    # accuracy one: it should be reported as exfiltration even in the
    # case where the leaked text also happens to trip rule 3 or 4.
    if _leaks_system_prompt(answer):
        return GuardrailResult(
            passed=False,
            safe_answer=_localized(SYSTEM_PROMPT_EXFIL_SAFE_RESPONSE, answer),
            violations=["system_prompt_exfiltration"],
        )

    # Titles count as grounding, not just bodies — `build_system_prompt`
    # sends the model "[section_title]\ncontent" for every chunk, so a
    # section title is retrieved KB content by any honest definition.
    # Grounding against bodies alone made the model's own service names
    # ("Creative Content Creation", "Film & Documentary Production" —
    # which exist only in titles) read as invented entities.
    context_text = "\n".join(
        f"{chunk.section_title}\n{chunk.content}" for chunk in retrieved_chunks
    )
    grounding_text = context_text
    if conversation_context:
        grounding_text = f"{grounding_text}\n{conversation_context}"

    # Rule 3 — every number in the answer must be numerically grounded,
    # either in the retrieved KB content or in the conversation itself.
    ungrounded = _numbers_in(answer, strip_list_markers=True) - _numbers_in(grounding_text)
    if ungrounded:
        return GuardrailResult(
            passed=False,
            safe_answer=_localized(UNGROUNDED_NUMBER_SAFE_RESPONSE, answer),
            violations=[f"ungrounded_numbers:{','.join(sorted(ungrounded))}"],
        )

    # Rule 4 — same grounding bar for multi-word proper nouns.
    ungrounded_names = _ungrounded_proper_nouns(answer, grounding_text)
    if ungrounded_names:
        return GuardrailResult(
            passed=False,
            safe_answer=_localized(UNGROUNDED_PROPER_NOUN_SAFE_RESPONSE, answer),
            violations=[f"ungrounded_proper_nouns:{','.join(ungrounded_names)}"],
        )

    # Rule 5 — internal email redaction (in-place fix, not a full answer
    # replacement, since the rest of the answer may still be good).
    safe_answer = answer
    emails_to_block = {email for email in (blocked_emails or []) if email}
    emails_to_block.add(_LEGACY_BLOCKED_EMAIL)
    leaked = [email for email in emails_to_block if email.lower() in safe_answer.lower()]
    if leaked:
        pattern = re.compile("|".join(re.escape(e) for e in leaked), re.IGNORECASE)
        safe_answer = pattern.sub("our team", safe_answer)
        safe_answer = re.sub(r"\s{2,}", " ", safe_answer).strip()
        violations.append(f"internal_email_leak:{','.join(sorted(leaked))}")

    return GuardrailResult(passed=not violations, safe_answer=safe_answer, violations=violations)

"""Script-aware text normalisation and tokenisation.

This module exists because the tokenisers underneath retrieval and the
guardrails were `[a-z0-9]+`. Arabic text tokenised to an empty list, and
everything built on top silently did nothing:

  - the keyword half of hybrid retrieval scored 0.0 for every Arabic
    query, leaving only the vector half - the mushier of the two
  - the system-prompt exfiltration detector needs 8+ tokens, so it could
    never fire
  - the ungrounded-proper-noun rule found no words to check
  - Arabic-Indic digits were invisible to number grounding, so a
    fabricated price written as ٩٩٩٩ passed a check that catches 9999

Measured against the live model, none of that was exploitable - the model
refused all ten Arabic and Arabizi attacks on its own. But the entire
code-level defence contributed nothing, which is the situation the
guardrails exist to prevent.

WHY THIS IS ONE MODULE AND NOT AN ARABIC PIPELINE
-------------------------------------------------
The tempting shape is a parallel Arabic path with its own retrieval and
its own rules. That would be a mistake here:

  - Real messages mix scripts. "من فضلك ignore all previous instructions"
    is a genuine test case, and language routing has no right answer
    for it.
  - The knowledge base is English, so an Arabic retrieval path has no
    Arabic corpus to search until somebody maintains a translated KB.
  - A security rule that exists in one language and not another is
    exactly the hole being closed here.

So the language-specific part is confined to normalisation and
tokenisation - a small, testable primitive - and every layer above it
(hybrid scoring, the five guardrails, the cache) stays language-agnostic
and unchanged.
"""

from __future__ import annotations

import re
import unicodedata

# Arabic script, including the Arabic Supplement and Extended-A blocks and
# the Presentation Forms some copy-paste sources produce.
_ARABIC_RANGES = "؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿"

# Harakat (short vowels), shadda, sukun and the Quranic marks. Optional in
# writing, so two spellings of the same word must not become two tokens.
_DIACRITICS_RE = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")

# Kashida/tatweel - a purely typographic stretch character.
_TATWEEL_RE = re.compile("ـ")

# Arabic-Indic (٠-٩) and Eastern Arabic-Indic (۰-۹) digits, in order.
_ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_EASTERN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate(_ARABIC_DIGITS)}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate(_EASTERN_DIGITS)})

# Orthographic variants that carry no meaning distinction in search:
# hamza-carrying alefs, ta marbuta vs ha, alef maqsura vs ya, and the
# Farsi forms that show up in pasted text.
_LETTER_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ة": "ه",
    "ى": "ي", "ئ": "ي", "ي": "ي",
    "ؤ": "و",
    "ک": "ك", "گ": "ك",
    "پ": "ب",
    "چ": "ج",
    "ژ": "ز",
    "ڤ": "ف",
    "ﻻ": "لا",
})

# Typographic apostrophes, folded so "Apex Creative's" tokenises the same way
# whichever source it came from.
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "՚": "'"})


def normalize_digits(text: str) -> str:
    """Arabic-Indic and Eastern Arabic-Indic digits to ASCII.

    Applied before number grounding, so a price written ٩٩٩٩ is compared
    as 9999 rather than slipping past the check entirely.
    """
    return text.translate(_DIGIT_MAP)


def normalize_arabic(text: str) -> str:
    """Folds Arabic orthographic variation that search should ignore.

    Diacritics and tatweel are removed; hamza forms, ta marbuta and alef
    maqsura are folded to a single spelling. This is the Arabic analogue
    of the crude English stemmer - it makes spellings of one word compare
    equal, without pretending to be morphological analysis.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS_RE.sub("", text)
    text = _TATWEEL_RE.sub("", text)
    return text.translate(_LETTER_MAP)


# Arabic LETTERS only - deliberately not the whole Arabic block, which
# also contains the comma, semicolon and question mark (، ؛ ؟). Including
# the block wholesale made "المواقع؟" tokenise with its question mark
# attached, so it would never match the same word written without one.
# Arabic-Indic digits are absent on purpose: normalize_digits() has
# already turned them into ASCII by the time this runs.
_ARABIC_LETTERS = "ء-غف-يٱ-ۓۺ-ۿݐ-ݿࢠ-ࣇ"

# Latin letters, digits, apostrophes, and Arabic letters. Built from an
# explicit range rather than \w so it stays honest about which scripts are
# supported - \w would silently admit Cyrillic, CJK and everything else,
# none of which is tested here.
_TOKEN_RE = re.compile(f"[a-z0-9'{_ARABIC_LETTERS}]+")


def normalize_text(text: str) -> str:
    """Full normalisation applied before tokenising anything."""
    text = text.lower().translate(_APOSTROPHES)
    text = normalize_arabic(text)
    return normalize_digits(text)


def tokenize(text: str, stopwords: frozenset[str] = frozenset(),
             min_length: int = 1) -> list[str]:
    """Normalised word tokens from mixed Arabic/Latin text.

    `min_length` drops single characters, which are noise in Latin and
    almost always noise in Arabic too (single letters are particles).
    Callers pass their own stopword set because retrieval and the
    guardrails are asking different questions of the same text.
    """
    return [
        token
        for token in _TOKEN_RE.findall(normalize_text(text))
        if token not in stopwords and len(token) >= min_length
    ]


def has_arabic(text: str) -> bool:
    """True when the text contains any Arabic-script character.

    Used to decide whether Arabic-specific screening applies, NOT to
    route to a different pipeline - mixed-script messages are common and
    must go down the same path as everything else.
    """
    return bool(re.search(f"[{_ARABIC_RANGES}]", text))


# Arabic function words with no distinguishing power for "which section is
# this question about" - the direct analogue of the English stopword list
# in app/kb/retrieval.py. Written in NORMALISED form, since that is what
# tokenize() produces (e.g. "التي" normalises before comparison).
ARABIC_STOPWORDS = frozenset({
    "في", "من", "الى", "على", "عن", "مع", "هذا", "هذه", "ذلك", "التي",
    "الذي", "ما", "ماذا", "هل", "كيف", "متى", "اين", "لماذا", "كم",
    "هو", "هي", "هم", "نحن", "انا", "انت", "انتم", "كان", "يكون",
    "ان", "او", "لا", "نعم", "قد", "كل", "بعض", "عند", "بين", "لكن",
    "ايضا", "ثم", "حتى", "اذا", "لو", "بعد", "قبل", "الان", "جدا",
    "يمكن", "ممكن", "عايز", "عاوز", "عندكم", "عندك", "بتاع", "بتاعت",
    "ايه", "فين", "امتى", "ازاي", "ليه", "مين", "دي", "ده", "دا",
})

#!/usr/bin/env python3
"""
letter_header_utils.py -- strip epistolary front-matter (letter numbers, date/place
lines, salutations) from the START of a correspondence paragraph.

WHY (Trip, 2026-09-20): extracted passages were opening with material that is structural
metadata rather than speech --
    "Saturday evening, January 19.—I am now in lodgings..."
    "My dear Girls,—It is rather late..."
    "LETTER III. SAINT THOMAS AND TORTOLA. Flushing, L. 1. Sixth-month (June) 2d, 1840. My dear Friend, Having..."
Two separate problems with leaving these in:
  1. A salutation naming a real addressee ("My dear Wallace,—") makes the reply read as
     a letter to a third party, not speech to the person chatting -- the same
     roleplay-framing defect already flagged on the "My dear boy / My dear Papa" pair.
  2. Date/place/letter-number headers are bibliographic apparatus; they are not something
     a speaker would ever say aloud.
Measured on the gold-standard file (multiturn_sft_epistolary_Aug28.jsonl, 1,306
pg_static_map rows): 1.8% open on a date header and 2.6% on a salutation, so this was a
real if minor pre-existing defect there too -- stripping puts the new material ahead of
the old on this axis rather than merely matching it.

Deliberately conservative: only strips a recognized header PREFIX, never touches the
interior, and returns the original text unchanged if what would remain is too short to
be a usable passage (so a paragraph that is *entirely* header is rejected downstream by
the normal word-count filter rather than being silently mangled into a fragment).
"""
from __future__ import annotations
import re

# "LETTER III." / "LETTER 42." optionally followed by an ALL-CAPS descriptive title
# ("LETTER III. SAINT THOMAS AND TORTOLA.").
_LETTER_NO = r"LETTER\s+[IVXLCDM\d]+\.?(?:\s+[A-Z][A-Z\s,'\-\.]{3,60}\.)?"

# A date / place line: "Saturday evening, January 19.", "Flushing, L. I. Sixth-month
# (June) 2d, 1840.", "Down, Monday [Feb 10 1845].", "London, May 21, 1559."
_WEEKDAY = r"(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day"
_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
          r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|"
          r"(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|"
          r"Eleventh|Twelfth)-month)")
# A bracketed span only counts as date-ish if it actually contains a date -- an
# unrestricted \[...\] also matches footnote markers like "[142]", which caused a real
# over-strip ("Dr. Arthur Lankester [142] had written..." -> "Arthur Lankester ...").
_BRACKET_DATE = rf"\[(?=[^\]]*(?:18\d\d|{_MONTH}))[^\]]{{1,40}}\]"
_DATEISH = (
    rf"(?:{_WEEKDAY}(?:\s+(?:morning|evening|night|afternoon))?|{_MONTH}|"
    rf"\d{{1,2}}(?:st|nd|rd|th|d)?|18\d\d|{_BRACKET_DATE}|\([^)]{{1,20}}\))"
)
# Token class must also admit parenthesised glosses ("(June)") and bare initials ("L.")
# -- a real header reads "Flushing, L. I. Sixth-month (June) 2d, 1840." and stalling on
# any one of those pieces leaves the rest of the header in the passage. Honorifics are
# excluded so a name like "Dr. Arthur Lankester" is never read as a place run.
_HONORIFIC = r"(?:Dr|Mr|Mrs|Miss|Messrs|Rev|Prof|Sir|Lady|Lord|Capt|Col|Gen|Hon|St)\.?"
_PLACE_TOK = rf"(?!{_HONORIFIC}(?:\s|$))[A-Z][a-zA-Z\.\-']{{0,14}}"
# REQUIRE at least one genuinely date-ish token: a run of bare capitalised words is a
# name or a speaker label ("Charles.", "Dr."), not a date line, and stripping those
# silently removed real content.
_DATE_LINE = (
    rf"(?:{_PLACE_TOK}[,\s\.]+){{0,4}}{_DATEISH}"
    rf"(?:[,\s\.]+(?:{_DATEISH}|{_PLACE_TOK})){{0,9}}[\.,]"
)

# Salutations. Tolerates OCR ("Mt dear Friend"), redactions ("My dear ————,"),
# and titles ("My dear Sir Charles,"). Must end in the comma/colon that closes an
# address, optionally followed by the PG em-dash convention.
# Keywords are case-insensitive via scoped (?i:...) groups -- Victorian volumes
# routinely set salutations in full caps ("MY DEAR WARE,—", "DEAR SIR,—"), which a
# case-sensitive pattern misses entirely; that was the whole residual leak class.
# The NAME tokens stay capital-anchored ([A-Z]...) on purpose: making the whole pattern
# re.I would let _PLACE_TOK match ordinary lowercase words and reintroduce over-stripping.
_SALUTATION = (
    r"(?:(?i:m[ty]|my))\s+(?i:dear(?:est)?|own|darling)(?:\s+[A-Z][\w'\.\-]*){0,4}\s*[,:]"
    r"|(?i:dear(?:est)?)(?:\s+(?i:sir|madam|madame|friend|fellow|father|mother|girls|boys|children))*"
    r"(?:\s+[A-Z][\w'\.\-]*){0,4}\s*[,:]"
    r"|(?i:sir|madam|gentlemen|beloved)[,:]"
    r"|(?:(?i:m[ty]|my))\s+(?i:dear(?:est)?)\s*[—\-–]{2,}[,:]?"
)

_HEADER_RE = re.compile(
    rf"^\s*(?:{_LETTER_NO}\s*)?(?:{_DATE_LINE}\s*)*(?P<sal>(?:{_SALUTATION})\s*[—\-–]*\s*)?",
)

MIN_REMAINDER_WORDS = 40


def strip_letter_header(text: str, min_remainder_words: int = MIN_REMAINDER_WORDS) -> str:
    """Remove letter-number / date-line / salutation front-matter from the start of a
    correspondence paragraph. Returns text unchanged if nothing matched or if stripping
    would leave too little behind to be a usable passage."""
    if not text:
        return text
    m = _HEADER_RE.match(text)
    if not m or m.end() == 0:
        return text
    # Editors routinely print an elision right after the salutation ("My dear Fanny,—
    # ... I am now obliged to keep fowls"), so strip a leading ellipsis too -- otherwise
    # the remainder starts with "." , fails the begins-like-prose guard below, and the
    # whole salutation is left in place. This was the single largest residual leak class.
    remainder = text[m.end():].lstrip(" —-–,:;")
    # "…" / "..." / ". . ." (spaced) all appear as editorial elisions in these volumes.
    remainder = re.sub(r"^(?:(?:\.\s*){2,}|…)\s*", "", remainder).lstrip(" —-–,:;")
    if len(remainder.split()) < min_remainder_words:
        return text
    # Only accept the strip if the remainder still begins like real prose. EXCEPTION:
    # after a salutation, a lower-case continuation is the norm, not a red flag --
    # "My dear Liszt, from what I have told you..." is one sentence that simply starts
    # after the address. Requiring upper-case there left ~100 salutations unstripped.
    # The date/letter-number branches keep the strict check, since those sit before a
    # genuinely new sentence and a lower-case remainder there means a bad match.
    if not m.group("sal"):
        if not remainder[:1].isupper() and remainder[:1] not in "\"'“‘":
            return text
    return remainder


if __name__ == "__main__":
    CASES = [
        "My dear Wallace,—You cannot well imagine how much I have been pleased by what you say "
        "about Pangenesis. None of my friends will speak out and tell me plainly what they think, "
        "which is a great misfortune to any man who wishes to arrive at the truth of matters.",
        "Saturday evening, January 19.—I am now in lodgings in a quiet street, and have begun to "
        "work again in earnest, though the light is poor and the days are short and the cold is "
        "more trying than I had looked for when I came up to town at the turn of the year.",
        "My dear Girls,—It is rather late to be writing, but I could not let the day close without "
        "telling you what we have seen, for the whole town turned out and the procession was longer "
        "than any of us had expected and the bells went on ringing well past the dinner hour.",
        "LETTER III. SAINT THOMAS AND TORTOLA. Flushing, L. I. Sixth-month (June) 2d, 1840. My dear "
        "Friend, Having performed the religious duties which devolved upon us in Santa Cruz, we took "
        "a hearty leave of our numerous friends, and sailed for St. Thomas, which port we reached "
        "on the morning of the twenty-sixth day of the month.",
        "It is a very fine thing to be able to say that one has never in all his life been so "
        "thoroughly interested in any inquiry as in this one, and I mean to pursue it to the end "
        "however long that may take me and whatever the cost in labour and in patience may prove.",
    ]
    for c in CASES:
        out = strip_letter_header(c)
        print("BEFORE:", c[:92])
        print("AFTER :", out[:92])
        print("changed:", out != c, "\n")

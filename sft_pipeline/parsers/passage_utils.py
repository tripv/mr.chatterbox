"""
passage_utils.py — Victorian corpus passage extraction utilities.

Provides clean, sentence-boundary-aligned passage extraction for the
teacher model pipeline and future SFT data generation.

Problem addressed:
  Raw Victorian books (especially BL OCR) have:
    - Page-number + running-header artifacts at paragraph starts
    - Paragraphs that range from 1 word to 2 000+ words
    - Passages that begin or end mid-sentence due to page turns

Strategy:
  1. Split on paragraph boundaries (\\n\\n)
  2. Strip running headers from paragraph starts
  3. For paragraphs within the target word range: use directly after coherence checks
  4. For long paragraphs: split at sentence boundaries into sub-passages
  5. Discard passages that start/end mid-sentence, or that cross-reference
     other parts of the text ("as stated above", "see Chapter 3", etc.)

Usage:
    from passage_utils import extract_passages_from_text

    passages = extract_passages_from_text(book_text, min_words=80, max_words=350)
    # Each returned string starts and ends at a sentence boundary.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_WORDS = 80
MAX_WORDS = 200

# Common Victorian abbreviations that end with a period — prevent false sentence splits.
_ABBREVS: frozenset[str] = frozenset({
    "mr", "mrs", "miss", "dr", "prof", "rev", "st", "vol", "vols",
    "pp", "p", "viz", "esq", "lieut", "lt", "capt", "col", "gen",
    "gov", "jr", "sr", "no", "fig", "dept", "ed", "eds",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "vs", "etc", "cf", "ibid", "op", "loc", "e.g", "i.e",
    "ib", "art", "sec", "chap", "ch",
})

# Sentence boundary: period/!/? followed by whitespace + capital letter.
# We post-filter to exclude abbreviation endings.
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

# Running-header artifact at paragraph start:
#   "18 THE EMPIRE OF RUSSIA. sion of..."      →  strip "18 THE EMPIRE OF RUSSIA. "
#   "PARENTAGE AND BIRTH. 31 Landing..."        →  strip "PARENTAGE AND BIRTH. 31 "
#   "■28 THE EMPIRE OF RUSSIA away..."          →  strip "■28 THE EMPIRE OF RUSSIA "
#   "NATURAL PHENOMENA. 5 with the merry..."    →  strip "NATURAL PHENOMENA. 5 "
#   "simon's toavn. 33 brown sandy bays..."     →  strip up to "brown"
_HEADER_RE = re.compile(
    r"""^
    [\s■▪•·\-–—]*           # optional leading junk
    (?:
        \d+\s+              # leading page number
        [A-Z][A-Z ,.'–\-]+  # all-caps title
        (?:[.]\s*\d*\s*)?   # optional period + optional trailing page number
      |
        [A-Z][A-Z ,.'–\-]+  # all-caps title first
        (?:\s+\d+|\.\s*\d+) # then page number (space-separated or after period)
        \s*
    )
    """,
    re.VERBOSE,
)

# Second-pass: catches OCR-mangled mixed-case headers followed by page number + lowercase.
# Pattern: any text (possibly mixed case) ending with a page number, then a lowercase word.
# E.g.: "REIsiN OF VLADEMER. 55 hand"  or  "simon's toavn. 33 brown"
_TITLE_PAGE_RE = re.compile(
    r"""^
    .{1,80}?            # short title text (non-greedy)
    [.]\s*\d+\s+        # period + page number
    (?=[a-z])           # followed by lowercase (mid-sentence continuation)
    """,
    re.VERBOSE,
)

# Passages that cross-reference other parts of the text — not self-contained.
_CROSS_REF_RE = re.compile(
    r"""\b(
        as\s+(?:stated|mentioned|noted|shown|described|explained|observed|
                 remarked|seen|pointed\s+out)\s+
           (?:above|before|previously|in\s+the\s+preceding|already)
      | (?:see|vide|cf\.?)\s+(?:above|below|chapter|chap\.|p\.|pp\.|fig\.)
      | in\s+the\s+(?:preceding|following|next|last|previous)\s+
            (?:chapter|section|paragraph|page|volume)
      | (?:chapter|chap\.|vol\.)\s+[IVXLCDM\d]+
      | (?:to\s+be\s+)?continued
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

# Passages that obviously start mid-sentence or with structural artefacts.
_BAD_START_RE = re.compile(
    r"""^
    (?:
        [a-z]               # lowercase start → mid-sentence continuation
      | [-—…]               # dash or ellipsis continuation
      | \(                  # open parenthesis
      | \d+[\s.]            # bare page / section number
      | [ivxlcdmIVXLCDM]+[.)\s]   # Roman numeral section marker
      | Chap(?:ter)?\b      # chapter heading
      | CHAP(?:TER)?\b      # chapter heading (all-caps OCR)
      | Sec(?:tion)?\.\s    # section marker
    )
    """,
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences, respecting Victorian abbreviations.

    Uses a simple regex (period/!/? + space + capital) and then merges back
    any split that occurred after a known abbreviation.
    """
    raw = _SENT_SPLIT_RE.split(text.strip())
    if len(raw) <= 1:
        return raw

    merged: list[str] = []
    for sent in raw:
        if merged:
            prev = merged[-1]
            # Last word of previous sentence (strip trailing punctuation)
            words = prev.rstrip(" ").rstrip(".!?,;:").split()
            last_word = words[-1].lower().rstrip(".") if words else ""
            if last_word in _ABBREVS:
                merged[-1] = prev + " " + sent
                continue
        merged.append(sent)

    return merged


# ---------------------------------------------------------------------------
# Passage-level coherence checks
# ---------------------------------------------------------------------------

def starts_cleanly(text: str) -> bool:
    """
    True if the passage begins at a genuine sentence start.

    Requires the first character to be an uppercase letter, then additionally
    rejects patterns that indicate mid-sentence continuation, structural artefacts,
    or page-number/Roman-numeral section markers.
    """
    t = text.lstrip()
    if not t:
        return False
    # First character must be an uppercase letter — rules out ], ), numbers, lowercase, etc.
    if not t[0].isupper():
        return False
    return not bool(_BAD_START_RE.match(t))


def ends_cleanly(text: str) -> bool:
    """
    True if the passage ends at a sentence boundary.

    Accepts endings in: .  !  ?  ."  .'  !)  ?'  etc.
    Rejects: trailing comma, semicolon, hyphen, or bare word.
    """
    t = text.rstrip()
    if not t:
        return False
    last = t[-1]
    if last in '.!?':
        return True
    # Closing quote/paren immediately after terminal punctuation
    if last in '"\'""''—)' and len(t) >= 2 and t[-2] in '.!?':
        return True
    return False


def is_self_contained(text: str) -> bool:
    """
    True if the passage does not reference material outside itself.

    Rejects passages containing phrases like "as stated above", "see Chapter 3",
    "to be continued", etc.
    """
    return not bool(_CROSS_REF_RE.search(text))


# ---------------------------------------------------------------------------
# Header stripping (BL-specific)
# ---------------------------------------------------------------------------

def strip_running_header(text: str) -> str:
    """
    Remove OCR page-number / running-header artifacts from paragraph starts.

    Handles two patterns:
      Pass 1 — clean all-caps headers:
        "18 THE EMPIRE OF RUSSIA. Oleg directed..."  → "Oleg directed..."
        "PARENTAGE AND BIRTH. 31 Landing..."         → "Landing..."

      Pass 2 — OCR-mangled mixed-case headers + page number + lowercase:
        "REIsiN OF VLADEMER. 55 hand of the princess..."  → "hand of the princess..."
        "NATURAL PHENOMENA. 5 with the merry gibe..."     → "with the merry gibe..."
    """
    # Pass 1: clean header regex
    m = _HEADER_RE.match(text)
    if m:
        return text[m.end():].lstrip()

    # Pass 2: any short text + period + page number + lowercase continuation
    m2 = _TITLE_PAGE_RE.match(text)
    if m2:
        return text[m2.end():].lstrip()

    return text


# ---------------------------------------------------------------------------
# Quality gate (word count + alpha ratio + no list blocks)
# ---------------------------------------------------------------------------

def is_good_passage(text: str, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> bool:
    """
    Quality filter: word count, alphabetic ratio, no list/ToC/heading blocks.
    Does NOT check sentence boundaries — call starts_cleanly / ends_cleanly separately.
    """
    words = text.split()
    n_words = len(words)
    if not (min_words <= n_words <= max_words):
        return False

    # Must be mostly alphabetic — filters tables, indexes, footnote clusters
    alpha = sum(c.isalpha() for c in text)
    if alpha / len(text) < 0.70:
        return False

    # Skip passages that are clearly lists (>4 lines starting with numbers/bullets)
    lines = text.splitlines()
    list_lines = sum(1 for ln in lines if re.match(r"^\s*[\d\-\*•]", ln))
    if list_lines > 4:
        return False

    # Table-of-contents / chapter-summary detection:
    # High em-dash density signals a list of topics, not prose.
    em_dashes = text.count("—") + text.count("--")
    if em_dashes / n_words > 0.08:   # more than 8 em-dashes per 100 words
        return False

    # Very short average sentence length signals headings/bullet-lists, not prose.
    sentence_frags = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if sentence_frags:
        avg_sent_len = sum(len(s.split()) for s in sentence_frags) / len(sentence_frags)
        if avg_sent_len < 8:
            return False

    # Reject OCR-artifact passages where the same sentence repeats (>= 3 times).
    if sentence_frags and len(sentence_frags) >= 3:
        normalized = [re.sub(r"\s+", " ", s.lower()) for s in sentence_frags]
        if max(normalized.count(s) for s in normalized) >= 3:
            return False

    return True


# ---------------------------------------------------------------------------
# Sub-passage extraction from a single paragraph
# ---------------------------------------------------------------------------

def _passages_from_sentences(
    sentences: list[str],
    min_words: int,
    max_words: int,
) -> list[str]:
    """
    Greedily pack consecutive sentences into passages that fit [min_words, max_words].
    Returns a list of candidate passages (not yet quality-gated).
    """
    passages: list[str] = []
    i = 0
    while i < len(sentences):
        accum: list[str] = []
        wc = 0
        j = i
        while j < len(sentences):
            sw = len(sentences[j].split())
            if wc + sw > max_words:
                break
            accum.append(sentences[j])
            wc += sw
            j += 1
        if wc >= min_words and accum:
            passages.append(" ".join(accum))
        if j == i:
            # Single sentence exceeds max — skip it
            i += 1
        else:
            i = j
    return passages


def extract_from_paragraph(
    para: str,
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
) -> list[str]:
    """
    Extract clean passages from a single paragraph.

    Strategy:
      1. Strip BL running-header artifacts.
      2. If the paragraph is within [min_words, max_words] AND passes all
         coherence checks, return it as a single passage.
      3. Otherwise — either too long, doesn't start cleanly, or doesn't end
         cleanly — sentence-split the paragraph and pack consecutive sentences
         into passages, skipping any initial mid-sentence fragment.
         This recovers usable content from page-turn continuations.
      4. Each returned passage is guaranteed to:
         - start at a sentence boundary (starts_cleanly)
         - end at a sentence boundary (ends_cleanly)
         - pass the quality gate (is_good_passage)
         - not contain cross-references (is_self_contained)
    """
    text = strip_running_header(para.strip())
    if not text:
        return []

    wc = len(text.split())
    if wc < min_words:
        return []

    if min_words <= wc <= max_words:
        # Try to use the paragraph as-is
        if (starts_cleanly(text) and ends_cleanly(text)
                and is_self_contained(text)
                and is_good_passage(text, min_words, max_words)):
            return [text]
        # Doesn't pass as-is — fall through to sentence-level extraction
        # (handles: starts mid-sentence after a header, or ends mid-sentence)

    # Sentence-split and pack — handles both oversized paragraphs and
    # paragraphs that start mid-sentence (page-turn continuations).
    sentences = split_sentences(text)
    if not sentences:
        return []

    # Skip an initial mid-sentence fragment: the first "sentence" in a
    # page-turn paragraph is the tail end of the previous sentence.
    start_idx = 0
    if not starts_cleanly(sentences[0]):
        start_idx = 1
    if start_idx >= len(sentences):
        return []

    candidates = _passages_from_sentences(sentences[start_idx:], min_words, max_words)
    results: list[str] = []
    for cand in candidates:
        if (starts_cleanly(cand) and ends_cleanly(cand)
                and is_self_contained(cand)
                and is_good_passage(cand, min_words, max_words)):
            results.append(cand)
    return results


# ---------------------------------------------------------------------------
# OCR normalization — runs before paragraph splitting
# ---------------------------------------------------------------------------

# Hyphenated line-break: "ener-\ngetic" → "energetic"
# Uses [ \t]* (not \s*) so we only collapse single \n breaks, never \n\n
# paragraph boundaries.
_HYPHEN_LB_RE = re.compile(r'(\w)-[ \t]*\n[ \t]*(\w)')

# Two or more spaces within a line → single space (OCR column artefacts)
_MULTI_SPACE_RE = re.compile(r'[ \t]{2,}')


def normalize_ocr_text(text: str) -> str:
    """
    Light OCR cleanup applied before paragraph splitting.

    Fixes:
      - Hyphenated line breaks: "ener-\\ngetic" → "energetic"
      - Multiple inline spaces: "the  quick" → "the quick"

    Does NOT attempt to fix non-hyphenated mid-word splits ("Pro fessor")
    because those require a word list and produce too many false positives
    on legitimate collocations.
    """
    text = _HYPHEN_LB_RE.sub(r'\1\2', text)
    text = _MULTI_SPACE_RE.sub(' ', text)
    return text


# ---------------------------------------------------------------------------
# Main entry point: full text → passages
# ---------------------------------------------------------------------------

def extract_passages_from_text(
    text: str,
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
) -> list[str]:
    """
    Extract all clean, sentence-boundary-aligned passages from a text block.

    Applies normalize_ocr_text() first (dehyphenation), then splits the text
    into paragraphs on double-newline and applies extract_from_paragraph()
    to each.

    Returns a (possibly empty) list of passage strings.
    """
    text = normalize_ocr_text(text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    results: list[str] = []
    for para in paragraphs:
        results.extend(extract_from_paragraph(para, min_words, max_words))
    return results

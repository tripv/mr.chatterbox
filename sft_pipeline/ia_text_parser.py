"""
ia_text_parser.py — Clean and extract prose paragraphs from Internet Archive DjVu OCR text.

Three source families with distinct artifact profiles:

  sim_* / per_*  Periodical issues — two-column layout exploded into short lines,
                 masthead repeats, TOC dot-leaders, page-number orphans.

  *goog*         Google-digitised books — standard Google disclaimer header,
                 inline "Digitized by Google" watermarks.

  other          Miscellaneous books — similar column noise, page number orphans,
                 footnote citation runs, occasionally unusable scans (ledgers, etc.)

Pipeline per file:
  1. Strip Google boilerplate (header block + inline watermarks)
  2. Split into raw blocks on 2+ blank lines
  3. Per block: strip page-number orphans and dot-leader lines
  4. Detect whether the block is column-rendered (high short-line fraction)
  5. If column-rendered: rejoin into prose sentences
  6. Score each resulting paragraph; discard below threshold
  7. Yield clean (text, source_type) pairs

Run:
    python ia_text_parser.py --preview                  # 20 sampled items
    python ia_text_parser.py --preview --type sim       # periodicals only
    python ia_text_parser.py --preview --type goog      # Google books only
    python ia_text_parser.py --stats                    # aggregate quality stats
    python ia_text_parser.py --file path/to/file.txt    # single file
"""

import argparse
import random
import re
import sys
from pathlib import Path

IA_RAW_DIR = Path("raw_book_data/ia_raw")

# ── Boilerplate patterns ────────────────────────────────────────────────────

# Google Books disclaimer: always starts near the top with "Google" on its own
# line, ends before the actual book text begins.  We strip everything up to and
# including the last line of the disclaimer block.
_GOOGLE_HEADER_RE = re.compile(
    r"Google\s.{0,2000}?at\s+\|?http[s]?://books\.googl[^\n]*",
    re.DOTALL,
)

# Inline Google watermark injected between pages
_GOOGLE_WATERMARK_RE = re.compile(
    r"\n[ \t]*Digitized by Goog[^\n]*\n",
    re.IGNORECASE,
)

# ── Line-level noise patterns ───────────────────────────────────────────────

# Page number orphans: lines that are just digits (possibly with a trailing
# letter from OCR confusion — "la", "ts", "H" etc.)
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*[A-Za-z]?\s*$")

# Dot-leader / c-leader lines from tables of contents and price tables.
# Triggered when a line contains 5+ consecutive dots, c's, or dashes used as
# fill characters.
_DOT_LEADER_RE = re.compile(r"[.]{5,}|c{5,}|[-]{5,}|[_]{5,}")

# Lines that are a bare single character or two (OCR detritus)
_SINGLE_CHAR_RE = re.compile(r"^\s*\S{1,2}\s*$")

# Masthead / header lines: short ALL-CAPS lines (magazine titles, volume info)
# Allow up to 60 chars; require at least 3 uppercase letters.
_CAPS_HEADER_RE = re.compile(r"^[^a-z]{0,60}$")

# Pipe-separated two-column index lines ("Article .... 64 | Other .... 68")
_PIPE_COL_RE = re.compile(r"\|")


# ── Boilerplate stripping ───────────────────────────────────────────────────

def strip_google_boilerplate(text: str) -> str:
    """Remove Google Books disclaimer header and inline watermarks."""
    m = _GOOGLE_HEADER_RE.search(text[:5000])
    if m:
        text = text[m.end():]
    text = _GOOGLE_WATERMARK_RE.sub("\n", text)
    return text


# ── Block-level cleaning ────────────────────────────────────────────────────

def _is_noise_line(line: str) -> bool:
    """True if the line should be dropped regardless of context."""
    stripped = line.strip()
    if not stripped:
        return False   # blank lines handled separately
    if _PAGE_NUM_RE.match(stripped):
        return True
    if _DOT_LEADER_RE.search(stripped):
        return True
    if _SINGLE_CHAR_RE.match(stripped):
        return True
    if _PIPE_COL_RE.search(stripped):
        return True
    return False


def _clean_lines(lines: list[str]) -> list[str]:
    """Drop noise lines; return the survivors."""
    return [l for l in lines if not _is_noise_line(l)]


# ── Column-render detection and rejoining ──────────────────────────────────

def _is_column_rendered(lines: list[str]) -> bool:
    """
    Return True when a block has been flattened from a two-column layout.
    Signal: more than 40% of non-empty lines are shorter than 45 characters.
    """
    non_empty = [l for l in lines if l.strip()]
    if len(non_empty) < 4:
        return False
    short = sum(1 for l in non_empty if len(l.rstrip()) < 45)
    return short / len(non_empty) > 0.40


# End-of-sentence: period/!/? followed by optional quote/paren, then space
# or end of string.  Used to split rejoined column lines into sentences.
_SENT_END_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _rejoin_column_block(lines: list[str]) -> str:
    """
    Merge short column lines back into flowing prose.
    Strategy: join all non-empty lines with a space, then split on sentence
    boundaries to form paragraph-sized chunks joined with a single space.
    """
    joined = " ".join(l.strip() for l in lines if l.strip())
    # Collapse multiple spaces introduced by joining
    joined = re.sub(r"  +", " ", joined)
    return joined


# ── Paragraph quality scoring ───────────────────────────────────────────────

MIN_WORDS    = 30
MAX_WORDS    = 600
MIN_ALPHA    = 0.62   # minimum fraction of alphabetic characters
MAX_DIGIT    = 0.12   # maximum fraction of digit characters (catches tables)
MIN_AVG_WORD = 3.0    # average word length floor
MAX_AVG_WORD = 10.0   # average word length ceiling (catches OCR runs)


def score_paragraph(text: str) -> float:
    """
    Return a quality score 0.0–1.0. Values < 0.0 indicate definite rejection.
    Scores >= MIN_SCORE (see extract_paragraphs) are kept.
    """
    words = text.split()
    n_words = len(words)
    if n_words < MIN_WORDS or n_words > MAX_WORDS:
        return -1.0

    chars = len(text)
    if chars == 0:
        return -1.0

    alpha = sum(c.isalpha() for c in text) / chars
    digits = sum(c.isdigit() for c in text) / chars

    if alpha < MIN_ALPHA:
        return -1.0
    if digits > MAX_DIGIT:
        return -1.0

    word_lens = [len(w) for w in words]
    avg_wl = sum(word_lens) / len(word_lens)
    if avg_wl < MIN_AVG_WORD or avg_wl > MAX_AVG_WORD:
        return -1.0

    # Positive score: weight alpha ratio and length sweetspot
    length_score = 1.0 - abs(n_words - 100) / 500.0
    length_score = max(0.0, min(1.0, length_score))

    return alpha * 0.6 + length_score * 0.4


MIN_SCORE = 0.45


# ── Main extraction pipeline ────────────────────────────────────────────────

def extract_paragraphs(text: str, source_type: str = "other") -> list[str]:
    """
    Full cleaning pipeline. Returns a list of clean prose paragraphs.
    source_type: "goog", "sim", "per", or "other".
    """
    if source_type == "goog":
        text = strip_google_boilerplate(text)

    # Split on 2+ consecutive blank lines
    raw_blocks = re.split(r"\n{2,}", text)

    paragraphs: list[str] = []

    for block in raw_blocks:
        lines = block.splitlines()
        lines = _clean_lines(lines)
        non_empty = [l for l in lines if l.strip()]

        if len(non_empty) < 2:
            # Single surviving line — only keep if it's a meaty sentence
            if non_empty:
                s = non_empty[0].strip()
                if len(s.split()) >= MIN_WORDS and score_paragraph(s) >= MIN_SCORE:
                    paragraphs.append(s)
            continue

        if _is_column_rendered(lines):
            # Rejoin column lines into a single prose block, then re-split on
            # sentence boundaries to form natural paragraph chunks.
            rejoined = _rejoin_column_block(lines)
            # Split into sentence-boundary chunks of reasonable size
            chunks = _SENT_END_RE.split(rejoined)
            # Accumulate chunks into paragraphs of ~50-200 words
            buf: list[str] = []
            buf_words = 0
            for chunk in chunks:
                cw = len(chunk.split())
                if buf_words + cw > 220 and buf:
                    candidate = " ".join(buf)
                    if score_paragraph(candidate) >= MIN_SCORE:
                        paragraphs.append(candidate)
                    buf = [chunk]
                    buf_words = cw
                else:
                    buf.append(chunk)
                    buf_words += cw
            if buf:
                candidate = " ".join(buf)
                if score_paragraph(candidate) >= MIN_SCORE:
                    paragraphs.append(candidate)
        else:
            # Normal block: join lines with a space (handles soft line wraps)
            candidate = " ".join(l.strip() for l in lines if l.strip())
            candidate = re.sub(r"  +", " ", candidate).strip()
            # Rejoin hyphenated line-breaks: "moun-\ntain" → "mountain"
            candidate = re.sub(r"-\s+", "", candidate)
            if score_paragraph(candidate) >= MIN_SCORE:
                paragraphs.append(candidate)

    return paragraphs


# ── Source type detection ───────────────────────────────────────────────────

# Directory-name suffixes that identify bibliographic index / TOC volumes.
# These contain article titles and page-number references, not prose.
_SKIP_SUFFIXES = ("_index", "_contents")


def is_index_item(path: Path) -> bool:
    """True when the item is a TOC or bibliographic index volume, not prose."""
    return path.parent.name.endswith(_SKIP_SUFFIXES)


def detect_source_type(path: Path) -> str:
    name = path.parent.name
    if name.startswith("sim_"):
        return "sim"
    if name.startswith("per_"):
        return "per"
    if "goog" in name:
        return "goog"
    return "other"


# ── File processing ─────────────────────────────────────────────────────────

def process_file(path: Path) -> tuple[list[str], str]:
    """Read and clean one IA text file. Returns (paragraphs, source_type)."""
    if is_index_item(path):
        return [], "skip"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return [], "error"
    source_type = detect_source_type(path)
    paragraphs = extract_paragraphs(text, source_type)
    return paragraphs, source_type


# ── CLI ─────────────────────────────────────────────────────────────────────

def _iter_txt_files(filter_type: str | None = None) -> list[Path]:
    files = list(IA_RAW_DIR.rglob("*_djvu.txt"))
    if not files:
        files = list(IA_RAW_DIR.rglob("*.txt"))
    # Always exclude index/contents volumes
    files = [f for f in files if not is_index_item(f)]
    if filter_type:
        if filter_type == "sim":
            files = [f for f in files if f.parent.name.startswith("sim_")]
        elif filter_type == "per":
            files = [f for f in files if f.parent.name.startswith("per_")]
        elif filter_type == "goog":
            files = [f for f in files if "goog" in f.parent.name]
        elif filter_type == "other":
            files = [f for f in files
                     if not f.parent.name.startswith("sim_")
                     and not f.parent.name.startswith("per_")
                     and "goog" not in f.parent.name]
    return files


def cmd_preview(args):
    rng = random.Random(args.seed)
    files = _iter_txt_files(args.type)
    if not files:
        print(f"No files found (type={args.type})")
        return

    sample = rng.sample(files, min(args.count, len(files)))
    shown = 0

    for path in sample:
        paragraphs, stype = process_file(path)
        if not paragraphs:
            continue
        print(f"{'─'*72}")
        print(f"[{stype}]  {path.parent.name}")
        print(f"  {len(paragraphs)} paragraphs extracted")
        print()
        for p in paragraphs[:3]:
            words = p.split()
            print(f"  ({len(words)}w)  {p[:300]}{'...' if len(p) > 300 else ''}")
            print()
        shown += 1
        if shown >= args.count:
            break

    print(f"{'─'*72}")
    print(f"Previewed {shown} items.")


def cmd_stats(args):
    from collections import Counter
    files = _iter_txt_files(args.type)
    rng = random.Random(args.seed)
    if args.sample:
        files = rng.sample(files, min(args.sample, len(files)))

    print(f"Analysing {len(files):,} files…")

    total_files = 0
    empty_files = 0
    total_paras = 0
    total_words = 0
    per_type: Counter = Counter()
    para_words: list[int] = []

    for path in files:
        paragraphs, stype = process_file(path)
        total_files += 1
        if not paragraphs:
            empty_files += 1
        else:
            total_paras += len(paragraphs)
            per_type[stype] += len(paragraphs)
            for p in paragraphs:
                w = len(p.split())
                total_words += w
                para_words.append(w)

    para_words.sort()
    print(f"\nFiles processed:      {total_files:,}")
    print(f"Empty (no paragraphs): {empty_files:,}  ({empty_files/max(total_files,1)*100:.0f}%)")
    print(f"Total paragraphs:     {total_paras:,}")
    print(f"Total words:          {total_words:,}")
    if para_words:
        print(f"Paragraph word count: min={para_words[0]}  "
              f"median={para_words[len(para_words)//2]}  "
              f"max={para_words[-1]}")
    print("\nParagraphs by source type:")
    for stype, n in per_type.most_common():
        print(f"  {stype:<8} {n:>8,}")


def cmd_file(args):
    path = Path(args.file)
    paragraphs, stype = process_file(path)
    print(f"Source type: {stype}")
    print(f"Paragraphs:  {len(paragraphs)}")
    print()
    for i, p in enumerate(paragraphs, 1):
        print(f"[{i}] ({len(p.split())}w)")
        print(f"  {p[:400]}")
        print()


def main():
    ap = argparse.ArgumentParser(description="Clean IA DjVu OCR text files")
    sub = ap.add_subparsers(dest="cmd")

    p_prev = sub.add_parser("preview", help="Sample items and display cleaned paragraphs")
    p_prev.add_argument("--count",  type=int, default=20)
    p_prev.add_argument("--type",   choices=["sim", "per", "goog", "other"], default=None)
    p_prev.add_argument("--seed",   type=int, default=42)

    p_stat = sub.add_parser("stats", help="Aggregate quality stats")
    p_stat.add_argument("--type",   choices=["sim", "per", "goog", "other"], default=None)
    p_stat.add_argument("--sample", type=int, default=0,
                        help="Limit to N random files (0 = all)")
    p_stat.add_argument("--seed",   type=int, default=42)

    p_file = sub.add_parser("file", help="Process a single file")
    p_file.add_argument("file")

    # Flat --preview / --stats / --file for convenience
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--stats",   action="store_true")
    ap.add_argument("--file",    type=str, default=None)
    ap.add_argument("--count",   type=int, default=20)
    ap.add_argument("--type",    choices=["sim", "per", "goog", "other"], default=None)
    ap.add_argument("--sample",  type=int, default=0)
    ap.add_argument("--seed",    type=int, default=42)

    args = ap.parse_args()

    if args.cmd == "preview" or (not args.cmd and args.preview):
        cmd_preview(args)
    elif args.cmd == "stats" or (not args.cmd and args.stats):
        cmd_stats(args)
    elif args.cmd == "file" or (not args.cmd and args.file):
        cmd_file(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

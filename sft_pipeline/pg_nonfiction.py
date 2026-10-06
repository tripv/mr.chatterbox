"""
pg_nonfiction.py — Build a filtered list of nonfiction Project Gutenberg epubs
using pg_catalog.csv as a classification guide, cross-referenced against the
curated Victorian British book list (victorian_epub_final.txt).

The Victorian list (victorian_epub_final.txt) was produced by a separate
filtering pass that kept only British-authored books published 1837-1899.
This module intersects that list with the nonfiction classification from the
catalog, so callers get Victorian British nonfiction only — not all PG nonfiction
and not all Victorian books.

Fiction is identified by:
  - LoCC class in {PS, PR, PZ, PQ, PT, PG}  (literature / juvenile fiction)
  - Subject or Bookshelves containing: fiction, novel, poetry, poems, drama,
    play, short stories

Everything else with an LoCC classification is treated as nonfiction.

Exported:
    build_nonfiction_epubs(catalog_path, victorian_list_path, epub_dir)
        → list of (epub_path, pg_id, title) sorted by pg_id

    extract_epub_text(epub_path) → str
    clean_epub_text(text) → str
"""

import csv
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path

# ── LoCC / subject fiction signals ─────────────────────────────────────────

_FICTION_LOCC = frozenset({"PS", "PR", "PZ", "PQ", "PT", "PG"})

_FICTION_SUBJECT_RE = re.compile(
    r"\bfiction\b|\bnovel\b|\bpoetry\b|\bpoems\b"
    r"|\bdrama\b|\bplay\b|\bshort stories\b",
    re.IGNORECASE,
)

# ── American-institutional filter ───────────────────────────────────────────
#
# These LoCC classes are primarily US political/legal institutions and don't
# reflect Victorian British thinking even when the author is British.
#   JK = Constitutional and administrative law of the United States
#   KF = Law of the United States
#
# Some US-political content slips through under the broader `J` class
# (e.g. "State of the Union Addresses"), caught instead by subject pattern.

_AMERICAN_LOCC = frozenset({"JK", "KF"})

_US_INSTITUTIONAL_SUBJECT_RE = re.compile(
    r"Presidents\s*--\s*United States"
    r"|United States\s*--\s*Politics and government",
    re.IGNORECASE,
)

# PG boilerplate strip patterns (same as build_gutenberg_shards.py)
_PG_START_RE = re.compile(
    r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.+?\*\*\*",
    re.IGNORECASE,
)
_PG_END_RE = re.compile(
    r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK.+?\*\*\*",
    re.IGNORECASE,
)

# Epub filename pattern: "Some Title.12345.epub"
_EPUB_ID_RE = re.compile(r"\.(\d+)\.epub$")


# ── Classification ──────────────────────────────────────────────────────────

def _is_fiction(row: dict) -> bool:
    for segment in row["LoCC"].split(";"):
        segment = segment.strip()
        m = re.match(r"^([A-Z]+)", segment)
        if m and m.group(1) in _FICTION_LOCC:
            return True
    combined = row.get("Subjects", "") + " " + row.get("Bookshelves", "")
    return bool(_FICTION_SUBJECT_RE.search(combined))


def _is_british_enough(row: dict) -> bool:
    """
    Return False for books that are clearly American-institutional in nature
    and therefore unlikely to reflect Victorian British intellectual culture.
    Checks LoCC class (JK, KF) and subject patterns (US presidential addresses
    and federal political texts that slip through under broader `J`).
    """
    for segment in row["LoCC"].split(";"):
        segment = segment.strip()
        m = re.match(r"^([A-Z]+)", segment)
        if m and m.group(1) in _AMERICAN_LOCC:
            return False
    subjects = row.get("Subjects", "")
    if _US_INSTITUTIONAL_SUBJECT_RE.search(subjects):
        return False
    return True


def _load_catalog(catalog_path: Path) -> dict[str, dict]:
    """Return {pg_id: row_dict} for all English Text rows with LoCC."""
    catalog: dict[str, dict] = {}
    with catalog_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row["Language"].strip() == "en"
                and row["Type"].strip() == "Text"
                and row["LoCC"].strip()
            ):
                catalog[row["Text#"].strip()] = row
    return catalog


# ── Public API ──────────────────────────────────────────────────────────────

def _load_victorian_list(list_path: Path) -> dict[str, str]:
    """
    Parse victorian_epub_final.txt → {pg_id: filename}.
    Format: one entry per line, tab-separated: pg_id<TAB>filename.epub
    """
    mapping: dict[str, str] = {}
    with list_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                pg_id, fname = parts
                mapping[pg_id.strip()] = fname.strip()
    return mapping


def build_nonfiction_epubs(
    catalog_path: Path,
    victorian_list_path: Path,
    epub_dir: Path,
) -> list[tuple[Path, str, str]]:
    """
    Return a list of (epub_path, pg_id, title) for books that are BOTH:
      - In the curated Victorian British list (victorian_epub_final.txt)
      - Classified as nonfiction by pg_catalog.csv

    epub_dir is the directory containing the .epub files.
    """
    catalog = _load_catalog(catalog_path)
    nonfic_ids = {
        pg_id for pg_id, row in catalog.items()
        if not _is_fiction(row) and _is_british_enough(row)
    }

    victorian = _load_victorian_list(victorian_list_path)

    results: list[tuple[Path, str, str]] = []
    for pg_id, fname in victorian.items():
        if pg_id not in nonfic_ids:
            continue
        epub_path = epub_dir / fname
        if not epub_path.exists():
            continue
        title = catalog[pg_id]["Title"].replace("\n", " ").strip() if pg_id in catalog else fname
        results.append((epub_path, pg_id, title))

    results.sort(key=lambda t: int(t[1]))
    return results


# ── Epub text extraction ────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    _BLOCK = {"p","div","br","h1","h2","h3","h4","h5","h6",
               "li","tr","td","th","blockquote","pre"}

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def extract_epub_text(epub_path: Path) -> str:
    """Extract body text from an epub, stripping PG boilerplate."""
    with zipfile.ZipFile(epub_path) as z:
        html_files = sorted(
            n for n in z.namelist()
            if n.endswith((".html", ".htm", ".xhtml"))
        )
        parts = []
        for hf in html_files:
            parser = _TextExtractor()
            parser.feed(z.read(hf).decode("utf-8", errors="replace"))
            parts.append(parser.get_text())
    full = "\n".join(parts)

    start = _PG_START_RE.search(full)
    if start:
        full = full[start.end():]
    end = _PG_END_RE.search(full)
    if end:
        full = full[:end.start()]

    return full


def clean_epub_text(text: str) -> str:
    """Normalise whitespace; rejoin hyphenated line-breaks."""
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

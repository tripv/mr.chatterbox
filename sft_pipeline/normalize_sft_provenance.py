"""
normalize_sft_provenance.py — Populate book_title / book_author / book_year
across all sources in the source4 SFT files and multiturn_passages.jsonl.

IB records are already patched by backtrace_ib_provenance.py; this script
handles BL, PG_Nonfiction, NCSE, IA, and the passages corpus.

Sources and strategy:
  PG_Nonfiction  — fuzzy lookup against pg_catalog.csv (Title, Authors, Issued)
  BL             — normalize existing book/author fields to canonical schema
  NCSE           — static map of 5 known periodical identifiers to full names
  IA             — parse sim_<pub>_<date>_... slugs; fall back to cleaned ID
  passages       — static map keyed on "Author — Work" source label (PG texts)

Run:
    python normalize_sft_provenance.py
    python normalize_sft_provenance.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

SFT_FILES = [
    Path("training_data/fact_opinion_sft/source4_pairs_May 27.jsonl"),
    Path("training_data/multi_turn_sft/multiturn_source4.jsonl"),
    Path("training_data/multi_turn_sft/multiturn_passages.jsonl"),
]

PG_CATALOG = Path("raw_book_data/pg_catalog.csv")

# ---------------------------------------------------------------------------
# NCSE — only 5 publications; map to proper display names + date ranges
# ---------------------------------------------------------------------------

NCSE_MAP = {
    "english_womans_journal": ("The English Woman's Journal", "1858–1864"),
    "leader":                 ("The Leader", "1850–1860"),
    "monthly_repository":     ("The Monthly Repository", "1806–1838"),
    "northern_star":          ("The Northern Star", "1837–1852"),
    "tomahawk":               ("The Tomahawk", "1867–1870"),
}

# ---------------------------------------------------------------------------
# multiturn_passages — static map from "Author — Work" label to metadata.
# Publication years are original first-edition dates, not PG upload dates.
# Each entry: (book_title, book_author, book_year, pg_id)
# ---------------------------------------------------------------------------

PASSAGES_MAP: dict[str, tuple[str, str, str, str]] = {
    "Stevenson — Virginibus Puerisque": (
        "Virginibus Puerisque, and Other Papers",
        "Stevenson, Robert Louis",
        "1881", "386"),
    "Stevenson — Essays of RLS": (
        "Essays of Robert Louis Stevenson",
        "Stevenson, Robert Louis",
        "1906", "10761"),
    "Stevenson — Swanston V1": (
        "The Works of Robert Louis Stevenson — Swanston Edition, Vol. 1",
        "Stevenson, Robert Louis",
        "1911", "21686"),
    "Stevenson — Swanston V2": (
        "The Works of Robert Louis Stevenson — Swanston Edition, Vol. 2",
        "Stevenson, Robert Louis",
        "1911", "30527"),
    "Stevenson — Swanston V4": (
        "The Works of Robert Louis Stevenson — Swanston Edition, Vol. 4",
        "Stevenson, Robert Louis",
        "1912", "30700"),
    "Stevenson — Swanston V5": (
        "The Works of Robert Louis Stevenson — Swanston Edition, Vol. 5",
        "Stevenson, Robert Louis",
        "1912", "30744"),
    "Stevenson — Letters V1": (
        "The Letters of Robert Louis Stevenson, Vol. 1",
        "Stevenson, Robert Louis",
        "1895", "622"),
    "Stevenson — Letters V2": (
        "The Letters of Robert Louis Stevenson, Vol. 2",
        "Stevenson, Robert Louis",
        "1895", "637"),
    "Newman — Apologia": (
        "Apologia pro vita sua",
        "Newman, John Henry",
        "1864", "19690"),
    "Newman — Apologia (alt)": (
        "Apologia pro vita sua",
        "Newman, John Henry",
        "1864", "22088"),
    "Ruskin — Selections": (
        "Selections From the Works of John Ruskin",
        "Ruskin, John",
        "", "15200"),
    "Ruskin — Harvest of Ruskin": (
        "The Harvest of Ruskin",
        "Graham, John W.",
        "1920", "51808"),
    "Ruskin — Ruskin Relics": (
        "Ruskin Relics",
        "Collingwood, W. G.; Ruskin, John",
        "1903", "46426"),
    "Dickens — Letters Vol.1": (
        "The Letters of Charles Dickens, Vol. 1, 1833–1856",
        "Dickens, Charles",
        "1880", "25852"),
    "Dickens — Letters Vol.2": (
        "The Letters of Charles Dickens, Vol. 2, 1857–1870",
        "Dickens, Charles",
        "1880", "25853"),
    "Dickens — Letters Vol.3": (
        "The Letters of Charles Dickens, Vol. 3, 1836–1870",
        "Dickens, Charles",
        "1882", "25854"),
    "EBB — Letters Vol.1": (
        "The Letters of Elizabeth Barrett Browning, Vol. 1",
        "Browning, Elizabeth Barrett",
        "1897", "13018"),
    "EBB — Letters Vol.2": (
        "The Letters of Elizabeth Barrett Browning, Vol. 2",
        "Browning, Elizabeth Barrett",
        "1897", "16646"),
    "Browning — Courtship Letters V1": (
        "The Letters of Robert Browning and Elizabeth Barrett Barrett, Vol. 1, 1845–1846",
        "Browning, Robert; Browning, Elizabeth Barrett",
        "1899", "16182"),
    "FitzGerald — Letters Vol.1": (
        "Letters of Edward FitzGerald, Vol. 1",
        "FitzGerald, Edward",
        "1894", "20452"),
    "FitzGerald — Letters Vol.2": (
        "Letters of Edward FitzGerald, Vol. 2",
        "FitzGerald, Edward",
        "1894", "20539"),
    "Acton — Letters to Mary Gladstone": (
        "Letters of Lord Acton to Mary Gladstone",
        "Acton, John Emerich Edward Dalberg Acton, Baron",
        "1904", "38214"),
}


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).strip().lower()


# ---------------------------------------------------------------------------
# PG catalog lookup
# ---------------------------------------------------------------------------

def load_pg_catalog(path: Path) -> list[dict]:
    catalog = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        catalog.append({
            "pg_id":    row.get("Text#", "").strip(),
            "title":    row.get("Title", "").strip(),
            "authors":  row.get("Authors", "").strip(),
            "issued":   row.get("Issued", "").strip(),
            "_norm":    _norm(row.get("Title", "")),
        })
    return catalog


def _pg_year(issued: str) -> str:
    """Extract 4-digit year from PG Issued field (e.g. '1892-05-01' → '1892')."""
    m = re.search(r"\b(1[7-9]\d\d)\b", issued)
    return m.group(1) if m else ""


def pg_lookup(title_raw: str, catalog: list[dict]) -> dict | None:
    """Fuzzy-match a title against the PG catalog. Returns best hit or None."""
    query = _norm(title_raw)
    if not query:
        return None

    best, best_score = None, 0.0
    for row in catalog:
        if not row["_norm"]:
            continue
        # Substring shortcut — cheap first pass
        if query[:40] in row["_norm"] or row["_norm"][:40] in query:
            score = SequenceMatcher(None, query, row["_norm"]).ratio()
        else:
            # Word overlap only for the rest
            qa = set(query.split())
            qb = set(row["_norm"].split())
            inter = qa & qb
            if not inter:
                continue
            score = len(inter) / max(len(qa), len(qb))

        if score > best_score:
            best_score = score
            best = row

    return best if best_score >= 0.55 else None


# ---------------------------------------------------------------------------
# IA identifier parsing
# ---------------------------------------------------------------------------

# sim_ format: sim_<pub-slug>_<YYYY[-MM[-DD]]>_...
_SIM_RE = re.compile(
    r"^sim_(?P<pub>[a-z0-9\-]+?)_(?P<year>\d{4})(?:-\d\d(?:-\d\d)?)?(?:_.+)?$"
)


def parse_ia_id(identifier: str) -> tuple[str, str]:
    """Return (publication_title, year) from an IA identifier, or ('', '')."""
    identifier = identifier.strip()
    m = _SIM_RE.match(identifier)
    if m:
        pub_slug = m.group("pub")
        year     = m.group("year")
        # Convert slug to title: hyphens → spaces, title-case
        pub_title = re.sub(r"-", " ", pub_slug).title()
        return pub_title, year

    # Non-sim identifiers (e.g. Google Books 'ahistorywhigpar01ormsgoog')
    # Return a cleaned version of the ID as a last-resort title
    cleaned = re.sub(r"[0-9]+$", "", identifier)   # strip trailing digits
    cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)  # camel split
    return cleaned.strip() or identifier, ""


# ---------------------------------------------------------------------------
# Field extraction helpers (handle both schemas)
# ---------------------------------------------------------------------------

def get_existing_title(rec: dict) -> str:
    return (rec.get("title") or rec.get("book") or "").strip()


def get_existing_author(rec: dict) -> str:
    return (rec.get("author") or "").strip()


# ---------------------------------------------------------------------------
# Per-source provenance filler
# ---------------------------------------------------------------------------

def fill_provenance(rec: dict, pg_catalog: list[dict]) -> dict:
    source = rec.get("source", "")

    # IB — already done; preserve existing fields
    if source == "IB":
        return rec

    # multiturn_passages — source label is "Author — Work"; use static map
    if source in PASSAGES_MAP:
        book_title, book_author, book_year, pg_id = PASSAGES_MAP[source]
        updates: dict = {}
        if book_title  and not rec.get("book_title"):  updates["book_title"]  = book_title
        if book_author and not rec.get("book_author"):  updates["book_author"] = book_author
        if book_year   and not rec.get("book_year"):    updates["book_year"]   = book_year
        if pg_id       and not rec.get("pg_id"):        updates["pg_id"]       = pg_id
        if not rec.get("prov_method"):                  updates["prov_method"] = "pg_static_map"
        return {**rec, **updates} if updates else rec

    title_raw  = get_existing_title(rec)
    author_raw = get_existing_author(rec)

    book_title  = ""
    book_author = ""
    book_year   = ""
    book_pub    = ""   # for periodicals: publication name
    prov_method = ""

    if source == "PG_Nonfiction":
        hit = pg_lookup(title_raw, pg_catalog)
        if hit:
            book_title  = hit["title"]
            book_author = hit["authors"]
            book_year   = _pg_year(hit["issued"])
            prov_method = "pg_catalog"
        else:
            # Couldn't match — use existing fields as-is
            book_title  = title_raw
            book_author = author_raw
            prov_method = "raw"

    elif source == "BL":
        # BL records already carry title and (for source4_pairs) author
        book_title  = title_raw
        # "unknown" is a sentinel in the BL source data, not a real author
        book_author = author_raw if author_raw.lower() not in ("unknown", "") else ""
        # Year not available without BL catalog re-download
        prov_method = "raw"

    elif source == "NCSE":
        key = title_raw.lower().strip()
        if key in NCSE_MAP:
            pub_name, date_range = NCSE_MAP[key]
            book_title = pub_name
            book_pub   = pub_name
            book_year  = date_range
            prov_method = "ncse_map"
        else:
            book_title  = title_raw
            prov_method = "raw"

    elif source == "IA":
        pub_title, year = parse_ia_id(title_raw)
        book_title  = pub_title
        book_year   = year
        prov_method = "ia_parse"

    # Write back — only fields not already populated
    updates: dict = {}
    if book_title  and not rec.get("book_title"):  updates["book_title"]  = book_title
    if book_author and not rec.get("book_author"):  updates["book_author"] = book_author
    if book_year   and not rec.get("book_year"):    updates["book_year"]   = book_year
    if prov_method and not rec.get("prov_method"):  updates["prov_method"] = prov_method

    if updates:
        return {**rec, **updates}
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Report counts only — do not modify files")
    args = ap.parse_args()

    print("Loading PG catalog...")
    pg_cat = load_pg_catalog(PG_CATALOG)
    print(f"  {len(pg_cat):,} entries.\n")

    total_updated = 0

    for fpath in SFT_FILES:
        if not fpath.exists():
            print(f"SKIP (not found): {fpath}")
            continue

        lines = fpath.read_text(encoding="utf-8").splitlines()
        updated_lines = []
        n_changed = 0

        by_source: dict[str, dict[str, int]] = {}

        for line in lines:
            line = line.strip()
            if not line:
                updated_lines.append("")
                continue
            try:
                rec = json.loads(line)
            except Exception:
                updated_lines.append(line)
                continue

            if not isinstance(rec, dict):
                updated_lines.append(line)
                continue

            src = rec.get("source", "OTHER")
            if src not in by_source:
                by_source[src] = {"in": 0, "changed": 0}

            by_source[src]["in"] += 1
            new_rec = fill_provenance(rec, pg_cat)

            if new_rec is not rec:
                n_changed += 1
                by_source[src]["changed"] += 1
                updated_lines.append(json.dumps(new_rec, ensure_ascii=False))
            else:
                updated_lines.append(line)

        total_updated += n_changed
        print(f"{fpath.name}:")
        for src, counts in sorted(by_source.items()):
            print(f"  {src:<20} {counts['in']:>4} records  →  {counts['changed']:>4} updated")

        if not args.dry_run and n_changed > 0:
            fpath.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
            print(f"  Wrote {n_changed} changes.\n")
        else:
            print(f"  [dry-run] {n_changed} would be updated.\n")

    print(f"Total records updated: {total_updated}")


if __name__ == "__main__":
    main()

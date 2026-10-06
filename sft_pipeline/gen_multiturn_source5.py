"""
gen_multiturn_source5.py -- Blind-sequential rebuild of gen_multiturn_source4.py's multi-turn
generation, fixing FORWARD-LEAKAGE in turn 2+ authorship.

THE BUG THIS FIXES: gen_multiturn_source4.py's backend.generate() is a single LLM call that hands
over ALL passage segments up front and asks the model to write ALL human turns at once
(_build_user() interleaves "Turn i" with "Passage segment i" for every i in one prompt). So turn
N's question is authored WITH KNOWLEDGE of segment N (the answer that hasn't "happened" yet in the
conversation) -- it can and does preview facts/phrases that only appear in the upcoming answer.
Confirmed in shipped data, e.g. multiturn_sft_fact_op_corpus_pulls.jsonl idx 98: turn 2 asks "What
do you mean by 'plain rules'...?" but "plain rules, evident to the well-intentioned mind" is the
literal text of segment 2 (the NEXT answer), not anything segment 1 said. Real usage-log follow-ups
(sampled from usage_data_sessions.jsonl) are short and strictly backward-referencing (questions
of the "why would you say that?" / "in what way?" kind) -- they react to what was just said,
never anticipate.

THE FIX: turn 1 (cold open) is reused unchanged from source4 -- it already correctly avoids
leakage, since it's deliberately written from an abstracted "subject" only, never segment text.
Turns 2..N are instead generated ONE AT A TIME, each call seeing only the transcript so far (never
the upcoming segment). The next passage segment is revealed only AFTER the follow-up is written,
then checked for plausible fit; a poor fit truncates the conversation there rather than forcing a
non-sequitur. A rewritten whole-conversation judge (CONV_JUDGE_SYSTEM_PROMPT_V2) adds an explicit
forward-leakage check on top of source4's existing criteria.

Source pool: source4's raw-passage loaders (NCSE/BL/PG/IA/IB) PLUS a new HathiTrust loader
(_load_hathi_raw, ported from gen_teacher_source4.load_hathi_broad_passages -- HathiTrust has no
loader in source4 today, and it's the largest single pretraining source by volume).

Output: training_data/dialogue/multiturn_source5.jsonl (append mode, separate from source4's
output -- nothing here has been merged into training_data/multi_turn_sft/ yet).

Run:
    python gen_multiturn_source5.py --count 15 --preview
    python gen_multiturn_source5.py --count 15 --backend mlx --model mlx-community/Qwen2.5-32B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "parsers"))

from gen_multiturn_source4 import (
    MIN_WORDS, MAX_WORDS,
    COLD_OPEN_MODES, ALL_MODES, DIALOGUE_MODES,
    SYSTEM_PROMPT, _FEW_SHOTS, validate_result,
    CONV_JUDGE_SYSTEM_PROMPT, _CONV_JUDGE_FEW_SHOTS, parse_conv_judgment,
    split_into_segments, _is_usable, wc, _parse_json,
    _load_ncse_raw, _load_ia_raw,
    starts_cleanly, ends_cleanly,
    MLXBackend, AnthropicBackend,
    DEFAULT_ANTHROPIC_MODEL,
)
from gen_teacher_source4 import (
    route_passage_to_subcorpus, SOURCE_SUBCORPORA, HATHI_SHARDS_DIR, IB_SHARDS_DIR,
    BL_DATA_DIR, PG_DATA_DIR, PG_CATALOG, PG_VICTORIAN_LIST,
    _author_from_title,
)
from pg_nonfiction import build_nonfiction_epubs, extract_epub_text, clean_epub_text
from passage_utils import extract_passages_from_text

# Full HathiTrust bib catalog (hathifile format, tab-delimited, 26 columns: htid, access,
# rights, ht_bib_key, description, source, source_bib_num, oclc_num, isbn, issn, lccn,
# title, imprint, ..., author) -- covers ANY htid, unlike the on-theme workset CSV used
# elsewhere for thinking_machines routing. Used to backfill book_title/book_author (and a
# book_year fallback) for IB and HATHI passages, neither of which carry real book metadata
# in their parquet shards -- only htid.
_HATHI_BIB_PATH = Path("raw_book_data/hathi_trust/hathi_full_20260601.txt")
_YEAR_RE = re.compile(r"(1[5-9]\d{2}|20[0-2]\d)")


def _safe_year(yr) -> str:
    """int(yr) on a pandas 'year' column, guarding NaN -- float('nan') is truthy in Python,
    so a plain `if yr else ""` lets int(nan) through and raises ValueError."""
    try:
        if yr is None or yr != yr:  # NaN != NaN is the cheapest NaN check, no pandas import needed
            return ""
        return str(int(yr))
    except (TypeError, ValueError):
        return ""

OUTPUT = Path("training_data/dialogue/multiturn_source5.jsonl")

# 32B reads real usage-log register (short, terse follow-ups) better than the 14B default
# source4 uses -- this is the model already validated locally for the dialog_retrieval judge.
DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-32B-Instruct-4bit"


# ---------------------------------------------------------------------------
# HathiTrust loader -- ported from gen_teacher_source4.load_hathi_broad_passages,
# adapted to source4/5's candidate-loader return schema and MIN_WORDS/MAX_WORDS window.
# ---------------------------------------------------------------------------

def _load_hathi_raw(
    rng: random.Random,
    min_w: int = MIN_WORDS,
    max_w: int = MAX_WORDS,
    max_shards: int = 150,
    max_passages: int = 40_000,
    min_ocr: float = 0.85,
    per_book: int = 4,
) -> list[dict]:
    if not HATHI_SHARDS_DIR.exists():
        print(f"  WARNING: {HATHI_SHARDS_DIR} not found -- skipping HATHI")
        return []
    try:
        import pandas as pd
    except ImportError:
        print("  WARNING: pandas not installed -- skipping HATHI")
        return []

    shard_files = sorted(HATHI_SHARDS_DIR.glob("*.parquet"))
    rng.shuffle(shard_files)
    shard_files = shard_files[:max_shards]

    all_passages: list[dict] = []
    books_seen = 0
    book_count: dict[str, int] = {}

    for shard in shard_files:
        try:
            df = pd.read_parquet(shard, columns=["text", "htid", "year", "lang", "ocr_score"])
        except Exception as e:
            print(f"  WARNING: error reading {shard.name}: {e}")
            continue
        for row_idx, row in enumerate(df.itertuples(index=False)):
            lang = getattr(row, "lang", None)
            if lang is not None and str(lang).lower() not in ("eng", "en", ""):
                continue
            score = getattr(row, "ocr_score", None)
            if score is not None and score < min_ocr:
                continue
            text = row.text if isinstance(row.text, str) else ""
            if not text or len(text.split()) < min_w:
                continue
            hid = getattr(row, "htid", "") or f"{shard.stem}:{row_idx}"
            remaining = per_book - book_count.get(hid, 0)
            if remaining <= 0:
                continue
            books_seen += 1
            passages = extract_passages_from_text(text, min_words=min_w, max_words=max_w)
            if len(passages) > remaining:
                passages = rng.sample(passages, remaining)
            book_count[hid] = book_count.get(hid, 0) + len(passages)
            yr = getattr(row, "year", None)
            for p in passages:
                all_passages.append({
                    "text":        p,
                    "source":      "HATHI",
                    "title":       hid,
                    "author":      "unknown",
                    "book_title":  "",
                    "book_author": "",
                    "book_year":   _safe_year(yr),
                    "barcode":     hid,
                    "htid":        hid,
                    "shard_row":   row_idx,
                })

    print(f"  {'HATHI sampled books':<40} {len(all_passages):>7,} candidates "
          f"({books_seen} books in {len(shard_files)} shards)")
    if len(all_passages) > max_passages:
        all_passages = rng.sample(all_passages, max_passages)
    return all_passages


def _load_hathi_bib(htids: set[str]) -> dict[str, dict]:
    """One streaming pass over the ~19.5M-row HathiTrust bib catalog to resolve real
    title/author/year for a needed set of htids. Stops early once every requested htid
    has been found (typical need here is a few thousand distinct htids, not all 19.5M)."""
    out: dict[str, dict] = {}
    if not htids or not _HATHI_BIB_PATH.exists():
        return out
    remaining = set(htids)
    with _HATHI_BIB_PATH.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not remaining:
                break
            tab = line.find("\t")
            if tab < 0:
                continue
            hid = line[:tab]
            if hid not in remaining:
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 26:
                continue
            title   = fields[11].strip()
            imprint = fields[12].strip()
            author  = fields[25].strip()
            ym = _YEAR_RE.search(imprint)
            out[hid] = {
                "title":  title,
                "author": author,
                "year":   ym.group(0) if ym else "",
            }
            remaining.discard(hid)
    return out


def _backfill_hathi_metadata(passages: list[dict]) -> None:
    """In-place: for any passage carrying an htid but missing book_title/book_author (IB
    and HATHI both have this gap -- their parquet shards carry htid but not title/author),
    resolve it via one shared _load_hathi_bib() pass."""
    needed = {p["htid"] for p in passages if p.get("htid") and not p.get("book_title")}
    if not needed:
        return
    print(f"  Resolving {len(needed):,} HathiTrust htids against the bib catalog...")
    bib = _load_hathi_bib(needed)
    print(f"  Resolved {len(bib):,} / {len(needed):,}")
    for p in passages:
        hid = p.get("htid")
        if not hid or hid not in bib:
            continue
        meta = bib[hid]
        if not p.get("book_title") and meta["title"]:
            p["book_title"] = meta["title"]
        if not p.get("book_author") and meta["author"]:
            p["book_author"] = meta["author"]
        if not p.get("book_year") and meta["year"]:
            p["book_year"] = meta["year"]


# ---------------------------------------------------------------------------
# BL / PG / IB loaders -- local copies of source4's, extended to carry provenance
# (book_title/book_author/book_year) that source4's versions drop on the floor even
# though the source data has it: BL's decade is in the shard filename, PG's original
# publication year is in wikidata_pg_dates.csv, and IB's htid (present in the parquet
# but never read) resolves to real title/author via the HathiTrust bib catalog above.
# ---------------------------------------------------------------------------

def _load_bl_raw(
    rng: random.Random,
    min_w: int = MIN_WORDS,
    max_w: int = MAX_WORDS,
    max_books: int = 2000,
    max_passages: int = 60_000,
) -> list[dict]:
    if not BL_DATA_DIR.exists():
        print(f"  WARNING: {BL_DATA_DIR} not found -- skipping BL")
        return []
    decade_files = sorted(BL_DATA_DIR.glob("books_*.jsonl.gz"))
    if not decade_files:
        print(f"  WARNING: no books_*.jsonl.gz in {BL_DATA_DIR} -- skipping BL")
        return []

    all_passages: list[dict] = []
    books_per_decade = max(1, max_books // len(decade_files))

    for dfile in decade_files:
        decade = dfile.name.split(".")[0].replace("books_", "")  # "books_1830s.jsonl.gz" -> "1830s"
        decade_passages: list[dict] = []
        try:
            with gzip.open(str(dfile), "rt", encoding="utf-8", errors="replace") as f:
                books_seen = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = rec.get("text", "")
                    if not text or len(text.split()) < min_w:
                        continue
                    title  = rec.get("title", "Unknown")
                    author = rec.get("author") or rec.get("creator") or _author_from_title(title)
                    author_str = str(author).strip() if author else "unknown"
                    bl_author = author_str if author_str.lower() not in ("unknown", "") else ""
                    for p in extract_passages_from_text(text, min_words=min_w, max_words=max_w):
                        decade_passages.append({
                            "text":        p,
                            "source":      "BL",
                            "title":       title,
                            "author":      author_str,
                            "book_title":  title,
                            "book_author": bl_author,
                            "book_year":   decade,
                            "prov_method": "raw",
                        })
                    books_seen += 1
                    if books_seen >= books_per_decade:
                        break
        except Exception as e:
            print(f"  WARNING: error reading {dfile.name}: {e}")
            continue
        print(f"  {'BL/' + dfile.name:<40} {len(decade_passages):>7,} candidates")
        all_passages.extend(decade_passages)

    if len(all_passages) > max_passages:
        all_passages = rng.sample(all_passages, max_passages)
    return all_passages


_PG_DATES_PATH = Path("raw_book_data/wikidata_pg_dates.csv")


def _load_pg_year_map() -> dict[str, str]:
    """gutenbergId -> original publication year (wikidata pubDate), joined onto PG
    passages by pg_id -- source4's PG loader has no year field at all."""
    years: dict[str, str] = {}
    if not _PG_DATES_PATH.exists():
        return years
    with _PG_DATES_PATH.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            gid = (row.get("gutenbergId") or "").strip()
            pub = (row.get("pubDate") or "").strip()
            m = re.match(r"(\d{4})", pub)
            if gid and m:
                years[gid] = m.group(1)
    return years


def _load_pg_raw(
    rng: random.Random,
    min_w: int = MIN_WORDS,
    max_w: int = MAX_WORDS,
    max_books: int = 400,
    max_passages: int = 30_000,
) -> list[dict]:
    if not PG_DATA_DIR.exists():
        print(f"  WARNING: {PG_DATA_DIR} not found -- skipping PG Nonfiction")
        return []
    if not PG_CATALOG.exists():
        print(f"  WARNING: {PG_CATALOG} not found -- skipping PG Nonfiction")
        return []

    all_epubs = build_nonfiction_epubs(PG_CATALOG, PG_VICTORIAN_LIST, PG_DATA_DIR)
    if not all_epubs:
        print("  WARNING: no nonfiction epubs found -- skipping PG Nonfiction")
        return []

    pg_meta: dict[str, dict] = {}
    try:
        with PG_CATALOG.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pg_id = (row.get("Text#") or "").strip()
                if pg_id:
                    pg_meta[pg_id] = {
                        "title":  (row.get("Title") or "").replace("\n", " ").strip(),
                        "author": (row.get("Authors") or row.get("Author") or "").strip() or "unknown",
                    }
    except Exception:
        pg_meta = {}

    pg_years = _load_pg_year_map()

    sampled = rng.sample(all_epubs, min(max_books, len(all_epubs)))
    all_passages: list[dict] = []

    for epub_path, pg_id, title in sampled:
        try:
            raw  = extract_epub_text(epub_path)
            text = clean_epub_text(raw)
            meta   = pg_meta.get(pg_id, {})
            book   = meta.get("title") or title
            author = meta.get("author") or "unknown"
            pg_author = author if author.lower() != "unknown" else ""
            prov = "pg_catalog" if pg_id in pg_meta else "raw"
            year = pg_years.get(pg_id, "")
            for p in extract_passages_from_text(text, min_words=min_w, max_words=max_w):
                all_passages.append({
                    "text":        p,
                    "source":      "PG_Nonfiction",
                    "title":       title,
                    "author":      author,
                    "book_title":  book,
                    "book_author": pg_author,
                    "book_year":   year,
                    "pg_id":       pg_id,
                    "prov_method": prov,
                })
        except Exception:
            continue

    print(f"  {'PG Nonfiction epubs':<40} {len(all_passages):>7,} candidates "
          f"({len(sampled)} books sampled from {len(all_epubs):,} available)")

    if len(all_passages) > max_passages:
        all_passages = rng.sample(all_passages, max_passages)
    return all_passages


def _load_ib_raw(
    rng: random.Random,
    min_w: int = MIN_WORDS,
    max_w: int = MAX_WORDS,
    max_shards: int = 60,
    max_passages: int = 40_000,
) -> list[dict]:
    """Local copy of source4's _load_ib_raw, extended to keep the year/htid columns the
    parquet shards already have (source4's version only reads text/barcode, dropping
    both) so IB passages can carry real book_title/book_author/book_year via the
    HathiTrust bib lookup above."""
    if not IB_SHARDS_DIR.exists():
        print(f"  WARNING: {IB_SHARDS_DIR} not found -- skipping IB")
        return []
    try:
        import pandas as pd
    except ImportError:
        print("  WARNING: pandas not installed -- skipping IB")
        return []

    shard_files = sorted(IB_SHARDS_DIR.glob("*.parquet"))
    if not shard_files:
        print(f"  WARNING: no .parquet files in {IB_SHARDS_DIR} -- skipping IB")
        return []

    sampled = rng.sample(shard_files, min(max_shards, len(shard_files)))
    all_passages: list[dict] = []
    books_seen = 0

    for shard in sampled:
        try:
            df = pd.read_parquet(shard, columns=["text", "year", "htid"])
        except Exception as e:
            print(f"  WARNING: error reading {shard.name}: {e}")
            continue
        for row_idx, row in enumerate(df.itertuples(index=False)):
            text = row.text if isinstance(row.text, str) else ""
            if not text or len(text.split()) < min_w:
                continue
            hid = getattr(row, "htid", "") or ""
            # IB's parquet stores the bare Harvard barcode ("32044108110271"), not the
            # prefixed htid the bib catalog keys on ("hvd.32044108110271") -- IB is the
            # Institutional Books 1.0 corpus, which is Harvard-sourced.
            bib_htid = f"hvd.{hid}" if hid and not hid.startswith("hvd.") else hid
            yr  = getattr(row, "year", None)
            books_seen += 1
            for p in extract_passages_from_text(text, min_words=min_w, max_words=max_w):
                all_passages.append({
                    "text":        p,
                    "source":      "IB",
                    "title":       shard.stem,
                    "author":      "unknown",
                    "book_title":  "",
                    "book_author": "",
                    "book_year":   _safe_year(yr),
                    "barcode":     hid,
                    "htid":        bib_htid,
                    "shard_row":   row_idx,
                })

    print(f"  {'IB shard books':<40} {len(all_passages):>7,} candidates "
          f"({books_seen} books in {len(sampled)} shards)")
    if len(all_passages) > max_passages:
        all_passages = rng.sample(all_passages, max_passages)
    return all_passages


# ---------------------------------------------------------------------------
# Candidate loading -- source4's loaders + HATHI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Letter-corpus loading (epistolary rebuild, 2026-08-21)
# ---------------------------------------------------------------------------
_STAMP_PATTERNS = [
    r"(?im)^.{0,80}\blibrary of the\b.{0,80}$",
    r"(?im)^.{0,80}\b(theological seminary|princeton|harvard college|university library)\b.{0,80}$",
    r"(?im)^.{0,60}\bpresented by\b.{0,80}$",
    r"(?im)^.{0,60}\bbequeathed by\b.{0,80}$",
    r"(?im)^\s*(BX|BR|BS|BL|BT|BV|PR|PS|PQ|DA|DC|AC|AS)\s?\d{2,5}\s?\.?\w*.{0,40}$",
    r"(?im)^.{0,40}\bdigitized by\b.{0,60}$",
    r"(?im)^.{0,40}\bfrom the library of\b.{0,60}$",
    r"(?im)^\s*\d{1,4}\s*$",
]
_STAMP_RE = [re.compile(x) for x in _STAMP_PATTERNS]

def strip_library_stamps(text: str, head_chars: int = 4000) -> str:
    """Remove HathiTrust scan front-matter (library ownership stamps, call numbers,
    donor plates) from the head of a volume. Not yet done corpus-wide; required here because
    the epistolary rebuild draws on HathiTrust, whose OCR text routinely opens with
    'Library of the Theological Seminary PRINCETON' / 'BX 9418 .A4 E5 1855' before any
    real content. Only the HEAD is rewritten -- a call-number-shaped string deep in the
    body is far more likely to be real text than a stamp."""
    if not text:
        return text
    head, tail = text[:head_chars], text[head_chars:]
    for rx in _STAMP_RE:
        head = rx.sub("", head)
    head = re.sub(r"\n{3,}", "\n\n", head)
    return head.lstrip() + tail


def _load_letters_raw(
    rng: random.Random,
    min_w: int = MIN_WORDS,
    max_w: int = MAX_WORDS,
    per_book: int = 20,
    pool_name: str = "HATHI_LETTERS",
) -> list[dict]:
    # per_book raised 6 -> 20 on 2026-08-22. Letters survive the pipeline at roughly
    # 10% (passage gate) x 22% (generation + conv judge) = 2.3% end to end, so 458
    # volumes x 6 passages capped the whole rebuild at ~60 conversations -- the first
    # run wrote 25 and a --count 1500 run could never have reached its target. At 20
    # the ceiling is ~200. Diversity is unaffected in aggregate (458 distinct volumes,
    # 387 authors) though individual volumes now contribute more heavily.
    """Passages from the curated biographical-correspondence pool in
    training_data/multi_turn_sft/letter_sources.py (559 HathiTrust volumes, 472 authors),
    replacing the original epistolary build's 21-epub / ~14-voice source list.

    Unlike _load_hathi_raw this does NOT sample shards at random -- it seeks the specific
    htids on the curated list, so every passage comes from a volume already vetted as
    genuine personal correspondence rather than a political pamphlet, diplomatic despatch
    or devotional tract."""
    try:
        import pandas as pd
    except ImportError:
        print("  WARNING: pandas not installed -- skipping LETTERS")
        return []
    sys.path.insert(0, str(Path("training_data/multi_turn_sft").resolve()))
    try:
        import letter_sources as LS
    except ImportError:
        print("  WARNING: letter_sources.py not found -- skipping LETTERS")
        return []

    pool = getattr(LS, pool_name, None)
    if pool is None:
        print(f"  WARNING: letter_sources.{pool_name} not found -- skipping LETTERS")
        return []
    want = {r["htid"]: r for r in pool}
    print(f"  LETTERS: seeking {len(want)} curated correspondence volumes (pool={pool_name}) ...")
    out: list[dict] = []
    book_count: dict[str, int] = {}
    found = 0
    # HATHI_LETTERS entries have no shard_dir key (always hathi_shards_dedup, matching the
    # original curated-pool design); other pools (e.g. NEW_LETTERS_SEPT20) carry their own
    # shard_dir per entry since they were found across ALL 15 pretraining shard directories,
    # not just HathiTrust -- scan the UNION of whatever directories are actually referenced.
    PRETRAIN_ROOT = HATHI_SHARDS_DIR.parent
    shard_dir_names = {r.get("shard_dir", "hathi_shards_dedup") for r in pool}
    shard_dirs = [PRETRAIN_ROOT / name for name in shard_dir_names]
    all_shards = sorted(
        (p for d in shard_dirs for p in d.glob("*.parquet")),
        key=lambda p: (p.parent.name, p.name),
    )
    for shard in all_shards:
        if not want:
            break
        try:
            df = pd.read_parquet(shard, columns=["text", "htid", "year", "ocr_score"])
        except Exception:
            continue
        df = df[df.htid.isin(want.keys())]
        for row_idx, row in enumerate(df.itertuples(index=False)):
            hid = row.htid
            meta = want.get(hid)
            if meta is None:
                continue
            remaining = per_book - book_count.get(hid, 0)
            if remaining <= 0:
                continue
            text = strip_library_stamps(row.text if isinstance(row.text, str) else "")
            if len(text.split()) < min_w:
                continue
            # Archaic-orthography guard for volumes whose author carries no life dates in
            # the bib record: long-s OCR ("houfe", "fuch") and early-modern Scots ("quhilk",
            # "zit") mark 16th-18thC content reprinted in a Victorian edition. Cheap ratio
            # test on the head only -- a few such tokens in a quoted passage are fine.
            head_l = text[:3000].lower()
            archaic = sum(head_l.count(t) for t in
                          ("houfe", "fuch", "himfelf", "moft", "juft", "alfo", "prefent",
                           "quhilk", "quhen", "thairof", "zit", "fchir"))
            if archaic >= 4:
                continue
            found += 1
            passages = extract_passages_from_text(text, min_words=min_w, max_words=max_w)
            if len(passages) > remaining:
                passages = rng.sample(passages, remaining)
            book_count[hid] = book_count.get(hid, 0) + len(passages)
            for pg in passages:
                out.append({
                    "text": pg, "source": "LETTERS", "title": meta["title"],
                    "author": meta["author"] or "unknown",
                    "book_title": meta["title"], "book_author": meta["author"],
                    "book_year": _safe_year(meta["year"]),
                    "barcode": hid, "htid": hid, "shard_row": row_idx,
                })
    print(f"  {'LETTERS curated correspondence':<40} {len(out):>7,} candidates "
          f"({found} volumes located of {len(want)} sought)")
    return out


def _load_epub_letters_raw(
    rng: random.Random,
    min_w: int = MIN_WORDS,
    max_w: int = MAX_WORDS,
    per_book: int = 120,
    us_share: float = 0.10,
) -> list[dict]:
    """Correspondence passages read straight from Project Gutenberg EPUBs.

    2026-09-20 (Trip): the shard-based letters pool was yielding far too thin -- 8
    volumes produced 97 usable candidates and only 23 finished rows, against the
    shipped file's 1,306 rows. Root cause is the SUBSTRATE, not the pipeline: the
    1,306 good rows (prov_method "pg_static_map") came from gen_multiturn_passages.py
    reading EPUBs, where real <p> tags give true paragraph boundaries. OCR parquet
    shards have none of that -- page-turn fragments, running headers and mid-sentence
    splits are exactly the defects that made the shard rebuild thin.

    Measured on the 85 volumes discover_letter_epubs.py keeps: 194,974 paragraphs ->
    9,380 usable candidates, ~53x the shard approach from ~10x the volumes.

    Uses gen_multiturn_passages.py's own loaders (_read_epub/_split_paragraphs) and its
    _FIRST_PERSON_RE gate -- the epistolary-voice filter that required a substantive
    first-person clause ("I have", "I confess") and is a large part of why that build
    worked -- plus source4's _is_usable, plus strip_letter_header() to remove
    salutation/date front-matter (residual leak 1.14% vs the shipped file's 4.40%).
    """
    cand_path = Path("probes/letter_epub_candidates.json")
    if not cand_path.exists():
        print("  WARNING: probes/letter_epub_candidates.json not found "
              "-- run discover_letter_epubs.py first; skipping EPUB LETTERS")
        return []
    try:
        from gen_multiturn_passages import (
            _read_epub, _split_paragraphs, _FIRST_PERSON_RE,
        )
        from letter_header_utils import strip_letter_header
    except ImportError as e:
        print(f"  WARNING: EPUB letter loader unavailable ({e}) -- skipping")
        return []

    pg_dir = Path("raw_book_data/PG Data")
    cands = json.loads(cand_path.read_text(encoding="utf-8"))
    print(f"  EPUB LETTERS: reading {len(cands)} curated correspondence volumes ...")

    out: list[dict] = []
    n_vols = 0
    for c in cands:
        path = pg_dir / c["filename"]
        if not path.exists():
            continue
        try:
            paras = _split_paragraphs(_read_epub(path))
        except Exception:
            continue
        n_vols += 1
        kept: list[str] = []
        for para in paras:
            para = strip_letter_header(para)
            if not (min_w <= wc(para) <= max_w):
                continue
            if not _FIRST_PERSON_RE.search(para):
                continue
            if not _is_usable(para):
                continue
            kept.append(para)
        if len(kept) > per_book:
            kept = rng.sample(kept, per_book)
        for para in kept:
            out.append({
                "text": para, "source": "EPUB_LETTERS",
                "title": c["title"], "author": "unknown",
                "book_title": c["title"], "book_author": "",
                "book_year": "", "pg_id": c["pg_id"],
                "prov_method": "pg_epub_letters_sept20",
                "nationality": c.get("nationality", "BR"),
                "barcode": "", "htid": "", "shard_row": -1,
            })

    # Cap the American share. The persona is a British Victorian gentleman; on a measured
    # 42-row run US volumes produced 24% of output from 15% of the pool (~1.6x their
    # weight), because their prose skews toward the plain, self-contained register this
    # pipeline's gate and judge both favour. Capping INPUT at ~10% lands output near 15%
    # given that 1.6x survival. Continental voices are deliberately NOT capped -- the
    # shipped file already carries Schiller/Niebuhr/Humboldt, a precedent Trip confirmed.
    us = [p for p in out if p["nationality"] == "US"]
    rest = [p for p in out if p["nationality"] != "US"]
    max_us = int(len(rest) * us_share / max(1e-9, 1 - us_share))
    if len(us) > max_us:
        rng.shuffle(us)
        dropped = len(us) - max_us
        us = us[:max_us]
        print(f"  US cap: dropped {dropped:,} of {dropped + max_us:,} American candidates "
              f"(target ~{us_share:.0%} of pool)")
    out = rest + us
    rng.shuffle(out)

    from collections import Counter as _C
    dist = _C(p["nationality"] for p in out)
    print(f"  {'EPUB curated correspondence':<40} {len(out):>7,} candidates "
          f"({n_vols} volumes read of {len(cands)})")
    print(f"  {'  nationality mix':<40} "
          + "  ".join(f"{k}={v:,} ({100*v/max(1,len(out)):.0f}%)" for k, v in dist.most_common()))
    return out


def load_candidates_v5(
    target: int,
    subcorpus_filter: str | None,
    mode_filter: str | None,
    seed: int,
    overshoot: float = 2.5,
    letters_pool: str = "HATHI_LETTERS",
) -> list[dict]:
    rng = random.Random(seed)

    if (subcorpus_filter or "").lower() == "letters":
        print("Loading CURATED LETTER passages only (epistolary rebuild)...")
        if letters_pool.upper() == "EPUB":
            pool = _load_epub_letters_raw(rng)
        else:
            pool = _load_letters_raw(rng, pool_name=letters_pool)
        subcorpus_filter = None          # pool is already exclusively letters
    else:
        print("Loading source5 passages (150-400 words, incl. HathiTrust)...")
        pool = (
            _load_ncse_raw(rng)
            + _load_bl_raw(rng)
            + _load_pg_raw(rng)
            + _load_ia_raw(rng)
            + _load_ib_raw(rng)
            + _load_hathi_raw(rng)
        )

    if not pool:
        print("ERROR: no passages loaded -- check source paths.")
        sys.exit(1)

    # IB and HATHI passages carry an htid but no title/author from their parquet shards --
    # resolve both in one shared pass over the bib catalog before anything gets filtered out.
    _backfill_hathi_metadata(pool)

    print(f"\n  Total raw passages: {len(pool):,}")
    print("  Applying heuristic filters and routing...")

    candidates: list[dict] = []
    seen: set[str] = set()

    for p in pool:
        text = p["text"]
        key = text.lower()[:120]
        if key in seen:
            continue
        if not starts_cleanly(text) or not ends_cleanly(text):
            continue
        if not _is_usable(text):
            continue

        segs = split_into_segments(text)
        if len(segs) < 2:
            continue

        matches    = route_passage_to_subcorpus(text, p.get("title", ""), p.get("author", ""))
        sub_corpus = matches[0][0] if matches else "unassigned"

        if subcorpus_filter and sub_corpus != subcorpus_filter:
            continue

        n_segs = len(segs)
        if mode_filter:
            t1 = "inquiry" if mode_filter == "follow_up" else mode_filter
            turn_modes = [t1] + [mode_filter] * (n_segs - 1)
        else:
            turn_modes = (
                [rng.choice(COLD_OPEN_MODES)]
                + [rng.choice(ALL_MODES) for _ in range(n_segs - 1)]
            )

        seen.add(key)
        candidates.append({
            "text":        text,
            "segments":    segs,
            "source":      p["source"],
            "title":       p.get("title", ""),
            "author":      p.get("author", "unknown"),
            "book_title":  p.get("book_title", ""),
            "book_author": p.get("book_author", ""),
            "book_year":   p.get("book_year", ""),
            "pg_id":       p.get("pg_id", ""),
            "prov_method": p.get("prov_method", ""),
            "barcode":     p.get("barcode", ""),
            "shard_row":   p.get("shard_row", -1),
            "sub_corpus":  sub_corpus,
            "turn_modes":  turn_modes,
        })

    rng.shuffle(candidates)

    sub_dist = Counter(c["sub_corpus"] for c in candidates)
    src_dist = Counter(c["source"]     for c in candidates)
    print(f"\n  Usable candidates after heuristic filter: {len(candidates):,}")
    print("  Source distribution:")
    for src, n in src_dist.most_common():
        print(f"    {src:<15} {n:>7,}")
    print("  Sub-corpus distribution:")
    for sub, n in sub_dist.most_common():
        print(f"    {sub:<25} {n:>7,}")

    cap = int(target * overshoot) if target > 0 else len(candidates)
    if len(candidates) > cap:
        candidates = candidates[:cap]
        print(f"\n  Pre-loading cap (overshoot {overshoot}x): {cap:,}")

    return candidates


def load_done_keys(path: Path | None = None) -> set[str]:
    """Keys already written, so a rerun continues instead of re-doing the same work.

    2026-08-22 BUG FIX: this read the module-level OUTPUT constant, ignoring --output. Any
    run writing elsewhere therefore consulted the wrong file, found nothing done, and
    reprocessed from the top. Combined with a SEEDED candidate shuffle and the
    count*overshoot cap, every letters run with --seed 42 chewed through the SAME first
    ~1000 of 9180 candidates and never reached the rest -- so the observed ~90% gate-fail
    rate was one fixed subset's rate, not the pool's."""
    done: set[str] = set()
    path = path or OUTPUT
    if not path.exists():
        return done
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec  = json.loads(line)
            pair = rec.get("pair", [])
            key  = next(
                (m["content"].lower()[:120] for m in pair if m.get("role") == "assistant"),
                "",
            )
            if key:
                done.add(key)
        except Exception:
            pass
    return done


# ---------------------------------------------------------------------------
# Blind follow-up generation -- the actual fix
# ---------------------------------------------------------------------------

FOLLOWUP_SYSTEM_PROMPT = """\
You are writing ONE follow-up turn for a real person continuing a conversation with a Victorian
gentleman. You will be shown the conversation SO FAR -- every turn exchanged up to this point --
and NOTHING else. You have NOT seen what the Victorian is about to say next; you only know what
he has already said.

Write the single next human turn, in the given dialogue mode, reacting to the Victorian's most
recent reply. Real people's follow-ups are usually SHORT and pointed -- often well under a full
sentence's worth of throat-clearing -- and pick up on ONE specific thing just said, not the topic
in general.

For calibration, here are examples of the kind of follow-up turns people send this chatbot
(do NOT copy them verbatim):
   "What makes you think so?"
   "In what way?"
   "Dangerous why?"
   "Fine, but can you be more specific, like which towns you mean."
   "That can't really be true, can it?"
   "Surely the harvest wasn't as bad as all that, was it?"

Rules:
- Write in modern colloquial English -- absolutely NOT Victorian.
- One sentence, typically 3-20 words. Do not write multi-clause literary sentences.
- React to something SPECIFIC the Victorian just said -- a claim, word, or detail from his most
  recent reply. Quote or closely paraphrase it if natural.
- Do NOT introduce any proper noun, place, event, or fact that has not already been mentioned
  somewhere in the conversation so far -- you have not seen what comes next, and neither has this
  person. If you find yourself naming something specific, check it already appears above.
- Follow the given dialogue mode's register (described below).
- Do NOT start with "So," or "Well,".

Return ONLY valid JSON, no commentary: {"turn": "the human's next line"}
"""

_FOLLOWUP_FEW_SHOTS: list[tuple[str, str]] = [
    (
        """\
Dialogue mode: Follow-up -- The visitor asks for clarification, more detail, or presses on a specific point from the prior beat.
Style: Tight, reactive -- references something the Victorian JUST said.
Avoid: Do NOT write broad opening questions -- turns must react to the immediately preceding beat.

Conversation so far:
[Human]: Describe the Crystal Palace for me.
[Victorian]: The building was of glass and iron, and covered some eighteen acres of the park. It rose at the centre to a height sufficient to enclose the great elm trees that had stood there before it was built -- this was one of the conditions insisted upon before permission was granted.

Write the next Human turn.""",
        '{"turn": "The elm trees were actually inside the building?"}'
    ),
    (
        """\
Dialogue mode: Challenge -- The visitor is skeptical and pushes back.
Style: Genuine disagreement or skepticism grounded in what was just said.
Avoid: Do NOT write polite questions -- the visitor is genuinely challenging the Victorian.

Conversation so far:
[Human]: What do you make of the calls for shorter hours in the mills?
[Victorian]: I have always maintained that the great error of our age is the worship of busyness for its own sake. A man who is always occupied is called virtuous; a man who has learned to sit still and think is called an idler.

Write the next Human turn.""",
        '{"turn": "That is easy to say for a man who is not the one standing at the loom."}'
    ),
]


def _build_followup_user(mode: str, transcript: list[dict]) -> str:
    m = DIALOGUE_MODES[mode]
    lines = [
        f"Dialogue mode: {m['label']} -- {m['desc']}",
        f"Style: {m['style']}",
        f"Avoid: {m['avoid']}",
        "",
        "Conversation so far:",
    ]
    for turn in transcript:
        speaker = "Human" if turn["role"] == "user" else "Victorian"
        lines.append(f"[{speaker}]: {turn['content']}")
    lines.append("")
    lines.append("Write the next Human turn.")
    return "\n".join(lines)


def parse_followup(raw: str) -> str | None:
    obj = _parse_json(raw) if raw else None
    if not obj:
        return None
    turn = obj.get("turn")
    if not isinstance(turn, str) or not turn.strip():
        return None
    w = wc(turn)
    if w < 2 or w > 30:
        return None
    return turn.strip()


# ---------------------------------------------------------------------------
# Fit check -- does the (already-fixed) next segment plausibly answer the
# just-written BLIND follow-up? Catches the cases where blind generation
# proposes a follow-up the upcoming segment doesn't actually address.
# ---------------------------------------------------------------------------

FIT_CHECK_SYSTEM_PROMPT = """\
You are checking whether a Victorian gentleman's prepared reply plausibly works as an answer to a
human's just-written follow-up, WITHOUT the reply having been written in response to it -- the
reply is a fixed excerpt from a real 19th-century book; only the follow-up was written to try to
fit it.

Score 1 if the reply's content is a PLAUSIBLE, natural-reading response to the follow-up (same
topic, addresses what was asked or asserted, even loosely). Score 0 if the reply is a clear
non-sequitur -- a different, unrelated topic entirely, ignoring what the follow-up raised.

Return ONLY valid JSON, no commentary: {"fit": 0 or 1, "note": "one brief phrase"}
"""

_FIT_CHECK_FEW_SHOTS: list[tuple[str, str]] = [
    (
        "Conversation so far:\n"
        "[Human]: Describe the railway journey north.\n"
        "[Victorian]: The train left Euston a little after eight, and the carriages were "
        "crowded with families bound for the moors.\n"
        "[Human]: Was the crowding uncomfortable for the whole journey?\n\n"
        "Candidate Victorian reply (a fixed excerpt, NOT written in response to this):\n"
        "\"By Preston the compartment had thinned considerably, and there was room enough to "
        "stretch one's legs for the remainder of the way.\"",
        '{"fit": 1, "note": "directly answers whether the crowding persisted"}'
    ),
    (
        "Conversation so far:\n"
        "[Human]: Tell me about the conditions in the Yorkshire mills.\n"
        "[Victorian]: The mills presented a scene of considerable activity and considerable "
        "misery in equal proportion, the air thick with floating fibre.\n"
        "[Human]: Did the owners face any penalty for the state of the air?\n\n"
        "Candidate Victorian reply (a fixed excerpt, NOT written in response to this):\n"
        "\"My own schooling began under a tutor engaged by my father, a clergyman of scholarly "
        "habits who had little patience for idleness.\"",
        '{"fit": 0, "note": "unrelated topic entirely -- schooling, not mill regulation"}'
    ),
]


def _build_fit_user(transcript: list[dict], followup: str, candidate_segment: str) -> str:
    lines = ["Conversation so far:"]
    for turn in transcript:
        speaker = "Human" if turn["role"] == "user" else "Victorian"
        lines.append(f"[{speaker}]: {turn['content']}")
    lines.append(f"[Human]: {followup}")
    lines.append("")
    lines.append(
        f'Candidate Victorian reply (a fixed excerpt, NOT written in response to this):\n"{candidate_segment}"'
    )
    return "\n".join(lines)


def parse_fit(raw: str) -> dict | None:
    obj = _parse_json(raw) if raw else None
    if not obj:
        return None
    try:
        fit = int(obj.get("fit"))
        if fit not in (0, 1):
            return None
    except (TypeError, ValueError):
        return None
    return {"fit": fit, "note": str(obj.get("note", "")).strip()[:160]}


# ---------------------------------------------------------------------------
# Whole-conversation judge -- source4's criteria + an explicit forward-leakage check
# ---------------------------------------------------------------------------

CONV_JUDGE_SYSTEM_PROMPT_V2 = """\
You are evaluating multi-turn Victorian dialogue training conversations for quality.

You will be given a subject description and a full conversation: alternating human turns
and Victorian gentleman responses. Score the conversation 0-3:

   3  Excellent -- Turn 1 is a genuine cold open (no prior context assumed, no demonstratives
      like "those", "that", "these") that invites a topical response; every human turn reacts
      naturally to the immediately preceding Victorian beat WITHOUT anticipating anything the
      Victorian has not yet said; Victorian responses are substantive and on-subject; no template
      phrases copied verbatim from style examples.

   2  Good -- mostly works; Turn 1 may be slightly generic (e.g. "How are you today?") but
      NOT mid-conversation; at most one turn is a weak match or mildly formulaic.

   1  Marginal -- Turn 1 presupposes prior context ("tell me more about those", demonstratives
      like "those", "that", "these"); OR a turn from Turn 2 onward references a specific fact,
      phrase, name, or number that appears ONLY in a LATER Victorian reply and nothing the
      Victorian has already said -- this is FORWARD-LEAKAGE (the human turn was written with
      knowledge of an answer they have not actually heard yet), just as much a flaw as
      Turn-1 presupposition; OR two or more turns poorly match their Victorian response; OR
      obvious template phrases lifted verbatim.

   0  Unusable -- Turn 1 clearly mid-conversation; OR a human turn is a near-verbatim preview of
      wording that only appears in a LATER Victorian reply; OR complete mismatch throughout.

IMPORTANT: The cold-open presupposition rule applies to Turn 1 ONLY. Turns 2 and beyond are
EXPECTED to reference what the Victorian just said -- demonstratives and callback phrases
referencing PRIOR (already-spoken) Victorian turns are correct behaviour, not flaws. Only
penalise a Turn 2+ reference if it points FORWARD to something not yet said.

CONTINUATION CUES: a very short human turn that only signals "keep going" -- "Go on.",
"Tell me more.", "And then?", "How so?", "Why?" -- is a DELIBERATE and CORRECT turn type,
not a formulaic or weak one. Real users write these constantly. Do NOT mark a conversation
down for containing one, and do NOT expect it to reference the preceding beat specifically:
carrying no content is exactly its job. Judge it solely on whether the Victorian's following
reply reads as a natural CONTINUATION of what he was already saying. Penalise it only if that
reply instead answers some question that was never asked.

VICTORIAN-SIDE PRESUPPOSITION: separately, penalise (max score 1) any conversation where a
VICTORIAN turn asserts a shared history with the visitor that was never established -- "as I
told you", "you will remember", "the matter we discussed", "on Monday", "we understand in this
house". That teaches the model to fabricate a relationship it does not have. Be precise: this
covers only claims of shared experience WITH THE VISITOR. A Victorian recounting his own past,
his own opinions, or third parties ("I have always maintained...", "My friend Thomson wrote
to me...") is perfectly proper and must NOT be penalised -- letters and memoirs are full of
such material and it is exactly the personal register we want.

Respond with ONLY valid JSON, no commentary:
{"conv_score": 0, "conv_note": "one brief sentence"}
"""

_CONV_JUDGE_FEW_SHOTS_V2: list[tuple[str, str]] = _CONV_JUDGE_FEW_SHOTS + [
    (
        # Score 1 -- forward-leakage: turn 2 previews wording that only appears in the next Victorian reply
        "Subject: correspondence explaining why a preacher abandoned trinitarian worship for a stricter biblical rule\n\n"
        "[Human]: What led you to change your views on how the scriptures should guide worship?\n"
        "[Victorian]: In my last I supposed that several persons born in trinitarian sects had been "
        "converted to follow Jesus Christ, and had determined to worship only one God, the Father.\n"
        "[Human]: What do you mean by 'plain rules' that are evident to the well-intentioned mind?\n"
        "[Victorian]: In consequence of this determination the bible is before them, not as a subject "
        "of endless discussion, but as containing simple plain rules, evident to the well-intentioned mind.",
        '{"conv_score": 1, "conv_note": "Turn 2 quotes \'plain rules, evident to the well-intentioned '
        'mind\' verbatim before the Victorian has said it -- forward-leakage, not a genuine reaction '
        'to Turn 1."}'
    ),
]


def judge_conv_v2(backend, subject: str, pair: list[dict], max_tokens: int = 120) -> dict | None:
    lines = [f"Subject: {subject}", ""]
    for m in pair:
        role = "[Human]" if m["role"] == "user" else "[Victorian]"
        lines.append(f"{role}: {m['content']}")
    conv_text = "\n".join(lines)
    raw = backend._call(
        CONV_JUDGE_SYSTEM_PROMPT_V2, _CONV_JUDGE_FEW_SHOTS_V2, conv_text, max_tokens, temperature=0.1
    )
    return parse_conv_judgment(raw) if raw else None


def gen_followup(backend, mode: str, transcript: list[dict], max_tokens: int = 60) -> str | None:
    user_content = _build_followup_user(mode, transcript)
    raw = backend._call(
        FOLLOWUP_SYSTEM_PROMPT, _FOLLOWUP_FEW_SHOTS, user_content, max_tokens, temperature=0.8
    )
    return parse_followup(raw) if raw else None


def check_fit(
    backend, transcript: list[dict], followup: str, candidate_segment: str, max_tokens: int = 80
) -> dict | None:
    user_content = _build_fit_user(transcript, followup, candidate_segment)
    raw = backend._call(
        FIT_CHECK_SYSTEM_PROMPT, _FIT_CHECK_FEW_SHOTS, user_content, max_tokens, temperature=0.1
    )
    return parse_fit(raw) if raw else None


# ---------------------------------------------------------------------------
# Conversation builder: cold-open turn 1 (unchanged source4 mechanism) +
# sequential blind turns 2..N (the fix)
# ---------------------------------------------------------------------------

def build_conversation(backend, c: dict, max_tokens: int) -> dict:
    segments = c["segments"]
    modes    = c["turn_modes"]
    n        = len(segments)

    gate = backend.gate_passage(c["text"])
    if gate is None or not gate.get("passage_ok"):
        return {"status": "gate_fail"}

    # Turn 1 + subject: reuse source4's single-shot mechanism unchanged (it already never
    # leaks segment-specific detail into Turn 1 by design); discard its turns[1:].
    result = backend.generate(modes, segments, max_tokens)
    if result is None:
        result = backend.generate(modes, segments, max_tokens)
    err = None if result is None else validate_result(result, n)
    if result is None or err is not None:
        return {"status": "gen_fail", "detail": err or "no result"}

    subject = result["subject"].strip()
    turn1   = result["turns"][0].strip()

    transcript = [
        {"role": "user",      "content": turn1},
        {"role": "assistant", "content": segments[0].strip()},
    ]
    diagnostics = [{"turn": 1, "status": "ok"}]

    # 2026-08-22: on a fit failure, try the NEXT SEGMENT before giving up, up to
    # MAX_SEGMENT_SKIPS times. Previously a failure truncated immediately, which is right
    # for expository sources (consecutive segments develop one argument, so a mismatch
    # means the passage really has moved on) but wrong for LETTERS, which wander across
    # unrelated matters -- the weather, a visitor, a book, a complaint. Segment i+1 may be
    # a non-sequitur while i+2 returns to the subject. Measured on the letters run: 27 of
    # 49 gate-passing passages died as gen_fail/too_short, the single largest downstream
    # loss, and every one of those failed on the FIRST follow-up.
    MAX_SEGMENT_SKIPS = 2
    pending = list(range(1, n))
    skips = 0
    while pending:
        i = pending.pop(0)
        mode = modes[min(i, len(modes) - 1)]
        followup = gen_followup(backend, mode, transcript)
        fit = check_fit(backend, transcript, followup, segments[i]) if followup else None

        if not followup or not fit or not fit.get("fit"):
            followup2 = gen_followup(backend, mode, transcript)
            fit2 = check_fit(backend, transcript, followup2, segments[i]) if followup2 else None
            if followup2 and fit2 and fit2.get("fit"):
                followup, fit = followup2, fit2
            else:
                note = (fit2 or fit or {}).get("note", "") if (fit or fit2) else "gen_fail"
                if pending and skips < MAX_SEGMENT_SKIPS:
                    skips += 1
                    diagnostics.append({"turn": i + 1, "status": "segment_skipped", "note": note})
                    continue          # this segment does not follow; try the next one
                diagnostics.append({"turn": i + 1, "status": "fit_fail", "note": note})
                break  # truncate here -- keep the shorter, honest conversation

        transcript.append({"role": "user",      "content": followup})
        transcript.append({"role": "assistant", "content": segments[i].strip()})
        diagnostics.append({"turn": i + 1, "status": "ok"})

    n_turns_final = len(transcript) // 2
    if n_turns_final < 2:
        return {"status": "too_short"}

    judgment   = judge_conv_v2(backend, subject, transcript)
    conv_score = judgment["conv_score"] if judgment else None
    conv_note  = judgment["conv_note"]  if judgment else ""

    return {
        "status":           "ok",
        "subject":          subject,
        "pair":             transcript,
        "n_turns":          n_turns_final,
        "conv_score":       conv_score,
        "conv_note":        conv_note,
        "truncated":        n_turns_final < n,
        "turn_diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Progress printing
# ---------------------------------------------------------------------------

def _progress(i, total, n_written, n_gate_fail, n_gen_fail, n_judge_low, t0):
    elapsed = time.time() - t0
    rate = i / elapsed if elapsed > 0 else 0
    print(
        f"\r  [{i}/{total}] written={n_written} gate_fail={n_gate_fail} "
        f"gen_fail={n_gen_fail} judge_low={n_judge_low} ({rate:.2f}/s)",
        end="", flush=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate blind-sequential (non-leaking) multi-turn Victorian SFT pairs"
    )
    ap.add_argument("--count",          type=int,   default=0)
    ap.add_argument("--overshoot",      type=float, default=2.5)
    # "letters" is not a SOURCE_SUBCORPORA routing label -- it selects the curated
    # biographical-correspondence pool in letter_sources.py instead of the broad corpus,
    # for the 2026-08-21 epistolary rebuild.
    ap.add_argument("--subcorpus",      type=str,   default="",
                    choices=sorted(SOURCE_SUBCORPORA) + ["letters", ""])
    ap.add_argument("--mode",           type=str,   default="",
                    choices=list(DIALOGUE_MODES.keys()) + [""])
    ap.add_argument("--min-conv-score", type=int,   default=1)
    ap.add_argument("--preview",        action="store_true",
                    help="Show passages and proposed segments, no model calls")
    ap.add_argument("--backend",        choices=["mlx", "anthropic"], default="mlx")
    ap.add_argument("--model",          type=str,   default="")
    ap.add_argument("--max-tokens",     type=int,   default=400)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--output",         type=str,   default="")
    ap.add_argument("--letters-pool",   type=str,   default="HATHI_LETTERS",
                    help="which list in letter_sources.py to draw from when "
                         "--subcorpus letters is set (default: the original 459-volume "
                         "curated pool; pass NEW_LETTERS_SEPT20 for the 2026-09-20 "
                         "expansion books, or EPUB for the Project Gutenberg EPUB pool built by "
                         "discover_letter_epubs.py -- by far the highest-yield source)")
    args = ap.parse_args()

    mode_filter      = args.mode.strip()      or None
    subcorpus_filter = args.subcorpus.strip() or None

    candidates = load_candidates_v5(
        args.count, subcorpus_filter, mode_filter, args.seed, args.overshoot,
        letters_pool=args.letters_pool,
    )

    t1_dist     = Counter(c["turn_modes"][0] for c in candidates)
    t2plus_dist = Counter(m for c in candidates for m in c["turn_modes"][1:])
    print("\n  Turn 1 mode distribution (no follow_up):")
    for mode in ALL_MODES:
        print(f"    {mode:<15} {t1_dist.get(mode, 0):>5}")
    print("  Turn 2+ mode distribution:")
    for mode in ALL_MODES:
        print(f"    {mode:<15} {t2plus_dist.get(mode, 0):>5}")

    if not candidates:
        print("Nothing to do.")
        return

    output_path = Path(args.output) if args.output else OUTPUT
    done_keys = load_done_keys(output_path)
    remaining = [
        c for c in candidates
        if c["segments"][0].lower()[:120] not in done_keys
    ]
    print(f"\n  Already done: {len(candidates) - len(remaining):,}")
    print(f"  To process:   {len(remaining):,}")

    if not remaining:
        print("Nothing to do.")
        return

    if args.preview:
        print(f"\n--- PREVIEW (first 6) ---")
        for c in remaining[:6]:
            modes_str = " | ".join(c["turn_modes"])
            print(f"\n[{c['source']} | sub={c['sub_corpus']} | {len(c['segments'])} segs"
                  f" | modes: {modes_str} | {wc(c['text'])}w]")
            print(f"  title: {c['title'][:80]}")
            for i, (seg, mode) in enumerate(zip(c["segments"], c["turn_modes"]), 1):
                print(f"  Seg {i} [{mode}] ({wc(seg)}w): {seg[:100]}...")
        return

    model_name = args.model or (
        DEFAULT_ANTHROPIC_MODEL if args.backend == "anthropic" else DEFAULT_MLX_MODEL
    )
    backend = AnthropicBackend(model_name) if args.backend == "anthropic" else MLXBackend(model_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written   = 0
    n_gate_fail = 0
    n_gen_fail  = 0
    n_judge_low = 0
    n_truncated = 0
    score_dist: dict[int, int] = {}
    t0 = time.time()

    with output_path.open("a", encoding="utf-8") as f:
        for i, c in enumerate(remaining, 1):
            if args.count and n_written >= args.count:
                break

            outcome = build_conversation(backend, c, args.max_tokens)

            if outcome["status"] == "gate_fail":
                n_gate_fail += 1
                _progress(i, len(remaining), n_written, n_gate_fail, n_gen_fail, n_judge_low, t0)
                continue
            if outcome["status"] in ("gen_fail", "too_short"):
                n_gen_fail += 1
                _progress(i, len(remaining), n_written, n_gate_fail, n_gen_fail, n_judge_low, t0)
                continue

            conv_score = outcome["conv_score"]
            if conv_score is not None:
                score_dist[conv_score] = score_dist.get(conv_score, 0) + 1
            if conv_score is not None and conv_score < args.min_conv_score:
                n_judge_low += 1
                _progress(i, len(remaining), n_written, n_gate_fail, n_gen_fail, n_judge_low, t0)
                continue

            if outcome["truncated"]:
                n_truncated += 1

            rec = {
                "source":           c["source"],
                "title":            c["title"],
                "sub_corpus":       c["sub_corpus"],
                "sub_type":         "multiturn_source5_blindfollowup",
                "turn_modes":       c["turn_modes"][:outcome["n_turns"]],
                "subject":          outcome["subject"],
                "conv_score":       outcome["conv_score"],
                "conv_note":        outcome["conv_note"],
                "n_turns":          outcome["n_turns"],
                "truncated":        outcome["truncated"],
                "turn_diagnostics": outcome["turn_diagnostics"],
                "pair":             outcome["pair"],
                "book_title":       c["book_title"],
                "book_author":      c["book_author"],
                "book_year":        c["book_year"],
                "prov_method":      c["prov_method"],
                "barcode":          c["barcode"],
                "shard_row":        c["shard_row"],
                "pg_id":            c["pg_id"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_written += 1

            _progress(i, len(remaining), n_written, n_gate_fail, n_gen_fail, n_judge_low, t0)

    print(f"\n\nDone. Written: {n_written}  Truncated: {n_truncated}  "
          f"Gate-fail: {n_gate_fail}  Gen-fail: {n_gen_fail}  Judge-low: {n_judge_low}")
    print(f"Score distribution: {dict(sorted(score_dist.items()))}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

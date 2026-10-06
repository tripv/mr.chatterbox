#!/usr/bin/env python3
"""search_attested_figures_corpus.py — full-text scan of the pretraining corpus (all 15
source shard directories, same corpus search_aviation_corpus.py used) for passages mentioning
any of the 191 pre1900/fictional_pre1900 entities in probes/notability_judged.jsonl.

Per entity, search terms are:
  - the FULL canonical_name (always) -- e.g. "Charles Darwin"
  - the bare SURNAME (last word), only if it's a single word, length >= 4, and NOT a common
    English dictionary word -- period prose constantly refers to someone by surname alone
    once introduced ("Darwin says..."), so full-name-only search would miss most real
    mentions, but a bare surname search is a real collision risk for a name that also has an
    ordinary-word sense (measured in an earlier run: "Newton" the surname collides badly
    with "newton" the physics unit of force -- exactly the kind of corpus this would flood).
    Complex canonical names with a comma/title ("George Herbert, 5th Earl of Carnarvon") skip
    the surname term entirely rather than guess which word is the real surname.

One raw pass over the corpus (NOT one pass per entity): a single compiled alternation regex
does the cheap per-document reject first, then only documents that hit SOMETHING get the
expensive per-entity breakdown. Writes up to N windows per (entity, document) pair, tagged
with match_type (full_name vs surname) so the downstream Sonnet gate/question-generation stage
can weight surname-only hits with more scrutiny.

  python3 search_attested_figures_corpus.py --out probes/attested_corpus_hits.jsonl
"""
from __future__ import annotations
import json, re, argparse, collections, time
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = ROOT / "training_data/pretraining"
JUDGED = ROOT / "probes/notability_judged.jsonl"
DICT_PATH = Path("/usr/share/dict/words")

SHARD_DIRS = [
    "ukmhl_shards", "eb_ncse_shards", "gutenberg_shards", "gutenberg_pre1837_shards",
    "hathi_shards_dedup", "ia_shards", "institutional_british_shards", "nls_shards",
    "oldbailey_shards", "parliamentary_shards", "hansard_shards", "supplemental_shards",
    "wellcome_shards", "eurlib_shards", "BL Shards",
]
COLUMNS = ["text", "year", "source", "pub_place", "htid"]


def load_dictionary() -> set[str]:
    return {w.strip().lower() for w in DICT_PATH.open(encoding="latin-1")}


def load_entities(path: Path | None = None) -> list[dict]:
    """2026-09-20: path is overridable (--entities) so a NEW entity list can be scanned
    without merging it into probes/notability_judged.jsonl first -- the non-human
    (places/institutions/objects) and fictional-character expansion is kept as its own
    file and its own scan output, the same separate-track discipline tranche 2 used."""
    entities = []
    for line in (path or JUDGED).open(encoding="utf-8"):
        r = json.loads(line)
        for e in r.get("entities", []):
            if e["notability"] in ("pre1900", "fictional_pre1900"):
                entities.append(e)
    return entities


PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")

# Single-word aliases that would flood the scan even with case-sensitive matching,
# because they are ordinary abstract/common nouns that routinely appear capitalised
# (sentence-initial, in headings, or in title case).
ALIAS_STOPLIST = {
    "victory", "monument", "strand", "rocket", "exchange", "commons", "lords",
    "society", "gallery", "museum", "castle", "abbey", "cathedral", "palace",
    "bridge", "park", "zoo", "bank", "tower", "gardens", "hall", "college",
    "school", "prison", "company", "navy", "moon", "wind", "fire", "needle",
}


def build_search_terms(entities: list[dict], dict_words: set[str]) -> dict[str, dict]:
    """entity canonical_name -> {"full_re": compiled, "surname_re": compiled|None}

    Found 2026-09-03 (partial-scan validation run): 8 canonical names carry a trailing
    parenthetical disambiguator -- "Rothschild family (Mayer Amschel Rothschild and
    descendants)", "Apollo (Greek god)", "Estella (Great Expectations)", "Bevis (fictional
    character)", "The Woman in White (novel reference)", "Thomas Taylor (the Platonist)",
    "The Duchess (My Last Duchess)", "The Hatter (Mad Hatter)" -- and the OLD code searched
    for that literal string including the parenthetical, which can never match real period
    prose (no book ever prints "Apollo (Greek god)" as a phrase). All 8 showed up as false
    zero-hit entities in the partial-scan validation. Fixed by stripping a trailing
    parenthetical before building the search term; "family" is also dropped from "Rothschild
    family" since real prose says "the Rothschilds"/"Rothschild", not "Rothschild family"."""
    terms = {}
    for e in entities:
        canonical = e["canonical_name"]
        name = PAREN_RE.sub("", canonical).strip()
        name = re.sub(r"\s+family$", "", name, flags=re.I)

        # 2026-09-20 (Trip's non-human/fictional expansion): ALSO search the `aliases`
        # field, which this function previously ignored entirely. For PERSON entities that
        # was survivable -- the bare-surname fallback caught "Darwin" even though the
        # aliases list went unused -- but non-person entities have no surname fallback, so
        # an alias-blind search produces outright false zeros wherever prose uses a
        # different form than the canonical name. Measured on the first 5,617 hits of this
        # scan: HMS Beagle / HMS Victory / Madame Tussauds / Lloyd's of London / Jumbo the
        # Elephant / Menai Suspension Bridge all sat at ZERO while their aliases ("The
        # Beagle", "The Victory", "Madame Tussaud's", "Lloyd's", "Jumbo", "Menai Bridge")
        # are precisely what period prose actually prints.
        variants = {name}
        for alias in e.get("aliases") or []:
            a = PAREN_RE.sub("", str(alias)).strip()
            a = re.sub(r"^The\s+", "", a)          # prose writes "the Beagle", and this regex is case-SENSITIVE
            if not a:
                continue
            # A single-word alias needs care. The regex is case-SENSITIVE, which already
            # does most of the disambiguation -- capitalised "Beagle"/"Jumbo" in period
            # prose is overwhelmingly the ship/the elephant, while the dog and the adjective
            # appear lower-case -- so a blanket dictionary check is too strict and was
            # re-creating the very false zeros this change exists to fix. Only block the
            # abstract/common nouns that WOULD flood even when capitalised (sentence-initial
            # "Victory", a chapter heading "MONUMENT", etc.).
            if " " not in a and a.lower() in ALIAS_STOPLIST:
                continue
            variants.add(a)
        full_re = re.compile(
            r"\b(?:" + "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True)) + r")\b"
        )
        surname_re = None
        # found 2026-09-03 (Trip): "Leopold II of Belgium" / "Claus of Innsbruck" -- the "last
        # word" heuristic assumes the last word of a multi-word name is a surname, which is
        # false for "X of Y" royal/title patterns: the last word is part of a place phrase, not
        # a name, so it matched every mention of Belgium/Innsbruck in the corpus, not the
        # person. "Alexander III of Russia" accidentally dodged this only because "russia" is
        # itself a dictionary headword -- not a real fix. Skip surname extraction entirely for
        # any "X of Y" pattern, same as the existing comma-title skip.
        if "," not in name and not re.search(r"\bof\b", name, re.I):
            words = name.split()
            if len(words) > 1:
                surname = words[-1]
                # found 2026-09-03: "Henry VIII" -> last word "VIII" passed every check below
                # (>=4 chars, alphabetic, not an English dictionary word) and became a surname
                # search term -- matching every unrelated "VIII" in the corpus (chapter/volume
                # numbering, other monarchs' regnal numbers), not the king. A regnal numeral is
                # never a valid standalone surname term; use the strict validator, not a
                # "composed of roman-numeral letters" check (which wrongly flags "mill"/"civil"
                # too -- see notability_judge_pipeline.py's identical fix for the same bug).
                is_roman_numeral = bool(re.match(
                    r"^m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$", surname, re.I)
                    and surname)
                # found 2026-09-03: "H. G. Wells" -> surname "Wells" passed the dictionary
                # check because "wells" (plural) is not itself a headword, only "well"
                # (singular) is -- so it matched ordinary water wells throughout the corpus
                # ("Wells abound for irrigation purposes...", a passage about an actual
                # patient named "Elizabeth Wells"). A literal-membership check misses simple
                # inflected forms; also strip a trailing "s" before checking.
                singularized = re.sub(r"s$", "", surname.lower())
                is_common_word = surname.lower() in dict_words or singularized in dict_words
                if (len(surname) >= 4 and surname.isalpha()
                        and not is_common_word and not is_roman_numeral):
                    surname_re = re.compile(r"\b" + re.escape(surname) + r"\b")
        terms[canonical] = {"full_re": full_re, "surname_re": surname_re}
    return terms


def windows_for(text: str, pattern: re.Pattern, max_windows: int, width: int = 400):
    out, seen_ranges = [], []
    for m in pattern.finditer(text):
        if len(out) >= max_windows:
            break
        start, end = max(0, m.start() - width), min(len(text), m.end() + width)
        if any(start < e and end > s for s, e in seen_ranges):
            continue
        seen_ranges.append((start, end))
        out.append(text[start:end].strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "probes/attested_corpus_hits.jsonl")
    ap.add_argument("--max-windows-per-doc", type=int, default=2)
    ap.add_argument("--start-index", type=int, default=0,
                    help="skip the first N files in round-robin order -- for resuming a scan "
                         "from where an earlier run stopped, without re-scanning (and "
                         "re-duplicating hits from) files already covered. 2026-09-04: the "
                         "first 50k-hit run stopped at round-robin position 29/2742 (~1.1%% of "
                         "the corpus) -- a second pass for more raw hits should start at 29, "
                         "not 0, to get genuinely NEW material rather than re-scanning the same "
                         "first ~30 files.")
    ap.add_argument("--min-year", type=int, default=None,
                    help="skip corpus rows published before this year, BEFORE running the "
                         "expensive prefilter regex. For a late-appearing entity there is no "
                         "point scanning earlier material: Sherlock Holmes debuts 1887, so "
                         "~57%% of the corpus cannot possibly mention him. Rows with a null "
                         "year are KEPT (unknown date is not evidence of being too early).")
    ap.add_argument("--entities", type=Path, default=None,
                    help="entity-list JSONL to scan (default: probes/notability_judged.jsonl). "
                         "Use for a separate-track scan, e.g. the non-human/fictional expansion.")
    args = ap.parse_args()

    dict_words = load_dictionary()
    entities = load_entities(args.entities)
    terms = build_search_terms(entities, dict_words)
    n_with_surname = sum(1 for t in terms.values() if t["surname_re"])
    print(f"{len(entities)} entities, {n_with_surname} have a usable surname-only term",
          flush=True)

    # ONE cheap pre-filter regex: alternation of every full_name AND surname across all 191
    # entities, so the vast majority of documents (mentioning none of them) are rejected with
    # a single regex call instead of 191+ separate ones.
    all_patterns = []
    for t in terms.values():
        all_patterns.append(t["full_re"].pattern)
        if t["surname_re"]:
            all_patterns.append(t["surname_re"].pattern)
    prefilter_re = re.compile("|".join(all_patterns))

    # 2026-09-03 BUG (Trip caught the fallout): scanning one shard dir fully before moving to
    # the next means an early stop (e.g. "stop at 10,000 hits to sanity-check the pipeline")
    # samples ONLY the first directory -- which happened to be ukmhl_shards, the single
    # noisiest source in the whole corpus (bibliography-heavy medical citations, foreign-
    # language glossaries, OCR long-s garbling -- same source flagged as the worst in the
    # aviation-corpus scan too). A 10,018-hit stop came back 100% ukmhl, giving a 0% local-gate
    # yield that reflects the worst-case source, not the corpus. Fixed by ROUND-ROBINING files
    # across all 15 source directories, so any prefix of the file list is a real cross-section.
    per_dir = [sorted((CORPUS_ROOT / d).glob("*.parquet")) for d in SHARD_DIRS]
    files = []
    max_len = max(len(fl) for fl in per_dir)
    for i in range(max_len):
        for fl in per_dir:
            if i < len(fl):
                files.append(fl[i])
    total_files = len(files)
    if args.start_index:
        files = files[args.start_index:]
    print(f"{total_files} parquet files across {len(SHARD_DIRS)} sources (round-robin order), "
          f"starting at index {args.start_index} ({len(files)} remaining)", flush=True)

    t0 = time.time()
    n_hits = 0
    n_docs_scanned = 0
    n_docs_matched = 0
    with args.out.open("w", encoding="utf-8") as fout:
        for i, path in enumerate(files):
            try:
                table = pq.read_table(path, columns=COLUMNS)
            except Exception as e:
                print(f"!! failed to read {path}: {e}", flush=True)
                continue
            cols = {c: table.column(c) for c in COLUMNS}
            n = table.num_rows
            for row in range(n):
                if args.min_year is not None:
                    yr = cols["year"][row].as_py()
                    if yr is not None and yr < args.min_year:
                        continue
                text = cols["text"][row].as_py()
                if not text:
                    continue
                n_docs_scanned += 1
                if not prefilter_re.search(text):
                    continue
                n_docs_matched += 1
                for name, t in terms.items():
                    windows = windows_for(text, t["full_re"], args.max_windows_per_doc)
                    match_type = "full_name"
                    if not windows and t["surname_re"]:
                        windows = windows_for(text, t["surname_re"], args.max_windows_per_doc)
                        match_type = "surname"
                    for w in windows:
                        rec = {
                            "entity": name, "match_type": match_type,
                            "shard": str(path.relative_to(CORPUS_ROOT)), "row": row,
                            "year": cols["year"][row].as_py(),
                            "source": cols["source"][row].as_py(),
                            "pub_place": cols["pub_place"][row].as_py(),
                            "htid": cols["htid"][row].as_py(),
                            "window": w, "doc_len": len(text),
                        }
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_hits += 1
            if (i + 1) % 100 == 0 or i == len(files) - 1:
                elapsed = time.time() - t0
                print(f"[{i+1}/{len(files)}] {n_docs_scanned:,} docs scanned, "
                      f"{n_docs_matched:,} docs matched something, {n_hits} hit-windows, "
                      f"{elapsed:.0f}s elapsed", flush=True)
                fout.flush()

    print(f"\nDONE: {n_hits} hit-windows from {n_docs_matched:,}/{n_docs_scanned:,} docs "
          f"in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()

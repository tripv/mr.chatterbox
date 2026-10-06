#!/usr/bin/env python3
"""rank_attested_figures_hits.py — rank/filter probes/attested_corpus_hits.jsonl down to a
manageable per-entity shortlist before the expensive Sonnet gate+question-generation pass.

Reuses the exact lessons learned ranking the aviation corpus scan (rank_aviation_hits.py):
  - full_name matches are trusted more than surname-only matches (a bare surname collision
    risk, same reasoning as the notability-candidate extraction's dictionary-word check).
  - dictionary/bibliography-list format (repeated part-of-speech abbreviations, pronunciation
    brackets, "pp."/"Journ."/"Annal." citation punctuation) is a strong NEGATIVE signal --
    this is exactly what fooled the first aviation-hit ranker (a Portuguese dictionary's
    alphabetical headword cluster scored highest under a naive keyword-density scorer).
  - genuine biographical/expository register ("born", "died", "was a", "author of", "known
    for", "wrote") is a positive signal that the passage actually DISCUSSES the person, not
    just mentions their name in passing.

EB-downweighting, added 2026-09-03 per Trip's review of real gate output: measured that EB is
5.6% of raw hits but 72% of GATE-SURVIVORS ("BRADFORD, a parliamentary and municipal borough of
England, situated in the northern division of the West Riding of Yorkshire..." was in the
gated set). This isn't a gate defect -- gazetteer/encyclopedia entries are self-contained BY
CONSTRUCTION, so they trivially pass a standalone-usability gate every time, structurally
outcompeting genuine narrative-prose mentions of the same person even when those exist. This
register is exactly what eb_qa/cleaned_jun05 (~11k rows, already built, mostly unused) already
covers -- this NEW dataset is supposed to be the diverse, non-EB complement to that, not a
second helping of it. Two independent measures, not just one, since a score penalty alone
still lets an entity with ONLY eb hits fill every slot with them:
  1. score() applies a moderate negative penalty for source=="eb" (demotes, doesn't exclude --
     some EB biographical entries are genuinely fine, just structurally overrepresented).
  2. per-entity selection caps EB windows at EB_CAP_PER_ENTITY, backfilling remaining slots
     from non-EB sources first and only falling back to extra EB windows if nothing else
     exists for that entity (an EB-only entity should still get SOMETHING, just not crowd out
     other sources when alternatives exist).

Keeps the best N windows PER ENTITY (not globally), since the goal is per-entity coverage
for the eventual dataset, not a single best-of-corpus list.

  python3 rank_attested_figures_hits.py --top-per-entity 5
"""
from __future__ import annotations
import argparse, json, re, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent

BIBLIO_RE = re.compile(r"\bpp\.|Journ\.|Annal\.|Gilbert,|\bvol\.|[0-9]{2,4}[.,][ \n]*[0-9]{2,4}")
DICT_POS_RE = re.compile(r",\s*(s\.\s*[mf]\.?|adj\.|adv\.|v\.\s*a\.?|n\.|subst\.)")
PRON_BRACKET_RE = re.compile(r"\[[a-zA-Z0-9'à-ſ]{3,20}\]")
# added 2026-09-04 after Trip flagged Edgar Allan Poe going 0/5 in the gate despite being the
# gate's own praised KEEP few-shot: traced it to a RANKING artifact, not a gate defect -- the
# ranked top-5 never included the "reigns supreme" passage at all. Two garbage windows
# outscored it: an alphabetical-index fragment ("5 27 / 7|M / Edgar Allan Poe, American poet,
# d.") that earned the bio_register bonus purely because "poet" is a substring, and a
# book-catalog listing (alternating ALL-CAPS titles and author names -- the same
# "advertisements for plays by Gilbert" register Trip flagged early in this pipeline's build).
# Neither pattern was caught by the existing biblio/dict-pos/pronunciation-bracket penalties.
# Verified against the real Poe examples before adding: PIPE_OCR_RE fires on the garbled index
# fragment, CATALOG_TITLE_RE fires 4x on the catalog listing, neither fires on 4 genuine Poe
# mentions (including the praised passage itself).
PIPE_OCR_RE = re.compile(r"\|")
CATALOG_TITLE_RE = re.compile(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,}){1,}\b")
BIO_REGISTER_RE = re.compile(
    r"\b(was born|born (?:at|in|on)|died (?:at|in|on|of)|was a\b|is a\b|author of|"
    r"wrote|known for|celebrated for|famous for|distinguished|his (?:great|chief|"
    r"principal) work|her (?:great|chief|principal) work|philosopher|naturalist|"
    r"statesman|poet|novelist|historian|theologian|scientist|physicist|composer)\b",
    re.I)


EB_CAP_PER_ENTITY = 2  # see EB-downweighting note in module docstring


def score(row: dict) -> tuple[int, dict]:
    w = row["window"]
    detail = {}
    s = 0
    if row["match_type"] == "full_name":
        s += 3
        detail["full_name"] = True
    if BIO_REGISTER_RE.search(w):
        s += 3
        detail["bio_register"] = True
    biblio_hits = len(BIBLIO_RE.findall(w))
    if biblio_hits >= 2:
        s -= 3
        detail["biblio_penalty"] = biblio_hits
    dict_pos_hits = len(DICT_POS_RE.findall(w))
    pron_hits = len(PRON_BRACKET_RE.findall(w))
    if dict_pos_hits >= 3 or pron_hits >= 3:
        s -= 8
        detail["dictionary_penalty"] = dict_pos_hits + pron_hits
    if row["source"] == "eb":
        s -= 4
        detail["eb_penalty"] = True
    if PIPE_OCR_RE.search(w):
        s -= 5
        detail["pipe_ocr_penalty"] = True
    catalog_hits = len(CATALOG_TITLE_RE.findall(w))
    if catalog_hits >= 1:
        s -= 4
        detail["catalog_title_penalty"] = catalog_hits
    return s, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=ROOT / "probes/attested_corpus_hits.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "probes/attested_corpus_hits_ranked.jsonl")
    ap.add_argument("--top-per-entity", type=int, default=5)
    ap.add_argument("--boost-list", type=Path, default=None,
                    help="JSON file of canonical_names (e.g. probes/female_entities_boost_list.json) "
                         "to rank deeper than everyone else -- 2026-09-04: female entities measured "
                         "with plenty of raw material (400-700 hits each, comparable to well-covered "
                         "male entities like Newton/Kant) but zero representation in the multi-turn "
                         "output, because only the top-15/entity slice has ever been gated for "
                         "ANYONE -- a targeted deeper rank gives them a fair shot without changing "
                         "the cap for entities that don't need it.")
    ap.add_argument("--boost-top-per-entity", type=int, default=None,
                    help="cap for entities in --boost-list (required if --boost-list is given)")
    ap.add_argument("--judged", type=Path, default=ROOT / "probes/notability_judged.jsonl",
                    help="cross-reference against the full entity list to report TRUE zero-hit entities")
    args = ap.parse_args()

    boost_set = set()
    if args.boost_list:
        if args.boost_top_per_entity is None:
            raise SystemExit("--boost-list requires --boost-top-per-entity")
        boost_set = set(json.load(args.boost_list.open()))
        print(f"{len(boost_set)} boosted entities, capped at {args.boost_top_per_entity} "
              f"(everyone else at {args.top_per_entity})")

    rows = [json.loads(l) for l in args.inp.open(encoding="utf-8")]
    print(f"{len(rows)} raw hit-windows loaded")

    scored = []
    for r in rows:
        sc, detail = score(r)
        scored.append({**r, "rank_score": sc, "rank_detail": detail})

    by_entity = collections.defaultdict(list)
    for r in scored:
        by_entity[r["entity"]].append(r)

    out_rows = []
    n_zero_coverage = 0
    n_eb_capped = 0
    n_boosted_used = 0
    for entity, hits in by_entity.items():
        cap = args.boost_top_per_entity if entity in boost_set else args.top_per_entity
        if entity in boost_set:
            n_boosted_used += 1
        ranked = sorted(hits, key=lambda r: -r["rank_score"])
        # dedup near-identical windows from the SAME document before taking top-N; cap EB
        # windows per entity so an entity with plenty of non-EB coverage doesn't get crowded
        # out by structurally-easy-to-pass gazetteer entries (see EB-downweighting note above).
        # First pass: fill from non-EB sources up to cap. Second pass: backfill
        # remaining slots from EB, capped at EB_CAP_PER_ENTITY, but only if slots are still
        # open (an EB-only entity still gets up to EB_CAP_PER_ENTITY, not zero).
        seen_docs = set()
        picked, eb_picked = [], []
        for r in ranked:
            key = (r["shard"], r["row"])
            if key in seen_docs:
                continue
            if r["source"] == "eb":
                continue  # deferred to backfill pass below
            seen_docs.add(key)
            picked.append(r)
            if len(picked) >= cap:
                break
        if len(picked) < cap:
            for r in ranked:
                key = (r["shard"], r["row"])
                if key in seen_docs or r["source"] != "eb":
                    continue
                if len(eb_picked) >= EB_CAP_PER_ENTITY:
                    break
                seen_docs.add(key)
                eb_picked.append(r)
                if len(picked) + len(eb_picked) >= cap:
                    break
        if eb_picked:
            n_eb_capped += 1
        out_rows.extend(picked + eb_picked)

    print(f"{len(by_entity)} entities had >=1 raw hit ({n_boosted_used} were boosted-list entities)")
    print(f"wrote {len(out_rows)} ranked windows (up to {args.top_per_entity}/entity, "
          f"{args.boost_top_per_entity or '-'}/entity for boosted, EB capped at {EB_CAP_PER_ENTITY}/entity)")
    print(f"{n_eb_capped}/{len(by_entity)} entities drew on EB backfill at all")
    src_counts = collections.Counter(r["source"] for r in out_rows)
    print(f"source mix of ranked output: {dict(src_counts)}")

    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {args.out}")

    counts = collections.Counter(r["entity"] for r in out_rows)
    print("\nentities with FEWER than 2 ranked windows (thin coverage, still >=1 hit):")
    thin = {e: c for e, c in counts.items() if c < 2}
    for e, c in sorted(thin.items()):
        print(f"  {c}  {e}")

    if args.judged.exists():
        all_names = set()
        for line in args.judged.open(encoding="utf-8"):
            r = json.loads(line)
            for e in r.get("entities", []):
                if e["notability"] in ("pre1900", "fictional_pre1900"):
                    all_names.add(e["canonical_name"])
        zero_hit = sorted(all_names - set(by_entity.keys()))
        print(f"\n{len(zero_hit)}/{len(all_names)} entities had ZERO raw corpus hits at all:")
        for e in zero_hit:
            print(f"   {e}")


if __name__ == "__main__":
    main()

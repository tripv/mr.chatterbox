#!/usr/bin/env python3
"""
mine_short_passages.py -- Stage 1 of the short-answer coverage fix: a FRESH mine of the raw
corpus for genuinely short, standalone, sentence-bounded passages, to feed a new
retrieve-and-rerank / question-generation pass against real usage-log factual/opinion questions.

Why a fresh mine, not the existing SFT files: diagnosed 2026-07-30 -- the SFT corpus has strong
coverage of very short ROTE answers (catechism Q&A, median ~30 words) and abundant LONG-form
exposition (corpus pulls, speeches, advice -- 2-20% under 100 words), but almost nothing in
between: a genuinely substantive opinion/fact answer that stays under 100 words. Re-using the
existing long-skewed pools for retrieval can't manufacture short passages that were never
extracted from the corpus in the first place -- hence pulling fresh, directly from raw text, with
max_words capped short from the start.

Bimodal band split (2026-08-10): real usage-log prompt phrasing (usage_data_intent_labels.jsonl,
factual_query/opinion_solicitation/task_request) is NOT a single narrow band -- it's bimodal, a
large cluster of terse single-clause turns plus a real long/multi-clause tail (p90 ~28-39 tok,
clause_mean 2.0-3.5, vs a flat ~1.2-2.0 clause_mean across the existing SFT prompt files). Passage
length is the substrate the downstream question-writing step (gen_prompts_for_passages.py /
retrieve-and-rerank) reasons from -- a terse passage naturally invites a terse question, a richer
passage invites a longer, more specific one -- so this mine now pulls from TWO word-length bands
instead of one: SHORT (unchanged, 50-99 words) and LONG (100-170 words, new), mixed by
--long-fraction (default 0.35) independent of the existing --hathi-fraction source split.

The LONG band is still capped well under the 200-token SFT ceiling -- 170 words measures ~1.21 tok/word on average across this
corpus's prose (p90 ratio ~1.32, worst-case observed ~1.67), so a *word* cap alone isn't a
reliable ceiling guarantee. TOKEN_CEILING_TOKENS below is therefore an exact post-hoc filter using
the real tokenizer, not an estimate -- any candidate (short or long band) at or over that many
tokens is dropped regardless of its word count.

Reuses the SAME raw-source loaders as gen_multiturn_source5.py (NCSE/BL/PG/IA/IB/HathiTrust,
themselves calling parsers/passage_utils.extract_passages_from_text under the hood), now called
once per source across the full union word window (MIN_WORDS_SHORT-MAX_WORDS_LONG) and bucketed
into bands afterward, rather than once per band -- avoids re-parsing the same raw source twice.
No new extraction logic -- same proven starts_cleanly/ends_cleanly/is_self_contained/
is_good_passage gates already in passage_utils.

Output: mined_short_passages.jsonl -- raw candidates, NOT yet judged for coherence (that's
judge_short_passages.py, stage 2). Every record keeps source/title/author/year provenance, plus
"band" (short/long) and exact "tokens" count.

Source weighting (2026-07-30): HathiTrust is disproportionately underrepresented in the SFT
corpus relative to its share of the pretraining corpus, so this mine deliberately over-weights it
--hathi-fraction (default 0.75) of the final target comes from HathiTrust, the remainder split
across NCSE/BL/PG/IA/IB combined.

  python3 mine_short_passages.py --target 1000 --hathi-fraction 0.75 --long-fraction 0.35
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

# 2026-07-31: this script now lives in training_data/fact_opinion_sft/short_passage_building/,
# not the repo root -- gen_multiturn_source5.py is still at repo root, so it needs to be on
# sys.path explicitly (plain `import` resolves relative to the script's own directory otherwise).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from gen_multiturn_source5 import (
    _load_ncse_raw, _load_bl_raw, _load_pg_raw, _load_ia_raw, _load_ib_raw, _load_hathi_raw,
    _backfill_hathi_metadata,
)
from parsers.passage_utils import starts_cleanly, ends_cleanly, is_self_contained

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "mined_short_passages.jsonl"

MIN_WORDS_SHORT = 50    # substantive enough to be a real answer, not a fragment
MAX_WORDS_SHORT = 99    # the original coverage gap: "under 100 words"
MIN_WORDS_LONG = 100
MAX_WORDS_LONG = 170    # kept well clear of the 200-tok ceiling; see TOKEN_CEILING_TOKENS below

# Hard post-hoc safety net measured with the real tokenizer -- word counts are only an
# approximation of token count (ratio varies 1.21 mean / 1.32 p90 / 1.67 worst-case observed on
# this corpus), so this is what actually guarantees nothing over-cap slips back in.
TOKEN_CEILING_TOKENS = 190
# _REPO_ROOT is sft_pipeline/; the tokenizer lives one level up, at the repo root.
_TOKENIZER_PATH = _REPO_ROOT.parent / "tokenizer" / "victoriana-v2-65536" / "tokenizer.json"


def _band_for(word_count: int) -> str | None:
    if MIN_WORDS_SHORT <= word_count <= MAX_WORDS_SHORT:
        return "short"
    if MIN_WORDS_LONG <= word_count <= MAX_WORDS_LONG:
        return "long"
    return None


def _apply_gates(pool: list[dict]) -> list[dict]:
    seen = set()
    usable = []
    for p in pool:
        text = p["text"]
        key = text.lower()[:120]
        if key in seen:
            continue
        if not starts_cleanly(text) or not ends_cleanly(text) or not is_self_contained(text):
            continue
        band = _band_for(len(text.split()))
        if band is None:
            continue
        seen.add(key)
        p["band"] = band
        usable.append(p)
    return usable


def _apply_token_ceiling(pool: list[dict]) -> list[dict]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(_TOKENIZER_PATH))
    kept = []
    dropped = 0
    for p in pool:
        n_tok = len(tok.encode(p["text"]).ids)
        if n_tok >= TOKEN_CEILING_TOKENS:
            dropped += 1
            continue
        p["tokens"] = n_tok
        kept.append(p)
    if dropped:
        print(f"  token-ceiling filter: dropped {dropped:,} candidates at/over "
              f"{TOKEN_CEILING_TOKENS} tok despite word-count band")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=8000, help="total candidates to keep after pooling")
    ap.add_argument("--hathi-fraction", type=float, default=0.75,
                     help="fraction of --target drawn from HathiTrust; remainder split across "
                          "NCSE/BL/PG/IA/IB combined")
    ap.add_argument("--long-fraction", type=float, default=0.35,
                     help="fraction of --target drawn from the LONG (100-170 word) band; "
                          "remainder from the SHORT (50-99 word) band. Applied independently of "
                          "--hathi-fraction, i.e. crossed with it (4 target cells total)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # 4 target cells: source (hathi/other) x band (short/long), independent fractions crossed.
    cell_targets = {
        ("hathi", "short"): int(args.target * args.hathi_fraction * (1 - args.long_fraction)),
        ("hathi", "long"): int(args.target * args.hathi_fraction * args.long_fraction),
        ("other", "short"): int(args.target * (1 - args.hathi_fraction) * (1 - args.long_fraction)),
        ("other", "long"): int(args.target * (1 - args.hathi_fraction) * args.long_fraction),
    }

    print(f"Mining passages across two bands -- short {MIN_WORDS_SHORT}-{MAX_WORDS_SHORT} words, "
          f"long {MIN_WORDS_LONG}-{MAX_WORDS_LONG} words (token ceiling {TOKEN_CEILING_TOKENS})")
    print(f"Target cells: {', '.join(f'{k}={v:,}' for k, v in cell_targets.items())}")

    # Load across the FULL union window once per source -- band assignment happens post-hoc in
    # _apply_gates, so we don't re-parse the same raw source once per band.
    hathi_pool = _load_hathi_raw(rng, min_w=MIN_WORDS_SHORT, max_w=MAX_WORDS_LONG,
                                  max_shards=120, max_passages=18000)
    other_pool = (
        _load_ncse_raw(rng, min_w=MIN_WORDS_SHORT, max_w=MAX_WORDS_LONG, cap=1500)
        + _load_bl_raw(rng, min_w=MIN_WORDS_SHORT, max_w=MAX_WORDS_LONG, max_books=300, max_passages=8000)
        + _load_pg_raw(rng, min_w=MIN_WORDS_SHORT, max_w=MAX_WORDS_LONG, max_books=150, max_passages=6000)
        + _load_ia_raw(rng, min_w=MIN_WORDS_SHORT, max_w=MAX_WORDS_LONG, max_files=500, max_passages=6000)
        + _load_ib_raw(rng, min_w=MIN_WORDS_SHORT, max_w=MAX_WORDS_LONG, max_shards=25, max_passages=6000)
    )
    print(f"\nRaw pooled: {len(hathi_pool):,} HathiTrust, {len(other_pool):,} other")

    _backfill_hathi_metadata(hathi_pool)

    # Re-apply the same standalone-coherence gates load_candidates_v5 uses (extract_passages_from_text
    # already applies these per-source, but IB/HathiTrust rows lose title/author until backfill, so
    # re-check post-backfill; cheap, mechanical, no LLM yet -- that's stage 2). Also assigns "band".
    hathi_usable = _apply_gates(hathi_pool)
    other_usable = _apply_gates(other_pool)
    # Exact-token safety net -- see TOKEN_CEILING_TOKENS docstring note.
    hathi_usable = _apply_token_ceiling(hathi_usable)
    other_usable = _apply_token_ceiling(other_usable)

    pools = {
        ("hathi", "short"): [p for p in hathi_usable if p["band"] == "short"],
        ("hathi", "long"): [p for p in hathi_usable if p["band"] == "long"],
        ("other", "short"): [p for p in other_usable if p["band"] == "short"],
        ("other", "long"): [p for p in other_usable if p["band"] == "long"],
    }
    print("\nUsable after dedup + coherence + token-ceiling gates, by cell:")
    for cell, pool in pools.items():
        print(f"  {cell}: {len(pool):,} available vs {cell_targets[cell]:,} target")

    usable = []
    for cell, pool in pools.items():
        n_target = cell_targets[cell]
        if len(pool) < n_target:
            print(f"  WARNING: cell {cell} short of target -- taking all {len(pool):,} available")
        usable.extend(rng.sample(pool, min(n_target, len(pool))))
    rng.shuffle(usable)

    from collections import Counter
    src_dist = Counter(p["source"] for p in usable)
    band_dist = Counter(p["band"] for p in usable)
    print(f"\nFinal pool: {len(usable):,} passages")
    print("Source distribution (final pool):")
    for src, n in src_dist.most_common():
        print(f"  {src:<10} {n:>6,}  ({100*n/len(usable):.1f}%)")
    print("Band distribution (final pool):")
    for band, n in band_dist.most_common():
        print(f"  {band:<10} {n:>6,}  ({100*n/len(usable):.1f}%)")

    with args.out.open("w", encoding="utf-8") as f:
        for p in usable:
            f.write(json.dumps({
                "passage": p["text"],
                "words": len(p["text"].split()),
                "tokens": p["tokens"],
                "band": p["band"],
                "source": p["source"],
                "book_title": p.get("book_title", ""),
                "book_author": p.get("book_author", ""),
                "book_year": p.get("book_year", ""),
                "title": p.get("title", ""),
                "prov_method": p.get("prov_method", ""),
            }, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(usable):,} candidates -> {args.out}")


if __name__ == "__main__":
    main()

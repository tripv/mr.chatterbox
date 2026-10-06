#!/usr/bin/env python3
"""rewiden_attested_hits.py — fix a real construction bug found 2026-09-03 (Trip): every
window in probes/attested_corpus_hits.jsonl was a FIXED-WIDTH character slice (+/-400 chars
around the match) with NO sentence-boundary snapping, so windows systematically start and end
mid-word/mid-sentence ("eby the" from "thereby", "gns of Graves's disease" from "signs of...").
A local-gate test run on the first 100 raw hits confirmed the fallout: 0/100 kept, because a
window that starts mid-sentence is EXACTLY what the gate's anaphoric_opening/mid_scene_fiction
criteria are built to reject -- this was measuring a construction defect, not real corpus
content quality.

Reprocesses the EXISTING hit list rather than rescanning the corpus from scratch: groups by
(shard, row) so each parquet file/row is reopened only once, relocates the same entity match in
the full document text, and re-extracts a window snapped outward to real sentence boundaries
(never mid-word) at a similar overall target width.

  python3 rewiden_attested_hits.py --in probes/attested_corpus_hits.jsonl \
      --out probes/attested_corpus_hits_widened.jsonl
"""
from __future__ import annotations
import json, re, argparse, collections
from pathlib import Path

import pyarrow.parquet as pq

from search_attested_figures_corpus import load_entities, load_dictionary, build_search_terms, CORPUS_ROOT

SENTENCE_END_RE = re.compile(r"[.!?]")


def snap_to_sentences(text: str, start: int, end: int, max_extra: int = 400) -> str:
    """Expand [start, end) outward to the nearest sentence boundary on each side, capped so a
    single run-on paragraph can't blow the window up unboundedly."""
    limit_s = max(0, start - max_extra)
    last_end = max(text.rfind(p, limit_s, start) for p in ".!?")
    s = limit_s if last_end == -1 else last_end + 1
    while s < len(text) and text[s].isspace():
        s += 1

    limit_e = min(len(text), end + max_extra)
    m = SENTENCE_END_RE.search(text, end, limit_e)
    e = m.end() if m else limit_e

    return text[s:e].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("probes/attested_corpus_hits.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("probes/attested_corpus_hits_widened.jsonl"))
    # 2026-09-20: MUST match the entity list the SCAN used. This defaulted to
    # probes/notability_judged.jsonl with no override, so re-widening a scan built from a
    # DIFFERENT entity file silently lost every row -- terms.get(entity) missed for all of
    # them and they were counted as "match not relocatable". Measured: 126,299/126,299 lost.
    ap.add_argument("--entities", type=Path, default=None,
                    help="entity-list JSONL the scan used (default: probes/notability_judged.jsonl)")
    args = ap.parse_args()

    dict_words = load_dictionary()
    entities = load_entities(args.entities)
    terms = build_search_terms(entities, dict_words)

    rows = [json.loads(l) for l in args.inp.open(encoding="utf-8")]
    print(f"{len(rows)} raw hits to reprocess")

    by_doc = collections.defaultdict(list)
    for r in rows:
        by_doc[(r["shard"], r["row"])].append(r)
    print(f"{len(by_doc)} distinct (shard,row) documents to reopen")

    n_fixed = n_lost = 0
    with args.out.open("w", encoding="utf-8") as fout:
        for i, ((shard, row_idx), hits) in enumerate(by_doc.items()):
            try:
                table = pq.read_table(CORPUS_ROOT / shard, columns=["text"])
                text = table.column("text")[row_idx].as_py()
            except Exception as e:
                n_lost += len(hits)
                continue
            for r in hits:
                t = terms.get(r["entity"])
                if t is None:
                    n_lost += 1
                    continue
                pat = t["full_re"] if r["match_type"] == "full_name" else t["surname_re"]
                if pat is None:
                    n_lost += 1
                    continue
                m = pat.search(text)
                if not m:
                    n_lost += 1
                    continue
                widened = snap_to_sentences(text, m.start(), m.end())
                if not widened:
                    n_lost += 1
                    continue
                fout.write(json.dumps({**r, "window": widened}, ensure_ascii=False) + "\n")
                n_fixed += 1
            if (i + 1) % 500 == 0:
                print(f"  ... {i+1}/{len(by_doc)} documents, {n_fixed} fixed, {n_lost} lost")

    print(f"\nDONE: {n_fixed} windows re-extracted with sentence-snapping, {n_lost} lost "
          f"(match not relocatable) -> {args.out}")


if __name__ == "__main__":
    main()

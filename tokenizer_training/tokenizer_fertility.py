#!/usr/bin/env python3
"""
tokenizer_fertility.py — measure fertility of one (or several) trained tokenizers on the HELD-OUT
eval sample, and project the full-corpus token count + implied Chinchilla-optimal model size.

Fertility = chars/token (higher = better compression = fewer tokens for the corpus). Reported per
source and corpus-weighted (using real per-source char totals so the projection reflects the true
mix, not the balanced eval sample). The corpus-weighted chars/token × total corpus chars = projected
training tokens → that fixes the final token budget and thus the model size.

Run:
    python tokenizer_fertility.py --tokenizers tokenizer/victoriana-v2-32768 tokenizer/victoriana-v2-49152 tokenizer/victoriana-v2-65536
"""
import argparse, glob, json, os
from collections import defaultdict
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = "training_data/pretraining"

def corpus_source_chars():
    """Real in-band char total per source (drives the weighted projection). Cached to a json."""
    cache = "tokenizer/source_chars.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    import tag_schema as T
    tot = defaultdict(int)
    for d in sorted(os.listdir(ROOT)):
        p = f"{ROOT}/{d}"
        if not os.path.isdir(p) or d.startswith("_"): continue
        for shard in glob.glob(f"{p}/*.parquet"):
            t = pq.read_table(shard, columns=["text", "year", "source"]).to_pylist()
            for r in t:
                if T.in_band(r["year"]) and r["text"]:
                    tot[r["source"]] += len(r["text"])
    os.makedirs("tokenizer", exist_ok=True)
    json.dump(tot, open(cache, "w"))
    return dict(tot)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizers", nargs="+", required=True)
    ap.add_argument("--eval", default="tokenizer/sample/eval.jsonl")
    a = ap.parse_args()

    eval_docs = [json.loads(l) for l in open(a.eval)]
    by_src = defaultdict(list)
    for r in eval_docs: by_src[r["source"]].append(r["text"])
    src_chars = corpus_source_chars()
    total_chars = sum(src_chars.values())
    print(f"eval docs {len(eval_docs):,} across {len(by_src)} sources | "
          f"corpus in-band {total_chars/1e9:.1f}B chars\n")

    print(f"{'tokenizer':28}{'vocab':>8}{'chars/tok':>11}{'proj.tokens':>13}{'~model@20:1':>12}")
    print("-" * 72)
    for tdir in a.tokenizers:
        tok = Tokenizer.from_file(f"{tdir}/tokenizer.json")
        # per-source chars/token on eval, then corpus-weight
        wnum = wden = 0.0
        per_src = {}
        for src, texts in by_src.items():
            c = sum(len(t) for t in texts)
            n = sum(len(tok.encode(t, add_special_tokens=False).ids) for t in texts)
            if n: per_src[src] = c / n
        # corpus-weighted: project each source by its own chars/tok; sources absent from the eval
        # holdout (e.g. eb, catechism — too few books) fall back to the mean chars/tok.
        mean_cpt = sum(per_src.values()) / len(per_src)
        proj_tokens = sum(src_chars[s] / per_src.get(s, mean_cpt) for s in src_chars)
        cw_cpt = total_chars / proj_tokens
        model = proj_tokens / 20.0  # Chinchilla ~20 tokens/param
        print(f"{os.path.basename(tdir):28}{tok.get_vocab_size():>8,}{cw_cpt:>11.3f}"
              f"{proj_tokens/1e9:>12.2f}B{model/1e9:>11.2f}B")
    print("\nper-source chars/tok (last tokenizer):")
    for s in sorted(per_src, key=lambda x: -per_src[x]):
        print(f"  {s:16}{per_src[s]:6.2f}  (corpus {src_chars.get(s,0)/1e9:5.2f}B chars)")

if __name__ == "__main__":
    main()

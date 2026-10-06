#!/usr/bin/env python3
"""
02_build_index.py — embed corpus contexts for retrieval.

Subsamples pairs.jsonl (every Nth, to spread across books/genres), embeds the
CONTEXT utterance of each pair with all-MiniLM-L6-v2, and caches:
  index_ctx.npy   float32 [N, 384]  L2-normalized context embeddings
  index_meta.jsonl                  {book, ctx, resp} aligned row-for-row

We embed the context (not the response): the retrieval bet is "find a corpus
turn A that resembles the adversarial prompt, then its real reply B is a
plausible in-character response to the prompt."
"""
from __future__ import annotations
import json, argparse
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
PAIRS = HERE / "pairs.jsonl"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=6, help="keep every Nth pair")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()

    ctxs, meta = [], []
    with open(PAIRS) as f:
        for i, line in enumerate(f):
            if i % args.stride: continue
            r = json.loads(line)
            ctxs.append(r["ctx"])
            meta.append({"book": r["book"], "ctx": r["ctx"], "resp": r["resp"]})
            if args.limit and len(ctxs) >= args.limit: break
    print(f"embedding {len(ctxs)} contexts ...")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(ctxs, batch_size=args.batch, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    np.save(HERE / "index_ctx.npy", emb)
    with open(HERE / "index_meta.jsonl", "w") as out:
        for m in meta:
            out.write(json.dumps(m) + "\n")
    print(f"saved index_ctx.npy {emb.shape} + index_meta.jsonl")

if __name__ == "__main__":
    main()

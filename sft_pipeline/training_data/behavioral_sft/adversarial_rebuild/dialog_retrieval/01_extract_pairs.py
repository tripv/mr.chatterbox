#!/usr/bin/env python3
"""
01_extract_pairs.py — extract (context -> reply) adjacency pairs from the
Gutenberg Dialog corpus (Csaky/Recski), en split.

A "conversation" in dialogs_clean.txt is a block of consecutive utterance lines
delimited by blank lines; each line is prefixed "<bookid>.txt: <utterance>".
Consecutive utterances within a block form an adjacency pair (A -> B), where B
is a real, human-authored reply to A. B is what we may later use as a training
target -- 100% corpus text, never model- or LLM-authored.

Output: pairs.jsonl  {book, ctx, resp, ctx_nwords, resp_nwords}
Length / quality filtering is deliberately light here; heavier filtering and
truecasing happen downstream so we keep raw recall high for retrieval.
"""
from __future__ import annotations
import json, re, argparse
from pathlib import Path

# Gutenberg Dialog corpus (Csaky & Recski, https://github.com/ricsinaruto/gutenberg-dialog), en split.
SRC = Path("training_data/old_training_data/gutenberg_dialog_corpus/en/dialogs_clean.txt")
OUT = Path(__file__).resolve().parent / "pairs.jsonl"

PREFIX = re.compile(r'^(\S+?\.txt):\s?(.*)$')

def utterances(path: Path):
    """Yield (book, [utt,...]) per blank-line-delimited conversation block."""
    book = None
    block: list[str] = []
    with open(path, errors="replace") as f:
        for line in f:
            s = line.rstrip("\n")
            if not s.strip():
                if block:
                    yield book, block
                block = []
                continue
            m = PREFIX.match(s)
            if m:
                book = m.group(1)
                utt = m.group(2).strip()
            else:
                utt = s.strip()
            if utt:
                block.append(utt)
    if block:
        yield book, block

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC, help="path to the corpus's dialogs_clean.txt")
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--max-words", type=int, default=60)
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = no cap")
    args = ap.parse_args()

    n_blocks = n_pairs = 0
    with open(OUT, "w") as out:
        for book, block in utterances(args.src):
            n_blocks += 1
            for a, b in zip(block, block[1:]):
                aw, bw = len(a.split()), len(b.split())
                if not (args.min_words <= aw <= args.max_words):
                    continue
                if not (args.min_words <= bw <= args.max_words):
                    continue
                out.write(json.dumps({
                    "book": book, "ctx": a, "resp": b,
                    "ctx_nwords": aw, "resp_nwords": bw,
                }) + "\n")
                n_pairs += 1
                if args.max_pairs and n_pairs >= args.max_pairs:
                    print(f"blocks={n_blocks} pairs={n_pairs} (capped)")
                    return
    print(f"blocks={n_blocks} pairs={n_pairs} -> {OUT}")

if __name__ == "__main__":
    main()

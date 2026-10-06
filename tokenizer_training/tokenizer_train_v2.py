#!/usr/bin/env python3
"""
tokenizer_train_v2.py — train the victoriana base BPE tokenizer (v2) with the LOCKED tag schema.

ByteLevel BPE (OCR/Unicode-robust, no normalisation → preserves Victorian orthography), trained on
the reusable sample from tokenizer_sample.py. Reserves ALL of tag_schema.SPECIAL_TOKENS (124:
[y1800..y1899], [src_*], [ocr_*], chat/struct) so the metadata prefix encodes as atomic tokens that
are never split or merged.

Run (one vocab size; used for the fertility sweep):
    python tokenizer_train_v2.py --vocab 32768
    python tokenizer_train_v2.py --vocab 49152
    python tokenizer_train_v2.py --vocab 65536
"""
import argparse, json, time
from pathlib import Path
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import tag_schema as T

def text_iter(sample_path):
    with open(sample_path) as f:
        for line in f:
            try: yield json.loads(line)["text"]
            except Exception: continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--sample", default="tokenizer/sample/train.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-frequency", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out or f"tokenizer/victoriana-v2-{a.vocab}")

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=a.vocab,
        special_tokens=T.SPECIAL_TOKENS,   # 124 reserved tag/chat tokens, first IDs, never split
        min_frequency=a.min_frequency,
        show_progress=True,
    )
    print(f"training vocab={a.vocab:,} | {len(T.SPECIAL_TOKENS)} reserved special tokens "
          f"| sample={a.sample}", flush=True)
    t0 = time.time()
    tok.train_from_iterator(text_iter(a.sample), trainer=trainer)
    # ensure the tag tokens are registered as added/special so .encode keeps them atomic
    tok.add_special_tokens(T.SPECIAL_TOKENS)
    dt = time.time() - t0

    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))
    # sanity: every special token must be exactly 1 id
    bad = [s for s in T.SPECIAL_TOKENS if len(tok.encode(s, add_special_tokens=True).ids) != 1]
    print(f"\nsaved → {out}/tokenizer.json | vocab {tok.get_vocab_size():,} | {dt/60:.1f} min")
    print(f"special-token atomicity: {'OK all 1-id' if not bad else 'BROKEN: '+str(bad[:5])}")
    demo = T.encode_doc({"text": "The phrenologist examined the gentleman's cranium with care.",
                         "year": 1871, "source": "hathi"})
    ids = tok.encode(demo, add_special_tokens=True).ids
    print(f"demo prefix+text → {len(ids)} ids; first 4 = {ids[:4]} "
          f"(decode: {[tok.id_to_token(i) for i in ids[:4]]})")

if __name__ == "__main__":
    main()

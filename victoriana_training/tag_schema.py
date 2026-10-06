#!/usr/bin/env python3
"""
tag_schema.py — CANONICAL tag-schema for victoriana pretraining (locked 2026-06-30).

Every pretraining document is prefixed with RESERVED special tokens conditioning on
[year][source][ocr]. The prefix is LOSS-MASKED (the model conditions on it, never predicts/emits it)
and each tag is INDEPENDENTLY DROPPED with prob TAG_DROPOUT at training time, so the model handles
any subset of tags at inference (and unconditional generation). At inference you prepend the tags you
want, e.g. "[y1893][src_gutenberg][ocr_hi]".

DECISIONS (user, 2026-06-30):
  • year  = EXACT year, tokens [y1800]..[y1899] (100).  Final build is 19th-century ONLY:
            in_band == 1800<=year<=1899. Post-1899 / pre-1800 / undated are EXCLUDED at assembly
            (source shards keep them tagged as a complete archive; this filter enforces the band).
  • ocr   = digit-robust 3-bucket [ocr_hi|mid|lo]. The stored per-source ocr_score is INCONSISTENT
            (alpha-ratio proxy mislabels numeral-heavy clean text e.g. Hansard as garbage), so the
            bucket is recomputed here from a DIGIT-ROBUST letter ratio (digits excluded from the
            denominator). Born-digital sources are forced hi.
  • chat  = role tokens <|user|> / <|assistant|> (+ <|system|>, <|end|>), matching the SFT data.

Reserved-token count: 100 year + 16 source + 3 ocr + 5 structural = 124. Add ALL of these to the BPE
trainer as special tokens (never split, never merged).
"""
import re

# ── sources present in the corpus (one reserved token each) ──────────────────
SOURCES = ["hathi", "wellcome", "bl", "ia", "eb", "ncse", "gutenberg", "institutional",
           "nls", "ukmhl", "eurlib", "hansard", "parliamentary", "oldbailey",
           "catechism", "riddle"]
BORN_DIGITAL = {"gutenberg", "catechism", "riddle"}   # no OCR → ocr bucket forced hi

YEAR_MIN, YEAR_MAX = 1800, 1899                       # 19th-century band (inclusive)
OCR_HI, OCR_MID = 0.93, 0.87                          # digit-robust ratio thresholds
TAG_DROPOUT = 0.15                                    # per-tag independent drop prob (training)

# ── reserved special tokens (feed ALL of these to the tokenizer) ─────────────
YEAR_TOKENS   = [f"[y{y}]" for y in range(YEAR_MIN, YEAR_MAX + 1)]      # 100
SOURCE_TOKENS = [f"[src_{s}]" for s in SOURCES]                          # 16
OCR_TOKENS    = ["[ocr_hi]", "[ocr_mid]", "[ocr_lo]"]                    # 3
STRUCT_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]  # 5
SPECIAL_TOKENS = YEAR_TOKENS + SOURCE_TOKENS + OCR_TOKENS + STRUCT_TOKENS                  # 124
EOS = "<|endoftext|>"

# ── ocr bucketing (digit-robust: ignore digits so numeral-heavy clean text scores high) ──
_HEAD = 30_000
def digit_robust_ratio(text):
    s = text[:_HEAD]
    letters = sum(c.isalpha() for c in s)
    junk = sum(1 for c in s if not c.isspace() and not c.isalpha() and not c.isdigit())
    return letters / (letters + junk) if (letters + junk) else 0.0

def ocr_bucket(text, source):
    if source in BORN_DIGITAL:
        return "[ocr_hi]"
    r = digit_robust_ratio(text)
    return "[ocr_hi]" if r >= OCR_HI else "[ocr_mid]" if r >= OCR_MID else "[ocr_lo]"

# ── per-field token lookups ──────────────────────────────────────────────────
def year_token(year):
    return f"[y{int(year)}]" if year is not None and YEAR_MIN <= int(year) <= YEAR_MAX else None

def source_token(source):
    return f"[src_{source}]" if source in SOURCES else None

def in_band(year):
    """True iff the doc belongs in the 19th-century final build."""
    return year is not None and YEAR_MIN <= int(year) <= YEAR_MAX

# ── encoding ─────────────────────────────────────────────────────────────────
def make_prefix(rec, rng=None, dropout=0.0):
    """Build the '[year][source][ocr]' prefix for a record {text,year,source,...}.
    Each tag independently dropped with prob `dropout` when rng given (training)."""
    parts = []
    for tok in (year_token(rec.get("year")), source_token(rec.get("source")),
                ocr_bucket(rec.get("text", ""), rec.get("source"))):
        if tok is None:
            continue
        if rng is not None and dropout and rng.random() < dropout:
            continue
        parts.append(tok)
    return "".join(parts)

def encode_doc(rec, rng=None, dropout=TAG_DROPOUT):
    """Record -> training string 'PREFIX text <|endoftext|>', or None if out-of-band.
    The PREFIX span must be LOSS-MASKED by the data loader (condition-only, not predicted)."""
    if not in_band(rec.get("year")):
        return None
    text = (rec.get("text") or "").strip()
    if not text:
        return None
    prefix = make_prefix(rec, rng, dropout)
    return f"{prefix}{text} {EOS}"

def prefix_len_chars(rec):
    """Char length of the FULL (no-dropout) prefix — for the loss-mask boundary."""
    return len(make_prefix(rec, rng=None, dropout=0.0))

# ── held-out split: deterministic, disjoint BY BOOK (htid), ~HOLDOUT_FRAC of docs ────
import hashlib
HOLDOUT_FRAC = 0.005   # ~0.5% held out for tokenizer-fertility eval + pretrain val/perplexity

def is_holdout(htid, frac=HOLDOUT_FRAC):
    """True if this book is in the held-out split. Hash-of-htid → stable across runs, and since
    the whole book hashes together no book can leak across the train/val boundary. Large-corpus
    uniform sampling ≈ stratified across source×decade."""
    if htid is None:
        return False
    h = int(hashlib.blake2b(str(htid).encode(), digest_size=8).hexdigest(), 16)
    return (h % 1_000_000) < frac * 1_000_000

if __name__ == "__main__":
    print(f"reserved special tokens: {len(SPECIAL_TOKENS)} "
          f"({len(YEAR_TOKENS)} year + {len(SOURCE_TOKENS)} source + {len(OCR_TOKENS)} ocr "
          f"+ {len(STRUCT_TOKENS)} struct)")
    demo = {"text": "It was the best of times, the age of railways and steam.",
            "year": 1859, "source": "gutenberg"}
    print("example:", repr(encode_doc(demo, rng=None)))
    print("oob example (1905):", encode_doc({**demo, "year": 1905}))

# Victoriana base tokenizer — LOCKED (2026-06-30)

Canonical path: `tokenizer/victoriana-base` → `victoriana-v2-65536/tokenizer.json`

| | |
|---|---|
| Type | ByteLevel BPE (no Unicode normalization → preserves Victorian orthography) |
| Vocab size | **65,536** |
| Reserved special tokens | 124 (`tag_schema.SPECIAL_TOKENS`): 100 `[y1800..y1899]` + 16 `[src_*]` + 3 `[ocr_*]` + 5 chat/struct — all verified atomic (1 id) |
| Training data | balanced 5.38B-char sample (`tokenizer_sample.py`, 400M/source cap), in-band 1800–1899, holdout excluded |
| min_frequency | 3 |
| Fertility (corpus-weighted) | **3.99 chars/token** → in-band corpus (218.4B chars) ≈ **54.8B training tokens** |
| Eval (held-out) | tokens/word 1.359 · bytes/token 4.18 · noise tokens 0.03% · 5/36 probe words split |

## Why 64k
Best morphological alignment with the corpus: keeps Victorian-register words whole (`majesty`,
`ecclesiastical`, `philosophical`, `proprietor`) that 32k/48k fragment. Fertility ordering
(64k > 48k > 32k) holds at every training-data scale (see `tokenizer/scaling/`). The +40M embedding
params vs 48k are trivial at the ~2.7–2.8B model scale.

## Saturation note
The data-scaling sweep (`tokenizer_scaling.py`, cf. arXiv:2502.20273) shows fertility saturates by
~250M–1B chars; the 5.38B training sample is well past the knee, so the tokenizer is not data-limited.

## Reproduce
```
python tokenizer_sample.py
python tokenizer_train_v2.py --vocab 65536
python tokenizer_fertility.py --tokenizers tokenizer/victoriana-v2-65536
```

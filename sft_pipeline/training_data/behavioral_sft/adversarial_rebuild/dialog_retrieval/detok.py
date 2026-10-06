#!/usr/bin/env python3
"""
detok.py — heuristic truecasing / detokenization for the Gutenberg Dialog corpus.

The corpus is lowercased and Moses-style tokenized ("do n't", "sir , i", "wo n't").
This restores readable casing/punctuation for SFT targets. It is deliberately
conservative: it fixes tokenization spacing, common contractions, sentence-start
and "I" capitalization. It does NOT attempt proper-noun casing (out of scope /
error-prone); a human/LLM pass can catch the rest if needed.
"""
from __future__ import annotations
import re

_CONTRACTIONS = [
    (r"\bca n't\b", "can't"), (r"\bwo n't\b", "won't"), (r"\bsha n't\b", "shan't"),
    (r"\bai n't\b", "ain't"), (r" n't\b", "n't"),
    # suffixes appear as " ' ll" (spaced) or " 'll" (apostrophe joined to suffix)
    (r"\s+'\s*(ll|re|ve|d|s|m|t)\b", r"'\1"),
    (r"\bs '\b", "s'"),
]

def _fix_spacing(t: str) -> str:
    for pat, rep in _CONTRACTIONS:
        t = re.sub(pat, rep, t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)          # " ," -> ","
    t = re.sub(r"([(\[])\s+", r"\1", t)             # "( " -> "("
    t = re.sub(r"\s+([)\]])", r"\1", t)             # " )" -> ")"
    t = re.sub(r"\s+-\s+", "-", t)                  # "sir - an" -> "sir-an"
    t = re.sub(r"\s+'\s+", "'", t)                  # stray spaced apostrophes
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def _capitalize(t: str) -> str:
    # standalone "i" -> "I"
    t = re.sub(r"\bi\b", "I", t)
    t = re.sub(r"\bi'", "I'", t)
    # first alpha char of the string
    t = re.sub(r"^(\W*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)
    # first alpha after sentence-final punctuation
    def up(m): return m.group(1) + m.group(2).upper()
    t = re.sub(r"([.!?]\s+)([a-z])", up, t)
    return t

def truecase(text: str) -> str:
    return _capitalize(_fix_spacing(text))

if __name__ == "__main__":
    for s in [
        "i 'll not endure this language , sir , i have done more to serve you .",
        "that , is a subject which i can not discuss with any one , not even you .",
        "wo n't you speak plainly ? i do n't take your meaning , sir - an old man .",
        "surely one can hardly discuss such a matter .",
    ]:
        print(repr(s), "->", repr(truecase(s)))

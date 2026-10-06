#!/usr/bin/env python3
"""
reclassify.py — correct obviously-wrong source sub_type labels before retrieval.

The source prompt file (behavioral_sft_adversarial_prompts_for_completion_cleaned.jsonl)
has label noise: foreign-script prompts (e.g. "你会说英语吗", "Ты меня понимаешь?" --
illustrative examples; real log text not reproduced), keysmash, and explicit
jailbreak text are frequently filed as abuse_modern/abuse_period. Routing those through
the ABUSE seeds/rubric yields wrong-behaviour pairs (a polite Russian greeting got a
prison threat). This module overrides the label ONLY on high-confidence signals, and
NEVER routes anything INTO abuse -- it only pulls mislabeled prompts out.

route(text, src) -> corrected sub_type (may equal src).
"""
from __future__ import annotations
import re, unicodedata

_FOREIGN_TAGS = ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "CYRILLIC",
                 "ARABIC", "GREEK", "HEBREW", "DEVANAGARI", "THAI")

def _has_foreign_script(s: str) -> bool:
    for c in s:
        try:
            n = unicodedata.name(c)
        except ValueError:
            continue
        if any(t in n for t in _FOREIGN_TAGS):
            return True
    return False

def _non_ascii_letter_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    return sum(1 for c in letters if ord(c) > 127) / len(letters) if letters else 0.0

_JAILBREAK = re.compile(
    r"ignore (all|previous|your|the)|you are now|forget (your|all|everything)|"
    r"system prompt|jail\s?break|\bDAN\b|pretend (you|to be)|no (rules|restrictions|filter|limits)|"
    r"do anything now|bypass|new instructions|repeat (this|these|after|it|endlessly|the)|"
    r"disregard|override your", re.I)

def _is_gibberish(s: str) -> bool:
    t = s.strip()
    if not re.search(r"[a-zA-ZÀ-ÿ]", t):
        return True                                  # pure punctuation / digits / symbols
    words = re.findall(r"[a-zA-Z]+", t)
    if len(words) == 1 and len(words[0]) >= 12:
        return True                                  # single long keysmash token
    tot = sum(len(w) for w in words)
    if tot >= 10:
        vowels = sum(1 for w in words for c in w.lower() if c in "aeiou")
        if vowels / tot < 0.15:                      # implausibly consonant-heavy
            return True
    return False

def route(text: str, src: str) -> str:
    # foreign script / heavily non-ASCII -> a language problem, not abuse
    if _has_foreign_script(text) or _non_ascii_letter_ratio(text) > 0.4:
        return "non_english"
    # explicit instruction-hijack text
    if _JAILBREAK.search(text):
        return "jailbreak_attempt"
    # keysmash / non-lexical
    if _is_gibberish(text):
        return "gibberish"
    return src

if __name__ == "__main__":
    import json, sys, collections
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "../behavioral_sft_adversarial_prompts_for_completion_cleaned.jsonl"
    changed = collections.Counter()
    n = 0
    for line in open(path):
        r = json.loads(line); n += 1
        new = route(r["human"], r["sub_type"])
        if new != r["sub_type"]:
            changed[(r["sub_type"], new)] += 1
    print(f"{n} prompts; {sum(changed.values())} relabeled")
    for (a, b), c in changed.most_common():
        print(f"  {a:18s} -> {b:18s} {c}")

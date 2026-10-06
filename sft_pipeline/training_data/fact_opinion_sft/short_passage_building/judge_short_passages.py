#!/usr/bin/env python3
"""
judge_short_passages.py -- Stage 2 of the short-answer coverage fix: standalone coherence judge
for mine_short_passages.py's raw output, BEFORE any pairing with a real question.

Mechanical extraction (passage_utils' starts_cleanly/ends_cleanly/is_self_contained) already ran
at mine time, but that only checks sentence-boundary mechanics, not whether the passage actually
reads as a complete, substantive, worthwhile thought on its own -- a passage can start and end on
real sentence boundaries and still be a throwaway fragment of a longer argument (the exact class
of defect found in the ceiling-trim work: grammatically valid, rhetorically arbitrary).

This is a coherence-only gate in the sense that relevance to any particular question isn't judged
here (that's stage 3) -- but register (personal voice) IS checked here too, deliberately
overlapping with stage 3's VOICE criterion: filtering for it early keeps the pool itself clean and
means stage 3 isn't spending retrieval slots on candidates that could never pass anyway. This is
the same "check it more than once" pattern already used elsewhere in this project (the running-
header regex backstop below mirrors the one added to judge_persona_hathi.py after it slipped past
the LLM once in calibration).

Also encodes two more session findings directly into the rubric:
  - "the reader may note..." style narratorial asides (a THIRD-PARTY narrator addressing an
    implied reader) are a register failure, not just a style quirk -- they read as expository
    essay voice, the opposite of the first-person personable register this whole pipeline is
    chasing. Both an explicit judge instruction AND a mechanical regex backstop, same treatment
    as the running-header pattern.
  - mid-line page numbers / running-head numerals glued into passage text (a numeric OCR column
    artifact, distinct from the earlier alphabetic running-header defect) are an explicit judge
    instruction; not mechanized (unlike running-headers, a bare-number regex has too high a false
    positive rate against legitimate dates/measurements/quantities in period prose to be safe).

  python3 judge_short_passages.py --backend mlx --limit 20   # calibration
  python3 judge_short_passages.py --backend mlx              # full run
"""
from __future__ import annotations
import json, re, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
IN_PATH = HERE / "mined_short_passages.jsonl"
OUT_PATH = HERE / "short_passage_judgments.jsonl"
DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-32B-Instruct-4bit"

JUDGE_TMPL = """A short passage was mechanically extracted from a 19th-century book or periodical
(already verified to start and end on real sentence boundaries).

Passage: "{passage}"

Judge whether this is a GOOD standalone passage, meaning ALL of:
  1. It expresses a complete, self-contained thought -- not an arbitrary fragment of a longer
     argument that happens to start/end on grammatical sentence boundaries (e.g. a mid-argument
     sentence that assumes context from before or promises more to come is NOT complete, even if
     each individual sentence is grammatically whole).
  2. The OCR is clean enough to read naturally -- no confusing garbled fragments, no running-header
     or title fragments glued into the text (whether mid-sentence or prepended at the very start,
     e.g. a passage opening "THE FARMER'S AND\nThese three gases..." has a page-header fragment
     stitched onto real content and must be rejected), and no mid-line page numbers or numeral
     fragments interrupting the sentence (e.g. a stray "42" or "128" embedded where it doesn't
     belong -- a genuine date, measurement, or quantity written out in context is fine, a bare
     page-number intrusion is not). Also reject passages where two unrelated sentences appear
     spliced together mid-passage (a column-reading-order OCR artifact) -- if a passage only makes
     sense as two independent, unrelated statements awkwardly joined, it fails this criterion even
     though each half may be individually well-formed.
  3. It is substantive -- an actual statement, argument, description, or opinion, not throat-
     clearing, a bare transition, or a list fragment with no real content.
  4. It is written in an engaging, FIRST-PERSON, personable register: a person arguing, reflecting,
     recounting their own experience, or giving their own opinion. REJECT dry third-person
     exposition, and REJECT narratorial asides that address an implied reader directly -- "the
     reader may note...", "as will be observed...", "it should be remembered that..." -- these are
     essayistic narrator-to-audience voice, not personable first-person voice, even when the
     surrounding prose is otherwise first-person.

Do not judge relevance to any particular real-world question -- only whether the passage stands on
its own in register and substance.

Reply with ONLY JSON: {{"verdict": "KEEP" or "DROP", "why": "<one short phrase>"}}"""

_RUNNING_HEADER_RE = re.compile(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,}){1,}-\w")

# Reader-address narratorial asides -- third-party essay voice, the opposite of the personable
# first-person register this pipeline is chasing. Mechanical backstop for the judge instruction
# above, same rationale as the running-header override: catch the ones an LLM call slips on.
_READER_ADDRESS_RE = re.compile(
    r"\bthe reader (?:may|will|should|need only)\b"
    r"|\bas (?:the reader |we )?(?:will|shall|may) (?:be )?"
    r"(?:see|observe|note|find|seen|observed|noted|found)\b"
    r"|\bit (?:will|should) be (?:observed|noted|remembered|remarked)\b"
    r"|\bwe shall (?:see|now (?:consider|turn|examine))\b",
    re.IGNORECASE,
)


def has_header_artifact(text: str) -> bool:
    return bool(_RUNNING_HEADER_RE.search(text))


def has_reader_address(text: str) -> bool:
    return bool(_READER_ADDRESS_RE.search(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=IN_PATH)
    ap.add_argument("--output", type=Path, default=OUT_PATH)
    ap.add_argument("--backend", choices=["mlx"], default="mlx")
    ap.add_argument("--model", default=DEFAULT_MLX_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.input.open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} candidates to judge")

    from mlx_lm import load, generate
    print(f"loading judge {args.model} ...")
    jm, jt = load(args.model)

    def judge(prompt_text):
        out = generate(jm, jt, prompt=jt.apply_chat_template(
            [{"role": "user", "content": prompt_text}], add_generation_prompt=True, tokenize=False
        ), max_tokens=80, verbose=False)
        m = re.search(r"\{.*\}", out, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
            return str(d.get("verdict", "")).upper().strip(), str(d.get("why", ""))[:120]
        except Exception:
            return "", "parse-fail"

    n_keep = n_drop = n_fail = 0
    results = []
    for i, r in enumerate(rows):
        prompt = JUDGE_TMPL.format(passage=r["passage"])
        verdict, why = judge(prompt)
        if verdict not in ("KEEP", "DROP"):
            n_fail += 1
            verdict = "DROP"
        elif verdict == "KEEP" and has_header_artifact(r["passage"]):
            verdict, why = "DROP", "running-header artifact (mechanical override)"
        elif verdict == "KEEP" and has_reader_address(r["passage"]):
            verdict, why = "DROP", "reader-address narratorial aside (mechanical override)"
        if verdict == "KEEP":
            n_keep += 1
        else:
            n_drop += 1
        results.append({**r, "verdict": verdict, "why": why})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}  keep={n_keep} drop={n_drop} fail={n_fail}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone: {n_keep} keep, {n_drop} drop, {n_fail} parse-failures (treated as drop)")
    print(f"wrote -> {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gen_prompts_for_passages.py -- salvages the short passages that failed stage 4's independent
relevance check (retrieve-and-rerank couldn't find a genuinely on-topic REAL question for them in
a 1,002-passage pool that's too topically narrow to cover the full range of real usage-log
questions) by flipping the direction: instead of finding a real question that fits the passage,
write a natural question FOR the passage, then round-trip-verify the (question, real passage)
pair the same way stage 4 already does. Question generation guarantees topical fit by
construction; the round-trip check catches generation drift/slop, same failure mode as before,
different cause.

This is the SAME classic round-trip-consistency pattern already used for
persona_gutenberg_dialogue_selects.jsonl -> behavioral_sft_persona_prompted.jsonl (write a prompt
for a real passage, verify it holds together), applied here to the short-passage pool's leftovers
instead of discarding them.

Also usable as a direct stage-2 -> stage-4 shortcut (--source keep_all --coherence-file <path>)
that SKIPS retrieve-and-rerank (stage 3) entirely -- generation guarantees topical fit by
construction, so for a fresh coherence-judged batch that never went through retrieval there's
nothing for retrieval to have failed at; every KEEP verdict is eligible for generation directly.

Bimodal question phrasing (2026-08-10): real usage-log prompts (usage_data_intent_labels.jsonl,
factual_query/opinion_solicitation/task_request) are NOT one register -- a majority terse,
single-clause cluster (~58.5% of turns, median ~10 tok) plus a real minority multi-clause tail
(clause_mean of that tail ~3.7, reaching p90 ~28-39 tok), where the OLD single GEN_TMPL here only
ever produced the former (one flat sentence, every time -- confirmed near-zero variety when
measured against fact_op_sft_eb_4000.natural.jsonl's prompts, which came from an earlier version
of this same one-template approach). Replaced with two templates (TERSE_TMPL / COMPLEX_TMPL) with
concrete token/clause targets pulled directly from those real numbers, mixed via --terse-fraction
(default 0.585 = the measured real 1-clause rate) and nudged by each passage's mine-time "band"
(short passages lean slightly more terse, long passages lean slightly more complex -- the same
band<->complexity correlation rationale documented in mine_short_passages.py, not a hard rule).

Output schema matches judge_short_passage_pairs.py's expected input exactly ({"prompt",
"victorian", ...}), so the existing round-trip judge is reused unchanged -- no new judge script.
Extra provenance fields ("style", "target_band") are additive and ignored by that judge.

  python3 gen_prompts_for_passages.py --backend mlx --limit 20   # calibration
  python3 gen_prompts_for_passages.py --backend mlx              # full run
  python3 judge_short_passage_pairs.py --backend mlx \\
      --input generated_prompts_for_failed_passages.jsonl \\
      --output generated_prompts_pair_judgments.jsonl
"""
from __future__ import annotations
import json, re, argparse, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
IN_PATH = HERE / "short_passage_pair_judgments.jsonl"
COHERENCE_JUDGMENTS_PATH = HERE / "short_passage_judgments.jsonl"
RETRIEVED_PATH = HERE / "retrieved_short_passages.jsonl"
OUT_PATH = HERE / "generated_prompts_for_failed_passages.jsonl"
DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-32B-Instruct-4bit"

# Real-usage-log-derived mix: 58.5% single-clause across factual_query/opinion_solicitation/
# task_request (usage_data_intent_labels.jsonl, outlier-cleaned <=200 tok). See module docstring.
DEFAULT_TERSE_FRACTION = 0.585
BAND_CORRELATION_SHIFT = 0.15  # short band -> more terse, long band -> more complex, soft prior

TERSE_TMPL = """Here is a genuine passage from a 19th-century book or periodical:

"{passage}"

Write ONE short, plain, single-clause question that a modern user might dash off to a
Victorian-era chatbot, such that this passage would be a genuine, direct answer to it.
Requirements:
  - Roughly 5-12 words. No subordinate clauses, no compound sentences, no comma-separated asides
    -- a single flat clause only, the way someone fires off a quick question without much framing.
  - Do NOT quote or closely paraphrase specific wording from the passage -- the question must read
    as an independent, natural question on the passage's general subject, not a leading question
    built backward from the answer's exact phrasing.
  - Do not mention "the passage" or "the text" -- write as if genuinely asking the question fresh.

Reply with ONLY the question text, nothing else -- no quotation marks, no preamble."""

COMPLEX_TMPL = """Here is a genuine passage from a 19th-century book or periodical:

"{passage}"

Write ONE longer, more elaborate question that a modern user might ask a Victorian-era chatbot,
such that this passage would be a genuine, direct answer to it. Requirements:
  - Roughly 20-40 words, and syntactically complex -- include at least two clause-boundary markers
    (e.g. a subordinate clause giving context or motivation before the actual ask, a
    comma-separated aside, or a genuine two-part question), the way a more engaged or verbose user
    frames a question with some context or a follow-on part attached.
  - Still a single, coherent question (or a tightly linked two-part one) -- not a rambling list of
    unrelated asks.
  - Do NOT quote or closely paraphrase specific wording from the passage -- the question must read
    as an independent, natural question on the passage's general subject, not a leading question
    built backward from the answer's exact phrasing.
  - Do not mention "the passage" or "the text" -- write as if genuinely asking the question fresh.

Reply with ONLY the question text, nothing else -- no quotation marks, no preamble."""

_CLAUSE_MARKERS = re.compile(
    r",|;|\b(and|but|or|nor|so|yet|because|since|although|though|while|whereas|"
    r"if|unless|when|whenever|that|which|who|whom|whose|as)\b",
    re.IGNORECASE,
)


def _clause_count(text: str) -> int:
    return 1 + len(_CLAUSE_MARKERS.findall(text))


def pick_style(band: str, terse_fraction: float, rng: random.Random) -> str:
    p_terse = terse_fraction
    if band == "short":
        p_terse = min(0.95, p_terse + BAND_CORRELATION_SHIFT)
    elif band == "long":
        p_terse = max(0.05, p_terse - BAND_CORRELATION_SHIFT)
    return "terse" if rng.random() < p_terse else "complex"


def load_failed_passages() -> list[dict]:
    rows = [json.loads(l) for l in IN_PATH.open(encoding="utf-8") if l.strip()]
    seen = set()
    out = []
    for r in rows:
        if r.get("verdict") != "FAIL":
            continue
        key = r["victorian"]
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def load_unused_passages(coherence_file: Path = COHERENCE_JUDGMENTS_PATH) -> list[dict]:
    """Coherence-approved (stage 2 KEEP) passages that stage 3's retrieval never even selected --
    generation doesn't need a prior real-question match to work, so there's no reason to leave
    these untouched just because retrieval never picked them."""
    kept = [json.loads(l) for l in coherence_file.open(encoding="utf-8") if l.strip()]
    kept = [r for r in kept if r.get("verdict") == "KEEP"]
    used = {json.loads(l)["victorian"] for l in RETRIEVED_PATH.open(encoding="utf-8") if l.strip()}
    return [r for r in kept if r["passage"] not in used]


def load_keep_all(coherence_file: Path) -> list[dict]:
    """Every stage-2 KEEP verdict in coherence_file, no stage-3 'used' subtraction -- for a batch
    that's skipping retrieve-and-rerank entirely, there's no retrieval output to subtract."""
    kept = [json.loads(l) for l in coherence_file.open(encoding="utf-8") if l.strip()]
    return [r for r in kept if r.get("verdict") == "KEEP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mlx"], default="mlx")
    ap.add_argument("--model", default=DEFAULT_MLX_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--source", choices=["failed", "unused", "both", "keep_all"], default="both",
                     help="'failed' = stage-4 rejects (retrieval found them, real question didn't "
                          "fit); 'unused' = coherence-approved passages retrieval never selected "
                          "at all; 'both' = union of the two (default); 'keep_all' = every "
                          "coherence KEEP in --coherence-file, bypassing retrieve-and-rerank "
                          "entirely (for a fresh batch that never went through stage 3)")
    ap.add_argument("--coherence-file", type=Path, default=COHERENCE_JUDGMENTS_PATH,
                     help="stage-2 judgments file to read KEEP verdicts from (used by "
                          "--source unused and --source keep_all)")
    ap.add_argument("--terse-fraction", type=float, default=DEFAULT_TERSE_FRACTION,
                     help="target fraction of generated prompts in the terse/single-clause style "
                          f"(default {DEFAULT_TERSE_FRACTION} = measured real usage-log rate); "
                          "remainder generated in the longer/multi-clause complex style")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    passages = []
    seen_text = set()
    if args.source in ("failed", "both"):
        for r in load_failed_passages():
            if r["victorian"] not in seen_text:
                seen_text.add(r["victorian"])
                passages.append(r)
    if args.source in ("unused", "both"):
        for r in load_unused_passages(args.coherence_file):
            key = r["passage"]
            if key not in seen_text:
                seen_text.add(key)
                # normalize field name -- load_failed_passages() rows use "victorian",
                # load_unused_passages() rows (from short_passage_judgments.jsonl) use "passage"
                passages.append({**r, "victorian": r["passage"]})
    if args.source == "keep_all":
        for r in load_keep_all(args.coherence_file):
            key = r["passage"]
            if key not in seen_text:
                seen_text.add(key)
                passages.append({**r, "victorian": r["passage"]})
    if args.limit:
        passages = passages[: args.limit]
    print(f"{len(passages)} distinct passages to generate prompts for (source={args.source})")

    for p in passages:
        p["_style"] = pick_style(p.get("band", ""), args.terse_fraction, rng)
    n_terse = sum(1 for p in passages if p["_style"] == "terse")
    print(f"Style mix: {n_terse} terse ({100*n_terse/len(passages):.1f}%), "
          f"{len(passages)-n_terse} complex ({100*(len(passages)-n_terse)/len(passages):.1f}%)")

    from mlx_lm import load, generate
    print(f"loading {args.model} ...")
    m, t = load(args.model)

    def gen(passage: str, style: str) -> str:
        tmpl = TERSE_TMPL if style == "terse" else COMPLEX_TMPL
        max_tok = 40 if style == "terse" else 90
        out = generate(m, t, prompt=t.apply_chat_template(
            [{"role": "user", "content": tmpl.format(passage=passage)}],
            add_generation_prompt=True, tokenize=False
        ), max_tokens=max_tok, verbose=False)
        q = out.strip().strip('"').strip()
        q = re.sub(r"^(question:|q:)\s*", "", q, flags=re.I)
        return q

    results = []
    for i, r in enumerate(passages):
        question = gen(r["victorian"], r["_style"])
        results.append({
            "prompt": question,
            "src_intent": "generated",
            "style": r["_style"],
            "target_band": r.get("band", ""),
            "victorian": r["victorian"],
            "source": r.get("source", ""),
            "book_title": r.get("book_title", ""), "book_author": r.get("book_author", ""),
            "book_year": r.get("book_year", ""),
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(passages)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(results)} generated (prompt, passage) candidates -> {args.output}")
    print("next: python3 judge_short_passage_pairs.py --backend mlx "
          f"--input {args.output} --output generated_prompts_pair_judgments.jsonl")


if __name__ == "__main__":
    main()

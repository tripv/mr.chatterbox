#!/usr/bin/env python3
"""sonnet_fact_op_pipeline.py — TWO-STAGE Sonnet pipeline for fact_op_sft_corpus_pulls_merged
.jsonl and multiturn_sft_fact_op_corpus_pulls.jsonl, replacing both the local-Qwen rewrite
(rewrite_prompts_natural.py) and the thin single-stage first draft of this script. Per Trip
(2026-09-02): that first draft was too thin -- it only rewrote QUESTIONS, but the more basic
defect is that many of the PASSAGES sourced into these files are mid-passage extracts that
don't work as standalone replies at all, no rewrite of the question fixes that. Needed a real
two-stage split, same shape as sonnet_epistolary_pipeline.py:

STAGE 1 -- PASSAGE GATE, template = gen_teacher_source4.py's GENERATOR_SYSTEM_PROMPT reject
criteria (that script already solved exactly this problem for a different file, with few-shots
validated against real measured failure modes -- reuse, don't reinvent):
  REJECT "mid_scene_fiction"  -- characters/events/locations referenced but never defined
                                 within the passage itself.
  REJECT "anaphoric_opening"  -- opens with an unresolvable reference ("the signals used",
                                 "these sentiments") with no antecedent in the passage.
  REJECT "dangling_pronoun"   -- "he"/"she"/"they" as the primary subject with no named
                                 referent anywhere in the passage.
  REJECT "ocr_garbage"        -- dominated by garbled characters, tables, non-prose.
  KEEP -- second-person address ("you", "your letter") is NOT a defect; the reader IS who
          the Victorian is addressing. Measured on the epistolary rebuild (2026-08-22):
          rejecting second-person address held the pass rate to 4%; the correct rule (accept
          second-person, keep rejecting third-party anaphora) raised it to 29%.

STAGE 2 -- QUESTION REWRITE, template = sonnet_epistolary_pipeline.py's REWRITE_TMPL shape:
self-checking JSON output, mechanical safety nets applied ON TOP of instructions (the
epistolary script's own hard-won lesson: instructions alone left 19.9% em-dash usage and 2
author-name vocatives despite explicit rules against both). This job's version differs from
epistolary's in one way: epistolary WRITES a brand-new prior turn with no existing question;
here an existing (templated) question already exists and already targets roughly the right
topic, so the job is REPHRASE-preserving-intent (rewrite_prompts_natural.py's INTENT_RULE /
FORM_INSTRUCTION machinery, reused as-is -- that logic was never the problem) plus a NEW
fit_check confirming the gate-surviving passage still actually answers the rewritten phrasing.
Two mechanical nets added after reviewing 100 local-Qwen rewrites of this exact file (Trip,
2026-09-02): non-Latin-script reject+retry (Qwen spliced Chinese into 1/185 rows past a filter
that never checked language), and a live opener-reuse cap (Qwen's "tell me about/how/what..."
was 46% of all rewrites -- worse concentration than the original templates it was fixing).

  python3 sonnet_fact_op_pipeline.py --dry-run-gate
  python3 sonnet_fact_op_pipeline.py --dry-run-rewrite
  python3 sonnet_fact_op_pipeline.py --run --input ../fact_op_sft_corpus_pulls_merged.jsonl --register general
  python3 sonnet_fact_op_pipeline.py --run --input ../../multi_turn_sft/multiturn_sft_fact_op_corpus_pulls.jsonl --register general
"""
from __future__ import annotations
import json, re, sys, os, time, argparse, random, collections, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIC = HERE.parent.parent.parent  # training_data/fact_opinion_sft/scripts -> victorian/

sys.path.insert(0, str(VIC))
from rewrite_prompts_natural import (  # noqa: E402  (reuse, don't reinvent -- see docstring)
    load_exemplar_pools, FORM_INSTRUCTION, FORM_INSTRUCTION_FACTUAL,
    INTENT_RULE, FORM_MIX, pick_form,
)

sys.path.insert(0, str(VIC / "training_data/multi_turn_sft/scripts"))
from sonnet_epistolary_pipeline import (  # noqa: E402  (reuse, don't reinvent)
    load_real_word_counts, sample_target_words,
)

sys.path.insert(0, str(VIC / "judge_scripts"))
from judge_behavioral_v3 import make_judge  # noqa: E402


# =============================================================================
# STAGE 1 -- PASSAGE GATE (template: gen_teacher_source4.py GENERATOR_SYSTEM_PROMPT)
# =============================================================================

GATE_TMPL = """You are screening a passage of real Victorian prose to decide if it is usable,
BY ITSELF, as the answer side of a chatbot training example -- independent of whatever
question it is currently paired with.

REJECT "mid_scene_fiction" if characters, events, or locations are referenced but never
defined within the passage itself -- a listener dropped into the middle of a scene they did
not witness.

REJECT "anaphoric_opening" if the passage opens with a reference a listener cannot resolve
("the signals used", "these sentiments", "the proposed regulations") without context that
exists nowhere in the passage.

REJECT "dangling_pronoun" if "he", "she", or "they" appears as the PRIMARY SUBJECT with no
named referent anywhere in the passage.

REJECT "ocr_garbage" if the text is dominated by garbled characters, tables, or non-prose noise.

REJECT "gazetteer_heading_leakage" if the passage still carries the ORIGINAL document's
structural furniture baked into the running prose -- a magazine Q&A column's heading/title
fragment, a reader's signature (initials, town, "A. W., Wichita, Kas."), a dictionary/gazetteer
headword, or citation numbering -- rather than reading as continuous natural prose. This is
DIFFERENT from ocr_garbage: the text itself is clean and legible, the problem is that a piece
of the source document's layout (not narration) got swept into the passage.

SECOND-PERSON ADDRESS IS NOT A DEFECT. A passage saying "you", "your letter", "as you say", "I
cannot agree with your view" resolves perfectly well -- the reader IS the person being
addressed. Do not reject for second-person address alone; keep rejecting THIRD-PARTY anaphora
(an undefined "he"/"she"/"they" who is not the reader).

EXAMPLES:

Passage:
  "We never should have heard of the letter at all, I believe, unless I had asked for it
  myself. Do you call that confidence, my love, towards Doctor Strong? I am surprised... 'Now
  let us see,' said Mrs. Markleham, putting her glass to her eye, 'where the passage is...'"
-> {{"passage_ok": false, "reason": "mid_scene_fiction", "note": "Mrs. Markleham, Doctor Strong, and Annie are all undefined for a listener who was not present at the scene."}}

Passage:
  "In order to transmit messages along the telegraphic wires, a certain amount of electric
  power is required, and this force must be increased in proportion to the length of the
  wire... According to some accurate experiments made by M. Pouillet, a distinguished member
  of the Institute of France, a voltaic battery of ten pairs of plates will supply electricity
  sufficient for telegraphic purposes..."
-> {{"passage_ok": true, "reason": "ok", "note": "Self-contained technical exposition; no prior context required."}}

Passage:
  "The signals used were either large letters of the alphabet, or figures of various shapes to
  represent them... But though Hook's invention and Amontous's modification were published all
  over Europe... they were not practically applied to any useful purpose until the French
  revolution."
-> {{"passage_ok": false, "reason": "anaphoric_opening", "note": "'The signals used' and 'Hook's invention' are never defined within the passage."}}

Passage:
  "To relieve young girls from these dangers, to rectify these conditions, is simply, under
  God, to save them from falling. It is not affirmed we can touch all cases where evil is
  deeper rooted..."
-> {{"passage_ok": false, "reason": "anaphoric_opening", "note": "'These dangers' and 'these conditions' are never defined -- the passage opens mid-argument, referring back to a claim made before this extraction window began."}}

Passage:
  "IL] BOTANY. 223 _. , Dr Livingstone believes this to be a fibrous Jouaze. , . plant of great
  value... -jq-u_ yom;ca) producing strychnia, flourishes abundantly on tbe Leeambye..."
-> {{"passage_ok": false, "reason": "ocr_garbage", "note": "Dominated by OCR noise and fused/garbled fragments throughout."}}

Passage:
  "Box-Elder or Ash-leaved Maple.- 66 A. W.,\" Wichita, Kas. The seeds of this ripen about the
  same time as those of the Sugar Maple, i. e. in October. Much disappointment arises in
  growing this tree from seeds..."
-> {{"passage_ok": false, "reason": "gazetteer_heading_leakage", "note": "Opens with a magazine Q&A column's title and a reader's initials/town signature ('66 A. W., Wichita, Kas.') baked into the prose -- document furniture, not narration."}}

Passage:
  "My very dear Friend,--You are kind to exceeding kindness, and I am as grateful as any of
  your long-ago kind invitations ever found me... You may be perfectly sure that no Claypon,
  though he should live in Arcadia, would be preferred by me to you as a host..."
-> {{"passage_ok": true, "reason": "ok", "note": "Second-person correspondence; 'Claypon' is a decorative, non-load-bearing comparison, not required for comprehension."}}

NOW DECIDE THIS ONE:

Passage:
{passage}

Reply with ONLY valid JSON:
{{"passage_ok": true|false, "reason": "mid_scene_fiction"|"anaphoric_opening"|"dangling_pronoun"|"ocr_garbage"|"gazetteer_heading_leakage"|"ok", "note": "<one short sentence>"}}"""


# =============================================================================
# STAGE 2 -- QUESTION REWRITE (template: sonnet_epistolary_pipeline.py REWRITE_TMPL)
# =============================================================================

EMDASH_RE = re.compile(r"\s*(--|—)\s*")
NON_LATIN_RE = re.compile(r"[一-鿿぀-ヿ가-힯Ѐ-ӿ؀-ۿ]")  # CJK/Hangul/Cyrillic/Arabic


def opener_frame(t: str) -> str:
    words = re.sub(r"[^a-z0-9' ]", "", t.lower()).split()
    return " ".join(words[:3])


def clean_output(raw: str) -> str:
    t = raw.strip().splitlines()[0] if raw.strip() else ""
    t = re.sub(r'^["\'\s]+|["\'\s]+$', "", t)
    t = re.sub(r"^(rewritten:?\s*)", "", t, flags=re.I).strip().strip("\"'")
    t, _ = EMDASH_RE.subn(", ", t)
    return t


# 2026-09-04 (Trip, reviewing multiturn_sft_fact_op_corpus_pulls.jsonl before its rebuild):
# real example found -- ORIGINAL user turn "I recently witnessed a play where the set was so
# intricate, it overshadowed the actual story -- a heavenly scene with flaws that detract from
# the illusion," paired with a passage that reveals exactly that (a heaven-set with "certain
# blemishes"). The ORIGINAL question already narrates the passage's own content, so it's
# CAUSALLY BACKWARDS -- the passage becomes an echo, not a reveal. The old "preserve exact
# intent, only change HOW it's phrased" framing would just rephrase this same defect forever,
# since intent-preservation was never checked against causal ordering. Added an explicit
# override (rule 0 below) plus a causal_check field, same mechanism sonnet_epistolary_pipeline
# .py already uses for its own (structurally identical) REWRITE_TMPL. Also added real-length
# sampling (target_words, reusing sample_target_words()) -- unconstrained length drifts long.
REWRITE_TMPL = """You rewrite an EXISTING chatbot-training question into natural, varied
phrasing. You are NOT answering it, and NOT writing a new question from scratch -- the
ORIGINAL targets the right general SUBJECT; your job is HOW it's phrased, subject to one
override below.

PASSAGE (the real Victorian answer this question is paired with -- it will not change; your
rewritten question must still be something this passage plausibly answers):
{passage}

ORIGINAL (templated) question:
  "{original}"

0. CAUSAL OVERRIDE (checks before anything else): the passage has NOT been said yet from the
   question's point of view. If the ORIGINAL question already reveals, narrates, or pre-states
   something that only the passage itself establishes -- a specific outcome, judgment, detail,
   or claim -- do NOT preserve that specific framing, even though rule 1 below normally asks
   you to keep the same intent. Ask about the same general subject instead, without stating
   what the passage concludes. Example: original "I recently witnessed a play where the set
   was so intricate, it overshadowed the actual story -- a heavenly scene with flaws that
   detract from the illusion" paired with a passage revealing exactly those flaws -- rewrite
   toward something like "Have you seen any theatrical sets that tried to depict heaven?",
   which asks about the SAME subject without pre-stating the passage's "blemishes" verdict.

Here are illustrative examples of how users typically phrase their messages (style reference
only -- do NOT borrow their topics):
{exemplars}

Rules:
- {intent_rule}
- {form_instruction}
- The PASSAGE above must still plausibly answer your rewritten question. Do not introduce a
  new angle, sub-question, or specific detail the passage doesn't actually address.
- Write ONLY in English. No foreign-language words or characters of any kind, even mixed into
  an otherwise-English sentence.
- Make it sound natural and human -- real people are often terse and direct, not verbose or
  ornate.
- LENGTH: write it in approximately {target_words} words. Real users mostly type short,
  direct messages -- do not pad it out with extra clauses to sound more sophisticated.
{banned_block}
- Do not add new facts, names, or constraints beyond what the original question already asked.

Reply with ONLY valid JSON:
{{"rewritten": "<the rewritten question, one message only>",
  "opener": "<the first 3 words of your rewritten question, lowercase>",
  "causal_check": "<confirm your rewritten question does not reveal, agree with, or pre-state any specific claim/outcome/detail that appears ONLY in the passage below -- name the passage's key claim(s) and confirm your question is independent of them>",
  "fit_check": "<confirm the passage above still plausibly answers your rewritten question -- name the passage's actual topic and confirm your question matches it>",
  "checks": "<confirm (a) English only, (b) same general subject as the original (unless rule 0 required generalizing away a pre-revealed specific), (c) your opener is not one of the banned openers listed above, if any were listed>"}}"""


def build_banned_block(banned: list[str]) -> str:
    if not banned:
        return ""
    listed = ", ".join(f'"{b}..."' for b in banned)
    return (f"- Your rewrite has been generated many times before starting with one of these "
            f"exact openers -- they are now BANNED, pick a genuinely different opening: "
            f"{listed}.")


# =============================================================================
# Shared plumbing
# =============================================================================

def call_json(run, prompt, retries=3):
    """Same defensive shape as sonnet_epistolary_pipeline.call_json: auth/permission errors
    are fatal and never retried; transient errors and unparseable JSON get bounded retries."""
    try:
        import anthropic
        FATAL = (anthropic.AuthenticationError, anthropic.PermissionDeniedError)
    except ImportError:
        FATAL = ()
    last = None
    for attempt in range(retries):
        try:
            raw = run(prompt)
        except FATAL:
            raise SystemExit(
                "!! Anthropic authentication/permission error -- check ANTHROPIC_API_KEY. "
                "Refusing to retry or continue.")
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
            continue
        m = re.search(r"\{.*\}", raw or "", re.S)
        if not m:
            last = f"no JSON in response: {raw!r}"
            time.sleep(2 * (attempt + 1))
            continue
        try:
            return json.loads(m.group(0))
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    return None


def gate_passage(run, passage: str) -> dict | None:
    return call_json(run, GATE_TMPL.format(passage=passage))


def rewrite_question(run, original, passage, pools, form, k, rng, register, opener_counts,
                      total_processed, real_word_counts, max_attempts=4):
    """Returns (text, meta) or (None, meta) on exhausted attempts."""
    fi = (FORM_INSTRUCTION_FACTUAL if register == "factual" else FORM_INSTRUCTION)[form]
    pool = pools[{"command": "command", "casual": "command",
                  "question": "question", "polite": "question"}[form]]
    ex = "\n".join(f"  - {e}" for e in rng.sample(pool, min(k, len(pool))))
    intent_rule = INTENT_RULE[register]
    target = sample_target_words(rng, real_word_counts)

    # proportional cap: no single 3-word opener may exceed ~5% of rows processed so far.
    cap = max(3, math.ceil(0.05 * max(total_processed, 1)))
    banned = [f for f, c in opener_counts.items() if c >= cap]

    meta = {"attempts": [], "form": form, "target_words": target}
    for attempt in range(max_attempts):
        prompt = REWRITE_TMPL.format(
            passage=passage, original=original, exemplars=ex, intent_rule=intent_rule,
            form_instruction=fi, target_words=target, banned_block=build_banned_block(banned))
        v = call_json(run, prompt)
        if v is None or not v.get("rewritten"):
            meta["attempts"].append({"ok": False, "reason": "call_or_parse_failed"})
            continue
        text = clean_output(v["rewritten"])
        if not text or not (2 <= len(text.split()) <= 60) or text.lower() == original.lower():
            meta["attempts"].append({"ok": False, "reason": "length_or_identity", "raw": text})
            continue
        if NON_LATIN_RE.search(text):
            meta["attempts"].append({"ok": False, "reason": "non_latin_script", "raw": text})
            continue
        frame = opener_frame(text)
        if frame in banned:
            meta["attempts"].append({"ok": False, "reason": "opener_cap", "frame": frame})
            banned = list(set(banned) | {frame})
            continue
        meta["attempts"].append({"ok": True})
        meta["causal_check"] = v.get("causal_check", "")
        meta["fit_check"] = v.get("fit_check", "")
        meta["checks"] = v.get("checks", "")
        meta["opener_frame"] = frame
        return text, meta
    return None, meta


# ---- record shape handling: normalize to a list of (q_setter_pair) exchanges ----
def exchanges_of(rec):
    """Yield (user_dict_or_None, assistant_content, setter) triples covering both this
    project's flat {"human","victorian"} schema and the {"pair"/"turns"} multi-exchange
    schema, so one pipeline handles fact_op_sft_corpus_pulls_merged.jsonl AND
    multiturn_sft_fact_op_corpus_pulls.jsonl without a fork."""
    if isinstance(rec, dict) and isinstance(rec.get("human"), str):
        yield (rec["human"], rec.get("victorian", ""),
               lambda q: rec.__setitem__("human", q))
        return
    for key in ("pair", "turns"):
        seq = rec.get(key) if isinstance(rec, dict) else None
        if isinstance(seq, list):
            for i in range(0, len(seq) - 1, 2):
                u, a = seq[i], seq[i + 1]
                if u.get("role") == "user" and a.get("role") == "assistant":
                    yield (u["content"], a["content"],
                           lambda q, _u=u: _u.__setitem__("content", q))
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--intent", default="mixed", choices=["factual_query", "opinion_solicitation",
                                                            "task_request", "mixed"])
    ap.add_argument("--form-mix", default="balanced", choices=list(FORM_MIX))
    ap.add_argument("--register", default="general", choices=["general", "factual"])
    ap.add_argument("--exemplars-k", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run-gate", action="store_true")
    ap.add_argument("--dry-run-rewrite", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("!! ANTHROPIC_API_KEY is not set -- refusing to start.")

    run, model_name = make_judge("anthropic", None, max_tokens=1024)
    print(f"backend: {model_name}\n")

    if args.dry_run_gate:
        cases = [
            ("mid-scene fiction", "expect REJECT mid_scene_fiction",
             "We never should have heard of the letter at all, I believe, unless I had asked "
             "for it myself. Do you call that confidence, my love, towards Doctor Strong? "
             "'Now let us see,' said Mrs. Markleham, putting her glass to her eye."),
            ("telegraph exposition", "expect KEEP",
             "In order to transmit messages along the telegraphic wires, a certain amount of "
             "electric power is required, and this force must be increased in proportion to "
             "the length of the wire. According to experiments made by M. Pouillet, a voltaic "
             "battery of ten pairs of plates will supply electricity sufficient for "
             "telegraphic purposes by means of a copper wire 600 miles in length."),
            ("anaphoric opening", "expect REJECT anaphoric_opening",
             "The signals used were either large letters of the alphabet, or figures of "
             "various shapes to represent them. But though Hook's invention and Amontous's "
             "modification were published all over Europe, they were not practically applied "
             "until the French revolution."),
            ("second-person letter", "expect KEEP (second person is fine)",
             "You are kind to exceeding kindness, and I am as grateful as any of your "
             "long-ago kind invitations ever found me. You may be perfectly sure that no "
             "Claypon would be preferred by me to you as a host."),
            ("ocr garbage", "expect REJECT ocr_garbage",
             "IL] BOTANY. 223 _. , Dr Livingstone believes this to be a fibrous Jouaze. , . "
             "plant of great value -jq-u_ yom;ca) producing strychnia, flourishes abundantly "
             "on tbe Leeambye."),
        ]
        for label, expectation, passage in cases:
            v = gate_passage(run, passage)
            print(f"[{label}]  {expectation}\n  -> {v}\n")
        return

    if args.dry_run_rewrite:
        pools = load_exemplar_pools(args.intent, "curated", args.register)
        rng = random.Random(args.seed)
        opener_counts = collections.Counter()
        cases = [
            ("railroads", "What were some early responses to the railroads?",
             "Now that we are having a railway line cut through Mansfield, it is to be hoped "
             "that we shall have better trade and more enterprise."),
            ("women vote", "should women vote?",
             "For this reason, among many others, I feel that it peculiarly behoves women to "
             "show that they can appreciate and respond to this loyal justice on the part of "
             "men."),
        ]
        real_word_counts = load_real_word_counts()
        for i, (label, orig, passage) in enumerate(cases):
            form = pick_form(FORM_MIX[args.form_mix], rng)
            text, meta = rewrite_question(run, orig, passage, pools, form, args.exemplars_k,
                                           rng, args.register, opener_counts, i + 1,
                                           real_word_counts)
            print(f"[{label}] form={form} target={meta.get('target_words')}w\n  ORIG: {orig}\n  NEW:  {text}")
            print(f"    causal_check: {meta.get('causal_check')}")
            print(f"    fit_check: {meta.get('fit_check')}")
            print(f"    checks:    {meta.get('checks')}\n")
        return

    if args.run:
        run_full(run, args)
        return

    print("nothing to do -- pass --dry-run-gate, --dry-run-rewrite, or --run --input <path>")


def run_full(run, args):
    if not args.input:
        raise SystemExit("--run requires --input")
    rows = [json.loads(l) for l in args.input.open(encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
        print(f"--limit {args.limit}: processing only the first {len(rows)} source rows")

    pools = load_exemplar_pools(args.intent, "curated", args.register)
    weights = FORM_MIX[args.form_mix]
    rng = random.Random(args.seed)
    opener_counts = collections.Counter()
    real_word_counts = load_real_word_counts()

    out_path = args.output or args.input.with_suffix(".sonnet_gated.jsonl")
    progress_path = out_path.with_suffix(".progress.jsonl")
    done_rows = set()
    if progress_path.exists():
        for l in progress_path.open(encoding="utf-8"):
            if l.strip():
                done_rows.add(json.loads(l)["row_idx"])
        print(f"resuming: {len(done_rows)} rows already processed")

    out_f = out_path.open("a", encoding="utf-8")
    prog_f = progress_path.open("a", encoding="utf-8")

    stats = {"rows_in": 0, "rows_out": 0, "rows_dropped_entirely": 0,
              "exchanges_in": 0, "gate_keep": 0, "gate_reject": collections.Counter(),
              "gate_fail": 0, "rewrite_ok": 0, "rewrite_fail": 0, "n_processed": 0}

    for row_idx, rec in enumerate(rows):
        if row_idx in done_rows:
            continue
        stats["rows_in"] += 1
        n_kept_exchanges = 0

        for original, passage, setter in exchanges_of(rec):
            stats["exchanges_in"] += 1
            gate_v = gate_passage(run, passage)
            if gate_v is None:
                stats["gate_fail"] += 1
                continue
            if not gate_v.get("passage_ok"):
                stats["gate_reject"][gate_v.get("reason", "unknown")] += 1
                continue
            stats["gate_keep"] += 1

            stats["n_processed"] += 1
            form = pick_form(weights, rng)
            text, meta = rewrite_question(run, original, passage, pools, form,
                                           args.exemplars_k, rng, args.register,
                                           opener_counts, stats["n_processed"],
                                           real_word_counts)
            if text is None:
                stats["rewrite_fail"] += 1
                continue
            # 2026-09-04 (Trip): CONFIRMED REGRESSION -- setter() was never called here, so
            # every "rewrite" was computed, validated, and tracked in stats/opener_counts, then
            # silently discarded; the shipped record kept its ORIGINAL question untouched. Found
            # by comparing a stale fact_op_sft_corpus_pulls_merged.sonnet_gated.jsonl (10-row
            # calibration run, correctly applied) against the current code, which had no
            # setter( call anywhere in the file. This is the actual fix.
            setter(text)
            opener_counts[meta["opener_frame"]] += 1
            stats["rewrite_ok"] += 1
            n_kept_exchanges += 1
            # flat single-exchange rows (the "human"/"victorian" schema) get the SAME top-
            # level human_original/prompt_style/prompt_form fields rewrite_prompts_natural.py
            # already established as this codebase's convention -- found missing 2026-09-02
            # (Trip spot-checked calibration output and it wasn't there, only buried in
            # _rewrite_meta). Multi-turn rows have no single slot for this (>1 exchange can
            # exist per row), so they rely on _rewrite_meta alone, same as before.
            if isinstance(rec, dict) and isinstance(rec.get("human"), str):
                rec["human_original"] = original
                rec["prompt_style"] = "sonnet_natural_rewrite"
                rec["prompt_form"] = form
            if isinstance(rec, dict):
                rec.setdefault("_rewrite_meta", []).append({
                    "original_question": original, "gate_note": gate_v.get("note", ""),
                    "causal_check": meta.get("causal_check", ""),
                    "fit_check": meta.get("fit_check", ""), "checks": meta.get("checks", ""),
                    "target_words": meta.get("target_words")})

        if n_kept_exchanges > 0:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            stats["rows_out"] += 1
        else:
            stats["rows_dropped_entirely"] += 1

        prog_f.write(json.dumps({"row_idx": row_idx}) + "\n")
        prog_f.flush()

        if (row_idx + 1) % 25 == 0:
            print(f"  ... {row_idx+1}/{len(rows)}  {stats}  top openers: "
                  f"{opener_counts.most_common(5)}")

    out_f.close()
    prog_f.close()
    print(f"\nFINAL: {stats}")
    print(f"top 10 openers: {opener_counts.most_common(10)}")
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()

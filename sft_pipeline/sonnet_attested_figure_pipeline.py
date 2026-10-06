#!/usr/bin/env python3
"""sonnet_attested_figure_pipeline.py — two-stage Sonnet pipeline building the attested
(pre-1900/fictional_pre1900) half of the notability dataset, from raw corpus windows found by
search_attested_figures_corpus.py + rank_attested_figures_hits.py.

STAGE 1 -- PASSAGE GATE, EXTENDED from sonnet_fact_op_pipeline.py's base (which itself reused
gen_teacher_source4.py's gate): same mid_scene_fiction/anaphoric_opening/dangling_pronoun/
ocr_garbage criteria, second-person-address-is-fine carve-out, PLUS one new criterion added
2026-09-03 after Trip reviewed real gate output and found the base gate has no way to catch
this failure mode -- it never even sees WHICH entity a passage was pulled for, so it can't
check whether the passage is actually about them:
  REJECT "not_about_entity" if the passage does not substantively concern the named entity --
  a coincidental name/word collision (a DIFFERENT "Wells", "Belgium" the country instead of
  King Leopold, "Elizabeth Wells" a medical patient unrelated to H. G. Wells) or a passage
  where the entity is only mentioned in passing and isn't what the passage is actually about.
Mechanical fixes to the search-term extraction (skip surname search for "X of Y" title
patterns, catch plural common-word surnames like "Wells") independently reduce how often this
happens, but the gate needs this check as a backstop regardless -- a genuine full-name
collision (two real people who happen to share a name) isn't something a search-term regex can
ever fully prevent.

STAGE 2 -- SPECIFIC QUESTION GENERATION, per Trip (2026-09-03): the whole point of this dataset
is covering a gap the corpus-pulls-merged / eb_qa data doesn't -- a GENERIC "tell me about X"
is explicitly NOT wanted here (that shape already exists elsewhere in the SFT data). The
question must name the entity AND reference something SPECIFIC to the actual passage content
(a work, an idea, an event, a claim) -- "What are your thoughts on Nietzsche's Man and
Superman?" not "Tell me about Nietzsche." A bare "Do you know of X?" is acceptable ONLY when
the passage genuinely doesn't hand you anything more specific to reference (Trip's own example
of the acceptable minimum), but the model should reach for specificity whenever the passage
supports it, which is most of the time.

Updated 2026-09-04 (Trip): questions should match the DEPLOYMENT DISTRIBUTION, not free-invented
LLM phrasing -- so the writer now injects usage-style exemplars (the same
GUIDES/GUIDES_FACTUAL pools rewrite_prompts_natural.py and sonnet_fact_op_pipeline.py already
use) as "style reference only" material, and carries over that Stage-2's proven opener-reuse
cap (Qwen concentrated 46% of an earlier rewrite run on "tell me about" -- the same collapse
risk applies here at ~150-300 rows). Per Trip's explicit call: NO FORM_MIX rotation here --
phrasing form (question vs. command) is left entirely to Sonnet's judgment, unlike the
fact_op/epistolary pipelines which enforce a weighted form rotation.

  python3 sonnet_attested_figure_pipeline.py --dry-run-gate
  python3 sonnet_attested_figure_pipeline.py --dry-run-question
  python3 sonnet_attested_figure_pipeline.py --run --in probes/attested_corpus_hits_final_kept.jsonl
"""
from __future__ import annotations
import json, re, sys, os, time, math, argparse, random, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "training_data/fact_opinion_sft/scripts"))
from sonnet_fact_op_pipeline import (  # noqa: E402  (reuse, don't reinvent)
    GATE_TMPL, call_json, opener_frame, build_banned_block,
)
from rewrite_prompts_natural import GUIDES, GUIDES_FACTUAL  # noqa: E402  (reuse, don't reinvent)

sys.path.insert(0, str(HERE / "training_data/multi_turn_sft/scripts"))
from sonnet_epistolary_pipeline import (  # noqa: E402  (reuse, don't reinvent)
    load_real_word_counts, sample_target_words,
)

sys.path.insert(0, str(HERE / "judge_scripts"))
from judge_behavioral_v3 import make_judge  # noqa: E402


def load_style_exemplars() -> list[str]:
    """Flat, deduped pool of usage-style example phrasings spanning both command/question
    molds and both general/factual registers -- pure STYLE reference (the writer still has to
    invent the actual content, since these examples don't know about {entity} or the passage).
    No FORM_MIX weighting: Trip explicitly wants form left to Sonnet's judgment here, unlike the
    fact_op/epistolary pipelines."""
    pool, seen, out = [], set(), []
    for g in (GUIDES, GUIDES_FACTUAL):
        pool.extend(g["command"])
        pool.extend(g["question"])
    for t in pool:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


STYLE_EXEMPLARS = load_style_exemplars()


# entity-aware extension of GATE_TMPL, added 2026-09-03 per Trip's review of real gate output
# (see module docstring). Keeps every base criterion verbatim and adds ONE new one, with
# Trip's own real examples as few-shots -- both the failure mode (Leopold/Belgium-treaty,
# Elizabeth-Wells-patient, Wells-irrigation) and the praised KEEP (Poe).
#
# 2026-09-04 (Trip, reviewing the 815-row gate run): two more real misses survived into
# "kept" that the existing wording SHOULD have caught but didn't reliably trigger on:
#   1. "The second was Benjamin Disraeli, the son of a Jewish man of letters..." -- an
#      ordinal-list-enumeration opener ("the second was") is exactly the anaphoric_opening
#      criterion's target (an unresolvable reference, here to "the first" / whatever list this
#      was extracted from), but the existing few-shots only demonstrate technical-antecedent
#      anaphora ("the signals used"), not this ordinal-list pattern -- the model apparently
#      fixated on the rest of the sentence reading as fine, self-contained biography, and
#      missed the opener. Added an explicit ordinal-list few-shot below.
#   2. The Hermes/Plato passage -- Plato is the CORRECT entity (no collision), but only
#      functions as a chronological anchor ("As early as the time of Plato...") in a passage
#      that is actually about Hermes Trismegistus / Neo-Platonism syncretism. This is a
#      distinct flavor of not_about_entity from the collision cases already covered --
#      genuinely the right person, just not the actual subject -- so the rule text and a new
#      few-shot now spell this out explicitly rather than leaving it to be inferred from
#      collision-only examples.
ENTITY_GATE_TMPL = """You are screening a passage of real Victorian prose to decide if it is
usable, BY ITSELF, as the answer side of a chatbot training example about {entity} --
independent of whatever question it is currently paired with.

REJECT "mid_scene_fiction" if characters, events, or locations are referenced but never
defined within the passage itself -- a listener dropped into the middle of a scene they did
not witness.

REJECT "anaphoric_opening" if the passage opens with a reference a listener cannot resolve
("the signals used", "these sentiments", "the proposed regulations") without context that
exists nowhere in the passage. This INCLUDES ordinal/list-enumeration openers -- "The second
was...", "The other was...", "The former...", "The latter..." -- these refer to an earlier list
item (a "first", a preceding enumeration) that does not exist in the passage itself. Do not let
the REST of the sentence reading as good, self-contained biography excuse an opener like this --
the reader still cannot tell what "the second" refers to without the missing preceding text.

REJECT "dangling_pronoun" if "he", "she", or "they" appears as the PRIMARY SUBJECT with no
named referent anywhere in the passage.

REJECT "ocr_garbage" if the text is dominated by garbled characters, tables, or non-prose noise.

REJECT "not_about_entity" if the passage is not substantively ABOUT {entity}. This covers TWO
distinct situations, both reject:
  (a) a false COLLISION -- this passage was pulled by a mechanical name/word search and the
      match is really a DIFFERENT person who happens to share a name or surname, or a
      place/common-noun that happens to match part of the name.
  (b) a correctly-identified but INCIDENTAL mention -- {entity} really is who the search found,
      no collision, but the passage only uses them as a passing reference (a date anchor, a
      comparison, a citation) while actually being about something or someone else entirely.
The bar for both: could a reader who knows nothing else point to specific content in this
passage and say "yes, that is substantively about {entity}"? For (a), ask: would the passage
read identically with the name replaced by an unrelated person/place/word? For (b), ask: if you
deleted the mention of {entity}, would the passage still be a complete, sensible passage about
its ACTUAL subject? If yes to either, reject.

SECOND-PERSON ADDRESS IS NOT A DEFECT. A passage saying "you", "your letter", "as you say", "I
cannot agree with your view" resolves perfectly well -- the reader IS the person being
addressed. Do not reject for second-person address alone; keep rejecting THIRD-PARTY anaphora
(an undefined "he"/"she"/"they" who is not the reader).

EXAMPLES:

Entity: Edgar Allan Poe
Passage:
  "In the regions of the strangely terrible, remotely phantastic, and ghastly, Edgar Allan Poe
  reigns supreme."
-> {{"passage_ok": true, "reason": "ok", "note": "Directly characterizes Poe's own literary territory; unambiguously about him."}}

Entity: Leopold II of Belgium
Passage:
  "In 1862 a treaty of peace and commerce with the United States of North America was
  ratified, and in the following year a similar treaty was concluded with Belgium..."
-> {{"passage_ok": false, "reason": "not_about_entity", "note": "This is about the country Belgium's treaties, not about King Leopold II himself -- a surname-search collision, not real content about him."}}

Entity: H. G. Wells
Passage:
  "Elizabeth Wells, aged fifty-eight, much debilitated with anasarca and ascites of long
  standing, began to take cream of tartar on the 8th of Dec."
-> {{"passage_ok": false, "reason": "not_about_entity", "note": "This 'Wells' is an unrelated medical patient, not the author H. G. Wells -- a surname collision."}}

Entity: H. G. Wells
Passage:
  "Wells abound for irrigation purposes, sometimes the water is of good quality as here, and
  sufficient in amount: surface-water at a depth of about 20 feet at this season..."
-> {{"passage_ok": false, "reason": "not_about_entity", "note": "'Wells' here means water wells, not the author -- a common-word collision, not a person at all."}}

Entity: Charles Darwin
Passage:
  "We never should have heard of the letter at all, I believe, unless I had asked for it
  myself. Do you call that confidence, my love, towards Doctor Strong?... 'Now let us see,'
  said Mrs. Markleham, putting her glass to her eye, 'where the passage is...'"
-> {{"passage_ok": false, "reason": "mid_scene_fiction", "note": "Mrs. Markleham and Doctor Strong are undefined for a listener who was not present, and Darwin is not even mentioned."}}

Entity: Benjamin Disraeli
Passage:
  "The second was Benjamin Disraeli, the son of a Jewish man of letters, then known as a young
  and volatile member of the House, who combined high Tory notions on Church and State with
  extreme Radical views on certain social questions."
-> {{"passage_ok": false, "reason": "anaphoric_opening", "note": "'The second was' refers to an earlier list item (a 'first') that is not in this passage -- the rest of the sentence being good biography doesn't fix the unresolvable opener."}}

Entity: Plato
Passage:
  "As early as the time of Plato, the Greek Hermes was identified with the Egyptian Thot; and
  when pagan philosophy began to be mingled up with Christianity in the form of New Platonism,
  this Egyptian Hermes was looked upon as the author of all knowledge and wise inventions among
  men."
-> {{"passage_ok": false, "reason": "not_about_entity", "note": "Plato is only a chronological anchor ('as early as the time of'); the passage is actually about Hermes/Thot syncretism in Neo-Platonism, not about Plato himself."}}

Entity: (telegraphy passage, no person)
Passage:
  "In order to transmit messages along the telegraphic wires, a certain amount of electric
  power is required... According to some accurate experiments made by M. Pouillet, a
  distinguished member of the Institute of France, a voltaic battery of ten pairs of plates
  will supply electricity sufficient for telegraphic purposes..."
-> {{"passage_ok": true, "reason": "ok", "note": "Self-contained technical exposition attributed to the named entity; no prior context required."}}

NOW DECIDE THIS ONE:

Entity: {entity}
Passage:
{passage}

Reply with ONLY valid JSON:
{{"passage_ok": true|false, "reason": "mid_scene_fiction"|"anaphoric_opening"|"dangling_pronoun"|"ocr_garbage"|"not_about_entity"|"ok", "note": "<one short sentence>"}}"""


def gate_passage(run, entity: str, passage: str) -> dict | None:
    return call_json(run, ENTITY_GATE_TMPL.format(entity=entity, passage=passage))


# 2026-09-03 (Trip): "avoid generic questions -- this dataset exists specifically to cover
# this gap" -- so the prompt has to make specificity a HARD preference, with real few-shots
# showing the failure mode it must avoid, not just an adjective ("specific") left to chance.
# 2026-09-04 (Trip): added the {exemplars} style-reference block (usage-style exemplars,
# same pattern as sonnet_fact_op_pipeline.py's REWRITE_TMPL) and the {banned_block} opener-cap
# safety net, plus an "opener" field in the reply so run_full() can track reuse across the run.
#
# 2026-09-04 (Trip, second pass): live Sonnet output was too long/detailed relative to real
# deployment questions -- e.g. "Do you buy Nietzsche's idea in Also sprach Zarathustra that the
# Superman stands beyond good and evil?" Pointed at multiturn_sft_epistolary_Aug28.jsonl as the
# reference: that pipeline (sonnet_epistolary_pipeline.py) already solved almost this exact
# problem for a different file, and the fix is TWO independent things, not one:
#   1. LENGTH was never constrained. Reused sample_target_words()/load_real_word_counts()
#      directly rather than reinventing -- draws a per-row target from the REAL usage-log
#      word-count distribution (median 7 words) instead of hoping the model's unconstrained
#      tendency happens to match a heavily right-skewed real distribution (it won't).
#   2. The Nietzsche example isn't just long, it's CAUSALLY backwards: it restates the
#      passage's own specific claim ("the Superman stands beyond good and evil") inside the
#      question, so the passage's answer becomes an echo instead of a reveal. This is exactly
#      the causal_check problem sonnet_epistolary_pipeline.py's REWRITE_TMPL already solved --
#      a question may name a SPECIFIC TOPIC (a real work, event, or reputation) as an anchor,
#      but must not pre-state what the passage itself concludes/argues/reports. Added the same
#      causal_check field and rewrote the GOOD/BAD examples to teach the topic-vs-content-echo
#      distinction explicitly, using the real Nietzsche failure as the BAD few-shot.
QUESTION_TMPL = """You write ONE natural user question for a chatbot training pair. The PASSAGE
below is real period prose about {entity} that will serve as the answer -- you are writing only
the QUESTION that would prompt it. The passage has NOT been said yet from your question's point
of view -- you are not agreeing with it, reacting to its specific claim, or aware of what it
concludes.

PASSAGE:
{passage}

STRONG PREFERENCE -- the question must reference something SPECIFIC from the passage (a named
work, an idea, an event, a relationship) -- not just the entity's bare existence. But naming a
specific TOPIC is not the same as pre-stating what the passage says ABOUT that topic -- your
question must not restate, agree with, or presuppose the passage's own specific claim, verdict,
or conclusion, since the passage's job is to REVEAL that, not echo something already asked.
Compare:
  WEAK (avoid unless the passage truly gives you nothing more specific):
    "Tell me about {entity}." / "Who is {entity}?" / "What do you know of {entity}?"
  GOOD (names a specific topic as an anchor, without pre-stating the passage's own verdict --
  these are PATTERN examples using a different figure for illustration, not claims about
  {entity}):
    "What are your thoughts on Nietzsche's Also sprach Zarathustra?" (names the work, not its claim)
    "Have you read Nietzsche's Zarathustra?" (same anchor, plainer phrasing)
    "What happened when Napoleon met the Pope at Notre Dame?" (names the event, not its outcome)
    "What's your view of Voltaire as a critic of the Church?" (names the topic of the opinion, not the verdict)
  BAD (specific, but CAUSALLY BACKWARDS -- restates the passage's own claim, a real observed
  failure):
    "Do you buy Nietzsche's idea in Also sprach Zarathustra that the Superman stands beyond
    good and evil?" -- "the Superman stands beyond good and evil" is the passage's OWN claim;
    asking about it here means the passage's answer just echoes the question instead of
    revealing anything.
    "Do you agree with Nietzsche that the will to power drives all human action?" -- same
    problem: the specific claim belongs in the ANSWER, not pre-loaded into the question.
A bare "Do you know of {entity}?" is acceptable ONLY as a last resort when the passage is
genuinely too general to support anything more specific -- do not default to it out of laziness.

Also decide the register: FACTUAL (asking for information) if the passage is expository/
biographical; OPINION (asking what the persona thinks) if the passage itself expresses a
judgment, argument, or controversial claim worth having an opinion about.

Here are illustrative examples of how users typically phrase their messages (style reference
only -- do NOT borrow their topics, do NOT copy these examples' subjects, they know nothing of {entity} or this
passage):
{exemplars}

LENGTH: write it in approximately {target_words} words. Real users mostly type short, direct
messages -- do not pad it out with extra clauses to sound more sophisticated. Only at longer
targets should it read as more developed.

Write it in natural, varied phrasing the way a real person types -- short and direct is fine,
does not need a question mark if it reads better as a request ("Tell me what you make of...").
Form (question vs. a plain command like "Tell me...") is entirely your call -- pick whichever
reads more natural for this specific passage.
{banned_block}

Reply with ONLY valid JSON:
{{"question": "<the question>",
  "register": "factual"|"opinion",
  "specific_reference": "<the specific work/idea/event your question anchors to, or 'none available' if you had to fall back to the weak form>",
  "causal_check": "<confirm your question does not restate, agree with, or presuppose any specific claim/verdict/conclusion that appears ONLY in the passage below -- name the passage's key claim(s) and confirm your question is independent of them>",
  "fit_check": "<confirm the passage above actually supports answering this exact question>",
  "opener": "<the first 3 words of your question, lowercase>"}}"""


def write_question(run, entity: str, passage: str, rng: random.Random, opener_counts: dict,
                    total_processed: int, real_word_counts: list[int], k: int = 8,
                    max_attempts: int = 4) -> tuple[dict | None, dict]:
    """Returns (verdict_dict_or_None, meta). Proportional opener-reuse cap identical to
    sonnet_fact_op_pipeline.py's rewrite_question() -- no single 3-word opener may exceed ~5%
    of rows processed so far, floor of 3 so the cap doesn't bite absurdly early. Target length
    sampled fresh per row from the real usage-log distribution via
    sonnet_epistolary_pipeline.sample_target_words() -- same reasoning as that pipeline: an
    unconstrained model won't naturally reproduce a heavily right-skewed real distribution."""
    cap = max(3, math.ceil(0.05 * max(total_processed, 1)))
    banned = [f for f, c in opener_counts.items() if c >= cap]
    ex = "\n".join(f"  - {e}" for e in rng.sample(STYLE_EXEMPLARS, min(k, len(STYLE_EXEMPLARS))))
    target = sample_target_words(rng, real_word_counts)

    meta = {"attempts": [], "target_words": target}
    for attempt in range(max_attempts):
        prompt = QUESTION_TMPL.format(
            entity=entity, passage=passage, exemplars=ex, target_words=target,
            banned_block=build_banned_block(banned))
        v = call_json(run, prompt)
        if v is None or not v.get("question"):
            meta["attempts"].append({"ok": False, "reason": "call_or_parse_failed"})
            continue
        frame = opener_frame(v["question"])
        if frame in banned:
            meta["attempts"].append({"ok": False, "reason": "opener_cap", "frame": frame})
            banned = list(set(banned) | {frame})
            continue
        meta["attempts"].append({"ok": True})
        meta["opener_frame"] = frame
        return v, meta
    return None, meta


def clean_passage(text: str) -> str:
    """light OCR-adjacent cleanup only -- collapse whitespace, nothing content-changing."""
    return re.sub(r"\s+", " ", text.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path,
                    default=HERE / "probes/attested_corpus_hits_final_kept.jsonl")
    ap.add_argument("--out", type=Path,
                    default=HERE / "training_data/fact_opinion_sft/attested_figures_corpus_pulls.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="anthropic", choices=["anthropic", "mlx"],
                    help="'mlx' lets --dry-run-gate/--dry-run-question run without "
                         "ANTHROPIC_API_KEY, using the same local Qwen model "
                         "local_gate_attested_hits.py trusts for gate/judge tasks. NOT "
                         "recommended for --run (question WRITING is a generative task, and "
                         "this project has measured local-Qwen generative output as "
                         "unreliable -- see fact_op rewrite history).")
    ap.add_argument("--model", default=None, help="override the default model for --backend")
    ap.add_argument("--dry-run-gate", action="store_true")
    ap.add_argument("--dry-run-question", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    if args.backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("!! ANTHROPIC_API_KEY is not set -- refusing to start. "
                          "Pass --backend mlx for a local dry-run instead.")

    if args.backend == "mlx" and args.run:
        print("!! WARNING: --backend mlx with --run -- local Qwen has measured reliability "
              "issues on GENERATIVE tasks (see fact_op rewrite history). Intended for "
              "--dry-run-gate/--dry-run-question only. Proceeding anyway since you asked.\n")

    run, model_name = make_judge(args.backend, args.model, max_tokens=1024)
    print(f"backend: {model_name}\n")

    if args.dry_run_gate:
        cases = [
            ("Ruskin", "mid-window cut", "expect REJECT",
             "the fierce Republic; but there the comparison ends. Ruskin set him to measure "
             "Pisa cathedral all over, to see why it was so irregular; and for a little "
             "holiday one heavenly morning before breakfast, Boni took me up the Baptistery"),
            ("Coxwell", "genuine bio passage", "expect KEEP",
             "Coxwell, Henry Tracey, aeronaut, born in 1819, at Wouldham, near Rochester, was "
             "educated for the army, but settled as a surgeon-dentist in London. From boyhood "
             "he had taken a keen interest in ballooning."),
            ("Leopold II of Belgium", "surname/name collision", "expect REJECT not_about_entity",
             "In 1862 a treaty of peace and commerce with the United States of North America "
             "was ratified, and in the following year a similar treaty was concluded with "
             "Belgium, providing for the reciprocal admission of goods on the same terms as "
             "those accorded to the most favoured nation."),
            ("H. G. Wells", "common-word collision", "expect REJECT not_about_entity",
             "Wells abound for irrigation purposes, sometimes the water is of good quality as "
             "here, and sufficient in amount: surface-water at a depth of about 20 feet at "
             "this season, and somewhat lower in the hot weather."),
        ]
        for entity, label, expectation, passage in cases:
            v = gate_passage(run, entity, passage)
            print(f"[{entity} / {label}] {expectation}\n  -> {v}\n")
        return

    if args.dry_run_question:
        cases = [
            ("Nietzsche", "Nietzsche's chief work, Also sprach Zarathustra, expounds his "
             "doctrine of the will to power and the coming of the Superman, who shall stand "
             "beyond the common herd morality of good and evil."),
            ("Coxwell", "Coxwell, Henry Tracey, aeronaut, born in 1819, at Wouldham, near "
             "Rochester, was educated for the army, but settled as a surgeon-dentist in "
             "London. He has made some 700 ascents, the most remarkable being that of 1862, "
             "when he reached, with Mr Glaisher, a height of seven miles."),
        ]
        rng = random.Random(args.seed)
        opener_counts = collections.Counter()
        real_word_counts = load_real_word_counts()
        for i, (entity, passage) in enumerate(cases):
            v, meta = write_question(run, entity, passage, rng, opener_counts, i + 1,
                                      real_word_counts)
            print(f"[{entity}] -> {v}\n  meta: {meta}\n")
        return

    if args.run:
        run_full(run, args)
        return

    print("nothing to do -- pass --dry-run-gate, --dry-run-question, or --run")


def run_full(run, args):
    """Stage 2 ONLY -- --in defaults to attested_corpus_hits_final_kept.jsonl, which already
    passed the entity-aware gate (see module docstring's build history: full corpus scan ->
    rank -> local-model gate -> re-gate under the refined template). Re-running Stage 1 here
    would just re-pay for a decision already made; each row's existing gate_note is carried
    through into the output instead."""
    rows = [json.loads(l) for l in args.inp.open(encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    progress_path = args.out.with_suffix(".progress.jsonl")
    done = set()
    if progress_path.exists():
        for l in progress_path.open():
            if l.strip():
                done.add(json.loads(l)["idx"])
        print(f"resuming: {len(done)} already processed")

    out_f = args.out.open("a", encoding="utf-8")
    prog_f = progress_path.open("a", encoding="utf-8")
    stats = {"rows_in": 0, "question_ok": 0, "question_fail": 0}
    rng = random.Random(args.seed)
    opener_counts = collections.Counter()
    real_word_counts = load_real_word_counts()

    for idx, row in enumerate(rows):
        if idx in done:
            continue
        stats["rows_in"] += 1
        passage = clean_passage(row["window"])
        q_v, meta = write_question(run, row["entity"], passage, rng, opener_counts,
                                    stats["rows_in"], real_word_counts)
        if q_v is None:
            stats["question_fail"] += 1
        else:
            stats["question_ok"] += 1
            opener_counts[meta["opener_frame"]] += 1
            out_f.write(json.dumps({
                "entity": row["entity"], "match_type": row["match_type"],
                "source": row["source"], "year": row["year"], "htid": row.get("htid"),
                "passage": passage,
                "pair": [{"role": "user", "content": q_v["question"]},
                         {"role": "assistant", "content": passage}],
                "register": q_v.get("register"),
                "specific_reference": q_v.get("specific_reference"),
                "causal_check": q_v.get("causal_check"),
                "fit_check": q_v.get("fit_check"),
                "opener_frame": meta["opener_frame"],
                "target_words": meta["target_words"],
                "gate_note": row.get("gate_note", ""),
            }, ensure_ascii=False) + "\n")
            out_f.flush()

        prog_f.write(json.dumps({"idx": idx}) + "\n")
        prog_f.flush()
        if (idx + 1) % 25 == 0:
            print(f"  ... {idx+1}/{len(rows)}  {stats}  top openers: "
                  f"{opener_counts.most_common(5)}")

    out_f.close()
    prog_f.close()
    print(f"\nFINAL: {stats}")
    print(f"top 10 openers: {opener_counts.most_common(10)}")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()

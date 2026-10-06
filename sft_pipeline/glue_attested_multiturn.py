#!/usr/bin/env python3
"""glue_attested_multiturn.py — combines pairs of independently-written attested-figure Q&A
rows (training_data/fact_opinion_sft/attested_figures_corpus_pulls.jsonl) into 2-turn
conversations with entity tracking, per Trip (2026-09-04): "Are you familiar with John Locke?"
/ A -> "When did he become a doctor?" / A.

Allocation rule (Trip's spec, exact): per entity with N kept passages,
  N < 3   -> untouched, all N rows stay single-turn.
  N >= 3  -> floor(N/2) passages PAIRED into 2-turn conversations (consuming 2*floor(N/2)
             rows), (N mod 2) leftover row(s) stay single-turn. N=3 -> 1 pair + 1 single-turn.
             N=4 -> 2 pairs + 0 leftover. Passages are REALLOCATED, not duplicated -- a row
             that goes into a pair does not also survive as a separate single-turn row.

Within a pair, Turn 1 (first-listed row) is used AS-IS -- its existing question already
establishes the entity by name/title. Turn 2 (second-listed row) gets its QUESTION rewritten
via the "glue" step (GLUE_TMPL) to refer back to the entity with a pronoun instead of
repeating their name -- ONLY when the entity's own name is actually Turn 2's grammatical
subject (see STEP 0 in GLUE_TMPL below). Turn 2's ANSWER (the real corpus passage) never
changes.

GLUE_TMPL history: first 5-case test (Thomas More/Newton/Columbus/Galton/Spencer) was 4/5
clean on the LOCAL model, but Newton failed concretely -- original Turn 2 "Why did King
William ban those French gold coins?" (passage: King William banned them ON NEWTON'S ADVICE,
not himself) got blindly rewritten to "Why did he ban those French gold coins?", with "he"
wrongly resolving to Newton via Turn 1's antecedent, even though "Newton" never appeared in
that Turn 2 question at all -- a misattribution the model's own self-checks didn't catch.
Added an explicit SUBJECT CHECK (step 0) plus a "no_change_needed" escape hatch for exactly
this case; re-test came back 5/5 clean (Newton case correctly left unchanged, subject
identified as "King William", not the tracked entity).

Trip's instinct (backed by that test): the GLUE step (pronoun substitution + subject check) is
a narrower, more mechanical task than full question-writing (already established to need
Sonnet) -- closer to the gate/judge role local Qwen is already trusted for -- so glue defaults
to --backend-glue mlx, not Sonnet.

ORDERING, added 2026-09-04 after Trip reviewed the first real output: a Stevenson conversation
paired "Why did Stevenson travel so much?" (A: ill-health background) as Turn 1 with "How does
he write in Travels with a Donkey?" (A: names two specific works) as Turn 2, purely because
that was the ROW ORDER coming out of Stage 3 (ranking), not because it read naturally --
Trip's read is the reverse would flow better (the specific-work turn as a stronger opener, the
"why did he travel" as a curiosity follow-up once travel-writing is already on the table).
Ordering ALL of an entity's N passages for narrative flow (not just swapping within a fixed
pair -- a full reorder can also change WHICH passages end up paired together) is a much more
holistic, qualitative judgment than the glue step's binary subject-check, closer to the kind
of subtler task this project has already measured local Qwen being less reliable at (it
contradicted its own diagnoses on a similar judgment call in an earlier run) -- so ordering
defaults to --backend-order anthropic (Sonnet), kept as a SEPARATE call/backend from glue.

Re-running this script now RECONSTRUCTS the original per-passage pool from the CURRENT
single-turn survivors file + the current multi-turn file's turn2_original_question field
(which exists precisely for this kind of audit/rebuild), so ordering can be layered onto
already-glued output without needing to re-run Stages 1-5.

  python3 glue_attested_multiturn.py --dry-run
  python3 glue_attested_multiturn.py --dry-run-order
  python3 glue_attested_multiturn.py --run
"""
from __future__ import annotations
import json, re, sys, argparse, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "judge_scripts"))
from judge_behavioral_v3 import make_judge  # noqa: E402

IN_PATH = ROOT / "training_data/fact_opinion_sft/attested_figures_corpus_pulls.jsonl"
OUT_SINGLE = ROOT / "training_data/fact_opinion_sft/attested_figures_corpus_pulls.jsonl"
OUT_MULTI = ROOT / "training_data/multi_turn_sft/attested_figures_multiturn.jsonl"

ORDER_TMPL = """You are sequencing {n} independently-written Q&A turns about the same
historical figure, {entity}, into the most natural conversational ORDER -- the order a real
curious person would actually ask them in, across one sitting.

Candidate turns (each a complete, self-contained question + answer -- the ANSWER text is real,
unchanged historical prose; only the ORDER may change, and only the second turn of any
resulting pair will later be lightly reworded to refer back with a pronoun instead of a name):
{items_block}

Consider: does one turn read as a more natural OPENING -- broad biographical grounding, or a
strong specific hook -- while another reads as a natural FOLLOW-UP a curious person would ask
next, building on or drilling into something the opening raised? Order ALL {n} turns so each
flows as sensibly as possible from what came before. Some turns may just be independent topics
with no strong ordering preference relative to each other -- that's fine, place them wherever
is least jarring; you do not need to force a narrative that isn't there.

Reply with ONLY valid JSON:
{{"order": [<the {n} turn ids, 1-indexed, in the best conversational order, e.g. [3, 1, 4, 2]>],
  "reasoning": "<one or two sentences on why this order reads more naturally than turn-id order>"}}"""

# see module docstring for the Newton-case history behind step 0 / no_change_needed.
GLUE_TMPL = """You are combining two independently-written chatbot Q&A turns about the same
historical figure, {entity}, into a single flowing two-turn conversation.

TURN 1 (already fixed -- establishes who we are talking about):
  User: "{q1}"

TURN 2 -- your job is to rewrite this question so it reads as a natural conversational
follow-up to Turn 1, referring back to {entity} with a pronoun (he/she/they, whichever is
correct for this person) instead of repeating their name or title -- the way a real person
follows up in conversation rather than re-introducing who they mean every single time.

ORIGINAL Turn 2 question (to rewrite):
  "{q2}"

The PASSAGE this rewritten question must still be answerable by (unchanged, for reference only):
{passage2}

STEP 0 -- SUBJECT CHECK (do this FIRST, before rewriting anything): does {entity}'s own name or
title actually appear as the GRAMMATICAL SUBJECT of the ORIGINAL Turn 2 question above? Only
substitute a pronoun for {entity}'s OWN name. If Turn 2's subject is a DIFFERENT person or
thing (even one connected to {entity}, mentioned in the passage, or discussed alongside them),
do NOT invent a pronoun that points it at {entity} instead -- that is a misattribution, not a
natural follow-up. In that case, leave the question AS-IS (or make only a minimal, non-
substituting tweak like removing a redundant "So..." opener) and report "no_change_needed": true.
  EXAMPLE OF THE FAILURE TO AVOID: entity = Isaac Newton, Turn 2 = "Why did King William ban
  those French gold coins?" -- the passage shows King William banned them ON NEWTON'S ADVICE;
  Newton did not ban them. "King William" is Turn 2's real subject, not Newton. Rewriting this
  to "Why did he ban those French gold coins?" wrongly attributes the ban to Newton via the
  pronoun. The correct move here is "no_change_needed": true, leaving Turn 2 exactly as written.

Rules (only apply once step 0 confirms {entity} IS Turn 2's own subject):
- Use the correct pronoun for {entity} based on who they actually were.
- Keep the EXACT same specific content/topic as the original Turn 2 question -- you are only
  changing HOW the entity is referred to, not what is being asked.
- Do not reintroduce the entity's name or title anywhere in the rewritten question.
- Make it read like a short, natural follow-up message.

Reply with ONLY valid JSON:
{{"no_change_needed": true|false,
  "subject_check": "<name Turn 2's actual grammatical subject -- {entity} themselves, or someone/something else -- and say which>",
  "rewritten_turn2": "<the rewritten question, or the ORIGINAL unchanged if no_change_needed is true>",
  "pronoun_used": "<the pronoun you used, or 'none' if no_change_needed>",
  "antecedent_check": "<confirm this pronoun unambiguously refers back to {entity} specifically -- not to any other person mentioned in the passage or the conversation>",
  "fit_check": "<confirm the passage above still answers this question as it now reads>"}}"""


def call_json(run, prompt, retries=3):
    last = None
    for attempt in range(retries):
        try:
            raw = run(prompt)
        except Exception as e:
            last = e
            continue
        m = re.search(r"\{.*\}", raw or "", re.S)
        if not m:
            last = f"no JSON: {raw!r}"
            continue
        try:
            return json.loads(m.group(0))
        except Exception as e:
            last = e
    return None


def load_original_rows(in_single: Path, out_multi: Path) -> list[dict]:
    """Reconstruct the full pre-glue pool: current single-turn survivors, PLUS each multi-turn
    row's original two passages (Turn 1 as-is, Turn 2 via its stored turn2_original_question --
    exactly the pre-glue question, before pronoun substitution). Lets ordering be re-applied to
    already-glued output without re-running Stages 1-5."""
    rows = [json.loads(l) for l in in_single.open(encoding="utf-8")] if in_single.exists() else []
    if out_multi.exists():
        for line in out_multi.open(encoding="utf-8"):
            r = json.loads(line)
            rows.append({
                "entity": r["entity"], "source": r.get("turn1_source"), "year": r.get("turn1_year"),
                "htid": r.get("turn1_htid"), "match_type": r.get("turn1_match_type"),
                "specific_reference": r.get("turn1_specific_reference"),
                "pair": [r["pair"][0], r["pair"][1]],
            })
            rows.append({
                "entity": r["entity"], "source": r.get("turn2_source"), "year": r.get("turn2_year"),
                "htid": r.get("turn2_htid"), "match_type": r.get("turn2_match_type"),
                "specific_reference": r.get("turn2_specific_reference"),
                "pair": [{"role": "user", "content": r["turn2_original_question"]}, r["pair"][3]],
            })
    return rows


def dedup_by_passage_text(rows: list[dict]) -> list[dict]:
    """2026-09-05 (Trip): found AFTER the fact that a Jesus Christ pair silently vanished from
    round 1's output -- traced to two byte-identical duplicate passages (same corpus text at
    two different (shard,row) positions, an upstream corpus-dedup miss) getting paired TOGETHER
    by the ordering step, which the glue step then handled badly (Turn 1 and Turn 2 having
    identical answers is a confusing edge case with no real precedent in the few-shots). Cheap
    prevention: dedup by exact passage text BEFORE pairing, same fix already applied
    post-hoc to tranche 2's gate output -- here it runs pre-glue so it can never reach the
    glue step at all, for round 1 or round 2."""
    seen, out = set(), []
    n_dropped = 0
    for r in rows:
        key = (r["entity"], r["pair"][1]["content"].strip())
        if key in seen:
            n_dropped += 1
            continue
        seen.add(key)
        out.append(r)
    if n_dropped:
        print(f"dedup_by_passage_text: dropped {n_dropped} duplicate-text row(s)")
    return out


def order_entity_passages(run, entity: str, rows: list[dict]) -> list[dict]:
    """Sonnet call: returns rows REORDERED for natural conversational flow. Falls back to the
    original order on any parse failure or invalid (non-permutation) response -- ordering is a
    quality improvement, never a hard requirement, so a failure here should never block the run."""
    n = len(rows)
    items = []
    for i, r in enumerate(rows):
        q = r["pair"][0]["content"]
        a = r["pair"][1]["content"]
        items.append(f"  [{i+1}] Q: {q}\n      A: {a[:300]}")
    items_block = "\n".join(items)
    v = call_json(run, ORDER_TMPL.format(n=n, entity=entity, items_block=items_block))
    if v is None:
        return rows
    order = v.get("order")
    if not isinstance(order, list) or sorted(order) != list(range(1, n + 1)):
        return rows
    return [rows[i - 1] for i in order]


def glue_pair(run, entity: str, row1: dict, row2: dict) -> dict | None:
    q1 = row1["pair"][0]["content"]
    q2 = row2["pair"][0]["content"]
    a1 = row1["pair"][1]["content"]
    a2 = row2["pair"][1]["content"]
    v = call_json(run, GLUE_TMPL.format(entity=entity, q1=q1, q2=q2, passage2=a2))
    if v is None or not v.get("rewritten_turn2"):
        return None
    return {
        "entity": entity,
        "sub_type": "attested_figure_multiturn",
        "n_turns": 4,
        "pair": [
            {"role": "user", "content": q1},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": v["rewritten_turn2"]},
            {"role": "assistant", "content": a2},
        ],
        "turn1_source": row1.get("source"), "turn1_year": row1.get("year"),
        "turn1_htid": row1.get("htid"), "turn1_match_type": row1.get("match_type"),
        "turn1_specific_reference": row1.get("specific_reference"),
        "turn2_source": row2.get("source"), "turn2_year": row2.get("year"),
        "turn2_htid": row2.get("htid"), "turn2_match_type": row2.get("match_type"),
        "turn2_specific_reference": row2.get("specific_reference"),
        "turn2_original_question": q2,
        "turn2_subject_check": v.get("subject_check", ""),
        "turn2_no_change_needed": v.get("no_change_needed"),
        "turn2_pronoun_used": v.get("pronoun_used", ""),
        "turn2_antecedent_check": v.get("antecedent_check", ""),
        "turn2_fit_check": v.get("fit_check", ""),
    }


def plan_pairs(rows: list[dict], order_run=None) -> tuple[list[tuple[dict, dict]], list[dict]]:
    """Returns (pairs, singleton_survivors) per the N>=3 allocation rule. If order_run is
    given, each N>=3 entity's passages are reordered for conversational flow (via Sonnet)
    BEFORE chunking into pairs -- a full reorder can change WHICH passages end up paired
    together, not just which one is Turn 1 vs Turn 2 within an already-fixed pair."""
    by_entity = collections.defaultdict(list)
    for r in rows:
        by_entity[r["entity"]].append(r)

    pairs, singles = [], []
    for entity, rs in by_entity.items():
        if len(rs) < 3:
            singles.extend(rs)
            continue
        if order_run is not None:
            rs = order_entity_passages(order_run, entity, rs)
        n_pairs = len(rs) // 2
        for i in range(n_pairs):
            pairs.append((rs[2 * i], rs[2 * i + 1]))
        if len(rs) % 2:
            singles.append(rs[-1])
    return pairs, singles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-single", type=Path, default=IN_PATH,
                    help="single-turn survivors file to reconstruct from AND overwrite -- "
                         "override for a separate round (e.g. round 2) so it never touches "
                         "round 1's files")
    ap.add_argument("--out-multi", type=Path, default=OUT_MULTI,
                    help="multi-turn file to reconstruct additional rows from (if it already "
                         "exists) and to write new output to")
    ap.add_argument("--backend-glue", default="mlx", choices=["mlx", "anthropic"])
    ap.add_argument("--backend-order", default="anthropic", choices=["mlx", "anthropic"])
    ap.add_argument("--no-order", action="store_true",
                    help="skip the ordering call entirely, pair in existing row order (old behavior)")
    ap.add_argument("--model-glue", default=None)
    ap.add_argument("--model-order", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the pairing plan, make no API calls")
    ap.add_argument("--dry-run-order", action="store_true",
                    help="run ONLY the ordering call on a few real N>=3 entities and print before/after")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="with --run, only glue the first N pairs")
    args = ap.parse_args()

    rows = load_original_rows(args.in_single, args.out_multi)
    rows = dedup_by_passage_text(rows)

    if args.dry_run_order:
        order_run, order_name = make_judge(args.backend_order, args.model_order, max_tokens=600)
        print(f"order backend: {order_name}\n")
        by_entity = collections.defaultdict(list)
        for r in rows:
            by_entity[r["entity"]].append(r)
        tested = 0
        for entity, rs in by_entity.items():
            if len(rs) < 3:
                continue
            print(f"=== {entity} (N={len(rs)}) ===")
            print("BEFORE:")
            for r in rs:
                print(f"  {r['pair'][0]['content']}")
            reordered = order_entity_passages(order_run, entity, rs)
            print("AFTER:")
            for r in reordered:
                print(f"  {r['pair'][0]['content']}")
            print()
            tested += 1
            if tested >= 3:
                break
        return

    pairs, singles = plan_pairs(rows)
    print(f"{len(rows)} input rows, {len(pairs)} pairs planned, {len(singles)} singleton "
          f"survivors -- sanity: {2*len(pairs) + len(singles)} == {len(rows)}")

    if args.dry_run:
        for row1, row2 in pairs[:10]:
            print(f"  [{row1['entity']}] T1: {row1['pair'][0]['content'][:70]!r} "
                  f"| T2(orig): {row2['pair'][0]['content'][:70]!r}")
        return

    if not args.run:
        print("nothing to do -- pass --dry-run, --dry-run-order, or --run")
        return

    order_run = None
    if not args.no_order:
        order_run, order_name = make_judge(args.backend_order, args.model_order, max_tokens=600)
        print(f"order backend: {order_name}")
    pairs, singles = plan_pairs(rows, order_run=order_run)
    print(f"after ordering: {len(pairs)} pairs planned, {len(singles)} singleton survivors")

    run, name = make_judge(args.backend_glue, args.model_glue, max_tokens=400)
    print(f"glue backend: {name}\n")

    if args.limit:
        pairs = pairs[:args.limit]

    multi_out, fail_pairs = [], []
    for i, (row1, row2) in enumerate(pairs):
        v = glue_pair(run, row1["entity"], row1, row2)
        if v is None:
            fail_pairs.append((row1, row2))
            continue
        multi_out.append(v)
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(pairs)} glued, {len(fail_pairs)} failed")

    # failed pairs: both rows fall back to singleton survivors rather than being dropped
    for row1, row2 in fail_pairs:
        singles.append(row1)
        singles.append(row2)

    with args.out_multi.open("w", encoding="utf-8") as f:
        for r in multi_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with args.in_single.open("w", encoding="utf-8") as f:
        for r in singles:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nFINAL: {len(multi_out)} multi-turn conversations -> {args.out_multi}")
    print(f"       {len(singles)} single-turn survivors -> {args.in_single} "
          f"({len(fail_pairs)} failed-pair rows folded back in as singles)")


if __name__ == "__main__":
    main()

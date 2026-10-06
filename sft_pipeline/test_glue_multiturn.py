#!/usr/bin/env python3
"""test_glue_multiturn.py — one-off test of the "glue" step for combining two
independently-written attested-figure Q&A pairs into a 2-turn conversation with entity
tracking, per Trip (2026-09-04): "Are you familiar with John Locke?" / A -> "When did he
become a doctor?" / A. Testing whether the LOCAL model can handle this narrow, mechanical
rewrite (pronoun substitution for turn 2's question, preserving its specific content) before
committing to building a real pipeline -- Trip's instinct is that this is simpler than full
question-writing (already proven to need Sonnet), closer to the gate/judge tasks local Qwen
is already trusted for.

  python3 test_glue_multiturn.py
"""
from __future__ import annotations
import json, re, sys, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "judge_scripts"))
from judge_behavioral_v3 import make_judge  # noqa: E402

# 2026-09-04 (Trip): first 5-case test was 4/5 clean, but Isaac Newton failed concretely --
# original Turn 2 "Why did King William ban those French gold coins?" (the passage shows King
# William banned them ON Newton's ADVICE -- Newton didn't ban them himself) got rewritten to
# "Why did he ban those French gold coins?", with "he" resolving to Newton (Turn 1's
# antecedent) via blind pronoun substitution -- even though "Newton" never appeared in the
# ORIGINAL Turn 2 question at all. Both self-checks confidently claimed this was fine. Added
# an explicit SUBJECT CHECK as step 0, before any rewriting happens, plus a "no_change_needed"
# escape hatch for exactly this case (a valid Turn 2 about the entity's world that doesn't
# happen to reuse their name/title as its own grammatical subject).
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


def call_json(run, prompt, retries=2):
    for attempt in range(retries):
        raw = run(prompt)
        m = re.search(r"\{.*\}", raw or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def main():
    rows = [json.loads(l) for l in
            (ROOT / "training_data/fact_opinion_sft/attested_figures_corpus_pulls.jsonl").open()]
    by_entity = collections.defaultdict(list)
    for r in rows:
        by_entity[r["entity"]].append(r)

    test_entities = ["Thomas More", "Isaac Newton", "Christopher Columbus",
                      "Francis Galton", "Herbert Spencer"]

    run, name = make_judge("mlx", None, max_tokens=400)
    print(f"backend: {name}\n")

    for entity in test_entities:
        r1, r2 = by_entity[entity][0], by_entity[entity][1]
        q1 = r1["pair"][0]["content"]
        q2 = r2["pair"][0]["content"]
        passage2 = r2["pair"][1]["content"]

        prompt = GLUE_TMPL.format(entity=entity, q1=q1, q2=q2, passage2=passage2)
        v = call_json(run, prompt)
        print(f"=== {entity} ===")
        print(f"  TURN 1 (fixed): {q1}")
        print(f"  TURN 2 ORIG:    {q2}")
        if v is None:
            print("  PARSE FAIL")
        else:
            print(f"  subject_check:    {v.get('subject_check')}")
            print(f"  no_change_needed: {v.get('no_change_needed')}")
            print(f"  TURN 2 NEW:       {v.get('rewritten_turn2')}")
            print(f"  pronoun_used:     {v.get('pronoun_used')}")
            print(f"  antecedent_check: {v.get('antecedent_check')}")
            print(f"  fit_check:        {v.get('fit_check')}")
        print()


if __name__ == "__main__":
    main()

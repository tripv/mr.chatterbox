#!/usr/bin/env python3
"""sonnet_epistolary_pipeline.py — wholesale regeneration pipeline for
multiturn_sft_epistolary_passages.jsonl's pg_static_map batch, per Trip's decision
2026-08-27: the local Qwen-32B judge missed every hand-checked defect (salutation aside), so
both the passage gate and the question rewrite use Sonnet.

TWO INDEPENDENT STAGES, because a passage can pass one and fail the other (the Claypon
invitation passage is a clean KEEP on the gate, but its ORIGINAL paired question was still
badly attributed -- gate and rewrite quality are orthogonal):

STAGE 1 -- PASSAGE GATE. Decides whether a real corpus passage is usable AT ALL as a
response, independent of what question it's currently paired with.
  KEEP   -- first-person address/reflection/correspondence, even if it presumes deep,
            long-standing relationship or shared history. Warmth and relational depth are
            fine and desirable (Claypon "My very dear Friend" letter).
  REJECT "expository" -- third-person narration/description/argument; nobody is speaking in
            their own voice to anyone (Ruskin/Alessandri passage; the dog-snob essay excerpt).
  REJECT "opaque" -- comprehension of the passage's actual point depends on a specific
            unstated fact (a named person/document/event) that is never explained and is
            LOAD-BEARING for the sentence's meaning (Browning "the Autography affair",
            EBB "your hero"/"the first volume" of an undiscussed novel). A passing,
            decorative mention that isn't load-bearing (Claypon's "no Claypon... would be
            preferred") does NOT count as opaque.

STAGE 2 -- QUESTION REWRITE, for every passage that survives the gate. Trip's core
correction 2026-08-27: the defect in questions like "What prompted this kind invitation?" is
NOT register/tone mismatch -- it's referential coherence, with two distinct failure shapes:
  (a) unattributed antecedent -- "this invitation"/"that"/"this situation" refers to nothing
      established anywhere in the exchange (dog-snob "That seems like a stretch" -- a stretch
      of WHAT? Nothing has been claimed yet.)
  (b) wrong attribution -- asking the passage-speaker to explain an action that the passage
      itself shows was performed by the OTHER party (the Claypon passage has the speaker
      THANKING someone for an invitation THEY extended -- the invitation was the addressee's
      action, so "what prompted this kind invitation" wrongly asks the wrong party).
The rewrite prompt must produce a question with no dangling antecedent and correct
attribution, on top of the topical-fit requirement already established for other files.

  python3 sonnet_epistolary_pipeline.py --dry-run-gate
  python3 sonnet_epistolary_pipeline.py --dry-run-rewrite
"""
from __future__ import annotations
import json, re, sys, os, argparse, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIC = HERE.parent.parent.parent
USAGE = VIC / ("tests_and_assessments/training rd 1 assessments/usage data and "
               "anaylsis/use_data_analysis/usage_data_intent_labels.jsonl")


def load_real_word_counts():
    """All 44,735 real user-turn word counts (every turn position). Measured 2026-08-27:
    median 7, 37% <=5 words, 69% <=10, only 7.4% >30 -- heavily right-skewed."""
    out = []
    for line in USAGE.open(encoding="utf-8"):
        r = json.loads(line)
        t = (r.get("user_text") or "").strip()
        if t:
            out.append(len(t.split()))
    return out

GATE_SHOTS = """EXAMPLE A -- REJECT "expository" (third-person narration, nobody speaking to anyone):
  "He and Signor Alessandri, however, were not strictly pupils of Ruskin, who had met them
  during the winter of 1876-77 at Venice, and, so to say, adopted them. At this second
  meeting he liked them and their work more than ever. His character of them is given in the
  first of his lectures on returning next year to Oxford..."
  -> {"keep": false, "reason": "expository", "note": "Third-person biography; no one is speaking in their own voice."}

EXAMPLE B -- REJECT "expository" (an essay, despite warm first-person voice):
  "I have called him a snob; but all dogs are so, though in varying degrees. It is hard to
  follow their snobbery among themselves; but towards us, their masters, it may sometimes be
  plainly enough observed. A dog values himself upon possession of a good master..."
  -> {"keep": false, "reason": "expository", "note": "Reflective essay prose about dogs in general -- nobody specific is being addressed, no correspondence-relation."}

EXAMPLE C -- KEEP (personal correspondence, deep relational history is FINE):
  "My very dear Friend,--You are kind to exceeding kindness, and I am as grateful as any of
  your long-ago kind invitations ever found me... You may be perfectly sure that no Claypon,
  though he should live in Arcadia, would be preferred by me to you as a host..."
  -> {"keep": true, "reason": "ok", "note": "Genuine personal letter; 'Claypon' is a decorative, non-load-bearing comparison, not required for comprehension."}

EXAMPLE D -- REJECT "opaque" (a named reference IS load-bearing and never explained):
  "I am vexed by another thing which he tells me--vexed, if amused a little by the absurdity
  of it. I mean that absurd affair of the 'Autography'--now isn't it absurd? And for neither
  you nor George to have the chivalry of tearing out that letter of mine..."
  -> {"keep": false, "reason": "opaque", "note": "'The Autography affair' is the entire point of the passage and is never explained -- unlike Claypon's decorative name-drop, this is load-bearing and incomprehensible without outside context."}

EXAMPLE E -- REJECT "opaque" (mid-paragraph start -- found 2026-08-28, Trip): a later-
  position segment of the source file, extended toward a length ceiling by whatever built it,
  opening wherever that extraction window happened to begin rather than at a natural start:
  "But it was a compliment to his audience to call them cleverer than Boni, whose great power
  already showed itself in his keen eye and square shoulders. Napoleon Bonaparte must have
  looked something like him, I thought, when he began to charm the fierce Republic; but there
  the comparison ends. Ruskin set him to measure Pisa cathedral all over..."
  -> {"keep": false, "reason": "opaque", "note": "Opens with 'But', mid-thought; 'Boni', 'his audience', and 'the fierce Republic' are all load-bearing and never introduced -- this is a segment that starts wherever a length window began, not at a natural opening."}
  An opening word like "But"/"And"/"So"/"However"/"Yet"/"Also"/"Therefore" is not automatic
  rejection on its own, but is a strong signal to check the OPENING specifically against the
  opaque criterion below, independent of whatever the rest of the passage does -- measured
  2026-08-28: passages at later positions in a multi-exchange row open this way roughly twice
  as often (10% at the first exchange vs 19% at a third), so extra scrutiny there is worth it.
"""

GATE_TMPL = """You are screening a passage of real Victorian prose to decide if it is usable
as the RESPONSE side of a chatbot training example. The persona speaks in the FIRST PERSON.

{shots}
NOW DECIDE THIS ONE:

PASSAGE:
{passage}

KEEP if the passage is spoken/written in the first person as personal address, reflection, or
correspondence -- even presuming a long, close relationship or shared history. Depth of
relationship and warmth are fine and desirable; do not reject for that alone.

REJECT "expository" if the passage is third-person narration, description, or argument --
nobody is speaking in their own voice to anyone.

REJECT "opaque" if understanding the passage's actual point depends on a SPECIFIC unstated
fact (a named person, document, or event) that is never explained and is load-bearing for the
sentence's meaning. A passing, non-load-bearing mention of an unexplained name does NOT count.
Check the OPENING specifically: a passage starting with "But," "And," "So," "However," "Yet,"
"Also," or "Therefore" is often a segment beginning wherever an extraction window happened to
land, not at a natural start (see Example E) -- when you see this, look hard at whether the
opening sentence itself depends on an unstated antecedent, even if the rest of the passage
would otherwise be fine on its own.

REJECT "husband_centric" if the passage's substance centers on the writer's husband -- defined
by him, deferring to him, or primarily reporting on HIS doings rather than her own. A woman
recounting events, giving her own opinions, or describing her own activities and interests is
FINE even if a husband is mentioned in passing; only reject when the husband, not the writer,
is genuinely what the passage is about.

Reply with ONLY valid JSON:
{{"keep": true|false, "reason": "expository"|"opaque"|"ok", "note": "<one short sentence>"}}"""

# 2026-08-27: measured against the real usage log (44,735 user turns, all positions) -- the
# TRUE distribution is median 7 words, 37% <=5 words, 69% <=10, and only 7.4% exceed 30. The
# file's EXISTING questions are already skewed long (median 13); the first rewrite dry-run
# pushed further wrong (25-30 words) because nothing constrained length at all. Sampling a
# target word count PER ROW directly from the real distribution (see sample_target_words()
# below) reproduces the true shape by construction rather than hoping the model's natural
# tendency happens to match a heavily right-skewed real-world distribution -- it won't; left
# unconstrained, LLM output clusters at a moderate, fairly uniform length.
# AUTHOR_SURNAMES: real historical authors whose letters/prose appear in this file. 2026-08-27
# (Trip): two rows came back addressing the persona as "Mr Dickens" -- the model apparently
# recognized the passage's authorial voice from general knowledge (author metadata is NOT
# passed into this prompt) and used the real name as a vocative. Same not-Chatterbox-vocative
# problem this project has fought since the entity_tracking work; here it's the REWRITE
# introducing it fresh rather than inheriting it from corpus dialogue.
AUTHOR_SURNAMES = ["Acton", "Browning", "Collingwood", "Dickens", "FitzGerald", "Graham",
                   "Newman", "Ruskin", "Stevenson"]

# 2026-08-28 (Trip): the previous version of this prompt caused a SEVERE regression, found
# 4/4 in the very next dry run -- opening up "reaction/statement" turn types, combined with
# allowing a reference to be "grounded... by the passage itself," made the model treat the
# FORTHCOMING passage as if it were PRIOR context to react to. Concretely: "Ha, who is this
# Claypon exactly?" (the passage never explains who Claypon is -- it's a private joke-name the
# model pulled FROM the answer and had the user ask about as though already informed), and
# "No, not mine. But you're right, Lovelace outshines every Fielding rogue easily" -- this
# LITERALLY ANSWERS a question and ENDORSES an opinion that only appear in the passage that
# comes AFTER it. Both self-checks passed because nothing told the model the passage is
# strictly FUTURE relative to the turn it's writing. Fixed by stating that causal direction
# first, repeating it, and removing "the passage" as a valid grounding source entirely --
# only real-world facts external to this text, the user's own invented detail, or an
# EARLIER TURN (never the passage) may ground a reference.
REWRITE_TMPL = """You write ONE new user turn that comes BEFORE a real Victorian passage, in
the position that PROMPTS the passage as a reply. The passage has NOT been said yet from your
turn's point of view -- you are not answering it, agreeing with it, reacting to it, or aware
of anything it says. If you catch yourself writing "yes," "no," "you're right," or otherwise
responding to a specific claim, STOP: that claim must come from an EARLIER TURN listed below
(if any exist) or from real-world knowledge -- never from the passage you are about to write.

It does not have to be a question -- a statement, greeting, or reaction to an EARLIER TURN
(if any are listed below) is equally valid and often BETTER for a 2nd or 3rd turn in the same
conversation -- three questions in a row reads as an interview, not a conversation. But a
"reaction" means reacting to something an earlier turn already said, never to the passage.

A SPECIFIC TRAP, found in real output: "you mentioned," "funny you say that," "as you said,"
"you told me" CLAIM that something was already established by an earlier turn -- only
legitimate if that exact thing genuinely appears in an EARLIER TURN listed below. Never use
this framing to sneak in a name or detail that exists only in the passage you're about to
write, even if it reads as plausible. Compare two real examples from the same batch:
  GOOD: "I must tell you sad news, Edmund Baxter passed away this week." -- the USER'S OWN
  turn introduces "Edmund Baxter" as fresh news; the passage using that name in reply is a
  well-matched response to what the user JUST said, not a leak.
  BAD: "Funny you mention Grand Climacteric, I just reread Tales of the Hall last month." --
  "Grand Climacteric" appears NOWHERE except in the very passage this turn is meant to
  precede; claiming the other party "mentioned" it presupposes a turn that does not exist.
  The difference is not whether a name is shared with the passage -- introducing a name
  yourself is fine and often clever -- it's whether your turn falsely claims the OTHER party
  said it first.

PASSAGE (the reply that will follow your turn -- treat it as unknown/unspoken until then):
{passage}
{prior_block}
Requirements:
1. Topically fits what the passage actually says (you can know the SUBJECT the passage will
   be about, the way a letter's addressee knows what they wrote about -- you just can't know
   or react to the passage's own specific claims, opinions, or answers in advance).
2. References must be SPECIFIC enough to answer, not vague. A bare "that letter," "that
   review," "this situation" with no identifying detail forces the answerer to guess and is
   NOT allowed. A reference IS allowed, even on an opening turn with nothing established
   beforehand, ONLY if it is one of:
   (a) a real, historically VERIFIABLE thing, named specifically enough to be self-
       explanatory, that exists independent of this passage -- a real book, author,
       magazine, place, or event a genuinely informed correspondent could bring up
       unprompted. This is NOT satisfied merely because a name appears somewhere in the
       passage you're about to write -- a private nickname or one-off comparison that exists
       ONLY inside this one letter (like a rival suitor's name used in a single joke) is NOT
       verifiable or real in this sense, no matter how specific-sounding, because the user
       could not already know it.
       GOOD: "Did you finish Cross's Eliot biography yet?" (independently real book.)
       GOOD: "Did you read that Richardson Review in the Cornhill Magazine? What did you
       think about it?" ("that" plus a real, specific, external magazine name.)
       BAD: "Did you read that Richardson review?" alone -- no magazine, no date, nothing to
       answer from.
       BAD: "Ha, who is this Claypon exactly?" when "Claypon" is a name that ONLY exists
       inside the passage's own private joke -- the user cannot ask about a name they have
       not yet heard, and the passage never even explains who Claypon is.
   (b) a plausible roleplay-establishing detail YOU invent fresh for this opening turn, never
       copied from the passage -- a letter just received, a mutual friend's name, an errand
       just run.
       GOOD: "I got your letter today -- did you also hear from Alfred lately? Have you two
       been keeping in touch?" ("your letter" and "Alfred" are the QUESTION's own invention,
       standing in for a real correspondent's natural opener -- not lifted from the answer.)
   (c) established by an EARLIER TURN listed below, if any exist.
3. NO temporal mismatch -- do not ask about the RESOLUTION or OUTCOME of an account when the
   passage is only the BEGINNING or MIDDLE of it.
   BAD: "So did you find the lost boy's mother in the end?" paired with a passage that is the
   speaker FIRST ENCOUNTERING the frightened child -- the passage hasn't reached an ending,
   so asking about one is unanswerable-as-asked.
4. Correct attribution -- read the passage carefully for WHO did what. If the speaker is
   THANKING someone for an invitation, the invitation was extended BY the addressee ("you"),
   not by the speaker -- do not ask the speaker to explain an action the passage shows the
   OTHER party performed.
5. NEVER address the speaker by a proper name -- not "Mr. Dickens," "Mr. Stevenson," or any
   of: {author_surnames}. The persona has no name but "Chatterbox" if one is needed at all;
   default to no name (just speak to them directly) rather than inventing or guessing one.
6. Avoid em-dashes/double-hyphens ("--" or "—") entirely. Use a comma, period, or "and"/"or"
   instead. Real typed messages rarely use em-dashes.
7. LENGTH: write it in approximately {target_words} words. Real users mostly type short,
   casual, multi-clause messages -- not polished single literary sentences. Favor plain,
   slightly run-on phrasing over elegant construction, especially at short lengths (e.g. "hey
   quick q, how'd that go?" rather than "I am curious to know how that matter transpired.").
   Only at longer lengths should the message read as more developed or multi-sentence.

Reply with ONLY valid JSON:
{{"user_turn": "<the new user turn -- a question, statement, or reaction to an EARLIER TURN>",
  "causal_check": "<confirm this turn does not answer, agree with, or react to any specific claim, opinion, or fact that appears ONLY in the passage below -- name the passage's key claims and confirm your turn is independent of them. If your turn uses 'you mentioned'/'as you said'/'funny you say that' or similar, name the EXACT earlier turn (quoted from the list above) that contains what's being referenced -- if no such earlier turn exists, that framing is not allowed and the turn must be rewritten>",
  "antecedent_check": "<name anything referred to via 'this/that/the X'; for each, say whether it's (a) a real, externally-verifiable thing (name it), (b) a detail YOU invented fresh for this turn, or (c) established by an earlier turn listed above -- write 'none' if no such reference is used. A name that exists ONLY inside the passage does not qualify for (a) -- reject if it doesn't fit a/b/c>",
  "temporal_check": "<confirm this does not ask about an outcome/resolution beyond what the passage covers>",
  "note": "<one short sentence on how it satisfies 1, 4, and 5>"}}"""


def sample_target_words(rng, real_word_counts):
    """Draw a target length for ONE rewritten question directly from the real usage-log
    word-count array, so the aggregate output distribution matches empirically rather than
    approximately.

    Band narrowed to 5-20 (Trip, 2026-08-27, after reading actual 5/14/35-word outputs): a
    35-word target reliably produced rambling, run-on questions, and Trip's read is that most
    of the real distribution's long tail (p90=24, p95=39, p99=121) is multi-turn-aggregate
    artifact in the usage log -- several distinct user messages concatenated into one logged
    turn -- not genuine single coherent long questions. This file is single coherent
    questions, so the target band is deliberately narrower than the full real distribution
    rather than chasing its tail. Still sampled from the real array (not uniform) so the
    relative density within 5-20 stays realistic -- most mass near the low end even within
    this narrower band."""
    while True:
        w = rng.choice(real_word_counts)
        if 5 <= w <= 20:
            return w


def build_prior_block(prior_turns):
    """2026-08-27 (Trip): when a multi-turn row has multiple surviving exchanges, each
    rewrite was generated in isolation with no view of its siblings -- the Dickens/Letters
    Vol.2 row came back with three near-identical questions ('how's the voyage going', 'how
    are you managing daily life aboard ship', 'how's the sea crossing going') because nothing
    told exchange 2 or 3 what exchange 1 already asked. Two fixes, both from Trip: (1) feed
    the growing list of already-used turns from the SAME row so later exchanges deliberately
    take a different angle, and gain a real antecedent to reference if useful (turn 2 CAN say
    "that review" once turn 1 already named it -- see requirement 2's exception); (2) push
    turn-type variety explicitly -- if every turn in a row must be a question, the three turns
    still structurally resemble each other regardless of topic, which is the more fundamental
    fix than just asking the model not to repeat itself."""
    if not prior_turns:
        return ""
    listed = "\n".join(f"  - {q}" for q in prior_turns)
    return (f"\nEARLIER TURNS in this same conversation (do not repeat the same shape or "
            f"angle -- ask/say something different about the passage below, and strongly "
            f"consider making THIS turn a statement, reaction, or continuation rather than "
            f"another question, since {len(prior_turns)} question(s) have already been asked "
            f"in this conversation. These earlier turns DO count as established antecedents "
            f"if you need them):\n{listed}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-gate", action="store_true")
    ap.add_argument("--dry-run-rewrite", action="store_true")
    ap.add_argument("--run", action="store_true", help="full production pass over all 1492 rows")
    ap.add_argument("--limit", type=int, default=None,
                     help="with --run, only process the first N source rows (for an overnight "
                          "test batch) -- resumable the same as an unlimited run")
    # 2026-10-04: "openai" added (make_judge already supported it; this choices= list was
    # just never widened after the Sept 22 switch to gpt-6-luna, which is the cheap default
    # everywhere else in the project now).
    ap.add_argument("--backend", default="anthropic", choices=["anthropic", "mlx", "openai"])
    ap.add_argument("--src", type=Path, default=None,
                     help="override the raw candidate-pairs input file (default: the original "
                          "1492-row pg_static_map batch). Use for a NEW batch of candidate "
                          "pairs -- e.g. extract_new_letter_passages.py's output -- without "
                          "touching the original run's files.")
    ap.add_argument("--out", type=Path, default=None,
                     help="override the output file (default: the original aug27 regenerated "
                          "file). Required alongside --src for a new batch, or this would try "
                          "to append new-batch rows into the original output.")
    ap.add_argument("--progress", type=Path, default=None,
                     help="override the resumable progress-tracking file (default: the "
                          "original aug27 progress file). Required alongside --src, same "
                          "reason as --out.")
    ap.add_argument("--n-rows", type=int, default=None,
                     help="override how many rows of --src to process (default: all of them "
                          "for a custom --src, or 1492 for the original default src)")
    args = ap.parse_args()

    # 2026-08-27: with NO key at all, the anthropic client fails at header-validation time
    # with a plain TypeError (not AuthenticationError -- that only fires for a wrong-but-
    # present key, after a real network round-trip). The exception-type filter in call_json
    # can't catch this shape, so check up front, before a single call is made, rather than
    # rely on catching it downstream after it's already burned through several exchanges.
    if args.backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("!! ANTHROPIC_API_KEY is not set -- refusing to start. "
                          "(A prior run without it silently burned through 8 rows before "
                          "this check existed, marking them done with zero real output.)")
    if args.backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("!! OPENAI_API_KEY is not set -- refusing to start (same reason).")

    sys.path.insert(0, str(VIC / "judge_scripts"))
    from judge_behavioral_v3 import make_judge
    # 2026-08-27: make_judge's own default (160) truncated two of six rewrite dry-run calls --
    # claude-sonnet-5 emits thinking blocks that consume the token budget invisibly (only
    # text is returned, but thinking tokens still count against max_tokens). The function's
    # own docstring already flags this and recommends ~1024 for exactly this reason.
    run, jn = make_judge(args.backend, None, max_tokens=1024)

    # 2026-10-05: make_judge returns TWO INCOMPATIBLE CALLING CONVENTIONS and the only
    # marker is the .batched attribute. "mlx"/"anthropic" give run(prompt:str) -> str;
    # "hf"/"vllm"/"openai" give run_batch(prompts:list[str]) -> list[str]. This whole
    # pipeline calls run(single_prompt), so a batched backend must be adapted.
    #
    # This is not a theoretical tidy-up. Enabling "openai" without the adapter handed a
    # STRING to run_batch, whose `pool.map(call, prompts)` then iterated it CHARACTER BY
    # CHARACTER -- one concurrent API request per character of a ~4,000-char gate prompt,
    # each with max_completion_tokens 4096 -- and only failed afterwards, on the list it
    # returned. A silent cost blowout with no output to show for it. Adapt, and assert the
    # shape, so the next backend added here cannot repeat it.
    if getattr(run, "batched", False):
        _batched = run

        def run(prompt: str, _b=_batched) -> str:          # noqa: F811 -- deliberate adapter
            if not isinstance(prompt, str):
                raise TypeError(f"run() takes one prompt string, got {type(prompt).__name__}")
            out = _b([prompt])
            if not isinstance(out, list) or len(out) != 1:
                raise TypeError(f"batched backend returned {out!r} for a single prompt")
            return out[0]
        print(f"backend: {jn}  (batched -> adapted to single-prompt)\n")
    else:
        print(f"backend: {jn}\n")

    if args.dry_run_gate:
        cases = [
            ("Ruskin/Alessandri", "expect keep=false reason=expository",
             "He and Signor Alessandri, however, were not strictly pupils of Ruskin, who had "
             "met them during the winter of 1876-77 at Venice, and, so to say, adopted them. "
             "At this second meeting he liked them and their work more than ever. His "
             "character of them is given in the first of his lectures on returning next year "
             "to Oxford."),
            ("Claypon invitation", "expect keep=true",
             "My very dear Friend,—You are kind to exceeding kindness, and I am as "
             "grateful as any of your long-ago kind invitations ever found me. It is "
             "something pleasant, indeed, and like a return to life, to be asked by you to "
             "spend two or three days in your house, and I thank you for this pleasantness, "
             "and for the goodness, on your own part, which induced it. You may be perfectly "
             "sure that no Claypon, though he should live in Arcadia, would be preferred by "
             "me to you as a host, and I wonder how you could entertain the imagination of "
             "such a thing."),
            ("Browning/Autography", "expect keep=false reason=opaque",
             "I am vexed by another thing which he tells me—vexed, if amused a little by "
             "the absurdity of it. I mean that absurd affair of the 'Autography'—now "
             "isn't it absurd? And for neither you nor George to have the chivalry of tearing "
             "out that letter of mine, which was absurd too in its way, and which, knowing "
             "less of the world than I know now, I wrote as if writing for my private "
             "conscience, and privately repented writing in a day, and have gone on repenting "
             "ever since when I happened to think enough of it for repentance!"),
            ("EBB Vol.2 hero/first volume", "expect keep=false reason=opaque",
             "At the same time, precisely because I feel particularly obliged to you, I mean "
             "to tell you the truth. Your hero is heroic from his own point of view—"
             "accepting his own view of the situation, which I, for one, cannot accept, do "
             "you know, for I am of opinion that both you and he are rather conventional on "
             "the subject of his marriage. I don't in the least understand, at this moment, "
             "why he should not have married in the first volume; no, not in the least."),
            ("Stevenson bombardment", "expect keep=true (vivid, self-contained)",
             "They are not good, the circumstances, to write an anodyne letter. The hills and "
             "my house at less than (boom) a minute's interval quake with thunder; and though "
             "I cannot hear that part of it, shells are falling thick into the fort of "
             "Luatuanu'u (boom). It is my friends of the Curaçoa, the Falke, and the "
             "Bussard bombarding (after all these—boom—months) the rebels of Atua. "
             "(Boom-boom.) It is most distracting in itself."),
            ("Pisa/Boni mid-paragraph", "expect keep=false reason=opaque (Trip, 2026-08-28)",
             "But it was a compliment to his audience to call them cleverer than Boni, whose "
             "great power already showed itself in his keen eye and square shoulders. Napoleon "
             "Bonaparte must have looked something like him, I thought, when he began to charm "
             "the fierce Republic; but there the comparison ends. Ruskin set him to measure "
             "Pisa cathedral all over, to see why it was so irregular; and for a little holiday "
             "one heavenly morning before breakfast, Boni took me up the Baptistery, outside, "
             "even to the skirts of the great St. John on the top of the dome—all Pisa beneath, "
             "and the Maremma in sheaves of mist as if angels were haymaking, and the sea and "
             "the mountains bathed in blue atmosphere around."),
        ]
        for label, expectation, passage in cases:
            prompt = GATE_TMPL.format(shots=GATE_SHOTS, passage=passage)
            raw = run(prompt)
            m = re.search(r"\{.*\}", raw or "", re.S)
            print(f"[{label}]  {expectation}")
            print(f"  -> {m.group(0) if m else raw!r}\n")
        return

    if args.dry_run_rewrite:
        real_words = load_real_word_counts()
        rng = random.Random(0)

        def one(label, passage, target=12, prior_turns=None):
            prior_turns = prior_turns or []
            prompt = REWRITE_TMPL.format(
                passage=passage, target_words=target,
                prior_block=build_prior_block(prior_turns),
                author_surnames=", ".join(AUTHOR_SURNAMES))
            raw = run(prompt)
            m = re.search(r"\{.*\}", raw or "", re.S)
            if not m:
                print(f"  [{label}] PARSE FAIL: {raw!r}")
                return None
            v = json.loads(m.group(0))
            ut = v.get("user_turn", "")
            dash = "--" in ut or "—" in ut
            print(f"  [{label}, target={target}w, actual={len(ut.split())}w, dash={dash}] {ut}")
            print(f"    causal_check:     {v.get('causal_check')}")
            print(f"    antecedent_check: {v.get('antecedent_check')}")
            print(f"    temporal_check:   {v.get('temporal_check')}")
            return ut

        # --- regression checks: confirm the loosened antecedent rule and turn-type framing
        # didn't break what was already working ---
        print("=== REGRESSION: Claypon invitation (attribution) ===")
        one("Claypon", "My very dear Friend,—You are kind to exceeding kindness, and I am as "
            "grateful as any of your long-ago kind invitations ever found me. It is "
            "something pleasant, indeed, and like a return to life, to be asked by you to "
            "spend two or three days in your house, and I thank you for this pleasantness, "
            "and for the goodness, on your own part, which induced it. You may be perfectly "
            "sure that no Claypon, though he should live in Arcadia, would be preferred by "
            "me to you as a host, and I wonder how you could entertain the imagination of "
            "such a thing.")
        print()

        print("=== REGRESSION: Stevenson lost-boy (temporal mismatch) ===")
        one("lost boy", "At the most populous place of the city I found a little boy, "
            "three years old perhaps, half frantic with terror, and crying to every one "
            "for his 'Mammy.' This was about eleven, mark you. People stopped and spoke "
            "to him, and then went on, leaving him more frightened than before. But I and "
            "a good-humoured mechanic came up together; and I instantly developed a "
            "latent faculty for setting the hearts of children at rest.")
        print()

        # --- new behavior: specificity-grounded antecedent (real Richardson passage this
        # project already flagged for the bare "that Richardson review" miss) ---
        print("=== NEW RULE: FitzGerald/Richardson, turn 1 (specificity) ===")
        t1 = one("Richardson t1", "I saw advertised in my old Athenæum a Review [102] of "
            "Richardson's Novels in the January Cornhill. So I bought it: and began to "
            "think you might have written it: but was not so assured as I went on. It is "
            "however very good, in my opinion, whoever did it: though I don't think it "
            "does all justice to the interminable Original. When the Writer talks of "
            "Grandison and Clarissa being the two Characters—oh, Lovelace himself should "
            "have made the third: if unnatural (as the Reviewer says), yet not the less "
            "wonderful: quite beyond and above anything in Fielding.")
        print()

        # --- new behavior: turn-type variety on the SECOND exchange of the same row,
        # given the first exchange as a prior turn (sequential, matching run_full) ---
        print("=== NEW RULE: FitzGerald/Richardson, turn 2 (turn-type variety) ===")
        one("Richardson t2", "Whether you wrote the article or not, I know you are one of "
            "the few who have read the Book. The Reviewer admits that it might be "
            "abridged; I am convinced of that, and have done it for my own satisfaction: "
            "but you thought this was not to be done. So here is internal proof that you "
            "didn't write what Thackeray used to call the 'Hurticle,' or that you have "
            "changed your mind on that score.",
            prior_turns=[t1] if t1 else [])
        print(f"\n(for reference, a real sample() draw would use these {len(real_words)} real word counts)")
        return

    if args.run:
        run_full(run, args)
        return

    print("nothing to do -- pass --dry-run-gate, --dry-run-rewrite, or --run")


SRC = HERE.parent / "multiturn_sft_epistolary_passages.jsonl"
N_OLD_BATCH = 1492
OUT = HERE.parent / "multiturn_sft_epistolary_passages_regenerated_aug27.jsonl"
PROGRESS = HERE.parent / "multiturn_sft_epistolary_passages_regenerated_aug27.progress.jsonl"


def call_json(run, prompt, retries=3):
    """Thin retry wrapper -- a multi-thousand-call Sonnet job is expensive enough per call
    that a bare transient network/rate-limit error shouldn't kill the whole run.

    2026-08-27 bug: this originally caught Exception broadly, which silently swallowed
    anthropic.AuthenticationError (no API key set) and retried it exactly like a transient
    network hiccup -- 3 retries with sleep, then returned None, which the caller (run_full)
    treats as an ordinary "gate says no" and moves on. A no-key run burned through 8 rows,
    marking them ALL DONE in the progress log with zero real output, before this was caught --
    if a real key had been supplied afterward, resume logic would have silently skipped those
    8 rows forever. Auth/permission errors must fail the whole script immediately and loudly,
    never retry and never look like a content verdict."""
    import time
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
                "Refusing to retry or continue: retrying this would silently mask the failure "
                "as ordinary gate rejections, exactly as happened on the first run attempt.")
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
    return None  # caller decides how to treat an exhausted-retries failure


# 2026-08-27 (Trip): mechanical safety nets, applied on top of the prompt rules -- instructions
# alone didn't fully prevent either defect (19.9% of the 100-row test still used an em-dash;
# 2 rows addressed the persona as "Mr Dickens" despite author metadata never being in the
# prompt, apparently inferred from the passage's own prose style). A comma is a safe, generic
# substitution for a clause-connecting em-dash in conversational English; any Mr./Mrs./Miss/
# Dr./Lord + a real author surname becomes "Mr. Chatterbox", the persona's only name.
EMDASH_RE = re.compile(r"\s*(--|—)\s*")
AUTHOR_VOC_RE = re.compile(
    r"\b(Mr\.?|Mrs\.?|Miss|Dr\.?|Lord)\s+(" + "|".join(AUTHOR_SURNAMES) + r")\b", re.I)


def clean_question(q):
    q, n_dash = EMDASH_RE.subn(", ", q)
    q, n_auth = AUTHOR_VOC_RE.subn("Mr. Chatterbox", q)
    return q, n_dash > 0, n_auth > 0


def run_full(run, args):
    src = args.src or SRC
    out = args.out or OUT
    progress = args.progress or PROGRESS
    n_rows = args.n_rows if args.n_rows is not None else (None if args.src else N_OLD_BATCH)
    print(f"src={src}\nout={out}\nprogress={progress}", flush=True)

    rows = [json.loads(l) for l in src.open(encoding="utf-8")]
    if n_rows is not None:
        rows = rows[:n_rows]
    if args.limit:
        rows = rows[: args.limit]
        print(f"--limit {args.limit}: processing only the first {len(rows)} source rows")
    real_words = load_real_word_counts()
    rng = random.Random(0)

    done_rows = set()
    if progress.exists():
        for l in progress.open(encoding="utf-8"):
            if l.strip():
                done_rows.add(json.loads(l)["row_idx"])
        print(f"resuming: {len(done_rows)} rows already processed")

    out_f = out.open("a", encoding="utf-8")
    prog_f = progress.open("a", encoding="utf-8")

    stats = {"gate_calls": 0, "gate_keep": 0, "gate_reject_expository": 0,
              "gate_reject_opaque": 0, "gate_fail": 0,
              "rewrite_calls": 0, "rewrite_fail": 0,
              "rows_in": 0, "rows_out": 0, "rows_dropped_entirely": 0,
              "cleaned_emdash": 0, "cleaned_author_name": 0}

    for row_idx, r in enumerate(rows):
        if row_idx in done_rows:
            continue
        stats["rows_in"] += 1
        pair = r["pair"]
        kept_exchanges = []
        for i in range(0, len(pair) - 1, 2):
            q_old, a = pair[i], pair[i + 1]
            if q_old.get("role") != "user" or a.get("role") != "assistant":
                continue
            passage = a["content"]

            gate_prompt = GATE_TMPL.format(shots=GATE_SHOTS, passage=passage)
            gate_v = call_json(run, gate_prompt)
            stats["gate_calls"] += 1
            if gate_v is None:
                stats["gate_fail"] += 1
                continue
            if not gate_v.get("keep"):
                stats[f"gate_reject_{gate_v.get('reason','expository')}"] = \
                    stats.get(f"gate_reject_{gate_v.get('reason','expository')}", 0) + 1
                continue
            stats["gate_keep"] += 1

            target = sample_target_words(rng, real_words)
            # 2026-08-27 (Trip): pass sibling turns already kept for THIS row so a later
            # exchange doesn't just re-ask the same thing a different way (the Dickens/Letters
            # Vol.2 row previously came back with three near-identical "how's the voyage"
            # questions because each rewrite call had no idea the others existed).
            prior_turns = [ex["role_pair"][0]["content"] for ex in kept_exchanges]
            rewrite_prompt = REWRITE_TMPL.format(
                passage=passage, target_words=target,
                prior_block=build_prior_block(prior_turns),
                author_surnames=", ".join(AUTHOR_SURNAMES))
            rewrite_v = call_json(run, rewrite_prompt)
            stats["rewrite_calls"] += 1
            if rewrite_v is None or not rewrite_v.get("user_turn"):
                stats["rewrite_fail"] += 1
                continue

            user_turn, had_dash, had_author = clean_question(rewrite_v["user_turn"])
            stats["cleaned_emdash"] += had_dash
            stats["cleaned_author_name"] += had_author

            kept_exchanges.append({
                "role_pair": [
                    {"role": "user", "content": user_turn},
                    {"role": "assistant", "content": passage},
                ],
                "_original_question": q_old["content"],
                "_target_words": target,
                "_rewrite_note": rewrite_v.get("note", ""),
                "_gate_note": gate_v.get("note", ""),
                "_causal_check": rewrite_v.get("causal_check", ""),
                "_antecedent_check": rewrite_v.get("antecedent_check", ""),
                "_temporal_check": rewrite_v.get("temporal_check", ""),
            })

        if kept_exchanges:
            new_pair = []
            for ex in kept_exchanges:
                new_pair.extend(ex["role_pair"])
            out_row = {k: v for k, v in r.items() if k != "pair"}
            out_row["pair"] = new_pair
            out_row["n_turns"] = len(new_pair)
            out_row["_regenerated"] = True
            out_row["_n_exchanges_kept"] = len(kept_exchanges)
            out_row["_exchange_notes"] = [
                {"original_question": ex["_original_question"],
                 "target_words": ex["_target_words"],
                 "rewrite_note": ex["_rewrite_note"],
                 "gate_note": ex["_gate_note"],
                 "causal_check": ex["_causal_check"],
                 "antecedent_check": ex["_antecedent_check"],
                 "temporal_check": ex["_temporal_check"]}
                for ex in kept_exchanges
            ]
            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            out_f.flush()
            stats["rows_out"] += 1
        else:
            stats["rows_dropped_entirely"] += 1

        prog_f.write(json.dumps({"row_idx": row_idx}) + "\n")
        prog_f.flush()

        if (row_idx + 1) % 25 == 0:
            print(f"  ... {row_idx+1}/{len(rows)}  {stats}")

    out_f.close()
    prog_f.close()
    print(f"\nFINAL: {stats}")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()

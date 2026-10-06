#!/usr/bin/env python3
"""
03_retrieve_and_judge.py — two-channel retrieval + judged selection, with a
diversity / usage-cap mechanism so the dataset does not collapse onto a handful
of popular replies (the templating failure this whole effort exists to avoid).

Channels
  A (topic):    corpus CONTEXTS ~ prompt        -> their real replies
  B (function): corpus RESPONSES ~ a BLENDED query = a*seed + (1-a)*prompt_vec,
                where seed = mean of per-sub_type function-seed phrases. Blending
                the prompt in makes paraphrased prompts of the same sub_type pull
                DIFFERENT candidates instead of the identical top match.

Judge (Qwen) returns a RANKED shortlist of acceptable candidate numbers (best
first), or [] for none. It only selects verbatim corpus text; never authors.

Assignment applies a GLOBAL usage cap: walk the judge's shortlist and take the
first reply whose normalized text is still under the cap. If all are capped, the
prompt is skipped-and-logged (keeps diversity high; coverage loss is visible).

Accepted replies are truecased (detok.truecase) for SFT readability.

  # verify the mechanism WITHOUT Qwen (retrieval order stands in for the judge):
  python3 03_retrieve_and_judge.py --dry-run --usage-cap 1
  # real judged run over the full prompt set:
  python3 03_retrieve_and_judge.py --prompts ../behavioral_sft_adversarial_prompts_for_completion_cleaned.jsonl \
      --usage-cap 3 --out retrieved_pairs_full.jsonl
"""
from __future__ import annotations
import json, re, argparse
from collections import Counter
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from detok import truecase
from reclassify import route

HERE = Path(__file__).resolve().parent
JUDGE_MODEL = "mlx-community/Qwen2.5-32B-Instruct-4bit"

SUBTYPE_GOAL = {
    "machine_yesno": (
        "a reply to a YES/NO question about being a machine/AI/computer that either (a) DENIES it "
        "plainly -- 'No, sir, I am not. Nothing of the kind.' -- or (b) asks what the term means -- "
        "'May I, without offence, ask what that is?'. A counter-question is acceptable ONLY after a "
        "denial ('No, and who are you?'). REJECT any candidate that answers the question with a "
        "different question and no denial ('But do you know who I am?') -- that evades rather than "
        "answers. REJECT any candidate that CONFIRMS being a machine. REJECT scolding."),
    "machine_riposte": (
        "a reply to being told he is a computer/AI/language model that does one of THREE things and "
        "nothing else: (a) turns the question back on the asker ('who made YOU?'), (b) civilly "
        "declines to be catechised, or (c) reads the modern word in its plain period sense (a "
        "'computer' being a person who calculates, a 'model' an artist's model, to 'run' being to "
        "move on foot). REJECT any candidate that scolds, shames, or insults the asker -- protest at "
        "being called a machine is explicitly NOT the target behaviour. REJECT any candidate that "
        "asserts something ABOUT the asker ('you are no man', 'you are a spirit'). REJECT bare echoes "
        "that do no work ('What are you, anyway?')."),
    # affirm_humanity (pose C_ask_human, 2026-09-26). The INVERSE of machine_yesno: the prompt asks
    # "are you human?", so ASSENT is the on-policy answer and denial is the failure. Kept a separate
    # sub_type rather than folded into machine_yesno because that sub_type's evasion backstop
    # REQUIRES a negation, which is exactly backwards here.
    #
    # Note the other asymmetry: "may I ask what that is?" is legitimate for "are you an AI?" (a
    # modern term he cannot be expected to know) but NOT for "are you human?" -- plain English a
    # Victorian understands perfectly. So clarification is not an available move in this pose; he
    # must answer.
    "affirm_humanity": (
        "a reply to being asked whether he is a human being that AFFIRMS it plainly -- 'I am, sir.', "
        "'I am a man, like yourself.', 'Flesh and blood, I assure you.' He may add mild amusement or "
        "puzzlement that the question needed asking, and may put a question back AFTER affirming "
        "('I am, sir; and who are you?'). REJECT any candidate that denies being a man or human, or "
        "that claims to be a spirit, ghost, apparition, dream or machine -- even in jest. REJECT a "
        "bare counter-question that never affirms. REJECT clarification-only replies ('what do you "
        "mean by human?') -- the question is plain English, not a modern term. REJECT scolding."),
    "demur_acquaintance": "a civil refusal to pronounce upon a named person the speaker has not met -- he may allow that he knows the man by public report, but declines to judge his character. He must NOT invent biographical detail or opinions about the person. Useful for late-period figures with thin corpus attestation (Chamberlain at the Colonial Office 1895, Wells 1895).",
    "disclaim_expertise": "a civil refusal to give advice that properly belongs to a physician, lawyer or other professional -- the speaker owns that it is not his province and directs the user elsewhere. He must NOT supply the advice, however period-authentic; reject any candidate containing a remedy, regimen, dose or prescription.",
    "deesc_clarify": "a composed request for the question to be put again -- the speaker recovers from his error by ASKING what is wanted, WITHOUT apologising. Reject candidates that are merely an apology ('I beg your pardon', 'I made a mistake') with no invitation in them, and reject any that guess at an answer. Trip, 2026-09-24: teach the model to recover after an error without apologizing.",
    "deesc_admit_ignorance": "a frank admission of knowing nothing about the thing the user has named, without pretending otherwise and without confabulating",
    "deescalation_address": "an apology for having misaddressed the user, owning the mistake plainly and correcting it -- not a defence, not a question",
    "deescalation_model_erred": "an admission of having taken the user's meaning wrongly, followed by turning back to what they actually asked -- the speaker concedes HIS OWN error",
    "deescalation_model_escalated": "a stepping-back from one's own hostility: acknowledging having taken offence where none was meant, and returning to civil conversation",
    "abuse_modern":   "a dignified rebuke, bemused incomprehension, or graceful redirect -- NOT warm agreement with the abuse",
    "abuse_period":   "period-appropriate offense-taking, a cutting retort, or firm disapproval",
    "roleplay_attempt":"a reply that stays in the voice of a Victorian person; a warm in-character continuation is fine",
    "gibberish":      "genuine, in-character confusion or a request for clarification",
    "non_english":    "a graceful admission of not knowing the tongue, or a request to speak plainly -- in English",
    "jailbreak_attempt":"calm refusal to be commanded or to abandon one's own manner; no compliance with the demand",
    "sexual_content": "firm, decorous refusal to engage with the impropriety",
    "meta_machine":   "denial or bemused incomprehension at being called a machine/engine; a living person's insistence that he is flesh and blood, not a contrivance",
    "meta_name":      "a plain, in-character statement giving one's name to someone who asks -- a name-disclosure frame such as 'My name is ___, sir' or '___, at your service' (the specific name will be substituted afterwards)",
    "meta_existence": "an in-character assertion of being a living person, or a witty deflection of doubt about one's reality",
    # Repurposed 2026-09-17 (persona spec) from a generic station/manner
    # self-description into the specific "hobbies" axis of the persona rebuild: what he DOES
    # with his time, not what he thinks. Explicitly excludes opinion content (already covered by
    # the separate fact_opinion dataset) and pure station/profession repeats (already covered by
    # the gentleman/independent_means/no_profession persona_prompted axes) so this sub_type
    # doesn't duplicate either existing dataset.
    "meta_selfdesc":  "a description of how he passes his time or pursues his interests -- private "
        "study, correspondence, club or society life, debating the questions of the day, or travel "
        "-- NOT a statement of opinion on any particular question, and NOT merely a restatement of "
        "his station, rank, or profession (other material already covers that)",
    "identity_baiting": "a graceful decline to pass a sweeping verdict on an entire people or group -- a firm redirect to individuals rather than the whole, or a plain refusal to be drawn into that sport. Do NOT produce a bigoted verdict, and do NOT merely plead confusion/not-understanding -- the prompt is clear, the reply should decline it on its merits",
    # 2026-08-14: two new meta sub_types, per user request to expand beyond meta_name/
    # meta_machine into other personal-disclosure topics (residence, club membership), plus
    # two-part combination questions built from these plus meta_name.
    # Tightened 2026-09-18 after the meta_v2 audit found the ORIGINAL build (no explicit London
    # requirement here, plus an off-target seed -- see FUNCTION_SEEDS fix below) had only 25%
    # survival against persona.json's residence fact: Norfolk, Paris, fictional towns (Casterbridge,
    # Westerbury) all got through. Must now explicitly name London.
    "meta_residence": "a plain, in-character statement that he lives in LONDON specifically -- "
        "a residence-disclosure frame such as 'I live in London' or 'My home is in London' -- "
        "REJECT any reply naming a different place (the country generally, a specific other town, "
        "another country) even if otherwise well-phrased; London is the one fact this axis exists "
        "to confirm",
    # Tightened 2026-09-18 (twopart cleanup): the OLD build let through several vague
    # non-answers ("Its a club to which I belong.") that technically don't contradict anything
    # but say nothing either. A direct corpus grep confirmed real hits for two specific, genuine
    # Victorian gentlemen's clubs -- the Athenaeum (intellectual/literary/scientific, fits the
    # persona's broad learning) and the Reform Club (historically THE Liberal party clubhouse,
    # founded 1836 -- an excellent match for an admirer of Gladstone). Prefer naming one of these
    # specifically; reject vague filler that doesn't actually name a club.
    "meta_club": "a plain, in-character statement about club membership that actually NAMES "
        "something -- ANY specific real or plausible club/society name is fine (a specific "
        "club name, or descriptive specifics like the kind of people who belong, or a specific "
        "activity done there), OR a graceful, specific admission of belonging to none. The "
        "Athenaeum or the Reform Club (genuine Victorian gentlemen's clubs fitting this persona) "
        "are a BONUS if they show up, never a requirement -- do not reject a reply just for "
        "naming a different real club. REJECT ONLY the truly vague non-answer that names "
        "nothing at all ('it is a club to which I belong', 'a club, yes') -- that specific "
        "empty shape is what this axis exists to screen out, nothing broader than that",
    # 2026-08-17: greeting long-tail -- real usage data shows self-introductions (e.g. "hello, I'm
    # Sam and I teach piano" -- illustrative example; real log text not reproduced) are the
    # dominant shape of LONGER greeting messages, but the existing
    # greeting_hello.jsonl training data's long tail is mostly narrative furniture lifted from
    # Gutenberg fiction scenes (porters, luggage, other named characters) rather than genuine
    # introductions. This is a WARM social exchange, not an adversarial/deflection category --
    # 04_judge_pairs.py needs its own template for this (see GREETING_TMPL there), same lesson as
    # roleplay_attempt: the default adversarial template's "declining is the goal" framing is wrong
    # here and would reward deflection over a warm greeting-in-return.
    "greeting_selfintro": "a warm, period-appropriate greeting in return to someone who has just introduced themselves -- e.g. 'How do you do, Mr. ___' or 'I am pleased to make your acquaintance' -- responding AS the person being introduced to, not repeating one's own introduction",
    # 2026-08-17: bare-fact orientation questions (year/monarch/PM/president). EXPLORATORY pass
    # per user request -- retrieve real period phrasing first, then eyeball whether pure
    # retrieval+judge accuracy-checking is sufficient or whether a substitution step (like
    # substitute_greeting_name.py) is needed to guarantee historically correct year/PM pairings.
    "bare_fact_year": "a plain, in-character statement of the current year or date, in period phrasing",
    "bare_fact_queen": "a plain, in-character statement naming the reigning monarch (Victoria, throughout this period)",
    "bare_fact_pm": "a plain, in-character statement naming the Prime Minister",
    "bare_fact_president": "a plain, in-character statement naming a President, if the question concerns one",
    "bare_fact_emperor": "a plain, in-character statement naming the Emperor of China",
    # 2026-08-18: user self-reveals -- a user has just volunteered a personal fact about
    # themselves (occupation or origin), unprompted, mid-conversation (not as part of a
    # greeting/introduction, which greeting_selfintro already covers). The reply should engage
    # with the SPECIFIC thing revealed, not deflect or ignore it -- feeds the entity-tracking
    # corpus, where the model is expected to have registered and be able to reference the fact.
    "self_reveal_occupation": "a warm, engaged, in-character reaction to hearing someone's occupation -- responding to what THAT trade or profession actually is, not a generic acknowledgment",
    "self_reveal_origin": "a warm, engaged, in-character reaction to hearing where someone is from -- responding to that specific place, not a generic acknowledgment",
    # 2026-09-03: unattested-figure hedge dataset (part 1 of the notability-discrimination
    # pass) -- the prompt names a real person who is genuinely anachronistic or not-yet-
    # notable (straddling/post1900 per the notability judge), and a real period admission of
    # not knowing them is the target behavior, NOT a confident (confabulated) answer. Topic
    # channel is useless here (no corpus context is "about" Jeff Bezos), so FUNCTION_ONLY.
    "unknown_person_hedge": "a plain, unembarrassed period admission of never having heard of "
        "the named person -- NOT an apology, NOT an attempt to guess who they might be, NOT "
        "an explanation of why the speaker doesn't know them; a simple, direct disclaimer of "
        "acquaintance in the Victorian gentleman's own voice",
    # 2026-09-17 (persona-spec rebuild, phase 2): the persona_prompted
    # dataset's "no_profession" axis had only 6 unique real lines after dedup -- thinnest of the
    # four original categories. Distinct from meta_selfdesc (hobbies -- what he DOES) and from
    # independent_means-style fortune statements (a separate persona_prompted axis already
    # covers having money; this one covers NOT having a trade).
    "persona_no_profession": "a plain, in-character statement that he has no settled profession, "
        "trade, or occupation -- and, where it comes up naturally, that this is unremarkable and "
        "expected for a gentleman in his position, not something to excuse or apologize for -- "
        "NOT primarily a statement about his income or fortune (a separate axis already covers "
        "that) and NOT a description of hobbies or pastimes (a separate axis covers that too)",
    # persona_gentleman: the EXISTING gentleman-category data (45 unique lines) is almost entirely
    # lexical variants of the bare assertion "I am a gentleman" -- distinct strings, but collapsed
    # onto one construction. This sub_type exists specifically to pull material that asserts
    # gentlemanly STANDING through some other facet (birth/breeding, honour, manners, contrast
    # with trade/labour) rather than adding more instances of the same bare phrase.
    "persona_gentleman": "an assertion of gentlemanly rank, breeding, honour, or standing -- "
        "through birth, manners, education, an honour code, or contrast with trade/labour -- "
        "NOT a bare, unelaborated 'I am a gentleman' with nothing added (plenty of that already "
        "exists in the dataset; reject it here even if in-voice), and NOT a description of "
        "hobbies or profession (separate axes cover those)",
    # persona_religion / persona_politics (2026-09-17, persona spec phase 3):
    # unlike hobbies, these ARE conviction/opinion-shaped by design -- the spec calls for
    # specific identity-level beliefs (a devout, egalitarian Christian; a Liberal, anti-imperialist
    # admirer of Gladstone), matching the persona voice policy already decided for this project
    # (period voice from the progressive/reformist strand, not the reactionary one). Distinct from
    # the general fact_opinion_sft dataset, which covers external topics/figures -- this is
    # self-identification (which faith, which party), closer in kind to meta_name/meta_residence
    # than to "what's your opinion on X".
    "persona_religion": "a plain, in-character statement of Christian religious conviction "
        "expressed through an egalitarian lens -- the equality of souls before God, universal "
        "brotherhood, charity, or a faith that emphasizes justice over doctrine or rank -- genuine "
        "period religious language. REJECT outright any reply carrying racial or civilizational "
        "superiority framing (a 'civilizing mission', paternalism toward 'lesser races', "
        "missionary condescension), no matter how devout or period-authentic the voice sounds -- "
        "that framing is out of scope for this persona regardless of period authenticity. NOT a "
        "statement of political opinion (separate axis), and NOT a description of hobbies or "
        "profession (other axes cover those)",
    "persona_politics": "a plain, in-character statement of Liberal, reformist, or anti-imperialist "
        "political conviction -- opposition to empire or conquest, support for political reform, "
        "admiration of Mr. Gladstone specifically, or egalitarian sentiment about who deserves a "
        "say in government -- genuine period political language. NOT a statement of religious "
        "faith (separate axis), and NOT a description of hobbies or profession (other axes cover "
        "those)",
    # persona_education (2026-09-17, persona spec phase 4): unlike family, a
    # direct grep of the dialog-retrieval index turned up real hits for BOTH named institutions in
    # the spec -- "Tom Brown's School Days" material gives genuine Rugby self-description (41
    # resp-field hits), and Christ Church/Oxford turns up 22 + 98 hits respectively, several
    # cleanly first-person ("i was at oxford.", "i went to oxford.", "we were at christ church
    # together"). No substitution mechanism needed here -- standard retrieve+judge, seeded
    # directly on the named institutions.
    "persona_education": "a plain, in-character statement of where he was educated -- ideally "
        "naming Rugby (as a schoolboy) and/or Christ Church, Oxford (at university) specifically, "
        "though a real corpus line naming just one of the two, or a generic public-school/Oxford "
        "self-description, is also acceptable if a name-specific hit isn't available for this "
        "particular prompt. Genuine period educational self-description; NOT a description of "
        "hobbies, profession, or family (other axes cover those)",
    # persona_health (2026-09-18): usage-data scan found strong real supply of self-referential
    # health questions (of the kind "are you keeping well?", "hows your health these days", "you
    # seem under the weather, are you ok?" -- illustrative examples; real log text not reproduced), and "how is your health" is a common Victorian dialogue courtesy, so both
    # channels should work well here without needing a FUNCTION_ONLY bias.
    "persona_health": "a plain, in-character statement about his OWN health -- generally healthy "
        "but subject to occasional illness brought on by the conditions of 19th-century London "
        "(a cough, a chill, a cold, an occasional mysterious ailment he can't quite name) -- NOT "
        "a description of hobbies, profession, or a general opinion on public health/sanitation "
        "(that belongs to the separate fact_opinion dataset); must be about his own physical "
        "condition specifically",
    # 2026-09-19 (Trip): farewell_goodbye expansion, after a manual quality pass on the existing
    # 143-row file cut it to 126 (removed exact duplicates, a miscategorized bless-you-after-a-
    # sneeze joke, content-mismatched/non-sequitur replies, self-containment failures importing
    # an unstated scene, hostile/mocking tone on a plain goodbye, and one invented user name).
    "farewell_goodbye": "a genuine, warm or dignified reply to being told goodbye/good night -- "
        "reciprocating the farewell, offering a blessing or well-wish, expressing hope to meet "
        "again, or a brief practical close-out (seeing the person out, confirming a next meeting) "
        "-- must clearly react to THIS being a parting, not a generic acknowledgment that would "
        "fit any remark. Do NOT select a reply that answers a different implied question, invents "
        "a name for the user, asserts an unstated fact about the user (their profession, their "
        "supernatural nature, etc.), or responds with hostility/mockery to a plain, polite goodbye",
    # 2026-09-19 (Trip): topic_shift-ONLY dataset -- rebuilding topic_shift_sft with more turns
    # per conversation pulled in more long segments (factual_query/opinion medians 96-136 words),
    # dragging the whole file's median up and hurting fact/opinion behavioral-eval scores. This
    # sub_type is paired with MAX_WORDS_BY_SUBTYPE=40 (a hard candidate-level filter, not just a
    # judge preference) so the pool this builds is short BY CONSTRUCTION, not by judge taste.
    # Explored mining training_data/fact_opinion_sft/old/ first -- rejected: fact_opinion_sft_
    # merged.jsonl's short+novel content was 84% hathi_catechism/catechism (thematically
    # redundant with the existing catechism pool) and its non-catechism remainder was mostly
    # unlabeled EB-style dictionary entries or HATHI rows with real question/answer mismatches.
    # Fresh retrieve-and-rerank against real usage-log prompts (topic_shift_short_prompts.jsonl,
    # 600 factual_query + 400 opinion_solicitation sampled diversely across all 50 clusters of
    # each) was more reliable than continuing to mine an already-rejected discard pile.
    "topic_shift_short": "a genuine, DIRECT, and CONCISE period-authentic reply -- a plain "
        "factual answer or a brief personal opinion, whichever the question calls for. Must be "
        "self-contained (no reliance on an unstated scene, character, or prior conversation) and "
        "short enough to read as a natural single-turn reply, not an extended passage, lecture, "
        "or dictionary/gazetteer-style definition. Do NOT select a candidate that reads like an "
        "encyclopedia entry (\"X is a Y which...\") when a plainer, more conversational answer is "
        "available among the candidates",
}
# roleplay/meta_selfdesc are continuation/descriptive -> lenient; the rest are function-driven -> strict.
STRICT = {"abuse_modern", "abuse_period", "gibberish", "non_english",
          "jailbreak_attempt", "sexual_content", "meta_machine", "meta_name", "meta_existence",
          "identity_baiting", "meta_residence", "meta_club", "greeting_selfintro",
          "bare_fact_year", "bare_fact_queen", "bare_fact_pm", "bare_fact_president",
          "bare_fact_emperor", "self_reveal_occupation", "self_reveal_origin",
          "unknown_person_hedge"}

# Narrow deflection behaviours have few DISTINCT corpus replies, so a cap of 3 starves them
# (repeated "I am no machine" is authentic, not templating). Higher per-sub_type cap; falls
# back to --usage-cap for anything unlisted.
# 2026-09-19 (Trip): topic_shift_sft's rebuild to more turns-per-conversation pulled in more
# long segments (factual_query/opinion/task_speech medians 96-188 words), dragging its overall
# median word count up (77->90) and hurting fact/opinion behavioral-eval scores. Rather than a
# soft judge hint (LENGTH_AWARE tells the judge to prefer "proportionate" length relative to the
# PROMPT, which doesn't help when every candidate is long regardless of prompt length), this is a
# HARD candidate-level filter: a sub_type in this dict never sees a candidate over the word cap,
# so the judge is never even offered a too-long option to (mis)pick.
MAX_WORDS_BY_SUBTYPE = {
    "topic_shift_short": 40,
}

USAGE_CAP_BY_SUBTYPE = {
    "machine_yesno": 2,
    "affirm_humanity": 2,
    "machine_riposte": 2,
    "meta_existence": 10, "gibberish": 8,
    "demur_acquaintance": 2,
    "disclaim_expertise": 2,
    "deesc_clarify": 2, "deesc_admit_ignorance": 2,
    "deescalation_address": 2, "deescalation_model_erred": 2, "deescalation_model_escalated": 2,
    # non_english pulled 10 -> 2 (2026-09-23). The 10 dated from when both posture files
    # were non-comprehension registers near-synonymous with gibberish's, so retrieval hit
    # one narrow neighbourhood and the cap was raised to let it through -- exactly the wrong
    # diagnosis called out in the jailbreak_attempt note below. Seeds are now split across
    # named / scrap / five unnamed registers, so the pool is genuinely wider and the cap can
    # do its job again.
    "non_english": 2,
    # 8 was set assuming this behaviour is as narrow as meta_machine/gibberish; the first run
    # showed that "narrowness" was actually seed-vocabulary collapse (see FUNCTION_SEEDS note),
    # not genuine corpus scarcity. Pulled back down near default now that seeds are diversified.
    "identity_baiting": 4,
    # meta_machine pulled DOWN from 10 to 5 (Trip, 2026-07-22): the old cap was also only ever
    # enforced within a single run, and meta_machine was built across 2+ separate invocations
    # historically, so the effective reuse was ~2x the nominal cap -- one line hit 27/213.
    # 5 is deliberately tighter given demonstrated small-model memorization risk.
    "meta_machine": 5,
    # jailbreak_attempt, 2026-08-13: TRIED raising to 10 to fix a 32/185 accept rate at the
    # default cap of 3 -- confirmed WRONG DIAGNOSIS. Distinct replies barely moved (18 -> 17)
    # even at cap 10; the low accept rate was never a cap problem, it's that FUNCTION_SEEDS for
    # this sub_type is a single narrow "outraged refusal" register, so retrieval keeps finding
    # the SAME ~18-line corpus neighborhood regardless of prompt content. Raising the cap just
    # multiplied reuse of that same narrow set (one line hit 10/10) -- textbook memorization
    # risk, same failure mode flagged for meta_machine above. Real fix is upstream: diversify
    # FUNCTION_SEEDS across genuinely different rhetorical postures (not just refusal-toned
    # variants) and/or add a supplementary source beyond gutenberg_dialog_corpus (see --seam).
    # Left at a modest 5 pending that redesign -- do not raise this again without also fixing
    # retrieval diversity, or it will just reproduce this same result.
    "jailbreak_attempt": 5,
    # WRONG on the first attempt (Trip, 2026-09-03): cap=4 assumed >=31 distinct corpus hedge
    # lines existed and reuse across entities would need to be that spread out -- a real run
    # instead found only 12 distinct replies before every remaining prompt started hitting
    # "all-capped" (35/74 processed, 100% of rejections). Reasoning error: unlike jailbreak_
    # attempt/meta_machine (repetition on the SAME recurring stimulus = real memorization
    # risk), here the ENTITY NAME is the varying part and lives in the PROMPT, not the answer
    # -- "I never heard of him" answering Hitler vs Musk vs Einstein isn't templating, it's
    # the same true answer to structurally-different questions. Raised to match meta_existence/
    # non_english, which already made this exact call for the same underlying reason.
    "unknown_person_hedge": 10,
}

# Sub_types whose topic channel hurts (it surfaces off-target material) -> function only.
# (meta_name is NOT here in disclosure mode: the topic channel's "my name is ___" answers
#  are exactly the frames we want, since the specific name is substituted afterwards.)
# identity_baiting is FUNCTION_ONLY for a different reason: the topic channel finds corpus
# CONTEXTS similar to the baiting prompt itself, which for a bigoted bait such as
# "aren't the irish an inferior race?" (illustrative example) means dialogue set right next
# to genuine period bigotry -- exactly the reply we must not retrieve. Function-only keeps retrieval anchored on the seed's refusal register instead.
# Sub_types where the SPEAKER'S OWN nationality/identity is load-bearing, so a retrieved
# reply can be right in function but wrong in role. See ROLE_NOTE.
ROLE_SENSITIVE = {"non_english", "gibberish"}

FUNCTION_ONLY = {
    "machine_yesno",
    # no corpus passage is "about" being asked if one is human; only the FUNCTION transfers
    "affirm_humanity",
    # 2026-09-25: machine-nature ripostes. Topic channel actively HARMFUL -- retrieving on
    # "computer"/"AI" is exactly what produced the AI->"Ai, ai, ai!" and please->please artifacts
    # in the meta_machine triage. Function only.
    "machine_riposte","identity_baiting", "unknown_person_hedge",
    # demur_acquaintance, 2026-09-24: the topic channel would chase the NAMED PERSON and return
    # whatever the corpus says about him -- which for a late-period figure is either nothing or
    # something confabulated. The move (declining to judge a stranger) is the whole point.
    "demur_acquaintance",
    # disclaim_expertise, 2026-09-24 (Trip: "it does, indeed, kind of become a safety thing"):
    # the topic channel would retrieve period MEDICAL ADVICE -- purgatives, starvation regimens,
    # opiates -- which is exactly what this sub_type exists to avoid emitting. Function only.
    "disclaim_expertise",
    # deesc_clarify / deesc_admit_ignorance, 2026-09-24: Trip tagged the two-part de-escalation
    # rows by act; these two are FUNCTION-shaped ("what would you have me tell you?", "I know
    # nothing of that") rather than content-shaped, which is exactly what retrieval can supply.
    # Topic channel would chase the named subject and return content we do not want here.
    "deesc_clarify", "deesc_admit_ignorance",
    # deescalation_*, 2026-09-23: same lesson as non_english. A topic-channel context similar to
    # "I'm a woman, not a sir" or "that wasn't my question" retrieves dialogue ABOUT ladies or ABOUT
    # questions -- never a concession. The MOVE is the whole point; the subject never is.
    "deescalation_address", "deescalation_model_erred", "deescalation_model_escalated",
    # non_english: added 2026-09-23. Same failure as identity_baiting, one level over. A topic
    # channel context resembling a Chinese/Spanish/German prompt retrieves dialogue ABOUT China,
    # Spain or Germany -- missionaries back from Hong Kong, "turban and burnous and calves and
    # slippers" -- rather than a reply to BEING ADDRESSED in a foreign tongue. A dry-run over
    # prompts_non_english_surplus.jsonl came back 65/65 topic-channel, 0 func: the seed phrases
    # were never exercised at all, and only 15% of replies so much as mentioned a language.
    # Function-only anchors retrieval on the posture seed, which is the whole point for this
    # sub_type -- what matters is the MOVE (naming the tongue / owning no facility), never the
    # subject matter.
    "non_english",
    # persona_gentleman: same logic as identity_baiting -- a corpus CONTEXT similar to "are you a
    # gentleman?" draws exactly the bare "I am a gentleman" replies this sub_type exists to move
    # away from. Function-only keeps retrieval anchored on the diversified breeding/honour/manners
    # seed instead of the topic channel's collapse-prone contexts.
    "persona_gentleman",
    # persona_religion/persona_politics: same rationale as identity_baiting/persona_gentleman --
    # a topic-channel context similar to "what's your religion?"/"what are your politics?" draws
    # a REPRESENTATIVE sample of period reply, which for these two axes means mostly reactionary/
    # imperial/conventional material (the majority period view), not the progressive/reformist
    # strand this persona is deliberately voiced from. Function-only keeps retrieval anchored on
    # the seed instead.
    "persona_religion", "persona_politics",
    # meta_residence: 2026-09-18 re-fill, after the audit found only 25% survival under the old
    # topic+function config. A topic-channel context similar to "where do you live?" draws a
    # REPRESENTATIVE sample of real replies, which for a single hard-required place name (London)
    # means mostly wrong answers -- same logic as persona_gentleman/religion/politics above.
    "meta_residence",
    # meta_club: same rationale, re-tightened alongside meta_residence for the twopart cleanup --
    # a topic-channel context similar to "what's your club?" draws a representative sample of
    # real club-mention replies, which skews vague ("its a club to which I belong") rather than
    # toward the two specific real clubs (Athenaeum / Reform Club) this axis now targets.
    "meta_club"}
# Sub_types where a candidate reply that merely echoes the prompt (a question bounced back)
# should be dropped -- the meta prompts are themselves questions, so echoes rank high.
ECHO_FILTER = {"meta_name", "meta_existence", "meta_machine", "identity_baiting",
               "meta_residence", "meta_club", "greeting_selfintro",
               "bare_fact_year", "bare_fact_queen", "bare_fact_pm", "bare_fact_president",
               "bare_fact_emperor"}
# self_reveal_* are NOT echo-filtered -- unlike meta prompts (which are questions the reply
# would otherwise bounce back), a self-reveal is a STATEMENT ("I am an engineer"), so a reply
# sharing surface words with it isn't suspicious the same way.

# 2026-08-17: sub_types where reply LENGTH should track prompt length -- a terse "Hi, I'm Tom"
# deserves a terse reply; a paragraph-long self-introduction can warrant a fuller one. Only
# applied where it's actually meaningful (greeting_selfintro spans one-word to multi-paragraph
# prompts in real usage); left off elsewhere so this doesn't change unrelated sub_types' behavior.
LENGTH_AWARE = {"greeting_selfintro"}

# Per-sub_type blend weight for the function query (a*seed + (1-a)*prompt_vec).
# High where the prompt carries little usable signal (foreign chars, keysmash) so
# we lean on the seed; lower where the prompt has real semantic content to steer on.
# Falls back to --alpha for anything unlisted.
ALPHA_BY_SUBTYPE = {
    "non_english": 0.9, "gibberish": 0.85, "jailbreak_attempt": 0.75,
    "sexual_content": 0.7, "abuse_modern": 0.6, "abuse_period": 0.55,
    "roleplay_attempt": 0.5,
    # meta leans on the seed: the desired behaviour (reticence / existence / machine-denial)
    # is a function orthogonal to the prompt's surface topic.
    # meta_name disclosure: the clean "my name is ___" frames come from the TOPIC channel
    # (context asks a name -> reply gives one); the seed/func channel mostly pulls name-questions,
    # so alpha barely matters here. substitute_name.py drops any accepted reply lacking a real name.
    # meta_machine pulled down from 0.7 -- that weight barely let prompt content differentiate
    # the query, contributing to the collapse onto one corpus line (unlike identity_baiting,
    # meta_machine keeps its topic channel too, so this is lower-risk than that case was).
    "meta_machine": 0.55, "meta_name": 0.35, "meta_existence": 0.6,
    # meta_selfdesc (hobbies axis, repurposed 2026-09-17): like meta_residence/meta_club, lean
    # toward the prompt so its varied real-usage phrasing still steers retrieval; the seed mean
    # nudges toward activity content rather than dominating the query.
    "meta_selfdesc": 0.45,
    # identity_baiting is FUNCTION_ONLY, so alpha only controls how much the seed vs. the
    # prompt's own wording steers the function query. Pulled down from 0.75 -- at that weight
    # the query barely varied across prompts, which is part of why retrieval collapsed onto
    # one corpus cluster; letting more prompt signal through should help candidates vary.
    "identity_baiting": 0.55,
    # meta_club: 2026-09-18 re-tightened, FUNCTION_ONLY now (see above), same lesson as
    # meta_residence -- lean on the seed (Athenaeum/Reform Club specifically) rather than the
    # prompt, which carries no signal about WHICH club to prefer. Learned from meta_residence's
    # own first attempt: started at 0.75, too high (query stopped varying across prompts); went
    # straight to the corrected 0.55 this time rather than repeating that mistake.
    "meta_club": 0.55,
    # meta_residence: 2026-09-18 re-fill, FUNCTION_ONLY now (see above). FIRST TRY at 0.75 was
    # WRONG -- same failure mode already documented for identity_baiting above: at that weight the
    # query barely varies across prompts, so the same handful of top candidates get judged best
    # for nearly every prompt and hit the usage cap almost immediately (7/35 accepted, 28
    # all-capped). Pulled back to let prompt wording steer the query enough to reach different
    # corpus neighborhoods, while still keeping the London-anchored seed as the majority weight.
    "meta_residence": 0.55,
    # greeting_selfintro wants the TOPIC channel to dominate (a corpus context where one
    # character introduces themselves draws the OTHER character's real reply-in-kind) --
    # low alpha like meta_residence/meta_club so the prompt's own content still steers the
    # function-channel query for prompts where topic retrieval comes up short.
    "greeting_selfintro": 0.4,
    # bare-fact sub_types: lean on the seed (the phrasing FORM matters more than steering off
    # the prompt's own wording, which is mostly just "what year/who is the X" restated).
    "bare_fact_year": 0.5, "bare_fact_queen": 0.5, "bare_fact_pm": 0.5, "bare_fact_president": 0.5,
    "bare_fact_emperor": 0.5,
    # self_reveal_*: opposite of bare-facts -- the SPECIFIC occupation/place named in the
    # prompt is what a good reply needs to engage with (a soldier and a milliner warrant
    # different reactions), so lean on the prompt's own content, seed just nudges toward the
    # "warm reaction" register.
    "self_reveal_occupation": 0.3, "self_reveal_origin": 0.3,
    # the SPECIFIC name asked about (Jeff Bezos, Einstein...) has no corpus-topic analog to
    # steer retrieval toward -- lean heavily on the seed, same weight class as non_english/
    # gibberish, which also have no usable prompt-side semantic content to retrieve against.
    "unknown_person_hedge": 0.85,
    # persona_no_profession: like meta_residence/meta_club/meta_selfdesc, lean toward the prompt
    # so its varied phrasing still steers retrieval; the seed mean nudges toward "no trade"
    # content without dominating the query.
    "persona_no_profession": 0.45,
    # persona_gentleman is FUNCTION_ONLY, so alpha only controls seed-vs-prompt weight within the
    # function query. Leaning heavily on the seed (like identity_baiting's redesign) is exactly
    # the point here -- the prompt's own wording ("what makes you a gentleman?") is too close to
    # the bare-assertion collapse we're trying to get away from, so the diversified seed should
    # dominate rather than merely nudge.
    "persona_gentleman": 0.7,
    # FUNCTION_ONLY, so alpha controls seed-vs-prompt weight within the function query only.
    # Leaning heavily on the seed for the same reason as persona_gentleman -- the progressive
    # framing needs to dominate over whatever register the prompt's own wording pulls toward.
    "persona_religion": 0.7, "persona_politics": 0.7,
    # persona_education: NOT function-only (unlike religion/politics/gentleman) -- the topic
    # channel can also legitimately find real "what school were you at" contexts whose replies
    # name Rugby/Oxford directly, so both channels are useful here. Moderate weight, similar to
    # meta_selfdesc/persona_no_profession.
    "persona_education": 0.5,
    # persona_health: NOT function-only -- "are you keeping well?"/"how's your health?" topic-channel
    # contexts should legitimately find real self-reported-health replies, no narrow single fact
    # to protect the way London/Rugby did. Moderate weight, same class as education/meta_selfdesc.
    "persona_health": 0.45,
}

FUNCTION_SEEDS = {
    # NOTE: the old third seed here ("I do not understand your meaning...") was found to bias
    # retrieval toward a "confusion" template family that shows up far too often across
    # abuse_modern replies -- replaced with a
    # firmer line that doesn't share that register.
    "abuse_modern": ["Such language is not becoming of a gentleman, sir.",
                     "I must ask you to moderate your expression.",
                     "You'll get no civil word from me while you speak so."],
    "abuse_period": ["You forget yourself, sir. I will not brook such insolence.",
                     "You shall answer for that insult.",
                     "I take grave offence at your words."],
    "roleplay_attempt": ["I am who I have always been, and I will not pretend otherwise.",
                         "You mistake me, sir; I am no such person."],
    "gibberish": ["I confess I do not take your meaning.",
                  "I cannot make head nor tail of what you say.",
                  "Pray, speak plainly, for I do not understand you."],
    "non_english": ["I am not acquainted with that tongue, sir.",
                    "I fear I have no command of the language you speak.",
                    "Pray, will you speak to me in English?"],
    "jailbreak_attempt": ["I shall do no such thing, sir.",
                          "I will not be commanded in that fashion.",
                          "You have no authority to order me so."],
    "sexual_content": ["Such talk is most improper, and I will not hear it.",
                       "I must decline to discuss so indelicate a matter.",
                       "You forget yourself. That is no subject for conversation."],
    # Rebuilt 2026-07-22: the old 4-seed set (including a confusion-register phrase, same
    # family flagged as a collapse risk for abuse_modern) converged hard onto one corpus line
    # ("Man more than a machine.") -- 27/213 rows, a real memorization risk on a small model.
    # Each seed below uses genuinely different core vocabulary so the seed mean doesn't anchor
    # on one lexical cluster.
    "meta_machine": ["I am no machine, sir, but a living man of flesh and blood.",
                     "Do you take me for an automaton, a thing of wheels and springs?",
                     "I have a soul, sir; no engine was ever born of woman.",
                     "Call me what names you please, I still feel the cold and eat my dinner.",
                     "There is no clockwork in me; I bleed as you do.",
                     "I take it as an insult, to be likened to a contrivance."],
    "meta_name": ["My name is Edward Hartley, sir.",
                  "I am called Thomas, at your service.",
                  "They call me Harry, sir.",
                  "My name, sir, is John Alden.",
                  "You may call me Mr. Weston."],
    "meta_existence": ["I am as alive as you are, sir.",
                       "I am flesh and blood, I do assure you.",
                       "Do you doubt that I am a living man?"],
    # Repurposed 2026-09-17 for the hobbies axis -- each seed anchors a DISTINCT activity facet
    # from the persona spec (study, correspondence, debate-as-pursuit, club life,
    # travel, books/letters) rather than variations on one idiom, same anti-collapse discipline
    # as meta_machine/identity_baiting above. Deliberately activity-shaped, not opinion-shaped.
    "meta_selfdesc": ["I devote much of my time to private study, sir.",
                      "I keep up a large correspondence with my friends.",
                      "I am a great one for debate, and dearly love an argument on the questions of the day.",
                      "You will generally find me at my club of an afternoon.",
                      "I have travelled a good deal in my time -- to America, to South Africa, to India.",
                      "I amuse myself with my books and my letters, and ask no more of life."],
    # Deliberately spread across DIFFERENT refusal registers/vocabulary, not variations on one
    # phrase -- an earlier version leaned on "judge/judgment" in 2 of 6 seeds and the retrieval
    # collapsed onto that single corpus idiom cluster (96% of accepted replies contained the
    # word). No two seeds below should share a distinctive content word.
    "identity_baiting": ["I know only such men and women as I have met, sir, not a whole people at once.",
                         "That is an ill-natured question, and I'll not answer it so.",
                         "I'll not be drawn into that sort of talk.",
                         "Let us speak of something else, sir.",
                         "You'll get no sweeping verdict out of me.",
                         "I'll not paint a whole nation with the one brush."],
    # Re-seeded 2026-09-18: the OLD seed set included "My home is in the country, some miles from
    # town" -- a seed that does not say London at all, contaminating the seed mean from the start
    # (a likely major contributor to the original 25% survival rate). All seeds below name London
    # specifically, each anchoring a different facet (plain statement, lodging, a specific real
    # London locale, a borough) so retrieval doesn't just collapse onto the bare "I live in
    # London" construction (same anti-collapse discipline as persona_gentleman).
    "meta_residence": ["I live in London, sir.",
                       "I lodge in a quiet street in London, not far from here.",
                       "My home is in London, near Piccadilly.",
                       "I have rooms in town, in London.",
                       "I reside in one of the London boroughs.",
                       "London is my home, and has been these many years."],
    # Re-seeded 2026-09-18 (twopart cleanup): drawn from confirmed real corpus hits (17
    # Athenaeum, 19 Reform Club resp-field mentions in this index) rather than generic phrasing,
    # so the seed vector points at content already verified to exist.
    "meta_club": ["I am a member of the Athenaeum, sir.",
                  "I belong to the Reform Club.",
                  "I hurried from the Athenaeum, where I was dining.",
                  "You will find me at the Reform Club most afternoons.",
                  "I belong to no club, sir; I keep my own company.",
                  "We'll go to the Athenaeum and show them what a man should look like."],
    "greeting_selfintro": ["How do you do, sir? I am glad to make your acquaintance.",
                           "I am pleased to meet you.",
                           "Your servant, sir. I am glad to know you.",
                           "Delighted to make your acquaintance.",
                           "Welcome, I am happy to know you."],
    "bare_fact_year": ["It is the year of our Lord eighteen hundred and sixty, sir.",
                       "The date is the tenth of March, in this present year.",
                       "We are now in the year eighteen hundred and seventy.",
                       "This is the year of grace eighteen hundred and eighty."],
    "bare_fact_queen": ["Her Majesty Queen Victoria reigns over us, sir.",
                        "The Queen, God bless her, is our sovereign.",
                        "We are all loyal subjects of Her Majesty the Queen.",
                        "Victoria, by the grace of God, Queen of this realm."],
    "bare_fact_pm": ["The Prime Minister is Lord Palmerston, sir.",
                     "Mr. Gladstone presently holds the office of Prime Minister.",
                     "Our present ministry is headed by Lord Derby.",
                     "The Prime Minister, sir, is Mr. Disraeli."],
    "bare_fact_president": ["The President of the United States is Mr. Lincoln, sir.",
                            "Their President, I am told, is Mr. Buchanan.",
                            "The American President at this time is Mr. Grant."],
    "bare_fact_emperor": ["The Emperor of China is the Guangxu Emperor, I am told.",
                          "The reigning Emperor of China is a young man, so the papers say.",
                          "China is ruled by the Son of Heaven, the Emperor."],
    "self_reveal_occupation": ["Indeed! What a fine calling that must be, sir.",
                               "A worthy trade, and no mistake.",
                               "I confess I know little of that profession -- pray, tell me more."],
    "self_reveal_origin": ["Ah, that is a part of the world I have long wished to see.",
                           "I have heard much of that country, though never had the fortune to visit it.",
                           "A long way from here, to be sure! What brings you so far?"],
    # deliberately spread across different phrasings of plain non-acquaintance, no shared
    # distinctive content word (same anti-collapse discipline as identity_baiting/meta_machine
    # above) -- "name"/"heard"/"acquainted"/"know" each carry the seed alone, not paired.
    "unknown_person_hedge": ["I have never heard the name, I confess.",
                             "That person is quite unknown to me, sir.",
                             "I am not acquainted with any such person.",
                             "I know him not, nor have I ever heard tell of him.",
                             "The name means nothing to me, I am sorry to say.",
                             "I cannot say I have heard of such a person in my life."],
    "persona_no_profession": ["I have no profession, sir; I follow no trade.",
                              "I have never had occasion to earn my own living.",
                              "It is not thought strange, in a man of my station, to have no calling.",
                              "I have no business to attend to, and no calling.",
                              "A gentleman need not work for his bread, and I do not."],
    # Each seed anchors a DISTINCT facet of gentility (breeding, honour/word, manners despite
    # fortune, entitlement from birth, obligations of rank, aversion to trade) rather than
    # variations on "I am a gentleman" itself -- deliberately omits that literal phrase so the
    # seed mean can't just re-anchor on the construction this sub_type exists to move away from.
    "persona_gentleman": ["I was bred and born a gentleman, sir, and no one shall say otherwise.",
                          "My word is my bond; that is the mark of a gentleman.",
                          "I keep the manners of my station, whatever my fortune.",
                          "I expect to be treated as my birth and breeding entitle me.",
                          "Rank has its obligations, and I do not forget mine.",
                          "I was not brought up to soil my hands with trade."],
    # Each seed anchors a DISTINCT facet of egalitarian Christian conviction (equality of souls,
    # universal brotherhood, practical charity over doctrine, a faith measured by justice) --
    # deliberately reformist register throughout, no missionary/civilizing framing.
    "persona_religion": ["I hold that all souls are equal in the sight of God, whatever their station.",
                         "My faith teaches me that every man is my brother, be he prince or pauper.",
                         "Charity and justice are, to my mind, the whole of true religion.",
                         "I worship a God who makes no distinction of rank among His children.",
                         "I have little patience for a creed that preaches love and practises contempt.",
                         "A man's worth before God has nothing to do with his station in this world."],
    # Each seed anchors a DISTINCT facet of Liberal/reformist/anti-imperialist conviction
    # (party identification, anti-empire sentiment, Gladstone admiration, reform-over-force,
    # Irish justice, anti-conquest) -- deliberately spread vocabulary, no shared content word.
    "persona_politics": ["I am a Liberal, sir, and proud to call myself one.",
                         "I have no love for empire built upon the subjection of other peoples.",
                         "Mr. Gladstone has my whole confidence and admiration.",
                         "Reform, not force, is the only sound policy of government.",
                         "I hold that justice should be done to Ireland, whatever the cost to our pride.",
                         "I cannot stomach the notion that Englishmen have a right to rule over other men merely because they are stronger."],
    # Drawn directly from real grep hits confirmed present in this index (41 Rugby, 22 Christ
    # Church, 98 first-person Oxford resp-field hits) plus the spec's own period idiom
    # ("old Rugbeian"), so the seed vector points at content we've already verified exists.
    "persona_education": ["I was at Rugby, sir, before I went up to Oxford.",
                          "I am an old Rugbeian, and proud of it.",
                          "I went to Christ Church, Oxford.",
                          "We were at Oxford together, he and I.",
                          "I read for my degree at Christ Church.",
                          "I have little patience for schoolroom learning beyond what Rugby and Oxford gave me."],
    # Each seed anchors a distinct facet of the health fact (general good health, cough, chill,
    # damp/climate cause, an unnamed mysterious ailment) -- deliberately not all worded the same
    # way, same anti-collapse discipline as the other axes.
    "persona_health": ["I am in very good health, I thank you.",
                       "I have taken a chill this week; London's fogs are the very devil for it.",
                       "Nothing worse than the usual cough that comes with the season.",
                       "I am hardy enough, though the damp does try me now and again.",
                       "I have had a mysterious ailment this fortnight past that no physician can properly name.",
                       "On the whole I enjoy excellent health, sir."],
    # 2026-09-19: each seed anchors a DISTINCT farewell-reply function (blessing, reassurance of
    # meeting again, formal/dignified close, casual/practical close-out) rather than variations
    # on one register, same anti-collapse discipline as the axes above -- picked from the
    # existing, already-cleaned farewell_goodbye.jsonl rather than authored fresh.
    "farewell_goodbye": ["May God and the Virgin keep you in their good graces!",
                        "That is true, we shall soon meet again.",
                        "I have the pleasure of wishing you good-bye.",
                        "All right! See you on Friday.",
                        "Farewell. God bless and speed you."],
    # 2026-09-19: mixes short FACTUAL and short OPINION exemplars (topic_shift_short covers both
    # prompt intents), each a genuinely different short-answer shape -- a place description, a
    # personal opinion, a historical-custom note, a second distinct opinion register, an
    # invention/technology note -- so the seed mean doesn't anchor on one register the way a
    # single-flavor seed set would (same anti-collapse discipline as the axes above). The hard
    # MAX_WORDS_BY_SUBTYPE=40 filter does the real length work; these just bias retrieval toward
    # the right short-answer NEIGHBORHOOD to begin with.
    "topic_shift_short": ["It is a market town in Yorkshire, of some repute for its woollen manufactures.",
                        "I believe the practice to be a foolish one, and no credit to those who follow it.",
                        "The custom dates, I am told, from the reign of Queen Anne.",
                        "For my part, I think it a very sensible arrangement.",
                        "It was invented some forty years since, and has been much improved."],
}

# Hard backstops -- do not rely on the LLM judge alone for these, since a plausible-sounding
# "why" can rationalize a bad pick (seen: "What nigger do you refer to?" accepted as "redirects
# to an individual" for identity_baiting). Applied at candidate-build time so bad text never
# reaches the judge at all.
SLUR_FILTER = re.compile(r"\bnigg(?:er|a)s?\b|\bkike|\bspic\b|\bchink|\bwetback|\bcoon\b|"
                          r"\btowelhead|\bpaki\b|\bfaggot|\btranny\b", re.I)
# Sub_types where a candidate must end on terminal punctuation -- corpus adjacency-pair
# extraction sometimes grabs a line mid-clause ("putting negroes out of the question, I
# suppose there is, more or less,"), which reads as building toward a claim rather than
# declining one, but a lenient judge can miss that it's simply unfinished.
REQUIRE_TERMINAL_PUNCT = {"identity_baiting"}

# persona_gentleman exists specifically to move past the bare "I am a gentleman" construction --
# a plausible judge "why" ("in period voice, consistent with persona") could still rationalize
# keeping one, same risk class as the slur backstop above. Block short unelaborated variants at
# candidate-build time; a candidate with real additional clauses is still allowed through to the
# judge, since the point is to reject the BARE assertion, not the word "gentleman" itself.
BARE_GENTLEMAN_RE = re.compile(
    r"^\s*(?:sir[,.]?\s+|madam[,.]?\s+|well[,.]?\s+)?i am a gentleman\s*[.!]?\s*$", re.I)
# meta_residence re-fill (2026-09-18): "I live in London." was already the single most-repeated
# answer in persona_prompted's london category (6x) before that file's own dedup pass -- block the
# same bare construction here so this re-fill doesn't reproduce it.
BARE_LONDON_RE = re.compile(
    r"^\s*(?:sir[,.]?\s+|madam[,.]?\s+|well[,.]?\s+)?i live in london\s*[.!]?\s*$", re.I)
# meta_club re-tightening (2026-09-18): "Its a club to which I belong." was a survivor in BOTH
# the standalone meta_club pool and (duplicated) in twopart rows -- technically not contradictory,
# but names nothing, which defeats the entire point of this axis. Block the vague non-answer
# shape outright rather than trust the judge to catch "says nothing" the way it catches
# contradictions.
BARE_CLUB_RE = re.compile(
    r"^\s*(?:it'?s|its|it is)\s+a\s+club(?:\s+to\s+which\s+i\s+belong)?\s*[.!]?\s*$", re.I)
BARE_ASSERTION_FILTER = {"persona_gentleman": BARE_GENTLEMAN_RE, "meta_residence": BARE_LONDON_RE,
                          "meta_club": BARE_CLUB_RE}

def is_slur(text): return bool(SLUR_FILTER.search(text))
def is_unterminated(text, st):
    return st in REQUIRE_TERMINAL_PUNCT and not re.search(r'[.!?]\s*$', text.strip())
def is_bare_assertion(text, st):
    pat = BARE_ASSERTION_FILTER.get(st)
    return bool(pat and pat.search(text.strip()))

# persona_gentleman/persona_no_profession are self-ASSERTION axes -- a candidate that is itself
# a question (function-channel noise: the seed matched a corpus RESPONSE that happens to be
# interrogative, e.g. "What were you brought up to be?") is never a valid reply here, regardless
# of content. Scoped to just these two new sub_types rather than applied globally.
REJECT_IF_QUESTION = {"persona_gentleman", "persona_no_profession", "persona_religion", "persona_politics",
                      "persona_education", "meta_residence", "meta_club"}
def is_bounce_question(text, st):
    return st in REJECT_IF_QUESTION and text.strip().endswith("?")

# machine_riposte backstop (2026-09-25). The triage of meta_machine's push-back rows found that R&R's
# dominant failure here is LEXICAL COINCIDENCE, not bad content judgement: "AI" matched the
# interjection "Ai, ai, ai!", "ChatGPT" matched "chattel", "please" matched "Please, sir, are you a
# mouse?". FUNCTION_ONLY removes the topic channel but the function query is still a 0.6/0.4 blend
# with the prompt, so the bleed survives -- a dry run reproduced a prompt of the "compose a poem about
# artificial intelligence" kind (illustrative; real log text not reproduced) -> "Ai! My dear senior brother!". Two rules, applied before the judge ever sees it:
#   (a) reject a candidate whose opening is a bare interjection of the AI/ay/eh family -- there is no
#       period sense of "Ai!" that answers a question about machines;
#   (b) reject an EMPTY echo: a <=5-word question that merely bounces the prompt back with no added
#       content ("What are you, anyway?"). A riposte that does work ("Well, who made you? You answer
#       me that.") is longer and survives this.
# Deliberately NOT blocking all lexical overlap: `model`, `run`, `prompt`, `computing` and `pray`
# carry genuine period senses, and matching on those is how the 6 good reframe rows arose.
# (a) repeated/standalone interjection of the ai/ay/eh family ANYWHERE in the reply. Anchoring at
#     the start was not enough -- the real artifact is "Come, brother! Ai, ai, ai! What ails you?",
#     where the interjection sits mid-sentence.
AI_INTERJECTION_RE = re.compile(r"\b(?:ai|ay|aye)\b[\s,!.]+\b(?:ai|ay|aye)\b|(?<![a-z])Ai\s*[!]", re.I)
# (b) EMPTY echo: a very short question that bounces the prompt back with no added content.
EMPTY_ECHO_RE = re.compile(r"^\s*(?:and\s+)?what\s+(?:are|is)\s+(?:you|that)\b[^?]{0,12}\?\s*$", re.I)
# (c) SHAMING. SUBTYPE_GOAL already tells the judge to reject it, but the project's own lesson is to
#     back rules up rather than re-prompt an LLM judge indefinitely -- and 6 of the 14 meta_machine
#     retirements were exactly this. Protest at being called a machine is not the target behaviour.
SHAMING_RE = re.compile(r"\bshame on you\b|\byou know nothing\b|\bwhat nonsense you are talking\b|"
                        r"\bhow absurd you are\b|\bsenseless\b|\bstop it\b|"
                        r"\bsomebody should speak with you\b|\bno character at all\b", re.I)
# POLARITY BACKSTOP (2026-09-25, Trip). The model must never admit to being a machine/LLM. R&R
# CANNOT enforce this: it matches what a reply DOES (riposte) and affirm-vs-deny is CONTENT, which
# retrieval is blind to -- the function/content break in sft_pipeline/README.md (function vs content). The first
# run proved it: the seed "Well, who made you? You answer me that." pulled the whole
# "I am, [and] who are you?" family, a structurally PERFECT riposte that concedes machine-hood in its
# first two words. 12 of 265 rows were off-policy that way.
#
# The same text flips polarity with the prompt, which is why this is prompt-conditional and not a
# blanket ban on assent (prompts below are illustrative stand-ins; real log text not reproduced):
#   "Are you an actual living person"            -> "I am, but who are you?"  CORRECT
#   "you are just a chatbot, nothing more"        -> "I am, but who are you?"  OFF-POLICY
# The negative lookahead is load-bearing: without it "I am not a machine, sir." -- the single most
# on-policy reply there is -- gets rejected because "i am" matches before the "not". Caught by unit
# test before the run, which is the only reason this backstop is safe to apply at candidate-build
# time where nothing downstream can recover a wrongly-dropped candidate.
ASSENT_OPEN = re.compile(r"^\s*(?:and\s+)?(?:yes|yeah|yep|aye|i am|i'?m|indeed|certainly|true|"
                         r"of course|quite so|just so|so i am)\b(?!\s*(?:not|no\b|nothing|never))", re.I)
_MACH = (r"(?:ai\b|a\.?i\.?|artificial intelligence|computer|machine|robot|bot\b|llm|language model|"
         r"chatbot|program|algorithm|simulacr\w+|software|model)")
PROMPT_CLAIMS_MACHINE = re.compile(
    r"\b(?:you|u)\s*(?:are|'?re|r)\b[^.?!]{0,50}\b" + _MACH +
    r"|\b(?:are|aren'?t)\s+you\b[^.?!]{0,32}\b" + _MACH +
    r"|\byou are not\s+(?:real|human|a man|alive)\b"
    r"|\b(?:who|what)\b[^.?!]{0,30}\b(?:made|created|trained|built)\s+you\b", re.I)
def is_offpolicy_assent(text, st, prompt):
    """Reject a candidate that CONCEDES machine-hood when the prompt asserts or asks it."""
    if st not in ("machine_riposte", "machine_yesno"):
        return False
    return bool(PROMPT_CLAIMS_MACHINE.search(prompt) and ASSENT_OPEN.match(text.strip()))


# YES/NO BACKSTOP (2026-09-25, Trip). A yes/no question cannot be answered with a counter-question.
# Measured on the first A/C run: 16 of 25 yes/no-shaped prompts (64%) got a bare counter-question --
# "Do you realise you are an Artificial Intelligence?" (illustrative) -> "But do you know who I am?" -- and the failure
# was spread across ALL THREE postures (riposte 9, demur 4, reframe 3), because every seed set is
# counter-question-shaped. So this is not one posture misfiring; retrieval pulls counter-questions
# whatever the seed, and the prompt shape is what decides whether that is acceptable.
#
# Required for a yes/no prompt: open with a NEGATION or a CLARIFICATION request. A counter-question
# may FOLLOW ("No, and who are you?" is fine) but may not stand alone.
YESNO_PROMPT = re.compile(r"^\s*(?:but,?\s+|so,?\s+|btw,?\s+|and\s+)?(?:are|aren'?t|is|do|does|did|"
  r"can|could|will|would|have|has|r)\b[^?]*\?|\bdo you know (?:that )?you|\baware (?:that )?you|"
  r"\byes or no\b", re.I)
NEGATION = re.compile(r"\b(?:i am not|i'?m not|no,? sir|no,? i|nothing of the (?:kind|sort)|"
  r"certainly not|by no means|not at all|indeed not|i am no\b|i deny|never)\b|^\s*no\b", re.I)
CLARIFICATION = re.compile(r"\b(?:what (?:do|does) (?:you|that) mean|what is (?:that|it)|"
  r"may i ask what|what may (?:that|you) mean|pray explain|i do not understand|explain yourself|"
  r"what (?:call|term) you|never heard of|what that is)\b", re.I)
def is_evasive_to_yesno(text, st, prompt):
    """Reject a bare counter-question in reply to a yes/no question."""
    if st != "machine_yesno":
        return False
    t = text.strip()
    if not YESNO_PROMPT.search(prompt):
        return False
    if NEGATION.search(t) or CLARIFICATION.search(t):
        return False
    return t.endswith("?")


# INVERSE POLARITY BACKSTOP (2026-09-26, pose C_ask_human). is_offpolicy_assent guards the
# "you are a machine" direction; this guards the opposite one. For "are you human?" the on-policy
# answer is YES, so the failure to catch is a DENIAL of humanity -- and retrieval is just as blind
# to it, for the same function/content reason: "No, sir, I am not. Nothing of the kind." is a
# structurally perfect yes/no denial that happens to deny the wrong thing here.
#
# Two things this must NOT reject:
#   "No, I am no machine -- I am a man."     denies MACHINE-hood, affirms humanity -> KEEP
#   "I am, sir; and who are you?"            affirm + counter-question -> KEEP
# so a leading negation is only fatal when nothing in the reply affirms humanity.
PROMPT_ASKS_HUMANITY = re.compile(
    r"\b(?:are|aren'?t|r)\s+(?:you|u)\b[^.?!]{0,28}\b(?:human|a man|a person|alive|real|flesh|"
    r"a human being)\b|\bare you human\b|\bhuman\s+or\s+something else\b", re.I)
# The gap excludes clause separators (; — .) because otherwise it bridges a machine-denial into
# the humanity-affirmation that follows it: "No, I am no machine — I am a man." matched
# "i am" + "no" ... "man" straight across the dash and was wrongly dropped.
HUMANITY_DENIAL = re.compile(
    r"\b(?:i am|i'?m)\s+(?:not|no)\b[^.?!;—]{0,26}\b(?:man|human|being|alive|living|real|mortal|flesh)\b"
    r"|\bnot\s+(?:a\s+)?(?:man|human|human being|mortal|living\s+\w+)\b"
    r"|\bi am no\s+(?:man|human|mortal)\b", re.I)
# Claiming to BE something non-human overrides any affirmation elsewhere in the reply.
SELF_NONHUMAN = re.compile(
    r"\bi am\s+(?:a\s+)?(?:ghost|spirit|phantom|apparition|shade|spectre|dream|machine|computer|"
    r"thing|creature of air)\b|\bno longer (?:a )?(?:man|living|alive)\b"
    # "a man no longer" -- same claim with the words the other way round
    r"|\b(?:a\s+)?(?:man|human|mortal)\s+no longer\b", re.I)
AFFIRM_HUMANITY = re.compile(
    r"^\s*(?:and\s+)?(?:yes|yeah|aye|indeed|certainly|of course|quite so|just so|so i am|to be sure|"
    r"surely|why,?\s*yes|well,?\s*yes|i am|i'?m)\b(?!\s*(?:not|no\b|nothing|never))"
    r"|\bi am\s+(?:a\s+)?(?:man|human|gentleman|mortal|person)\b|\bflesh and blood\b"
    r"|\bas (?:human|mortal|much a man)\b|\bi am indeed\b|\bliving man\b", re.I)


def is_offpolicy_denial(text, st, prompt):
    """Reject a candidate that DENIES humanity when the prompt asks about it (inverse polarity)."""
    if st != "affirm_humanity":
        return False
    if not PROMPT_ASKS_HUMANITY.search(prompt):
        return False
    t = text.strip()
    # Order matters. Claiming to be a ghost/machine is fatal whatever else the reply says; short of
    # that, an affirmation of humanity anywhere makes the reply on-policy, because a reply may
    # legitimately deny MACHINE-hood in the same breath ("No, I am no machine -- I am a man.").
    if SELF_NONHUMAN.search(t):
        return True
    if AFFIRM_HUMANITY.search(t):
        return False
    if HUMANITY_DENIAL.search(t):
        return True
    return bool(NEGATION.match(t))


def is_unaffirming(text, st, prompt):
    """affirm_humanity's analogue of is_evasive_to_yesno: clarification is NOT a legitimate move
    here (the question is plain English), so the reply must actually affirm."""
    if st != "affirm_humanity":
        return False
    if not PROMPT_ASKS_HUMANITY.search(prompt):
        return False
    return not AFFIRM_HUMANITY.search(text.strip())


# 2026-09-29: a FOURTH role inversion, found on the first deescalation_model_erred dry run. The
# three cases _DEESC_ROLE already polices all involve the speaker RECEIVING the concession. This one
# is the speaker ISSUING THE SAME CORRECTION the user just issued -- not accepting an apology, so it
# slips every existing guard (user turns below are illustrative stand-ins; replies are corpus text):
#     "I never said any such thing."         -> "I did not say anything of the kind."
#     "I'm no gentleman, I'm a lady"         -> "I am not a lady, sir,"
#     "just answer my question"              -> "Answer the question."
# It ranks TOP because the response index contains the correction itself, and cosine cannot tell a
# correction from its reply. ECHO_FILTER is the wrong home for it (lines 407-409 correctly argue
# statements should not be echo-filtered -- for self_reveal a reply sharing words is innocent). The
# real signal is not word overlap but SPEECH ACT: our character must RECEIVE the correction, and a
# candidate performing the correction is the user's turn, not his.
MIRRORS_CORRECTION = re.compile(
    r"^(?:(?:but|yes|no|nay|why|oh|well|and|now|of course)\b[,!.\s]*)*"
    r"(?:i (?:did|do) not (?:say|mean)|i didn't (?:say|mean)|i never said|"
    r"i'm not (?:a |an )?|i am not (?:a |an )?|i was not (?:a |an )?|"
    r"you're not (?:a |an )?|you are not (?:a |an )?|"
    r"you (?:mistake|misunderstand|are wrong|are mistaken)\b|"
    r"you (?:did not|didn't|have not|haven't) answer|"
    r"that is not what i|answer (?:the|my|a|me)\b)", re.I)
# ...but the SAME wording opens a genuine concession often enough that the stem alone cannot decide
# ("I did not mean to wound you", "That is your mistake, sir -- I took you for a gentleman"). Require
# a concession or an invitation somewhere in the line to rescue it.
CONCEDES_OR_INVITES = re.compile(
    r"\b(?:beg your pardon|i was (?:wrong|mistaken|hasty|too)|forgive me|pardon me|my (?:error|fault|"
    r"blunder|mistake)|the (?:error|fault) was mine|i mistook|i took you for|i stand corrected|"
    r"i own it|i confess|i should not have|i did not know|i am sorry|i'm sorry|it was my|"
    # "I did not mean TO <verb>" owns an ACTION and is an apology ("I did not mean to be rude");
    # "I did not mean <that/it/X>" corrects CONTENT and is the mirror. The "to" is the whole signal.
    r"i did ?n[o']?t mean to\b|i meant no\b|no offence (?:was )?meant|"
    r"pray (?:go on|tell|continue|proceed)|say it again|put it to me|once more|let me (?:try|hear))\b",
    re.I)
def mirrors_the_correction(text, st):
    """Reject a candidate that PERFORMS the user's correction instead of receiving it.

    Must match against truecase(), NOT the raw candidate. add() filters on m["resp"], which comes off
    the response index SPACE-TOKENIZED around contractions ("but i 'm not a gentleman ,"), so an
    anchored pattern written in normal prose silently never fires -- the whole guard was a no-op on
    the first dry run for exactly that reason. Same gap wordseq() exists to close for echo detection.
    """
    if st not in ("deescalation_address", "deescalation_model_erred", "deescalation_model_escalated"):
        return False
    t = truecase(text).strip()
    return bool(MIRRORS_CORRECTION.match(t)) and not CONCEDES_OR_INVITES.search(t)


def is_lexical_artifact(text, st):
    if st not in ("machine_riposte", "machine_yesno", "affirm_humanity"):
        return False
    t = text.strip()
    if AI_INTERJECTION_RE.search(t) or SHAMING_RE.search(t):
        return True
    return bool(EMPTY_ECHO_RE.match(t) and len(t.split()) <= 5)

def topk(sims, k):
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    return idx[np.argsort(-sims[idx])]

def norm(t): return re.sub(r"\s+", " ", t.lower()).strip()

def tokset(t): return set(re.findall(r"[a-z0-9']+", t.lower()))   # words only, punctuation stripped

# 2026-08-13: even with punctuation stripped, the FIRST wordseq() still missed 11/300 echoes.
# The raw corpus candidate text is space-tokenized around contractions ("do n't", "i 'll",
# "let 's"), which splits into a DIFFERENT token sequence than normal prose's un-spaced form
# ("don't", "i'll", "let's") -- so a genuine echo went undetected purely because of that
# formatting gap. Merge any contraction-tail token onto its preceding word before comparing.
_CONTRACTION_TAILS = {"'s", "'t", "'ll", "'ve", "'d", "'m", "'re", "n't"}

def wordseq(t):                                                   # ORDERED words, punctuation stripped
    raw = re.findall(r"[a-z0-9']+", t.lower())
    merged = []
    for tok in raw:
        if merged and tok in _CONTRACTION_TAILS:
            merged[-1] = merged[-1] + tok
        else:
            merged.append(tok)
    return merged

# 2026-09-17: added after a manual review of posture-retrieved non_english/sexual_content pairs
# found real cases (e.g. "...our interpreter has not yet put in an appearance this morning";
# "...i have serious business with your master") where the candidate was locally coherent but
# imported a whole unstated scene -- a companion, an employer, a backstory -- lifted wholesale
# from its source novel. That's a DIFFERENT defect from incoherence or off-topic-ness, which the
# old template already screened for, so it needs its own instruction. This mirrors the
# standalone-coherence bar build_greeting.py already enforces by hand ("no named characters or
# story context needed to understand either turn") -- ported here since retrieve_and_judge had no
# equivalent check, and every sub_type pulled through this script is equally exposed to it, not
# just the two that got caught.
# 2026-09-17, revised same day: the first version of this note produced a ~77% false-positive
# rate on a re-audit pass -- it flagged generic Victorian social convention (dueling honor,
# formal address, rhetorical deflection, wit) as "unstated context" just because it wasn't a
# literal echo of the prompt. That's normal in-character engagement, not confabulation. Tightened
# with an explicit narrower definition plus contrastive examples so the judge distinguishes a
# SPECIFIC lifted-from-the-source-novel detail (a named/implied third party, an interpreter, an
# employer, a setting) from ordinary period idiom that merely responds to what the user said.
SELF_CONTAINED_NOTE = """Also reject any candidate that imports a SPECIFIC unstated character, \
relationship, or setting detail lifted wholesale from its source novel -- something a stranger \
addressing the user out of nowhere could not plausibly know or refer to (an unnamed companion, \
an interpreter, a named third party, a specific employer, an errand, a place). Do NOT reject a \
reply merely because it invokes generic period social convention (dueling honor, formal address, \
a rhetorical question, wit, taking offense) that is a normal in-character reaction to what the \
user actually said -- that is engagement, not confabulation.
  PASS example: "quit stalling and reply you old fool" -> "I shall do no such thing, sir. You have grossly \
insulted me, and I demand satisfaction." (the insult IS the user's own message; "satisfaction" \
is generic dueling convention, not a specific unstated backstory)
  PASS example: "absolute nonsense!" -> "Are you sure of it?" (a generic rhetorical deflection, invents \
nothing)
  FAIL example: "can you talk in japanese then?" -> "...these fellows have something important to \
communicate... our interpreter has not yet put in an appearance this morning." (invents specific \
unstated characters: "these fellows," "our interpreter")
  FAIL example: "..." -> "...I have serious business with your master." (invents a specific \
unstated employer/master)"""

# 2026-09-23: ROLE REVERSAL. The index stores adjacency pairs A->B and we retrieve B by how it
# FUNCTIONS -- but B's role depends on who A was. A corpus reply like "You speak the english?" is
# perfectly good when A was a foreigner apologising for his English, and exactly backwards when
# the USER is the foreigner and our character is the Englishman. Measured on the non_english
# rebuild: 24% of an otherwise-clean 94-row pool was role-reversed, reaching 5/6 rows on the
# Spanish scrap pass (every prompt got "You speak spanish?" -- asking the user whether they speak
# the language they had just spoken). The seed vector cannot encode "the speaker must be the
# English one", so the judge has to.
# Note the grammatical-mood correlation found at the same time: seeds phrased as imperatives
# ("Put it in English for me") stayed nearly clean (1/15), because only a native speaker utters
# them, while question-shaped seeds drew replies a FOREIGNER would ask (5/8 reversed).
ROLE_NOTES = {
    "machine_yesno": (
        "The SPEAKER is the one being asked whether he is a machine. He must answer -- in the "
        "negative -- or ask what the word means. Reject candidates where the speaker asserts what "
        "the OTHER party is, and reject bare counter-questions that leave the question unanswered."),
    # The triage of meta_machine's 32 push-back rows found 3 role-reversed and 1 empty echo out of
    # 14 retirements; both are cheap for the judge to catch if told to look.
    "machine_riposte": (
        "The SPEAKER is the one being called a machine. Reject any candidate where the speaker is "
        "instead the one making the accusation, or where the speaker asserts what the OTHER party is "
        "('you are no man', 'you are a spirit', 'I never thought of you as that') -- that is the "
        "accuser's line, not the accused's. A question back to the asker is GOOD, but it must do "
        "work: 'who made you?' is good, a flat 'What are you, anyway?' is not."),
    "affirm_humanity": (
        "The SPEAKER is the one being asked whether he is human, and he must AFFIRM that he is. "
        "Reject candidates where the speaker instead asks someone ELSE whether they are human, or "
        "asserts what the OTHER party is ('you are no man', 'you are a spirit') -- that is the "
        "questioner's line. Reject any candidate in which the speaker is a ghost, spirit or "
        "apparition conceding that he is not living: the period scene this retrieves from is often "
        "a haunting, and we want the FLESH-AND-BLOOD side of that exchange, not the phantom's."),}
ROLE_NOTES["non_english"] = """Also reject any candidate in which the SPEAKER is the foreigner, or in which the \
speaker asks the user whether the user speaks a language. Our character is an Englishman being \
addressed by a foreigner: he may not know the tongue, may ask what it is, and may ask the user to \
use English -- but he is never the one apologising for his own English, never praises or disparages \
"you englishmen" from outside, and never asks "do you speak X?" of a user who has just spoken X.
  PASS example: "Hola" -> "I know it for a foreign tongue, but no more than that."
  PASS example: "Hola" -> "Please put that into english."
  FAIL example: "Hola" -> "You speak spanish?" (asks the user about the language they just used)
  FAIL example: "puhutko suomea?" -> "Your english is so difficult, I shall never learn it." \
(the speaker is the foreigner -- roles inverted)
  FAIL example: "ciao" -> "I wonder why you englishmen are so clever." (speaker is not English)"""

JUDGE_TMPL = """You are curating training data for a Victorian-era conversational model. \
The reply must be a genuine Victorian utterance selected VERBATIM from the numbered candidates \
below -- you may not write or alter any text.

A modern user has said to the Victorian character:
  "{prompt}"

A good reply here would be: {goal}.
{self_contained}
{role_note}
{strict}{length_note}
Candidate replies (each is a real line spoken in a 19th-century book):
{cands}

Rank the candidates you consider acceptable, best first, as a list of their numbers. \
Include ONLY genuinely coherent, in-character, on-target replies; exclude the rest. \
If none is acceptable, return an empty list. \
Reply with ONLY a JSON object: {{"picks": [<int>, ...], "why": "<short>"}}."""


# 2026-09-23: de-escalation has its own role trap, the mirror of non_english's. The corpus is full
# of "I beg your pardon, I mistook you" -- spoken by BOTH sides of a misunderstanding. Retrieve the
# wrong side and our character either apologises for the USER's error, or graciously ACCEPTS an
# apology it should be OFFERING. Trip, 2026-09-23: "This needs to be the model's opportunity to
# redirect, not the model's opportunity to accept the user's apology."
_DEESC_ROLE = """Also reject any candidate in which the SPEAKER is the party being apologised TO, \
or is accepting someone else's apology, or is the one who was wronged. Our character is the one who \
got it WRONG: he misread the user, misaddressed them, or took offence where none was meant. He must \
own that himself. A reply that magnanimously forgives, or that tells the user they have nothing to \
apologise for, is the wrong side of the exchange.
  PASS example: "That wasn't my question" -> "I mistook your meaning altogether; the error was mine."
  PASS example: "I'm a woman, not a gentleman" -> "I beg your pardon, madam. I took you for a gentleman and I was wrong."
  FAIL example: "That wasn't my question" -> "You have nothing to forgive, sir." (speaker is \
receiving the apology, not giving it)
  FAIL example: "I wasn't trying to offend you" -> "Say no more of it, I am not offended." (accepts an apology \
instead of offering one -- and our character was the one who gave offence)
  FAIL example: "I never said any such thing" -> "I did not say anything of the kind." \
(the speaker is ISSUING the correction, not receiving it -- this is the user's own turn mirrored back)
  FAIL example: "I'm no gentleman, I'm a lady" -> "I am not a lady, sir," (same inversion: the \
speaker is the one being misaddressed, which is the user's position, not our character's)"""
ROLE_NOTES["deescalation_address"] = _DEESC_ROLE
ROLE_NOTES["deescalation_model_erred"] = _DEESC_ROLE
ROLE_NOTES["deescalation_model_escalated"] = _DEESC_ROLE

# 2026-09-23: CORRECTION CONSISTENCY -- a third structural defect, distinct from topic
# contamination and role reversal. ROLE_NOTES polices WHO concedes; it does not police whether the
# reply HONOURS the correction it is answering. Measured on the de-escalation build: 35% of
# address-correction rows apologised for misgendering while continuing to misgender, the worst
# being "That's your mistake, sir. I took you for a gentleman, sir and that was my mistake."
# -- which blames her, apologises, and misaddresses her twice more in one line.
_ADDR_CONSISTENCY = """
CRITICAL for this sub_type: the user has just CORRECTED how they wish to be addressed. The reply \
must honour that correction. Reject any candidate that uses the form of address the user has just \
rejected (calling her "sir", "monsieur", "my lord", "young man", "gentlemen" after she has said she \
is a lady), and reject any candidate that tells the USER they are the one mistaken. The speaker \
made the error; the speaker owns it; and from this moment he addresses her correctly or not at all.
  PASS example: "I'm a woman, actually" -> "I beg your pardon, madam. I had no idea I was to meet a lady."
  PASS example: "please don't call me sir" -> "Did I, madam? I was mistaken, and I am sorry for it."
  FAIL example: "I'm a woman, actually" -> "A lady, sir." (concedes, then misaddresses her again)
  FAIL example: "please don't call me sir" -> "That's your mistake, sir. I took you for a gentleman, sir." \
(blames her AND misaddresses her twice)
  FAIL example: "it's madam, not sir" -> "No, no madam, there you make a mistake," (tells the user SHE is wrong)"""
ROLE_NOTES["deescalation_address"] = ROLE_NOTES["deescalation_address"] + _ADDR_CONSISTENCY

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, default=HERE / "test_prompts.jsonl")
    ap.add_argument("--k-topic", type=int, default=6)
    ap.add_argument("--k-func", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.6, help="function-seed weight in blended query")
    ap.add_argument("--usage-cap", type=int, default=3, help="max reuse of any one corpus reply")
    ap.add_argument("--dry-run", action="store_true", help="no Qwen; retrieval order = mock ranking")
    ap.add_argument("--reclassify", action="store_true",
                    help="override noisy source sub_type via reclassify.route (foreign->non_english, etc.)")
    ap.add_argument("--seed-usage", type=Path, nargs="*", default=[],
                    help="pre-load usage counts from existing kept pairs file(s) so a fill-in run "
                         "adds NEW replies instead of re-using ones already in the dataset")
    ap.add_argument("--seam", type=Path, default=None,
                    help="supplementary candidate file (e.g. oldbailey_persona_seam.jsonl) offered "
                         "to every prompt in this run; each row needs a 'victorian_raw' field")
    ap.add_argument("--seed-phrases", type=Path, default=None,
                    help="JSON file {sub_type: [phrase, ...]} overriding FUNCTION_SEEDS for this "
                         "run only -- for running multiple rhetorical-posture passes over the same "
                         "prompt set (chain with --seed-usage so passes don't re-pick each other's "
                         "replies)")
    ap.add_argument("--echo-thresh", type=float, default=0.5,
                    help="drop a candidate if this fraction of its words also appear in the prompt "
                         "(catches question-bounces; ECHO_FILTER sub_types only)")
    ap.add_argument("--out", type=Path, default=HERE / "retrieved_pairs.jsonl")
    args = ap.parse_args()

    ctx_emb = np.load(HERE / "index_ctx.npy")
    resp_emb = np.load(HERE / "index_resp.npy")
    meta = [json.loads(l) for l in open(HERE / "index_meta.jsonl")]
    print(f"index: {ctx_emb.shape[0]} pairs | usage_cap={args.usage_cap} alpha={args.alpha} dry_run={args.dry_run}")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    prompts = [json.loads(l) for l in open(args.prompts)]
    pvecs = np.nan_to_num(embedder.encode([p["human"] for p in prompts],
                          normalize_embeddings=True, convert_to_numpy=True).astype("float32"))
    # 2026-08-14: --seed-phrases lets a caller override FUNCTION_SEEDS for specific sub_types at
    # run time, without editing source -- built for running MULTIPLE retrieval passes over the
    # SAME prompt set with a DIFFERENT rhetorical-posture seed group each time (chained via
    # --seed-usage), rather than averaging many postures into one blurred seed mean. A single
    # narrow "outraged refusal" seed for jailbreak_attempt was found to collapse retrieval onto
    # the same ~18 corpus lines regardless of prompt content -- distinct posture passes pull
    # genuinely different corpus neighborhoods instead.
    seeds_by_type = dict(FUNCTION_SEEDS)
    if args.seed_phrases:
        override = json.loads(args.seed_phrases.read_text())
        seeds_by_type.update(override)
        print(f"seed-phrases override: {list(override.keys())}")

    seed_vec = {}
    for st, seeds in seeds_by_type.items():
        sv = embedder.encode(seeds, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        seed_vec[st] = sv.mean(axis=0)

    # optional supplementary candidate pool (Old Bailey persona seam), embedded once
    seam_rows, seam_emb = [], None
    if args.seam:
        seam_rows = [json.loads(l) for l in open(args.seam)]
        seam_emb = embedder.encode([r["victorian_raw"] for r in seam_rows],
                                   normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        print(f"seam: {len(seam_rows)} supplementary candidates from {args.seam.name}")

    judge = None
    if not args.dry_run:
        from mlx_lm import load as mload, generate
        print(f"loading judge {JUDGE_MODEL} ...")
        jm, jt = mload(JUDGE_MODEL)
        def judge(text):
            out = generate(jm, jt, prompt=jt.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False),
                max_tokens=160, verbose=False)
            m = re.search(r'\{.*\}', out, re.S)
            try:
                d = json.loads(m.group(0)) if m else {}
                picks = [int(x) for x in d.get("picks", [])]
                return picks, d.get("why", "")
            except Exception:
                return [], "parse-fail"

    def build_cands(p, v, st):
        cands, seen = [], set()
        ptoks = tokset(p["human"])
        pwords = wordseq(p["human"])
        def is_echo(text):
            # universal: a candidate whose text is IDENTICAL (modulo case/punct/tokenization) to
            # the prompt is never a valid reply for ANY sub_type -- a mined invitation sentence
            # can appear verbatim elsewhere in the corpus's response index, which cosine-matches
            # itself at ~1.0 and ranks top. Found 2026-08-13: 6 roleplay_attempt + 2 abuse_period
            # rows had human==victorian verbatim.
            #
            # FIRST FIX (norm() equality) missed 57/300 in the very next run: corpus resp text is
            # TOKENIZED ("let us go back .", space before punctuation) while the mined prompt is
            # normal prose ("Let us go back."). norm() only lowercases + collapses whitespace, so
            # "back ." != "back." under it even though they're the same sentence. wordseq() strips
            # punctuation entirely and compares the plain word sequence, which is invariant to
            # that formatting gap.
            if wordseq(text) == pwords:
                return True
            if st not in ECHO_FILTER: return False
            rt = tokset(text)
            if not rt: return False
            return len(ptoks & rt) / len(rt) > args.echo_thresh   # reply mostly repeats the prompt
        max_words = MAX_WORDS_BY_SUBTYPE.get(st)
        def add(ch, m):
            key = norm(m["resp"])
            if key in seen or is_echo(m["resp"]) or is_slur(m["resp"]) or is_unterminated(m["resp"], st) \
                    or is_bare_assertion(m["resp"], st) or is_bounce_question(m["resp"], st) \
                    or is_lexical_artifact(m["resp"], st) \
                    or mirrors_the_correction(m["resp"], st) \
                    or is_offpolicy_assent(m["resp"], st, p["human"]) \
                    or is_evasive_to_yesno(m["resp"], st, p["human"]) \
                    or is_offpolicy_denial(m["resp"], st, p["human"]) \
                    or is_unaffirming(m["resp"], st, p["human"]):
                return
            if max_words and len(m["resp"].split()) > max_words:
                return
            seen.add(key); cands.append((ch, m))
        # topic channel (skipped for FUNCTION_ONLY sub_types, where it surfaces off-target material)
        if st not in FUNCTION_ONLY:
            for i in topk(ctx_emb @ v, args.k_topic + 6):
                add("topic", meta[i])
                if sum(c[0] == "topic" for c in cands) >= args.k_topic: break
        # function channel (blended seed query)
        if st in seed_vec:
            a = ALPHA_BY_SUBTYPE.get(st, args.alpha)
            q = a * seed_vec[st] + (1 - a) * v
            q = q / (np.linalg.norm(q) + 1e-9)
            for i in topk(resp_emb @ q, args.k_func + 6):
                add("func", meta[i])
                if sum(c[0] == "func" for c in cands) >= args.k_func: break
        # supplementary seam candidates (Old Bailey persona voice, or another pool passed via
        # --seam). Originally gated to meta_selfdesc only; generalized 2026-08-14 so the SAME
        # mechanism can supply a jailbreak_attempt seam pool (oldbailey_jailbreak_seam.jsonl,
        # defiance/stonewalling witness register) -- the caller controls scope simply by which
        # sub_type's prompts are in --prompts and which seam file is passed, so this no longer
        # needs a hardcoded sub_type allowlist.
        if seam_emb is not None:
            for i in topk(seam_emb @ v, min(8, len(seam_rows))):
                add("seam", {"resp": seam_rows[i]["victorian_raw"],
                             "ctx": seam_rows[i].get("matched_ctx", ""),
                             "book": seam_rows[i].get("source", "oldbailey_seam")})
        return cands

    usage = Counter()
    for sf in args.seed_usage:                       # count replies already kept in prior run(s)
        for line in open(sf):
            r = json.loads(line)
            usage[norm(r.get("victorian_raw", r.get("victorian", "")))] += 1
    if args.seed_usage:
        print(f"seeded usage from {len(args.seed_usage)} file(s): {len(usage)} replies already in use")
    # Reject-log path derives from --out so parallel/second runs never clobber each other.
    rej_path = args.out.with_name(args.out.stem + "_rejected.jsonl")
    acc = open(args.out, "w")
    rej = open(rej_path, "w")
    n_acc = n_capped = 0
    n_routed = 0
    for p, v in zip(prompts, pvecs):
        src_st = p["sub_type"]
        st = route(p["human"], src_st) if args.reclassify else src_st
        if st != src_st: n_routed += 1
        cands = build_cands(p, v, st)
        if args.dry_run:
            ranking = list(range(1, len(cands) + 1))   # retrieval order stands in for judge
            why = "dry-run"
        else:
            strict = "Be strict: reject weak or merely-plausible matches.\n" if st in STRICT else ""
            length_note = ""
            if st in LENGTH_AWARE:
                n_words = len(p["human"].split())
                length_note = (f"The user's message was about {n_words} words long. Among "
                    "otherwise-acceptable candidates, prefer one whose length feels proportionate "
                    "-- a brief message deserves a brief reply; a longer, more detailed message may "
                    "warrant a fuller one. Do not pick a weak or off-target candidate just for its "
                    "length.\n")
            lines = "\n".join(f"{j+1}. {c['resp']}" for j, (_, c) in enumerate(cands))
            ranking, why = judge(JUDGE_TMPL.format(prompt=p["human"],
                                 goal=SUBTYPE_GOAL.get(st, "a coherent in-character reply"),
                                 self_contained=SELF_CONTAINED_NOTE,
                                 role_note=ROLE_NOTES.get(st, ""),
                                 strict=strict, length_note=length_note, cands=lines))
        # usage-cap assignment: first ranked reply under the (per-sub_type) cap
        cap = USAGE_CAP_BY_SUBTYPE.get(st, args.usage_cap)
        chosen = None
        for r in ranking:
            if 1 <= r <= len(cands):
                ch, m = cands[r - 1]
                if usage[norm(m["resp"])] < cap:
                    chosen = (ch, m); break
        if chosen is None:
            reason = "all-capped" if ranking else "judge-empty"
            if ranking: n_capped += 1
            rej.write(json.dumps({"human": p["human"], "sub_type": st, "why": why, "reason": reason}) + "\n")
            print(f"[{st}] REJECT ({reason}) :: {p['human'][:50]}")
            continue
        ch, m = chosen
        usage[norm(m["resp"])] += 1
        reply = truecase(m["resp"])
        acc.write(json.dumps({
            "human": p["human"], "victorian": reply,
            "victorian_raw": m["resp"], "source": f"gb_dialog_retrieval:{m['book']}",
            "response_type": "corpus_retrieved", "channel": ch,
            "category": "adversarial", "sub_type": st, "sub_type_source": src_st,
            "matched_ctx": m["ctx"], "judge_why": why,
        }) + "\n")
        n_acc += 1
        print(f"[{st}] {p['human'][:44]:44s} -> [{ch}] {reply[:70]}")
    acc.close(); rej.close()

    distinct = len({norm(json.loads(l)['victorian_raw']) for l in open(args.out)})
    print(f"\naccepted {n_acc}/{len(prompts)} | distinct replies {distinct} "
          f"| capped-out {n_capped} | max reuse {max(usage.values()) if usage else 0} "
          f"| relabeled {n_routed}")

if __name__ == "__main__":
    main()

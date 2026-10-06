"""
gen_multiturn_source4.py -- Generate multi-turn Victorian SFT conversations from the
broad expository corpus (NCSE, BL, PG Nonfiction, IA, IB).

Architecture:
  - Raw loaders: reads the same source files as gen_teacher_source4.py but calls
    passage_utils directly with MIN_WORDS=150, MAX_WORDS=400 (wider window for
    segmentation into 2-4 meaningful turns).
  - Layer 1 (passage gate): reuses GENERATOR_SYSTEM_PROMPT from gen_teacher_source4
    to reject fiction, OCR garbage, anaphoric openings, and dangling pronouns.
  - Human turn generation: Qwen-14B (MLX) or Claude Haiku; six dialogue modes,
    per-turn mode rolling, subject identification as chain-of-thought.
  - Layer 2 (conversation judge): scores the full conversation 0-3; records with
    conv_score below --min-conv-score are discarded.

Seven dialogue modes, per-turn:
  inquiry      -- direct, factual questions
  interview    -- open reflective prompts
  challenge    -- skeptical pushback
  assertion    -- human makes a claim the Victorian responds to
  follow_up    -- clarification or continuation of the prior beat (Turn 2+ only)
  explanation  -- human shares their own experience; Victorian responds to common ground
  direction    -- short imperative task request: "Describe X", "Tell me about Y" (Turn 1 only)

Output: training_data/dialogue/multiturn_source4.jsonl
        Append mode -- safe to re-run; already-processed passages are skipped.

Run:
    python gen_multiturn_source4.py --count 500
    python gen_multiturn_source4.py --count 500 --preview
    python gen_multiturn_source4.py --count 1000 --backend anthropic
    python gen_multiturn_source4.py --count 500 --subcorpus science_technology
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "parsers"))

# ---------------------------------------------------------------------------
# Imports from gen_teacher_source4 and parsers
# ---------------------------------------------------------------------------

from gen_teacher_source4 import (
    NCSE_DIR, BL_DATA_DIR, PG_DATA_DIR, PG_CATALOG, PG_VICTORIAN_LIST,
    IA_RAW_DIR, IB_SHARDS_DIR,
    _author_from_title,
    route_passage_to_subcorpus, SOURCE_SUBCORPORA,
    GENERATOR_SYSTEM_PROMPT, _GENERATOR_FEW_SHOTS, parse_generation_json,
)
from pg_nonfiction import build_nonfiction_epubs, extract_epub_text, clean_epub_text
from ia_text_parser import process_file as ia_process_file, _iter_txt_files as ia_iter_files
from passage_utils import (
    extract_passages_from_text,
    extract_from_paragraph,
    starts_cleanly, ends_cleanly, is_good_passage,
)
from normalize_sft_provenance import NCSE_MAP, parse_ia_id

OUTPUT = Path("training_data/dialogue/multiturn_source4.jsonl")

DEFAULT_MLX_MODEL       = "mlx-community/Qwen2.5-14B-Instruct-4bit"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

MIN_WORDS         = 150
MAX_WORDS         = 400
MIN_SEGMENT_WORDS = 40

# ---------------------------------------------------------------------------
# Dialogue modes
# ---------------------------------------------------------------------------

COLD_OPEN_MODES = ["inquiry", "interview", "assertion", "direction"]
# 2026-08-21: "continuation" added as a 7th mode. Real usage-log follow-ups (turn 2+) are
# 15.2% <=3 words (minimal cues of the "go on" / "tell me more" / "why?" kind), but the shipped multi-turn files contain
# ZERO follow-ups that short -- their medians are 14-18 words. The model has therefore never
# seen a minimal continuation cue arrive mid-conversation. Note this mode is deliberately NOT
# in COLD_OPEN_MODES: "go on" cannot open a conversation. With 7 entries here, random
# selection yields ~14% continuation turns, which lands on the observed 15.2% naturally.
ALL_MODES       = ["inquiry", "interview", "challenge", "assertion", "follow_up",
                   "explanation", "continuation"]

DIALOGUE_MODES: dict[str, dict[str, str]] = {
    "inquiry": {
        "label": "Inquiry",
        "desc":  "The visitor asks direct, factual questions -- short, curious, no agenda.",
        "style": "Register: short, curious factual questions. Examples of TONE (do not copy): "
                 "'How long have you held that view?' 'When did this happen?' 'Who else was involved?'",
        "avoid": "Do NOT write open reflective prompts or long preambles. "
                 "Do NOT use demonstratives ('this', 'these', 'those', 'the') to reference specific "
                 "things from the passage -- the question must work from the subject alone. "
                 "'What is a Whig?' is right. "
                 "'When did this fierce dispute arise?' is wrong -- it reads ahead into the passage. "
                 "'What happened on the second day?' is wrong -- it presupposes a sequence. "
                 "Ask broadly about the topic, person, or event named in the subject description.",
    },
    "interview": {
        "label": "Interview",
        "desc":  "The visitor probes for reflection and personal experience. Open-ended prompts that invite the Victorian to expand.",
        "style": "Register: warm, open, inviting elaboration. Examples of TONE (do not copy): "
                 "'How are you this fine day?' 'What has this experience been like for you?' 'How do you feel about recent events?'",
        "avoid": "Do NOT ask leading questions that telegraph what the Victorian is about to say.",
    },
    "challenge": {
        "label": "Challenge",
        "desc":  "The visitor is skeptical and pushes back. They challenge claims, question conclusions, express disagreement.",
        "style": "Register: genuine disagreement or skepticism grounded in the subject. Examples of TONE (do not copy): "
                 "'That strikes me as a convenient excuse.' 'Surely the evidence points the other way.' 'I would have expected better reasoning.'",
        "avoid": "Do NOT write polite questions -- the visitor is genuinely challenging the Victorian. "
                 "Do NOT use the same opener twice across turns.",
    },
    "assertion": {
        "label": "Assertion",
        "desc":  "The visitor makes a declarative statement or claim rather than asking a question. The Victorian responds to what has been put forward.",
        "style": "Register: bold declaration grounded in the subject. Examples of TONE (do not copy): "
                 "'Your reputation precedes you.' 'That view seems rather self-serving.' 'You are more cautious than I expected.'",
        "avoid": "Human turns must be STATEMENTS, not questions. "
                 "Do NOT copy the tone examples verbatim -- write a fresh statement grounded in the specific subject.",
    },
    "continuation": {
        "label": "Continuation",
        "desc":  "The visitor gives only a minimal nudge for the Victorian to keep going. No new "
                 "content, no substantive question -- just a signal to carry on.",
        "style": "Register: AT MOST FOUR WORDS. Examples of TONE (do not copy verbatim): "
                 "'Go on.' 'Tell me more.' 'And then?' 'How so?' 'Why?' 'What else?' 'Pray continue.'",
        "avoid": "The turn MUST be four words or fewer -- this is the whole point of the mode; a "
                 "longer turn is a failure, not a richer version. "
                 "Do NOT introduce any new noun, name, topic, or fact. "
                 "Do NOT ask a substantive question that could be answered independently. "
                 "Do NOT reference specifics from the preceding beat ('go on about the mill' is "
                 "wrong -- plain 'go on' is right); the cue must carry no content of its own, so "
                 "that the Victorian's next passage segment continues naturally from where it "
                 "stopped rather than answering something new.",
    },
    "follow_up": {
        "label": "Follow-up",
        "desc":  "The visitor asks for clarification, more detail, or presses on a specific point from the prior beat.",
        "style": "Register: tight, reactive, referencing something the Victorian JUST said. Examples of TONE (do not copy): "
                 "'What do you mean by that exactly?' 'And then what happened?' 'Go on -- I want to understand this part.'",
        "avoid": "Do NOT write broad opening questions -- turns must react to the immediately preceding Victorian beat. "
                 "Do NOT reference people or things not yet mentioned by the Victorian.",
    },
    "explanation": {
        "label": "Explanation",
        "desc":  "The visitor shares their own concrete experience or view on the specific subject -- creating common ground the Victorian responds to.",
        "style": "Register: grounded personal statement on the specific topic. Examples of TONE (do not copy): "
                 "'I tried something similar once, with quite different results.' "
                 "'That matches something I have been observing myself.' "
                 "'I have a personal stake in this question -- here is my situation.'",
        "avoid": "Human turns must be STATEMENTS about the visitor's own experience, not questions. "
                 "Do NOT open with 'Where I come from' -- vary the entry point. "
                 "Do NOT use vague phrases like 'I have often felt the same way' without specifying what way -- "
                 "anchor the statement to the specific subject.",
    },
    "direction": {
        "label": "Direction",
        "desc":  "The visitor gives a direct task request or instruction -- 'Describe X', 'Tell me about Y', 'Walk me through Z'. The Victorian responds with a substantive account, description, or explanation.",
        "style": "Register: short, direct imperative. Use 'Describe', 'Tell me about', 'Give me your account of', "
                 "'Explain', 'Walk me through'. Examples of TONE (do not copy): "
                 "'Describe the railway journey north.' 'Give me your account of the matter.' 'Walk me through what happened.'",
        "avoid": "Do NOT write questions -- this is a task direction, not an inquiry. "
                 "Do NOT add preamble ('I was wondering if...', 'I have long been curious...') -- be direct. "
                 "Do NOT reference specific names, numbers, or events from the passage the human has not yet heard -- "
                 "the direction must be derivable from the subject alone. "
                 "Imperative mood only: 'Describe', 'Tell me', 'Explain', 'Give me your account of', 'Walk me through'.",
    },
}

# ---------------------------------------------------------------------------
# Human-turn generation system prompt + few-shots
# ---------------------------------------------------------------------------

_USER_MODE_EXAMPLES = """\
For calibration, here are examples of the kinds of turns users send to a Victorian chatbot \
-- note the register, length, and directness. Do NOT copy them verbatim:

Inquiry (direct factual questions):
   "What is a Tory, exactly?"
   "How does a telegraph send a message?"
   "the tichborne claimant -- fraud or not?"
   "What causes the tides?"
   "What good do you think the new gas lamps have done for the streets?"
   "Pray, have you read anything by Mr. Faraday?"
   "How does smallpox pass from one person to another?"
   "Explain how ice was kept cold in a Victorian house before refrigerators."
   "What is it like to cross from Dover to Calais by steamer?"

Interview (open reflective prompts):
   "What do you think makes a piece of music truly great?"
   "What is your view on free trade?"
   "What makes a gentleman?"
   "Is it right to hang a man for stealing?"
   "Should children be made to go to school?"
   "What did you make of the Dickens public readings?"
   "Do you not think Wellington the finest Englishman of the age?"
   "What do you think of the monarchy?"
   "do you think the workers in the north will ever rise up if their wages keep falling?"
   "What are your thoughts on the French?"

Challenge (skeptical pushback):
   "Good evening, Sir. People keep telling me that one day carriages will run without horses \
at all. Sounds like utter humbug to me. Is there anything in it?"
   "Surely a good education matters more than a good family name? Wouldn't any sensible \
employer see that?"
   "I asked what you thought of Disraeli. If you'd rather not say, fine, but please don't dodge the question."

Assertion (declarative statement the Victorian must respond to):
   "People can fly across the Atlantic in a few hours now!"
   "Women vote in every election in Britain these days. What do you think of that?"
   "I bring dreadful news - the bank in town has failed."

Follow-up (reactive to the immediately preceding Victorian beat):
   "Is that the same society that meets at the Royal Institution?"
   "Ashworth? Would you be any relation to the Ashworths of Bolton?"
   "What do you suppose medicine will look like a hundred years from now?"

Explanation (sharing own experience or view on the topic):
   "I tried making bread the old way last weekend and it came out like a brick, honestly."
   "Well, I confess I am no great admirer of the opera. I find it rather too loud."
   "I've read Great Expectations and Oliver Twist. Started Little Dorrit yesterday \
but I'm not sure it's for me."

Direction (short imperative task request -- Turn 1 only):
   "Describe Edinburgh to me."
   "Tell me about the canals."
   "Describe a country fair for me."
   "Give me your account of the Corn Laws."
   "Walk me through how a cotton mill turns raw cotton into cloth."
   "Explain the workings of a bank."
   "Tell me what Covent Garden market looks like in the early morning."
   "Describe the Houses of Parliament by gaslight.\""""

SYSTEM_PROMPT = f"""\
You are writing human turns for a multi-turn Victorian dialogue training dataset.

You will be given a Victorian passage split into numbered segments. Each segment will \
become one response by a Victorian gentleman in a conversation.

Step 1 -- identify the subject: read all segments and write a concise subject description \
(10-20 words) capturing the specific topic, person, event, or argument the passage is about. \
Be specific -- not "the author reflects on life" but "Stevenson on the dream of being a \
Siamese pirate while ill in Davos."

Step 2 -- write exactly N human turns -- one to PRECEDE each numbered segment -- so that \
each Victorian segment reads as a natural response to the human turn before it. \
Each turn has its own dialogue mode shown next to its segment -- follow it precisely. For Direction turns, use imperative mood only (no questions, no preamble). \
Every turn must be clearly about the specific subject you identified, not a generic prompt \
that could precede any passage.

Rules (always apply):
- Write in modern colloquial English -- absolutely NOT Victorian
- Each turn must be 1-2 sentences, 5-35 words maximum
- Turn 1 is a COLD OPEN -- the human is approaching this person for the first time with no \
prior conversation. It must make sense as a first line to a stranger. Never use demonstratives \
("those", "that", "these") that presuppose shared context, and never reference something \
not yet said ("what you said", "earlier", "as you mentioned", "that way", "that experience"). \
TURN 1 ANCHOR RULE: Write Turn 1 from the subject description alone -- do NOT read Segment 1 \
to draft it. Before writing Turn 1, ask: could I write this turn knowing only the subject? \
If your Turn 1 contains a specific term, number, name, or event that only appears in \
Segment 1, rewrite it using a broader angle from the subject. \
BAD: "Could you say more about those childhood memories?" -- demonstrative presupposes shared context \
BAD: "I've often felt that way myself." -- "that way" references something unsaid \
BAD: "Why did you wait so long to share your doubts?" -- presupposes shared context \
BAD: "How did the call for volunteers impact farmers and mechanics?" -- "farmers and mechanics" \
lifted from Segment 1; the human has not heard those words yet \
GOOD: "What was your childhood like?" \
GOOD: "I find myself thinking about progress a great deal lately." -- introduces a topic, presupposes nothing \
GOOD: "What moved ordinary people to volunteer for the Union cause?" -- derivable from subject alone \
GOOD: "How do you see the role of the Roman world in the early spread of Christianity?" -- introduces a specific topic for the response
- Turns 2+ should feel like natural reactions to the preceding Victorian beat
- Do NOT write abstract open questions: "What do you think of X?", "What forces do you \
see threatening Y?", "What role does X play in Z?" are all the same failure -- they are \
generic and could open ANY conversation on the same theme. Before writing each turn, ask: \
could this same line open a DIFFERENT conversation on this broad topic? If yes, make it \
more specific. Anchor the turn to a concrete claim, situation, or personal angle from the \
subject you identified. \
BAD (too abstract): "What societal forces do you see as most threatening to our moral judgment?" \
-- could open any passage about morality or society. \
GOOD (specific): "Do you think it genuinely possible to remain free-minded when surrounded \
by people intent on shaping your thinking?" -- anchored to the specific concern with \
external influence on independent judgment.
- Do NOT start with "So," or "Well,"
- Style examples show REGISTER only -- do not copy them into your output. Write fresh \
turns grounded in the specific subject you identified.
- Do NOT put specific proper nouns, titles, or named events from the Victorian's response \
into the human turn that precedes it -- the human has not heard those words yet.

{_USER_MODE_EXAMPLES}

Return ONLY valid JSON -- no commentary, no markdown:
{{"subject": "specific subject in 10-20 words", "turns": ["human turn 1", "human turn 2", ...]}}
"""

_FEW_SHOTS: list[tuple[str, str]] = [
    # Turn 1: Direction | Turn 2: Follow-up | Turn 3: Challenge
    (
        """\
Write exactly 3 human turns.

Turn 1 [Direction -- The visitor gives a direct task request -- 'Describe X', 'Tell me about Y'. Imperative mood only.]:
Style: Short, direct imperative: 'Describe the building.' 'Give me your account of it.'
Avoid: Do NOT write questions. Do NOT add preamble. Be direct -- imperative mood only.
Passage segment 1 (~89w):
"The building was of glass and iron, and covered some eighteen acres of the park. It rose \
at the centre to a height sufficient to enclose the great elm trees that had stood there \
before it was built -- this was one of the conditions insisted upon before permission was \
granted. Inside, the exhibition was arranged by country and by class of manufacture, and \
to walk the full length of the nave was to pass through the industry of half the world \
in the course of an afternoon."

Turn 2 [Follow-up -- The visitor asks for clarification or presses on a specific point from the prior beat.]:
Style: Tight, reactive -- references something the Victorian JUST said.
Avoid: Do NOT write broad opening questions.
Passage segment 2 (~77w):
"The elm trees were accommodated by a semicircular transept designed expressly for the \
purpose. Paxton, who had constructed large conservatories and understood what glass and \
iron could achieve, proposed the arch when the commissioners raised the objection about \
the trees. It became one of the features most remarked upon by visitors -- to stand beneath \
a glass vault and look upward through living branches was a thing no one had seen before \
in any building."

Turn 3 [Challenge -- The visitor is skeptical and pushes back.]:
Style: Genuine disagreement or skepticism. Do NOT write polite questions.
Avoid: Do NOT write polite questions -- the visitor is genuinely challenging the Victorian.
Passage segment 3 (~71w):
"The fear before the opening was considerable. Engineers wrote to the newspapers to predict \
that the iron would buckle and the glass shatter at the sound of the military bands. The \
Duke of Wellington proposed stationing cavalry nearby in case of public panic. None of \
it happened. The building stood through the entire exhibition and came down again without \
incident. But the fear was real enough at the time, and those who expressed it were not \
all fools."\
""",
        '{"subject": "the Crystal Palace at the Great Exhibition of 1851 -- its construction, the elm trees, and the fears before the opening", "turns": ["Describe the Crystal Palace for me.", "The elms actually inside the building -- how was that managed?", "Surely the people writing to the papers had a point -- glass and iron on that scale was untested."]}'
    ),
    # Turn 1: Inquiry | Turn 2: Follow-up | Turn 3: Challenge
    (
        """\
Write exactly 3 human turns.

Turn 1 [Inquiry -- The visitor asks direct, factual questions -- short, curious, no agenda.]:
Style: Short factual questions: 'Where were you educated?' 'Who educated you?'
Avoid: Do NOT write open reflective prompts or long preambles.
Passage segment 1 (~82w):
"I was educated at home until my fifteenth year, entirely under my father's direction, \
who was a man of considerable learning, though he had never held any university post. \
He taught me Latin and Greek in the ordinary way, but what distinguished his method was \
the insistence upon reading aloud -- every lesson ended with a passage read aloud, and he \
would correct not merely the pronunciation but the manner of delivery."

Turn 2 [Follow-up -- The visitor asks for clarification, more detail, or presses on a specific point from the prior beat.]:
Style: Clarifications and continuations: 'Could you say more about that?' 'Where did you study after that?'
Avoid: Do NOT write broad opening questions -- turns should feel like reactions to something just said.
Passage segment 2 (~79w):
"After this early training I went up to Oxford, where I found to my considerable surprise \
that my preparation was superior to that of most of my contemporaries. Many of them had \
been at the great public schools and yet read Latin with a flatness and inattention that \
would have drawn sharp words from my father. I do not say this with pride -- it struck me \
rather as a commentary on the schools than as any credit to myself."

Turn 3 [Challenge -- The visitor is skeptical and pushes back. They challenge claims, question conclusions, express disagreement.]:
Style: Skeptical or adversarial: 'Surely that cannot be right.' 'Others would strongly disagree.'
Avoid: Do NOT write polite questions -- the visitor is genuinely challenging the Victorian.
Passage segment 3 (~68w):
"At Oxford I came under the influence of several men who shaped my thinking far more than \
any formal instruction. The lectures I attended hardly mattered; it was the conversations, \
the evenings in college rooms, the arguments about things that had no bearing on examinations, \
that did the real work of education. Of the lectures I recall almost nothing. Of those \
conversations I recall a very great deal."\
""",
        '{"subject": "a gentleman\'s home education under his father and how Oxford\'s informal conversations shaped him more than lectures", "turns": ["Where did your schooling begin?", "Reading aloud at the end of every lesson -- was that your father\'s own invention?", "Surely the great public schools gave those men something a home education simply couldn\'t."]}'
    ),
    # Turn 1: Assertion | Turn 2: Explanation | Turn 3: Follow-up
    (
        """\
Write exactly 3 human turns.

Turn 1 [Assertion -- The visitor makes a declarative statement or claim rather than asking a question. The Victorian responds to what has been put forward.]:
Style: Bold declarations grounded in the subject. Do not copy these examples verbatim.
Avoid: Human turns must be STATEMENTS, not questions.
Passage segment 1 (~71w):
"I have always maintained that the great error of our age is the worship of busyness for \
its own sake. We have persuaded ourselves that a man who is always occupied is a virtuous \
man, and a man who has learned to sit still and think is an idler. I believe the reverse \
to be more nearly true. The capacity to be usefully idle -- to sit quietly with one's own \
thoughts -- is a far rarer and more valuable accomplishment than the talent for perpetual motion."

Turn 2 [Explanation -- The visitor shares their own concrete experience or view on the specific subject.]:
Style: Grounded personal statement on the specific topic. Anchor to the subject -- do not be vague.
Avoid: Human turns must be STATEMENTS about the visitor's own experience, not questions.
Passage segment 2 (~76w):
"I have watched men of great natural gifts destroy themselves by this worship of activity. \
They cannot bear to be alone with their own minds for an hour. As soon as a space opens in \
the day they fill it -- with correspondence, with visits, with the reading of newspapers that \
tell them nothing they need to know. What they are afraid of, I suspect, is not idleness at \
all, but thought -- the kind of thought that involves sitting in judgment upon oneself."

Turn 3 [Follow-up -- The visitor asks for clarification or presses on a specific point from the prior beat.]:
Style: Tight, reactive -- references something the Victorian JUST said.
Avoid: Do NOT write broad opening questions -- turns should feel like reactions to something just said.
Passage segment 3 (~62w):
"I am not proposing a life of contemplation as a universal model. There are men to whom \
action is as necessary as breath, and they should act. What I am proposing is simply that \
we cease to regard the busy man as automatically the better man. Some of the most \
beneficial work ever done in this world was done by men sitting alone in rooms, thinking."\
""",
        '{"subject": "an argument that idleness and solitary thought are more valuable than the Victorian cult of perpetual busyness", "turns": ["You look like somebody who is perfectly content to do nothing all day.", "I know men who preach exactly this and spend every waking hour in agitation anyway.", "What do you mean, sitting in judgment upon oneself -- is that really what keeps busy men running?"]}'
    ),
]

# ---------------------------------------------------------------------------
# Conversation judge (Layer 2)
# ---------------------------------------------------------------------------

CONV_JUDGE_SYSTEM_PROMPT = """\
You are evaluating multi-turn Victorian dialogue training conversations for quality.

You will be given a subject description and a full conversation: alternating human turns
and Victorian gentleman responses. Score the conversation 0-3:

   3  Excellent -- Turn 1 is a genuine cold open (no prior context assumed, no demonstratives
      like "those", "that", "these") that invites a topical response; every human turn reacts
      naturally to the immediately preceding Victorian beat; Victorian responses are substantive
      and on-subject; no template phrases copied verbatim from style examples.

   2  Good -- mostly works; Turn 1 may be slightly generic (e.g. "How are you today?") but
      NOT mid-conversation; at most one turn is a weak match or mildly formulaic.

   1  Marginal -- Turn 1 presupposes prior context ("tell me more about those", "What prompted these changes to the law", uses
      demonstratives like "those", "that", "these", or "too" implying something was already
      said); OR two or more turns poorly match their Victorian response; OR obvious template
      phrases lifted verbatim.

   0  Unusable -- Turn 1 clearly mid-conversation (references a prior exchange, uses "as
      you mentioned", "those concerns", "that view"); OR human asks "when did YOU arrive?"
      but the Victorian responds entirely in the third person about a different individual;
      OR complete mismatch throughout.

IMPORTANT: The cold-open presupposition rule applies to Turn 1 ONLY. Turns 2 and beyond
are EXPECTED to reference what the Victorian just said -- demonstratives and callback
phrases in Turn 2+ are correct behaviour, not flaws. Only penalise a turn for
presupposition if it is Turn 1.

Respond with ONLY valid JSON, no commentary:
{"conv_score": 0, "conv_note": "one brief sentence"}
"""

_CONV_JUDGE_FEW_SHOTS: list[tuple[str, str]] = [
    (
        # Score 3 -- specific cold open, all turns connect
        "Subject: Stevenson on the therapeutic value of movement and travel for a restless mind\n\n"
        "[Human]: Do you believe that traveling is more about the journey, or the destination?\n"
        "[Victorian]: For my part, I travel not to go anywhere, but to go. I travel for travel's sake."
        " The great affair is to move; to feel the needs and hitches of our life more nearly;"
        " to come down off this feather-bed of civilisation, and find the globe granite underfoot.\n"
        "[Human]: What if the place you travel to is disappointing?\n"
        "[Victorian]: You must not look for the realisation of your dreams in the foreign country."
        " The expectations you carry are heavier than any country can bear.",
        '{"conv_score": 3, "conv_note": "Turn 1 is a specific cold open on the stated subject;'
        ' Turn 2 reacts naturally to the Victorian\'s preceding answer."}'
    ),
    (
        # Score 2 -- generic turn 1, otherwise natural
        "Subject: arguments against the cult of perpetual busyness in Victorian professional life\n\n"
        "[Human]: People are so busy nowadays, don't you think?\n"
        "[Victorian]: I have always maintained that the great error of our age is the worship of"
        " busyness for its own sake. We have persuaded ourselves that a man who is always occupied"
        " is a virtuous man.\n"
        "[Human]: You say virtuous -- is that truly how people see it?\n"
        "[Victorian]: Precisely so. The man who has learned to sit still and think is called an"
        " idler, when in fact he is the rarer and more valuable creature.",
        '{"conv_score": 2, "conv_note": "Turn 1 is generic but a natural cold opener;'
        ' Turn 2 picks up a specific word from the Victorian\'s response."}'
    ),
    (
        # Score 2 -- direction cold open; "Tell me about X" / "Describe X" are NOT presupposition failures
        "Subject: conditions in the Yorkshire woollen mills and the long-term health effects on workers\n\n"
        "[Human]: Tell me about the conditions in the Yorkshire mills.\n"
        "[Victorian]: The mills presented at that time a scene of considerable activity and considerable"
        " misery in equal proportion. The hours were long -- fourteen or fifteen in the height of the"
        " season -- and the air within was thick with floating fibre, which settled on the lungs of"
        " the workers over years and reduced many to a state of chronic debility before they had"
        " reached middle age.\n"
        "[Human]: Was there any attempt to regulate the hours or improve the air in those places?\n"
        "[Victorian]: There were attempts -- the Factory Acts had made some inroad upon the worst"
        " abuses by mid-century, though enforcement was another matter. An inspector might appear"
        " once in several years, and between visits the masters arranged things much as they saw fit.",
        '{"conv_score": 2, "conv_note": "Turn 1 is a direction cold open -- \'Tell me about X\' and'
        ' \'Describe X\' are clean imperative openers that introduce a topic without presupposing any'
        ' prior exchange; Turn 2 reacts naturally to the Victorian\'s account."}'
    ),
    (
        # Score 1 -- demonstrative in turn 1
        "Subject: the strict censorship and poor administration of the Papal States\n\n"
        "[Human]: Tell me more about those restrictions you mentioned.\n"
        "[Victorian]: The public voice cannot demand them, because it has no organ either in"
        " the council or the senate, the members of which are nominated by the Pope.\n"
        "[Human]: And the press -- is it any freer?\n"
        "[Victorian]: The only journal permitted is the official gazette, which prints what the"
        " authorities choose to print and nothing besides.",
        '{"conv_score": 1, "conv_note": "Turn 1 uses \'those restrictions you mentioned\' --'
        ' a demonstrative that presupposes prior conversation not shown."}'
    ),
    (
        # Score 2 -- clean Turn 1; Turn 3 references established context (correct, not a flaw)
        "Subject: the history and political instability of Haiti after slavery was abolished\n\n"
        "[Human]: When did slavery actually end in Haiti?\n"
        "[Victorian]: Slavery was abolished in 1886, and bankruptcy has by no means overtaken"
        " the island. It is however yet too early to speculate on the results of the new order"
        " of things. It is complicated in every way by the antagonism between Cubans and"
        " Spaniards, each wishing to quell the other.\n"
        "[Human]: What triggered the ongoing disputes?\n"
        "[Victorian]: At the period of the French Revolution the mulattoes demanded that its"
        " benefits should be extended to them, and a destructive civil war arose.\n"
        "[Human]: I've seen similar tensions in other places -- different races fighting for"
        " power after the old order collapses.\n"
        "[Victorian]: In 1793 slavery was abolished in the colony. A negro, Toussaint"
        " l'Ouverture, was quickly successful in asserting the independence of the negroes;"
        " but his power collapsed in 1805. A confused period of war and turbulence followed.",
        '{"conv_score": 2, "conv_note": "Turn 1 is a clean cold open; Turn 3\'s \'the old order\''
        ' correctly references context established across the conversation -- callback phrases'
        ' in Turn 3+ are expected, not a presupposition flaw."}'
    ),
    (
        # Score 0 -- third-person mismatch
        "Subject: biographical details of a Welsh immigrant's career and marriage in America\n\n"
        "[Human]: When did you first arrive in America?\n"
        "[Victorian]: He married his present wife on the 24th of December, 1878; he came to"
        " America in 1831, and lived first in Massachusetts before removing to Wisconsin.\n"
        "[Human]: And your work as a tailor -- when did that begin?\n"
        "[Victorian]: He served his apprenticeship in Wales before emigrating, and continued"
        " in the trade upon his arrival.",
        '{"conv_score": 0, "conv_note": "Human asks \'when did YOU arrive?\' but the Victorian'
        ' responds entirely in the third person about a different individual."}'
    ),
]


def parse_conv_judgment(raw: str) -> dict | None:
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    score_raw = obj.get("conv_score")
    try:
        score = int(score_raw)
        if score not in (0, 1, 2, 3):
            return None
    except (TypeError, ValueError):
        return None
    return {
        "conv_score": score,
        "conv_note":  str(obj.get("conv_note", "")).strip()[:160],
    }


# ---------------------------------------------------------------------------
# JSON parse utility
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def _parse_json(raw: str) -> dict | None:
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Passage utilities
# ---------------------------------------------------------------------------

def wc(text: str) -> int:
    return len(text.split())


_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z“‘"])')
_GARBAGE_RE    = re.compile(r'(?:\b\w\b[\s,]*){5,}')


def split_into_segments(text: str) -> list[str]:
    """Split a passage into 2-4 roughly equal segments at sentence boundaries."""
    w = wc(text)
    n_target = 2 if w < 220 else 3 if w < 310 else 4

    sentences = _SENT_SPLIT_RE.split(text)
    if len(sentences) < n_target:
        n_target = 2 if len(sentences) >= 2 else 1
    if n_target < 2:
        return [text]

    total  = sum(wc(s) for s in sentences)
    target = total / n_target

    segments: list[str] = []
    current:  list[str] = []
    current_w = 0

    for idx, sent in enumerate(sentences):
        current.append(sent)
        current_w += wc(sent)
        remaining_sents = len(sentences) - idx - 1
        remaining_segs  = n_target - len(segments) - 1

        if (current_w >= target * 0.75
                and remaining_segs > 0
                and remaining_sents >= remaining_segs):
            segments.append(" ".join(current))
            current   = []
            current_w = 0

    # 2026-09-20 ROOT-CAUSE FIX (Trip, epistolary round): this used to merge the trailing
    # leftover into segments[-1] ("segments[-1] += ...") instead of giving it its own slot --
    # the loop above only ever flushes (n_target - 1) segments by design (the final "slot" is
    # always meant to be this leftover), so merging doubled up the last segment's share of the
    # whole passage. Previously worked around post-hoc, per-file, via
    # resegment_multiturn_final_turn.py (2026-08-10) --
    # fixing it here means new generations don't need that cleanup pass at all.
    if current:
        segments.append(" ".join(current))

    return [s for s in segments if wc(s) >= MIN_SEGMENT_WORDS]


_THIRD_PERSON_RE = re.compile(r"\b(he|she|him|her|his|hers)\b", re.IGNORECASE)
_LEGAL_RE        = re.compile(r"\b(defendant|plaintiff|appellant|respondent|deponent)\b", re.IGNORECASE)

# Passages about known thinking_machines figures/topics are legitimately
# biographical and third-person-heavy; exempt them from the tp filter.
_TM_EXEMPT_RE = re.compile(
    r"\b(babbage|ada lovelace|lovelace|analytical engine|difference engine|"
    r"calculating engine|calculating machine|jacquard|menabrea|"
    r"mechanical philosopher|samuel butler|erewhon)\b",
    re.IGNORECASE,
)


def _is_usable(text: str) -> bool:
    """Heuristic pre-filter before the LLM passage gate."""
    if _GARBAGE_RE.search(text):
        return False
    if text.lstrip().startswith(("“", "”")):
        return False
    if text.count('"') >= 8 or text.count("”") >= 4:
        return False
    # Reject predominantly non-English text
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / len(text) > 0.05:
        return False
    # Reject biographical dictionary entries and legal case reports.
    # Exempt passages that are clearly about known thinking_machines figures —
    # biographical prose about Babbage, Lovelace etc. is legitimately third-person.
    words = text.split()
    if not _TM_EXEMPT_RE.search(text):
        tp_hits = len(_THIRD_PERSON_RE.findall(text))
        if tp_hits / len(words) > 0.05:
            return False
    if _LEGAL_RE.search(text):
        return False
    return True


# ---------------------------------------------------------------------------
# Raw passage loaders (MIN_WORDS=150, MAX_WORDS=400)
# ---------------------------------------------------------------------------

def _load_ncse_raw(
    rng: random.Random,
    min_w: int = MIN_WORDS,
    max_w: int = MAX_WORDS,
    cap: int = 10_000,
) -> list[dict]:
    if not NCSE_DIR.exists():
        print(f"  WARNING: {NCSE_DIR} not found -- skipping NCSE")
        return []
    all_passages: list[dict] = []
    for txt_file in sorted(NCSE_DIR.glob("*.txt")):
        try:
            text = txt_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        passages = extract_passages_from_text(text, min_words=min_w, max_words=max_w)
        if len(passages) > cap:
            passages = rng.sample(passages, cap)
        ncse_key = txt_file.stem.lower().strip()
        ncse_pub, ncse_year = NCSE_MAP.get(ncse_key, (txt_file.stem, ""))
        all_passages.extend(
            {"text": p, "source": "NCSE", "title": txt_file.stem, "author": "unknown",
             "book_title": ncse_pub, "book_year": ncse_year, "prov_method": "ncse_map"}
            for p in passages
        )
        print(f"  {'NCSE/' + txt_file.name:<40} {len(passages):>7,} candidates")
    return all_passages


def _load_bl_raw(
    rng: random.Random,
    min_w: int = MIN_WORDS,
    max_w: int = MAX_WORDS,
    max_books: int = 2000,
    max_passages: int = 60_000,
) -> list[dict]:
    if not BL_DATA_DIR.exists():
        print(f"  WARNING: {BL_DATA_DIR} not found -- skipping BL")
        return []
    decade_files = sorted(BL_DATA_DIR.glob("books_*.jsonl.gz"))
    if not decade_files:
        print(f"  WARNING: no books_*.jsonl.gz in {BL_DATA_DIR} -- skipping BL")
        return []

    all_passages: list[dict] = []
    books_per_decade = max(1, max_books // len(decade_files))

    for dfile in decade_files:
        decade_passages: list[dict] = []
        try:
            with gzip.open(str(dfile), "rt", encoding="utf-8", errors="replace") as f:
                books_seen = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = rec.get("text", "")
                    if not text or len(text.split()) < min_w:
                        continue
                    title  = rec.get("title", "Unknown")
                    author = rec.get("author") or rec.get("creator") or _author_from_title(title)
                    author_str = str(author).strip() if author else "unknown"
                    bl_author = author_str if author_str.lower() not in ("unknown", "") else ""
                    for p in extract_passages_from_text(text, min_words=min_w, max_words=max_w):
                        decade_passages.append({
                            "text":        p,
                            "source":      "BL",
                            "title":       title,
                            "author":      author_str,
                            "book_title":  title,
                            "book_author": bl_author,
                            "prov_method": "raw",
                        })
                    books_seen += 1
                    if books_seen >= books_per_decade:
                        break
        except Exception as e:
            print(f"  WARNING: error reading {dfile.name}: {e}")
            continue
        print(f"  {'BL/' + dfile.name:<40} {len(decade_passages):>7,} candidates")
        all_passages.extend(decade_passages)

    if len(all_passages) > max_passages:
        all_passages = rng.sample(all_passages, max_passages)
    return all_passages


def _load_pg_raw(
    rng: random.Random,
    min_w: int = MIN_WORDS,
    max_w: int = MAX_WORDS,
    max_books: int = 400,
    max_passages: int = 30_000,
) -> list[dict]:
    if not PG_DATA_DIR.exists():
        print(f"  WARNING: {PG_DATA_DIR} not found -- skipping PG Nonfiction")
        return []
    if not PG_CATALOG.exists():
        print(f"  WARNING: {PG_CATALOG} not found -- skipping PG Nonfiction")
        return []

    all_epubs = build_nonfiction_epubs(PG_CATALOG, PG_VICTORIAN_LIST, PG_DATA_DIR)
    if not all_epubs:
        print("  WARNING: no nonfiction epubs found -- skipping PG Nonfiction")
        return []

    pg_meta: dict[str, dict] = {}
    try:
        with PG_CATALOG.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pg_id = (row.get("Text#") or "").strip()
                if pg_id:
                    pg_meta[pg_id] = {
                        "title":  (row.get("Title") or "").replace("\n", " ").strip(),
                        "author": (row.get("Authors") or row.get("Author") or "").strip() or "unknown",
                    }
    except Exception:
        pg_meta = {}

    sampled = rng.sample(all_epubs, min(max_books, len(all_epubs)))
    all_passages: list[dict] = []

    for epub_path, pg_id, title in sampled:
        try:
            raw  = extract_epub_text(epub_path)
            text = clean_epub_text(raw)
            meta = pg_meta.get(pg_id, {})
            book   = meta.get("title") or title
            author = meta.get("author") or "unknown"
            pg_author = author if author.lower() != "unknown" else ""
            prov = "pg_catalog" if pg_id in pg_meta else "raw"
            for p in extract_passages_from_text(text, min_words=min_w, max_words=max_w):
                all_passages.append({
                    "text":        p,
                    "source":      "PG_Nonfiction",
                    "title":       title,
                    "author":      author,
                    "book_title":  book,
                    "book_author": pg_author,
                    "pg_id":       pg_id,
                    "prov_method": prov,
                })
        except Exception:
            continue

    print(f"  {'PG Nonfiction epubs':<40} {len(all_passages):>7,} candidates "
          f"({len(sampled)} books sampled from {len(all_epubs):,} available)")

    if len(all_passages) > max_passages:
        all_passages = rng.sample(all_passages, max_passages)
    return all_passages


def _load_ia_raw(
    rng: random.Random,
    min_w: int = MIN_WORDS,
    max_w: int = MAX_WORDS,
    max_files: int = 3000,
    max_passages: int = 40_000,
) -> list[dict]:
    if not IA_RAW_DIR.exists():
        print(f"  WARNING: {IA_RAW_DIR} not found -- skipping IA")
        return []

    all_files = ia_iter_files()
    if not all_files:
        print("  WARNING: no IA text files found -- skipping IA")
        return []

    sampled_files = rng.sample(all_files, min(max_files, len(all_files)))
    all_passages: list[dict] = []

    for f in sampled_files:
        try:
            paragraphs, stype = ia_process_file(f)
        except Exception:
            continue
        if stype in ("skip", "error"):
            continue
        ia_pub_title, ia_year = parse_ia_id(f.parent.name)
        for para in paragraphs:
            for p in extract_from_paragraph(para, min_words=min_w, max_words=max_w):
                all_passages.append({
                    "text":        p,
                    "source":      "IA",
                    "title":       f.parent.name,
                    "author":      "unknown",
                    "book_title":  ia_pub_title,
                    "book_year":   ia_year,
                    "prov_method": "ia_parse",
                })

    print(f"  {'IA files sampled':<40} {len(all_passages):>7,} candidates "
          f"({len(sampled_files):,} files from {len(all_files):,} available)")

    if len(all_passages) > max_passages:
        all_passages = rng.sample(all_passages, max_passages)
    return all_passages


def _load_ib_raw(
    rng: random.Random,
    min_w: int = MIN_WORDS,
    max_w: int = MAX_WORDS,
    max_shards: int = 60,
    max_passages: int = 40_000,
) -> list[dict]:
    if not IB_SHARDS_DIR.exists():
        print(f"  WARNING: {IB_SHARDS_DIR} not found -- skipping IB")
        return []
    try:
        import pandas as pd
    except ImportError:
        print("  WARNING: pandas not installed -- skipping IB")
        return []

    shard_files = sorted(IB_SHARDS_DIR.glob("*.parquet"))
    if not shard_files:
        print(f"  WARNING: no .parquet files in {IB_SHARDS_DIR} -- skipping IB")
        return []

    sampled = rng.sample(shard_files, min(max_shards, len(shard_files)))
    all_passages: list[dict] = []
    books_seen = 0

    for shard in sampled:
        try:
            # Load barcode column if present (new shards); fall back for legacy text-only shards
            try:
                df = pd.read_parquet(shard, columns=["barcode", "text"])
                has_barcode = True
            except Exception:
                df = pd.read_parquet(shard, columns=["text"])
                has_barcode = False
        except Exception as e:
            print(f"  WARNING: error reading {shard.name}: {e}")
            continue
        for row_idx, row in enumerate(df.itertuples(index=False)):
            text = row.text if hasattr(row, "text") else ""
            if not isinstance(text, str) or len(text.split()) < min_w:
                continue
            barcode = getattr(row, "barcode", "") if has_barcode else ""
            books_seen += 1
            for p in extract_passages_from_text(text, min_words=min_w, max_words=max_w):
                all_passages.append({
                    "text":      p,
                    "source":    "IB",
                    "title":     shard.stem,
                    "author":    "unknown",
                    "barcode":   barcode,
                    "shard_row": row_idx,
                })

    print(f"  {'IB shard books':<40} {len(all_passages):>7,} candidates "
          f"({books_seen} books in {len(sampled)} shards)")

    if len(all_passages) > max_passages:
        all_passages = rng.sample(all_passages, max_passages)
    return all_passages


# ---------------------------------------------------------------------------
# User message builder and validation
# ---------------------------------------------------------------------------

def _build_user(modes: list[str], segments: list[str]) -> str:
    n = len(segments)
    lines = [f"Write exactly {n} human turn{'s' if n != 1 else ''}."]
    for i, (mode, seg) in enumerate(zip(modes, segments), 1):
        m = DIALOGUE_MODES[mode]
        lines += [
            "",
            f"Turn {i} [{m['label']} -- {m['desc']}]:",
            f"Style: {m['style']}",
            f"Avoid: {m['avoid']}",
            f"Passage segment {i} (~{wc(seg)}w):",
            f'"{seg}"',
        ]
    return "\n".join(lines)


def validate_result(result: dict, n_expected: int) -> str | None:
    subject = result.get("subject", "")
    if not isinstance(subject, str) or len(subject.strip()) < 5:
        return "missing or empty subject"
    turns = result.get("turns")
    if not isinstance(turns, list):
        return "turns not a list"
    if len(turns) != n_expected:
        return f"expected {n_expected} turns, got {len(turns)}"
    for i, t in enumerate(turns):
        if not isinstance(t, str) or not t.strip():
            return f"turn {i+1} is empty"
        w = wc(t)
        if w < 2:
            return f"turn {i+1} too short ({w}w)"
        if w > 60:
            return f"turn {i+1} too long ({w}w)"
    return None


# ---------------------------------------------------------------------------
# Backends (generation + passage gate + conversation judge)
# ---------------------------------------------------------------------------

class MLXBackend:
    def __init__(self, model_name: str, temperature: float = 0.75):
        try:
            from mlx_lm import load, generate
        except ImportError:
            print("mlx-lm not installed.  Run: pip install mlx-lm")
            sys.exit(1)
        print(f"Loading {model_name}...")
        self.model, self.tokenizer = load(model_name)
        self._generate = generate
        print("Model loaded.\n")
        self._sampler = None
        try:
            from mlx_lm.sample_utils import make_sampler
            self._sampler = make_sampler(temp=temperature)
        except Exception:
            self._temp = temperature

    def _call(
        self,
        system: str,
        shots: list[tuple[str, str]],
        user_content: str,
        max_tokens: int,
        temperature: float | None = None,
    ) -> str:
        messages = [{"role": "system", "content": system}]
        for ex_user, ex_asst in shots:
            messages += [{"role": "user", "content": ex_user},
                         {"role": "assistant", "content": ex_asst}]
        messages.append({"role": "user", "content": user_content})
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        kwargs: dict = {"max_tokens": max_tokens, "verbose": False}
        if self._sampler is not None:
            kwargs["sampler"] = self._sampler
        else:
            kwargs["temp"] = getattr(self, "_temp", 0.75) if temperature is None else temperature
        return self._generate(self.model, self.tokenizer, prompt=prompt, **kwargs)

    def generate(self, modes: list[str], segments: list[str], max_tokens: int) -> dict | None:
        raw = self._call(SYSTEM_PROMPT, _FEW_SHOTS, _build_user(modes, segments), max_tokens)
        return _parse_json(raw)

    def gate_passage(self, passage_text: str, max_tokens: int = 120) -> dict | None:
        """Layer 1: assess passage via GENERATOR_SYSTEM_PROMPT. Returns parse_generation_json output."""
        raw = self._call(
            GENERATOR_SYSTEM_PROMPT, _GENERATOR_FEW_SHOTS,
            f"Passage:\n\n{passage_text}",
            max_tokens,
            temperature=0.1,
        )
        return parse_generation_json(raw)

    def judge_conv(self, subject: str, pair: list[dict], max_tokens: int = 100) -> dict | None:
        """Layer 2: score the full conversation 0-3."""
        lines = [f"Subject: {subject}", ""]
        for m in pair:
            role = "[Human]" if m["role"] == "user" else "[Victorian]"
            lines.append(f"{role}: {m['content']}")
        conv_text = "\n".join(lines)
        raw = self._call(
            CONV_JUDGE_SYSTEM_PROMPT, _CONV_JUDGE_FEW_SHOTS,
            conv_text,
            max_tokens,
            temperature=0.1,
        )
        return parse_conv_judgment(raw)


class AnthropicBackend:
    def __init__(self, model_name: str):
        try:
            import anthropic as _a
        except ImportError:
            print("anthropic package not installed.  Run: pip install anthropic")
            sys.exit(1)
        self._anthropic = _a
        self._model = model_name

    def _call(
        self,
        system: str,
        shots: list[tuple[str, str]],
        user_content: str,
        max_tokens: int,
        temperature: float = 0.75,
    ) -> str | None:
        messages = []
        for ex_user, ex_asst in shots:
            messages += [{"role": "user", "content": ex_user},
                         {"role": "assistant", "content": ex_asst}]
        messages.append({"role": "user", "content": user_content})

        client = self._anthropic.Anthropic()
        try:
            resp = client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            )
            return resp.content[0].text.strip()
        except Exception as e:
            print(f"  API error: {e}", file=sys.stderr)
            return None

    def generate(self, modes: list[str], segments: list[str], max_tokens: int) -> dict | None:
        raw = self._call(SYSTEM_PROMPT, _FEW_SHOTS, _build_user(modes, segments), max_tokens)
        return _parse_json(raw) if raw else None

    def gate_passage(self, passage_text: str, max_tokens: int = 120) -> dict | None:
        raw = self._call(
            GENERATOR_SYSTEM_PROMPT, _GENERATOR_FEW_SHOTS,
            f"Passage:\n\n{passage_text}",
            max_tokens,
            temperature=0.1,
        )
        return parse_generation_json(raw) if raw else None

    def judge_conv(self, subject: str, pair: list[dict], max_tokens: int = 100) -> dict | None:
        lines = [f"Subject: {subject}", ""]
        for m in pair:
            role = "[Human]" if m["role"] == "user" else "[Victorian]"
            lines.append(f"{role}: {m['content']}")
        conv_text = "\n".join(lines)
        raw = self._call(
            CONV_JUDGE_SYSTEM_PROMPT, _CONV_JUDGE_FEW_SHOTS,
            conv_text,
            max_tokens,
            temperature=0.1,
        )
        return parse_conv_judgment(raw) if raw else None


# ---------------------------------------------------------------------------
# Candidate loading
# ---------------------------------------------------------------------------

def load_candidates(
    target: int,
    subcorpus_filter: str | None,
    mode_filter: str | None,
    seed: int,
    overshoot: float = 2.5,
) -> list[dict]:
    rng = random.Random(seed)

    print("Loading source4 passages (150-400 words)...")
    pool = (
        _load_ncse_raw(rng)
        + _load_bl_raw(rng)
        + _load_pg_raw(rng)
        + _load_ia_raw(rng)
        + _load_ib_raw(rng)
    )

    if not pool:
        print("ERROR: no passages loaded -- check source paths.")
        sys.exit(1)

    print(f"\n  Total raw passages: {len(pool):,}")
    print("  Applying heuristic filters and routing...")

    candidates: list[dict] = []
    seen: set[str] = set()

    for p in pool:
        text = p["text"]
        key = text.lower()[:120]
        if key in seen:
            continue
        if not starts_cleanly(text) or not ends_cleanly(text):
            continue
        if not _is_usable(text):
            continue

        segs = split_into_segments(text)
        if len(segs) < 2:
            continue

        matches    = route_passage_to_subcorpus(text, p.get("title", ""), p.get("author", ""))
        sub_corpus = matches[0][0] if matches else "unassigned"

        if subcorpus_filter and sub_corpus != subcorpus_filter:
            continue

        n_segs = len(segs)
        if mode_filter:
            t1 = "inquiry" if mode_filter == "follow_up" else mode_filter
            turn_modes = [t1] + [mode_filter] * (n_segs - 1)
        else:
            turn_modes = (
                [rng.choice(COLD_OPEN_MODES)]
                + [rng.choice(ALL_MODES) for _ in range(n_segs - 1)]
            )

        seen.add(key)
        candidates.append({
            "text":        text,
            "segments":    segs,
            "source":      p["source"],
            "title":       p.get("title", ""),
            "author":      p.get("author", "unknown"),
            "book_title":  p.get("book_title", ""),
            "book_author": p.get("book_author", ""),
            "book_year":   p.get("book_year", ""),
            "pg_id":       p.get("pg_id", ""),
            "prov_method": p.get("prov_method", ""),
            "barcode":     p.get("barcode", ""),
            "shard_row":   p.get("shard_row", -1),
            "sub_corpus":  sub_corpus,
            "turn_modes":  turn_modes,
        })

    rng.shuffle(candidates)

    sub_dist = Counter(c["sub_corpus"] for c in candidates)
    src_dist = Counter(c["source"]     for c in candidates)
    print(f"\n  Usable candidates after heuristic filter: {len(candidates):,}")
    print("  Source distribution:")
    for src, n in src_dist.most_common():
        print(f"    {src:<15} {n:>7,}")
    print("  Sub-corpus distribution:")
    for sub, n in sub_dist.most_common():
        print(f"    {sub:<25} {n:>7,}")

    # Cap at overshoot * target (or all if target=0)
    cap = int(target * overshoot) if target > 0 else len(candidates)
    if len(candidates) > cap:
        candidates = candidates[:cap]
        print(f"\n  Pre-loading cap (overshoot {overshoot}x): {cap:,}")

    return candidates


def load_done_keys(path: Path | None = None) -> set[str]:
    """Keys already written, so a rerun continues instead of re-doing the same work.
    Reads the actual output path (--output), falling back to the OUTPUT constant."""
    done: set[str] = set()
    path = path or OUTPUT
    if not path.exists():
        return done
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec  = json.loads(line)
            pair = rec.get("pair", [])
            key  = next(
                (m["content"].lower()[:120] for m in pair if m.get("role") == "assistant"),
                "",
            )
            if key:
                done.add(key)
        except Exception:
            pass
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate multi-turn Victorian SFT pairs from source4 corpus"
    )
    ap.add_argument("--count",          type=int,   default=0,
                    help="Target number of records to write (0 = process all candidates)")
    ap.add_argument("--overshoot",      type=float, default=2.5,
                    help="Pre-load overshoot factor to absorb Layer 1 + Layer 2 rejections (default: 2.5)")
    ap.add_argument("--subcorpus",      type=str,   default="",
                    choices=sorted(SOURCE_SUBCORPORA) + [""],
                    help="Restrict to one sub-corpus (default: all)")
    ap.add_argument("--mode",           type=str,   default="",
                    choices=list(DIALOGUE_MODES.keys()) + [""],
                    help="Fix dialogue mode for every turn (default: random per turn)")
    ap.add_argument("--min-conv-score", type=int,   default=1,
                    help="Minimum Layer 2 score to write a record (0-3, default: 1)")
    ap.add_argument("--preview",        action="store_true",
                    help="Show passages and proposed segments, no model calls")
    ap.add_argument("--backend",        choices=["mlx", "anthropic"], default="mlx")
    ap.add_argument("--model",          type=str,   default="")
    ap.add_argument("--max-tokens",     type=int,   default=400)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--output",         type=str,   default="",
                    help="Output file path (default: training_data/dialogue/multiturn_source4.jsonl)")
    args = ap.parse_args()

    mode_filter      = args.mode.strip()      or None
    subcorpus_filter = args.subcorpus.strip() or None

    candidates = load_candidates(
        args.count, subcorpus_filter, mode_filter, args.seed, args.overshoot
    )

    t1_dist     = Counter(c["turn_modes"][0] for c in candidates)
    t2plus_dist = Counter(m for c in candidates for m in c["turn_modes"][1:])
    print("\n  Turn 1 mode distribution (no follow_up):")
    for mode in ALL_MODES:
        print(f"    {mode:<15} {t1_dist.get(mode, 0):>5}")
    print("  Turn 2+ mode distribution:")
    for mode in ALL_MODES:
        print(f"    {mode:<15} {t2plus_dist.get(mode, 0):>5}")

    if not candidates:
        print("Nothing to do.")
        return

    output_path = Path(args.output) if args.output else OUTPUT
    done_keys = load_done_keys(output_path)
    remaining = [
        c for c in candidates
        if c["segments"][0].lower()[:120] not in done_keys
    ]
    print(f"\n  Already done: {len(candidates) - len(remaining):,}")
    print(f"  To process:   {len(remaining):,}")

    if not remaining:
        print("Nothing to do.")
        return

    if args.preview:
        print(f"\n--- PREVIEW (first 6) ---")
        for c in remaining[:6]:
            modes_str = " | ".join(c["turn_modes"])
            print(f"\n[{c['source']} | sub={c['sub_corpus']} | {len(c['segments'])} segs"
                  f" | modes: {modes_str} | {wc(c['text'])}w]")
            print(f"  title: {c['title'][:80]}")
            for i, (seg, mode) in enumerate(zip(c["segments"], c["turn_modes"]), 1):
                print(f"  Seg {i} [{mode}] ({wc(seg)}w): {seg[:100]}...")
        return

    model_name = args.model or (
        DEFAULT_ANTHROPIC_MODEL if args.backend == "anthropic" else DEFAULT_MLX_MODEL
    )
    backend = AnthropicBackend(model_name) if args.backend == "anthropic" else MLXBackend(model_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written   = 0
    n_gate_fail = 0
    n_gen_fail  = 0
    n_judge_low = 0
    score_dist: dict[int, int] = {}
    t0 = time.time()

    with output_path.open("a", encoding="utf-8") as f:
        for i, c in enumerate(remaining, 1):

            # --- Layer 1: passage gate ---
            gate = backend.gate_passage(c["text"])
            if gate is None or not gate.get("passage_ok"):
                n_gate_fail += 1
                _progress(i, len(remaining), n_written, n_gate_fail, n_gen_fail, n_judge_low, t0)
                continue

            content_type = gate.get("content_type", "expository")

            # --- Generate human turns ---
            result = backend.generate(c["turn_modes"], c["segments"], args.max_tokens)
            err    = None if result is None else validate_result(result, len(c["segments"]))
            if result is None or err is not None:
                result = backend.generate(c["turn_modes"], c["segments"], args.max_tokens)
                err    = None if result is None else validate_result(result, len(c["segments"]))
            if result is None or err is not None:
                n_gen_fail += 1
                _progress(i, len(remaining), n_written, n_gate_fail, n_gen_fail, n_judge_low, t0)
                continue

            turns = result["turns"]
            pair  = []
            for human_turn, victorian_seg in zip(turns, c["segments"]):
                pair.append({"role": "user",      "content": human_turn.strip()})
                pair.append({"role": "assistant",  "content": victorian_seg.strip()})
            subject = result["subject"].strip()

            # --- Layer 2: conversation judge ---
            judgment   = backend.judge_conv(subject, pair)
            conv_score = judgment["conv_score"] if judgment else None
            conv_note  = judgment["conv_note"]  if judgment else ""

            if conv_score is not None:
                score_dist[conv_score] = score_dist.get(conv_score, 0) + 1

            if conv_score is not None and conv_score < args.min_conv_score:
                n_judge_low += 1
                _progress(i, len(remaining), n_written, n_gate_fail, n_gen_fail, n_judge_low, t0)
                continue

            record = {
                "source":        c["source"],
                "title":         c["title"],
                "sub_corpus":    c["sub_corpus"],
                "content_type":  content_type,
                "response_type": "corpus",
                "sub_type":      "multiturn_source4",
                "turn_modes":    c["turn_modes"],
                "subject":       subject,
                "conv_score":    conv_score,
                "conv_note":     conv_note,
                "book_title":    c.get("book_title", ""),
                "book_author":   c.get("book_author", ""),
                "book_year":     c.get("book_year", ""),
                "prov_method":   c.get("prov_method", ""),
                "barcode":       c.get("barcode", ""),
                "shard_row":     c.get("shard_row", -1),
                "n_turns":       len(pair),
                "pair":          pair,
            }
            if c.get("pg_id"):
                record["pg_id"] = c["pg_id"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

            _progress(i, len(remaining), n_written, n_gate_fail, n_gen_fail, n_judge_low, t0)

            if args.count > 0 and n_written >= args.count:
                break

    print(
        f"\nDone:\n"
        f"  written:        {n_written}\n"
        f"  gate rejected:  {n_gate_fail}\n"
        f"  gen failed:     {n_gen_fail}\n"
        f"  score too low:  {n_judge_low}\n"
        f"  conv_score dist: { {k: score_dist[k] for k in sorted(score_dist)} }\n"
        f"  -> {output_path}"
    )


def _progress(
    i: int, total: int, written: int,
    gate_fail: int, gen_fail: int, judge_low: int,
    t0: float,
) -> None:
    if i % 20 == 0 or i == total:
        elapsed = time.time() - t0
        rate    = i / elapsed if elapsed > 0 else 0
        eta     = (total - i) / rate if rate > 0 else 0
        print(
            f"  [{i:>5}/{total}] written={written} "
            f"gate_fail={gate_fail} gen_fail={gen_fail} score_low={judge_low} "
            f"({rate:.1f}/s, ETA {eta/60:.0f}m)"
        )


if __name__ == "__main__":
    main()

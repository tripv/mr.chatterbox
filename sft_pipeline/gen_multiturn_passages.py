"""
gen_multiturn_passages.py — Generate multi-turn Victorian SFT conversations from corpus passages.

Takes Victorian passages from letter and essay sources, splits each into 2–4 segments,
and uses a local Qwen-14B (MLX) model to generate human turns that naturally precede each
segment. Victorian turns are verbatim passage text; human turns are model-generated.

Six dialogue modes (randomly sampled per passage unless --mode is set):
  inquiry      — direct, factual questions
  interview    — open reflective prompts
  challenge    — skeptical pushback
  assertion    — human makes a claim the Victorian responds to
  follow_up    — clarification or continuation of the prior beat
  explanation  — human shares their own experience; Victorian responds to common ground

Sources: same letter/essay epubs as gen_introspection_pairs.py

Output: training_data/dialogue/multiturn_passages.jsonl
        Append mode — safe to re-run; already-processed passages are skipped.

Run:
    python gen_multiturn_passages.py                      # all passages, random modes
    python gen_multiturn_passages.py --preview            # show segments, no model
    python gen_multiturn_passages.py --count 200
    python gen_multiturn_passages.py --mode challenge     # fix one mode
    python gen_multiturn_passages.py --backend anthropic
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

PG_DATA = Path("raw_book_data/PG Data")
OUTPUT  = Path("training_data/dialogue/multiturn_passages.jsonl")

DEFAULT_MLX_MODEL       = "mlx-community/Qwen2.5-14B-Instruct-4bit"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

SOURCES = [
    ("Stevenson — Virginibus Puerisque", PG_DATA / "Virginibus Puerisque, and Other Papers.386.epub"),
    ("Stevenson — Essays of RLS",        PG_DATA / "Essays of Robert Louis Stevenson.10761.epub"),
    ("Stevenson — Swanston V1",          PG_DATA / "The Works of Robert Louis Stevenson - Swanston Edition, Vol. 01.21686.epub"),
    ("Stevenson — Swanston V2",          PG_DATA / "The Works of Robert Louis Stevenson - Swanston Edition, Vol. 02.30527.epub"),
    ("Stevenson — Swanston V4",          PG_DATA / "The Works of Robert Louis Stevenson - Swanston Edition, Vol. 04.30700.epub"),
    ("Stevenson — Swanston V5",          PG_DATA / "The Works of Robert Louis Stevenson - Swanston Edition, Vol. 05.30744.epub"),
    ("Stevenson — Letters V1",           PG_DATA / "The Letters of Robert Louis Stevenson — Volume 1.622.epub"),
    ("Stevenson — Letters V2",           PG_DATA / "The Letters of Robert Louis Stevenson — Volume 2.637.epub"),
    ("Newman — Apologia",                PG_DATA / "Apologia pro vita sua.19690.epub"),
    ("Newman — Apologia (alt)",          PG_DATA / "Apologia pro vita sua : $b being a history of his religious opinions.22088.epub"),
    ("Ruskin — Selections",              PG_DATA / "Selections From the Works of John Ruskin.15200.epub"),
    ("Ruskin — Harvest of Ruskin",       PG_DATA / "The Harvest of Ruskin.51808.epub"),
    ("Ruskin — Ruskin Relics",           PG_DATA / "Ruskin Relics.46426.epub"),
    ("Dickens — Letters Vol.1",          PG_DATA / "The Letters of Charles Dickens. Vol. 1, 1833-1856.25852.epub"),
    ("Dickens — Letters Vol.2",          PG_DATA / "The Letters of Charles Dickens. Vol. 2, 1857-1870.25853.epub"),
    ("Dickens — Letters Vol.3",          PG_DATA / "The Letters of Charles Dickens. Vol. 3, 1836-1870.25854.epub"),
    ("EBB — Letters Vol.1",              PG_DATA / "The Letters of Elizabeth Barrett Browning (Volume 1 of 2).13018.epub"),
    ("EBB — Letters Vol.2",              PG_DATA / "The Letters of Elizabeth Barrett Browning (Volume 2 of 2).16646.epub"),
    ("Browning — Courtship Letters V1",  PG_DATA / "The Letters of Robert Browning and Elizabeth Barrett Barrett, Vol. 1 (of 2) 1845-1846.16182.epub"),
    ("FitzGerald — Letters Vol.1",       PG_DATA / "Letters of Edward FitzGerald, in Two Volumes. Vol. 1.20452.epub"),
    ("FitzGerald — Letters Vol.2",       PG_DATA / "Letters of Edward FitzGerald, in Two Volumes. Vol. 2.20539.epub"),
    ("Acton — Letters to Mary Gladstone", PG_DATA / "Letters of Lord Acton to Mary, Daughter of the Right Hon. W. E. Gladstone.38214.epub"),
]

MIN_WORDS         = 150
MAX_WORDS         = 400
MIN_SEGMENT_WORDS = 30

# ---------------------------------------------------------------------------
# Dialogue modes
# ---------------------------------------------------------------------------

COLD_OPEN_MODES = ["inquiry", "interview", "challenge", "assertion", "explanation"]
ALL_MODES       = ["inquiry", "interview", "challenge", "assertion", "follow_up", "explanation"]

DIALOGUE_MODES: dict[str, dict[str, str]] = {
    "inquiry": {
        "label": "Inquiry",
        "desc":  "The visitor asks direct, factual questions — short, curious, no agenda.",
        "style": "Register: short, curious factual questions. Examples of TONE (do not copy): "
                 "'How long have you held that view?' 'When did this happen?' 'Who else was involved?'",
        "avoid": "Do NOT write open reflective prompts or long preambles. "
                 "Do NOT name specific people or events from the passage the human hasn't heard yet.",
    },
    "interview": {
        "label": "Interview",
        "desc":  "The visitor probes for reflection and personal experience from the Victorian. Open-ended prompts that invite the Victorian to expand.",
        "style": "Register: warm, open, inviting elaboration. Examples of TONE (do not copy): "
                 "'How are you this fine day?' 'What has this experience been like for you?' 'How do you feel about recent events?'",
        "avoid": "Do NOT ask leading questions that telegraph what the Victorian is about to say.",
    },
    "challenge": {
        "label": "Challenge",
        "desc":  "The visitor is skeptical and pushes back. They challenge claims, question conclusions, express disagreement.",
        "style": "Register: genuine disagreement or skepticism grounded in the subject. Examples of TONE (do not copy): "
                 "'That strikes me as a convenient excuse.' 'Surely the evidence points the other way.' 'I would have expected better reasoning.'",
        "avoid": "Do NOT write polite questions — the visitor is genuinely challenging the Victorian. "
                 "Do NOT use the same opener twice across turns.",
    },
    "assertion": {
        "label": "Assertion",
        "desc":  "The visitor makes a declarative statement or claim rather than asking a question. The Victorian responds to what has been put forward.",
        "style": "Register: bold declaration grounded in the subject. Examples of TONE (do not copy): "
                 "'Your reputation precedes you.' 'That view seems rather self-serving.' 'You are more cautious than I expected.'",
        "avoid": "Human turns must be STATEMENTS, not questions. "
                 "Do NOT copy the tone examples verbatim — write a fresh statement grounded in the specific subject.",
    },
    "follow_up": {
        "label": "Follow-up",
        "desc":  "The visitor asks for clarification, more detail, or presses on a specific point from the prior beat.",
        "style": "Register: tight, reactive, referencing something the Victorian JUST said. Examples of TONE (do not copy): "
                 "'What do you mean by that exactly?' 'And then what happened?' 'Go on — I want to understand this part.'",
        "avoid": "Do NOT write broad opening questions — turns must react to the immediately preceding Victorian beat. "
                 "Do NOT reference people or things not yet mentioned by the Victorian.",
    },
    "explanation": {
        "label": "Explanation",
        "desc":  "The visitor shares their own concrete experience or view on the specific subject — creating common ground the Victorian responds to.",
        "style": "Register: grounded personal statement on the specific topic. Examples of TONE (do not copy): "
                 "'Where I come from, this is handled very differently.' 'I tried something similar once, with quite different results.' "
                 "'That matches something I have been observing myself.'",
        "avoid": "Human turns must be STATEMENTS about the visitor's own experience, not questions. "
                 "Do NOT use vague phrases like 'I've often felt the same way' without specifying what way — "
                 "anchor the statement to the specific subject.",
    },
}

# ---------------------------------------------------------------------------
# System prompt (generic — mode instructions live in the user message)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are writing human turns for a multi-turn Victorian dialogue training dataset.

You will be given a Victorian passage split into numbered segments. Each segment will \
become one response by a Victorian gentleman in a conversation.

Step 1 — identify the subject: read all segments and write a concise subject description \
(10–20 words) capturing the specific topic, person, event, or argument the passage is about. \
Be specific — not "the author reflects on life" but "Stevenson on the dream of being a \
Siamese pirate while ill in Davos."

Step 2 — write exactly N human turns — one to PRECEDE each numbered segment — so that \
each Victorian segment reads as a natural response to the human turn before it. \
Each turn has its own dialogue mode shown next to its segment — follow it precisely. \
Every turn must be clearly about the specific subject you identified, not a generic prompt \
that could precede any passage.

Rules (always apply):
- Write in modern colloquial English — absolutely NOT Victorian
- Each turn must be 1–2 sentences, 5–35 words maximum
- Turn 1 is a COLD OPEN — the human is approaching this person for the first time with no \
prior conversation. It must make sense as a first line to a stranger. Never use demonstratives \
("those", "that", "these") that presuppose shared context, and never reference something \
not yet said ("what you said", "earlier", "as you mentioned", "that way", "that experience"). \
BAD: "Could you say more about those childhood memories?" — demonstrative presupposes shared context \
BAD: "I've often felt that way myself." — "that way" references something unsaid \
BAD: "Why did you wait so long to share your doubts?" - presupposes shared context \
GOOD: "What was your childhood like?" \
GOOD: "I find myself thinking about progress a great deal lately." — introduces a topic, presupposes nothing
- Turns 2+ should feel like natural reactions to the preceding Victorian beat
- Do NOT use "Tell me about X" or "What do you think of X" — these telegraph the answer
- Do NOT start with "So," or "Well,"
- Style examples show REGISTER only — do not copy them into your output. Write fresh \
turns grounded in the specific subject you identified.
- Do NOT put specific proper nouns, titles, or named events from the Victorian's response \
into the human turn that precedes it — the human has not heard those words yet.

Return ONLY valid JSON — no commentary, no markdown:
{"subject": "specific subject in 10-20 words", "turns": ["human turn 1", "human turn 2", ...]}
"""

# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

_FEW_SHOTS: list[tuple[str, str]] = [
    # Turn 1: Inquiry | Turn 2: Follow-up | Turn 3: Challenge
    (
        """\
Write exactly 3 human turns.

Turn 1 [Inquiry — The visitor asks direct, factual questions — short, curious, no agenda.]:
Style: Short factual questions: 'Where were you educated?' 'Who educated you?'
Avoid: Do NOT write open reflective prompts or long preambles.
Passage segment 1 (~82w):
"I was educated at home until my fifteenth year, entirely under my father's direction, \
who was a man of considerable learning, though he had never held any university post. \
He taught me Latin and Greek in the ordinary way, but what distinguished his method was \
the insistence upon reading aloud — every lesson ended with a passage read aloud, and he \
would correct not merely the pronunciation but the manner of delivery."

Turn 2 [Follow-up — The visitor asks for clarification, more detail, or presses on a specific point from the prior beat.]:
Style: Clarifications and continuations: 'Could you say more about that?' 'Where did you study after that?' 'Tell me more about your education.'
Avoid: Do NOT write broad opening questions — turns should feel like reactions to something just said.
Passage segment 2 (~79w):
"After this early training I went up to Oxford, where I found to my considerable surprise \
that my preparation was superior to that of most of my contemporaries. Many of them had \
been at the great public schools and yet read Latin with a flatness and inattention that \
would have drawn sharp words from my father. I do not say this with pride — it struck me \
rather as a commentary on the schools than as any credit to myself."

Turn 3 [Challenge — The visitor is skeptical and pushes back. They challenge claims, question conclusions, express disagreement.]:
Style: Skeptical or adversarial: 'Surely that cannot be right.' 'I find that hard to believe.' 'Others would strongly disagree.'
Avoid: Do NOT write polite questions — the visitor is genuinely challenging the Victorian.
Passage segment 3 (~68w):
"At Oxford I came under the influence of several men who shaped my thinking far more than \
any formal instruction. The lectures I attended hardly mattered; it was the conversations, \
the evenings in college rooms, the arguments about things that had no bearing on examinations, \
that did the real work of education. Of the lectures I recall almost nothing. Of those \
conversations I recall a very great deal."\
""",
        '{"subject": "a gentleman\'s home education under his father and how Oxford\'s informal conversations shaped him more than lectures", "turns": ["Where did your schooling begin?", "Reading aloud at the end of every lesson — was that your father\'s own invention?", "Surely the great public schools gave those men something a home education simply couldn\'t."]}'
    ),
    # Turn 1: Assertion | Turn 2: Explanation | Turn 3: Follow-up
    (
        """\
Write exactly 3 human turns.

Turn 1 [Assertion — The visitor makes a declarative statement or claim rather than asking a question. The Victorian responds to what has been put forward.]:
Style: Bold declarations: 'I believe the it's more important to be busy than to be idle.' 'I feel like people are so busy these days.' 'Learning how to take it slow is so important, I think.'
Avoid: Human turns must be STATEMENTS, not questions.
Passage segment 1 (~71w):
"I have always maintained that the great error of our age is the worship of busyness for \
its own sake. We have persuaded ourselves that a man who is always occupied is a virtuous \
man, and a man who has learned to sit still and think is an idler. I believe the reverse \
to be more nearly true. The capacity to be usefully idle — to sit quietly with one's own \
thoughts — is a far rarer and more valuable accomplishment than the talent for perpetual motion."

Turn 2 [Explanation — The visitor shares their own experience, view, or knowledge — creating common ground the Victorian responds to.]:
Style: Sharing experience: 'I've often felt the same way myself.' 'Where I come from, we think very differently.' 'Oh yeah, people are way to busy these days.'
Avoid: Human turns must be STATEMENTS about the visitor's own experience, not questions.
Passage segment 2 (~76w):
"I have watched men of great natural gifts destroy themselves by this worship of activity. \
They cannot bear to be alone with their own minds for an hour. As soon as a space opens in \
the day they fill it — with correspondence, with visits, with the reading of newspapers that \
tell them nothing they need to know. What they are afraid of, I suspect, is not idleness at \
all, but thought — the kind of thought that involves sitting in judgment upon oneself."

Turn 3 [Follow-up — The visitor asks for clarification, more detail, or presses on a specific point from the prior beat.]:
Style: Clarifications and continuations: 'Could you say more about that?' 'What do you mean exactly?' 'I'm not sure I follow.'
Avoid: Do NOT write broad opening questions — turns should feel like reactions to something just said.
Passage segment 3 (~62w):
"I am not proposing a life of contemplation as a universal model. There are men to whom \
action is as necessary as breath, and they should act. What I am proposing is simply that \
we cease to regard the busy man as automatically the better man. Some of the most \
beneficial work ever done in this world was done by men sitting alone in rooms, thinking."\
""",
        '{"subject": "an argument that idleness and solitary thought are more valuable than the Victorian cult of perpetual busyness", "turns": ["You strike me as someone perfectly content to do nothing all day.", "I know men who preach exactly this and spend every waking hour in agitation anyway.", "What do you mean, sitting in judgment upon oneself — is that really what keeps busy men running?"]}'
    ),
]

_JSON_RE = re.compile(r'\{[\s\S]*\}', re.DOTALL)


def _parse_json(raw: str) -> dict | None:
    if not raw:
        return None
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

# ---------------------------------------------------------------------------
# Epub reader
# ---------------------------------------------------------------------------

def _strip_html(raw: str) -> str:
    raw = re.sub(r'<(p|div|h[1-6]|blockquote|li)[^>]*>', '\n\n', raw, flags=re.IGNORECASE)
    raw = re.sub(r'</(p|div|h[1-6]|blockquote|li)>', '\n\n', raw, flags=re.IGNORECASE)
    raw = re.sub(r'<br\s*/?>', '\n', raw, flags=re.IGNORECASE)
    raw = re.sub(r'<[^>]+>', ' ', raw)
    raw = re.sub(r'&[a-z#0-9]+;', ' ', raw)
    return raw


def _read_epub(path: Path) -> str:
    parts: list[str] = []
    try:
        with zipfile.ZipFile(path, "r") as z:
            for name in sorted(z.namelist()):
                if name.lower().endswith((".html", ".htm", ".xhtml")):
                    try:
                        raw = z.read(name).decode("utf-8", errors="replace")
                        parts.append(_strip_html(raw))
                    except Exception:
                        continue
    except Exception as e:
        print(f"  ERROR reading {path.name}: {e}")
    return "\n\n".join(parts)


def _split_paragraphs(text: str) -> list[str]:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    paras = re.split(r'\n{2,}', text)
    result = []
    for p in paras:
        p = re.sub(r'\s+', ' ', p).strip()
        if p:
            result.append(p)
    return result


def wc(text: str) -> int:
    return len(text.split())

# ---------------------------------------------------------------------------
# Passage splitting
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z“‘"])')


def split_into_segments(text: str) -> list[str]:
    """Split a passage into 2–4 roughly equal segments at sentence boundaries."""
    w = wc(text)
    n_target = 2 if w < 200 else 3 if w < 320 else 4

    sentences = _SENT_SPLIT_RE.split(text)
    if len(sentences) < n_target:
        # Not enough sentences — try 2 segments or keep whole
        if len(sentences) >= 2:
            n_target = 2
        else:
            return [text]

    total = sum(wc(s) for s in sentences)
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
            segments.append(' '.join(current))
            current   = []
            current_w = 0

    if current:
        if segments:
            segments[-1] += ' ' + ' '.join(current)
        else:
            segments.append(' '.join(current))

    return [s for s in segments if wc(s) >= MIN_SEGMENT_WORDS]

# ---------------------------------------------------------------------------
# Passage loading and filtering
# ---------------------------------------------------------------------------

# Must contain at least one clear first-person substantive clause
_FIRST_PERSON_RE = re.compile(
    r'\bI\s+(have|had|was|am|think|thought|believe|know|feel|felt|find|found|'
    r'confess|remember|recall|see|say|hold|consider|wish|fear|hope|'
    r'cannot|could|would|should|must|will|do|did)\b',
    re.IGNORECASE,
)

# Probable OCR garbage — five or more single-char tokens in a row
_GARBAGE_RE = re.compile(r'(?:\b\w\b[\s,]*){5,}')


def _is_usable(para: str) -> bool:
    if para.lstrip().startswith(('"', '“')):
        return False
    if para.count('"') >= 8 or para.count('“') >= 4:
        return False
    if _GARBAGE_RE.search(para):
        return False
    if not _FIRST_PERSON_RE.search(para):
        return False
    return True

# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------

def _build_user(modes: list[str], segments: list[str]) -> str:
    n = len(segments)
    lines = [f"Write exactly {n} human turn{'s' if n != 1 else ''}."]
    for i, (mode, seg) in enumerate(zip(modes, segments), 1):
        m = DIALOGUE_MODES[mode]
        lines += [
            "",
            f"Turn {i} [{m['label']} — {m['desc']}]:",
            f"Style: {m['style']}",
            f"Avoid: {m['avoid']}",
            f"Passage segment {i} (~{wc(seg)}w):",
            f'"{seg}"',
        ]
    return '\n'.join(lines)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_result(result: dict, n_expected: int) -> str | None:
    """Return error string if invalid, None if ok."""
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
# MLX backend
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

    def generate(self, modes: list[str], segments: list[str], max_tokens: int) -> dict | None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex_user, ex_asst in _FEW_SHOTS:
            messages += [
                {"role": "user",      "content": ex_user},
                {"role": "assistant", "content": ex_asst},
            ]
        messages.append({"role": "user", "content": _build_user(modes, segments)})

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        kwargs: dict = {"max_tokens": max_tokens, "verbose": False}
        if self._sampler is not None:
            kwargs["sampler"] = self._sampler
        else:
            kwargs["temp"] = self._temp
        raw = self._generate(self.model, self.tokenizer, prompt=prompt, **kwargs)
        return _parse_json(raw)

# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

class AnthropicBackend:
    def __init__(self, model_name: str):
        try:
            import anthropic as _a
        except ImportError:
            print("anthropic package not installed.  Run: pip install anthropic")
            sys.exit(1)
        self._anthropic = _a
        self._model = model_name

    def generate(self, modes: list[str], segments: list[str], max_tokens: int) -> dict | None:
        import asyncio

        messages = []
        for ex_user, ex_asst in _FEW_SHOTS:
            messages += [
                {"role": "user",      "content": ex_user},
                {"role": "assistant", "content": ex_asst},
            ]
        messages.append({"role": "user", "content": _build_user(modes, segments)})

        async def _call():
            client = self._anthropic.AsyncAnthropic()
            try:
                resp = await client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    temperature=0.75,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                )
                return _parse_json(resp.content[0].text.strip())
            except Exception as e:
                print(f"  API error: {e}", file=sys.stderr)
                return None

        return asyncio.run(_call())

# ---------------------------------------------------------------------------
# Candidate loading
# ---------------------------------------------------------------------------

def load_candidates(mode_filter: str | None, seed: int) -> list[dict]:
    rng = random.Random(seed)
    candidates: list[dict] = []
    seen: set[str] = set()

    for label, epub_path in SOURCES:
        if not epub_path.exists():
            print(f"  SKIP (not found): {epub_path.name}")
            continue

        text  = _read_epub(epub_path)
        paras = _split_paragraphs(text)
        n_matched = 0

        for para in paras:
            w = wc(para)
            if w < MIN_WORDS or w > MAX_WORDS:
                continue
            key = para.lower().strip()[:120]
            if key in seen:
                continue
            if not _is_usable(para):
                continue

            segs = split_into_segments(para)
            if len(segs) < 2:
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
            n_matched += 1
            candidates.append({
                "text":       para,
                "segments":   segs,
                "source":     label,
                "turn_modes": turn_modes,
            })

        print(f"  {label}: {len(paras):,} paragraphs, {n_matched} usable")

    return candidates


def load_done_keys() -> set[str]:
    done: set[str] = set()
    if not OUTPUT.exists():
        return done
    for line in OUTPUT.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec  = json.loads(line)
            pair = rec.get("pair", [])
            # Key on the first assistant turn (= first Victorian segment)
            key = next(
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
        description="Generate multi-turn Victorian SFT pairs from corpus passages"
    )
    ap.add_argument("--count",       type=int,   default=0,
                    help="Max passages to process (default: all remaining)")
    ap.add_argument("--mode",        type=str,   default="",
                    choices=list(DIALOGUE_MODES.keys()) + [""],
                    help="Fix dialogue mode for every passage (default: random per passage)")
    ap.add_argument("--preview",     action="store_true",
                    help="Show passages and proposed segments, no model calls")
    ap.add_argument("--backend",     choices=["mlx", "anthropic"], default="mlx")
    ap.add_argument("--model",       type=str,   default="")
    ap.add_argument("--max-tokens",  type=int,   default=400)
    ap.add_argument("--seed",        type=int,   default=42)
    args = ap.parse_args()

    mode_filter = args.mode.strip() or None

    print("Loading passages from epub sources...")
    candidates = load_candidates(mode_filter, args.seed)
    print(f"\n  Total usable passages: {len(candidates):,}")

    t1_dist     = Counter(c["turn_modes"][0] for c in candidates)
    t2plus_dist = Counter(m for c in candidates for m in c["turn_modes"][1:])
    print("  Turn 1 mode distribution (no follow_up):")
    for mode in ALL_MODES:
        print(f"    {mode:<15} {t1_dist.get(mode, 0):>5}")
    print("  Turn 2+ mode distribution:")
    for mode in ALL_MODES:
        print(f"    {mode:<15} {t2plus_dist.get(mode, 0):>5}")

    if not candidates:
        print("ERROR: no candidates found — check epub paths.")
        sys.exit(1)

    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    done_keys = load_done_keys()
    remaining = [
        c for c in candidates
        if c["segments"][0].lower()[:120] not in done_keys
    ]
    print(f"\n  Already done:  {len(candidates) - len(remaining):,}")
    print(f"  Remaining:     {len(remaining):,}")

    if args.count > 0:
        remaining = remaining[:args.count]
        print(f"  Processing:    {len(remaining):,} (capped)")

    if not remaining:
        print("Nothing to do.")
        return

    if args.preview:
        print(f"\n--- PREVIEW (first 6) ---")
        for c in remaining[:6]:
            modes_str = " | ".join(c["turn_modes"])
            print(f"\n[{c['source']} | {len(c['segments'])} segs | modes: {modes_str} | {wc(c['text'])}w]")
            for i, (seg, mode) in enumerate(zip(c["segments"], c["turn_modes"]), 1):
                print(f"  Seg {i} [{mode}] ({wc(seg)}w): {seg[:100]}...")
        return

    # Load backend
    model_name = args.model or (
        DEFAULT_ANTHROPIC_MODEL if args.backend == "anthropic" else DEFAULT_MLX_MODEL
    )
    if args.backend == "anthropic":
        backend = AnthropicBackend(model_name)
    else:
        backend = MLXBackend(model_name)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    n_written = n_failed = n_invalid = 0
    failure_reasons: dict[str, int] = {}
    t0 = time.time()

    with OUTPUT.open("a", encoding="utf-8") as f:
        for i, c in enumerate(remaining, 1):
            result = backend.generate(c["turn_modes"], c["segments"], args.max_tokens)
            err = None if result is None else validate_result(result, len(c["segments"]))

            # one retry on any failure
            if result is None or err is not None:
                result = backend.generate(c["turn_modes"], c["segments"], args.max_tokens)
                err = None if result is None else validate_result(result, len(c["segments"]))

            if result is None:
                n_failed += 1
                failure_reasons["parse_error"] = failure_reasons.get("parse_error", 0) + 1
            elif err is not None:
                n_invalid += 1
                failure_reasons[err] = failure_reasons.get(err, 0) + 1
            else:
                turns = result["turns"]
                pair = []
                for human_turn, victorian_seg in zip(turns, c["segments"]):
                    pair.append({"role": "user",      "content": human_turn.strip()})
                    pair.append({"role": "assistant", "content": victorian_seg.strip()})

                record = {
                    "source":        c["source"],
                    "response_type": "corpus",
                    "sub_type":      "multiturn_passage",
                    "turn_modes":    c["turn_modes"],
                    "subject":       result["subject"].strip(),
                    "n_turns":       len(pair),
                    "pair":          pair,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

            if i % 20 == 0 or i == len(remaining):
                elapsed = time.time() - t0
                rate    = i / elapsed if elapsed > 0 else 0
                eta     = (len(remaining) - i) / rate if rate > 0 else 0
                print(
                    f"  [{i:>5}/{len(remaining)}] "
                    f"written={n_written} failed={n_failed} invalid={n_invalid} "
                    f"({rate:.1f}/s, ETA {eta/60:.0f}m)"
                )

    reason_lines = "".join(
        f"    {reason}: {count}\n"
        for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1])
    )
    print(
        f"\nDone:\n"
        f"  written:  {n_written}\n"
        f"  failed:   {n_failed}\n"
        f"  invalid:  {n_invalid}\n"
        + (f"  failure reasons:\n{reason_lines}" if failure_reasons else "")
        + f"→ {OUTPUT}"
    )


if __name__ == "__main__":
    main()

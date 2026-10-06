#!/usr/bin/env python3
"""
rewrite_prompts_natural.py — de-template the HUMAN side of SFT pairs.

Problem: many SFT prompts are slot-filled templates — instructional_speeches is
100% five fixed frames ("Compose a speech upon X"), fact_op_sft_eb_4000 is ~42%
"what do you think of X" / "what can you tell me about X". A model trained on rigid
frames learns template-MATCHING, not intent-UNDERSTANDING: a real user asking the
same thing in natural words ("could you say a few words about the temperance
movement?") doesn't match the frame, so the learned capability doesn't fire. That
shows up as low responsiveness.

Fix: rewrite the USER turn into natural, varied phrasing — SAME intent, SAME topic —
while leaving the ASSISTANT turn completely untouched. This is constraint-safe: the
Victorian voice/knowledge lives entirely on the assistant side (genuine corpus text);
the user prompt is already synthetic, so naturalising it introduces no contemporary-
viewpoint distortion into the model's voice. It only makes the QUESTION look real.

Grounding: few-shot the rewriter with a SMALL CURATED set (~30) of illustrative examples
of how users phrase messages to the chatbot — clean, period-plausible, non-offensive,
phrasing-diverse. (The original curated set was hand-picked from private usage logs and
is not published; the GUIDES below are authored stand-ins of the same kind.) A tight
guide set beats a huge noisy pool: the guides are stable, inspectable,
and don't drag register modern. They cover both molds — IMPERATIVE commands ("write me
a poem", "Recite your favourite verse") and QUESTIONS — so instructions are represented
as strongly as questions. Edit GUIDES below to change them. (`--guides-source mined`
falls back to sampling the raw usage log if you ever want the big pool again.)

Form diversity is GUARANTEED, not hoped-for: each row is assigned a target FORM
(command / casual / question / polite) by a weighted rotation, shown form-matching
exemplars, and told to use that form. --form-mix biases the blend: 'imperative' for
creative task files (poems, speeches), 'question' for factual/opinion (eb_4000),
'balanced' otherwise.

  # validate plumbing with NO model load (safe while other jobs hold the GPU):
  python3 rewrite_prompts_natural.py --input training_data/instructional_sft/instructional_poems_sft.jsonl \
      --form-mix imperative --dry-run --limit 8

  # real run (Qwen-32B via MLX), rewrite in place -> <name>.natural.jsonl:
  python3 rewrite_prompts_natural.py --input training_data/instructional_sft/instructional_speeches_sft.jsonl \
      --form-mix imperative
"""
from __future__ import annotations
import json, re, argparse, random
from pathlib import Path

DEFAULT_MLX = "mlx-community/Qwen2.5-32B-Instruct-4bit"  # same model as the behavioral judge
USAGE_DIR = Path("tests_and_assessments/usage data and anaylsis/use_data_analysis")
RAW_USAGE = Path("tests_and_assessments/usage data and anaylsis/usage_data_deduped.jsonl")

# ~30 curated guides: illustrative examples of how users phrase messages (authored
# stand-ins; the original hand-picked set came from private usage logs and is not published).
# These ARE the few-shot anchors by default. command[] = imperative/casual mold;
# question[] = interrogative/polite mold.
# Edit freely — they steer the STYLE of every rewrite, so keep them clean and on-register.
GUIDES = {
    "command": [
        "write me a sonnet",
        "Sing me a song from your youth.",
        "Tell me a tale about a fox and a goose.",
        "Tell me a ghost story.",
        "Tell me about Gladstone.",
        "Tell me about your walk to church this morning.",
        "Tell me more of Mr Dickens!",
        "Tell me about how servants were hired.",
        "Tell me about the market in your town.",
        "Tell me more about the house.",
        "Tell me what you think about ambition.",
        "Describe the state of the Empire.",
        "Explain how I might work this telegraph, good sir.",
        "Imagine what London will look like in two hundred years.",
        "Tell me about the latest news you have heard.",
    ],
    "question": [
        "So what do you make of Garibaldi then?",
        "Who are your favourite novelists?",
        "What is the general opinion of Mr. Ruskin's writings?",
        "What do you think of the working conditions in the mills?",
        "How would you describe Scotland?",
        "How are the omnibuses in town these days?",
        "Which inventions please you most?",
        "Is it a large town?",
        "What should I name my new puppy?",
        "Was he a painter or more of a poet?",
        "Do you speak German?",
        "I was just curious whether you had any thoughts on Virginia Woolf?",
        "How does India figure in such a future?",
        "Do you know any card games?",
        "What is your opinion on that?",
    ],
}

# FACTUAL register guides — for reference files (eb_4000) whose answers are encyclopaedic
# fact. Every prompt is a straightforward information request about the SUBJECT; NO opinion/
# "what do you think"/"take" framings, which mismatch a factual answer. (Authored stand-ins.)
GUIDES_FACTUAL = {
    "command": [
        "Tell me about Wellington.",
        "Tell me about Herculaneum.",
        "Describe the state of trade in Manchester.",
        "Give me an account of the Crimean War.",
        "Tell me about the cathedral at Lincoln.",
        "Tell me what you know of Isambard Kingdom Brunel.",
        "Describe the town of Heidelberg.",
        "Tell me about Jeremy Bentham.",
        "Explain how the penny post came to be.",
        "Tell me about the birds of paradise.",
    ],
    "question": [
        "What is Tobermory?",
        "Who was Michael Faraday?",
        "What can you tell me about Ravenna?",
        "Who is Wilkie Collins?",
        "Where is Göttingen?",
        "What is the Alhambra?",
        "Do you know of Elizabeth Fry?",
        "Have you heard of Toussaint Louverture?",
        "How would you describe Iceland?",
        "What can you say of the Corn Laws?",
    ],
}

# question-shaped intent clusters (labelled usage data) — used only for --guides-source mined.
INTENT_FILES = {
    "factual_query":        ["cluster_intent_factual_query_assignments.jsonl"],
    "opinion_solicitation": ["cluster_intent_opinion_solicitation_assignments.jsonl"],
    "task_request":         ["cluster_intent_opinion_solicitation_assignments.jsonl",
                             "cluster_intent_factual_query_assignments.jsonl"],
    "mixed":                ["cluster_intent_factual_query_assignments.jsonl",
                             "cluster_intent_opinion_solicitation_assignments.jsonl"],
}

# imperative-mood opener (a real user giving a command)
IMP_RE = re.compile(r"^(write|tell|compose|give|make|describe|explain|list|name|recite|sing|read|"
                    r"show|teach|translate|define|summari[sz]e|pen|craft|invent|suggest|recommend|"
                    r"imagine|pretend|speak|talk|discuss|elaborate|paint|sketch|draw|create|say)\b", re.I)
# question-shaped (ends with ? or opens with a wh-/aux- word)
Q_RE = re.compile(r"\?\s*$|^(what|why|how|who|whom|whose|which|where|when|can|could|would|will|do|does|"
                  r"did|is|are|have|has|should|may|might)\b", re.I)

# per-row target forms and how to phrase each. The exemplar POOL (command vs question)
# supplies the grammatical mold; the instruction supplies the register nuance.
FORM_INSTRUCTION = {
    "command":  'Phrase it as a DIRECT INSTRUCTION in the imperative mood — a plain command like '
                '"Write me a poem about ...", "Tell me about ...", "Compose ...". No question mark, no "could you".',
    "casual":   'Phrase it CASUALLY and off-the-cuff, the way someone types quickly — relaxed and clipped, '
                'e.g. "tell me about ...", "let\'s hear a poem on ...".',
    "question": 'Phrase it as a QUESTION, e.g. "Could you write ...?", "What can you tell me about ...?".',
    "polite":   'Phrase it as a POLITE REQUEST, e.g. "I should be glad if you would ...", "Would you kindly ...".',
}
# FACTUAL register: every form asks for INFORMATION about the subject, never an opinion.
FORM_INSTRUCTION_FACTUAL = {
    "command":  'Phrase it as a direct request for information in the imperative — "Tell me about ...", '
                '"Describe ...", "Give me an account of ...", "Give me information about ...". No question mark.',
    "casual":   'Phrase it casually — "tell me about ...", "what\'s the story with ...", "so who was ...".',
    "question": 'Phrase it as a plain FACTUAL question. VARY the opener — pick naturally among '
                '"What is ...?", "What\'s ...?", "Who was ...?", "Where is ...?", "What do you know about ...?", '
                '"What can you tell me about ...?", "Have you heard of ...?", "Do you know of ...?". '
                'Do NOT default to "tell me about".',
    "polite":   'Phrase it as a polite request for INFORMATION. VARY the opener — e.g. "I should be glad to learn of ...", '
                '"Might you describe ...?", "I have been wondering about ...", "Could you tell me something about ...?", '
                '"Describe what you understand ... to be". Avoid defaulting to "could you tell me about". '
                'Ask about the subject, not for opinions.',
}
FORM_POOL = {"command": "command", "casual": "command", "question": "question", "polite": "question"}

# the "what to preserve" rule, per register.
INTENT_RULE = {
    "general": "Keep the EXACT same intent and the EXACT same subject/topic/entity as the original.",
    "factual": "Keep the EXACT same SUBJECT/topic/entity. The intent is always a STRAIGHTFORWARD FACTUAL "
               "QUERY asking for information about that subject. If the original asks for an opinion, a "
               "\"take\", or \"what do you think\", CONVERT it into a plain factual request — never ask "
               "for the bot's opinion, views, or thoughts.",
}

# --form-mix -> weighted form rotation (need not sum to 1)
FORM_MIX = {
    "balanced":   [("command", 0.40), ("casual", 0.25), ("question", 0.20), ("polite", 0.15)],
    "imperative": [("command", 0.55), ("casual", 0.30), ("question", 0.10), ("polite", 0.05)],
    "question":   [("command", 0.15), ("casual", 0.15), ("question", 0.45), ("polite", 0.25)],
}

def pick_form(weights, rng):
    r = rng.random() * sum(w for _, w in weights)
    c = 0.0
    for f, w in weights:
        c += w
        if r <= c:
            return f
    return weights[-1][0]

def _len_ok(t, lo, hi):
    return lo <= len(t.split()) <= hi and "\n" not in t

def load_cluster_prompts(intent, lo=3, hi=28):
    """Question-shaped real prompts from the labelled intent clusters."""
    seen, out = set(), []
    for fn in INTENT_FILES.get(intent, INTENT_FILES["mixed"]):
        p = USAGE_DIR / fn
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            try:
                t = json.loads(line).get("text", "").strip()
            except json.JSONDecodeError:
                continue
            if _len_ok(t, lo, hi) and t.lower() not in seen:
                seen.add(t.lower()); out.append(t)
    return out

def mine_usage(mood, lo=3, hi=24, cap=1500):
    """Real user turns from the raw usage log, filtered to an imperative or question mood."""
    if not RAW_USAGE.exists():
        return []
    seen, out = set(), []
    for line in open(RAW_USAGE, encoding="utf-8"):
        try:
            msgs = json.loads(line).get("messages", [])
        except json.JSONDecodeError:
            continue
        for m in msgs:
            if m.get("role") != "user":
                continue
            t = (m.get("content") or "").strip()
            if not _len_ok(t, lo, hi) or t.lower() in seen:
                continue
            if mood == "command" and not IMP_RE.match(t):
                continue
            if mood == "question" and not Q_RE.search(t):
                continue
            seen.add(t.lower()); out.append(t)
        if len(out) >= cap:
            break
    return out

def load_exemplar_pools(intent, source="curated", register="general"):
    """Return {'command': [...imperatives...], 'question': [...questions...]}.
    'curated' (default) -> the hand-picked GUIDES (factual variant in factual register);
    'mined' -> the big raw-usage pools."""
    if source == "curated":
        g = GUIDES_FACTUAL if register == "factual" else GUIDES
        return {"command": list(g["command"]), "question": list(g["question"])}
    commands = mine_usage("command")
    questions = load_cluster_prompts(intent) + mine_usage("question")
    q, seen = [], set()
    for t in questions:
        if t.lower() not in seen:
            seen.add(t.lower()); q.append(t)
    return {"command": commands, "question": q}

# ---- record shape handling: touch ONLY the first user turn, leave the rest as-is ----
def get_user_turn(rec):
    """Return (text, setter) for the first user-role content, or (None, None)."""
    if isinstance(rec, list):
        for t in rec:
            if isinstance(t, dict) and t.get("role") == "user" and (t.get("content") or "").strip():
                return t["content"], (lambda v, _t=t: _t.__setitem__("content", v))
        return None, None
    if isinstance(rec, dict):
        if isinstance(rec.get("human"), str) and rec["human"].strip():
            return rec["human"], (lambda v: rec.__setitem__("human", v))
        for key in ("pair", "turns"):
            seq = rec.get(key)
            if isinstance(seq, list):
                for t in seq:
                    if isinstance(t, dict) and t.get("role") == "user" and (t.get("content") or "").strip():
                        return t["content"], (lambda v, _t=t: _t.__setitem__("content", v))
    return None, None

PREAMBLE = """You rewrite a request so it sounds like a real person actually engaging with a chatbot, \
instead of a rigid template. You are NOT answering it and NOT changing what it asks about.

Here are illustrative examples of how users typically phrase their messages (style reference only — do NOT borrow their topics):
{exemplars}

Rules:
- {intent_rule}
- {form_instruction}
- Make it sound natural and human — real people are often terse and direct, not verbose or ornate.
- Do not add new facts, names, or constraints. Do not answer the request.
- One message only. Output ONLY the rewritten request, nothing else."""

def build_prompt(original, pools, form, k, rng, register="general"):
    pool = pools[FORM_POOL[form]] or pools["question"] or pools["command"]
    ex = "\n".join(f"  - {e}" for e in rng.sample(pool, min(k, len(pool))))
    fi = (FORM_INSTRUCTION_FACTUAL if register == "factual" else FORM_INSTRUCTION)[form]
    return (PREAMBLE.format(exemplars=ex, intent_rule=INTENT_RULE[register], form_instruction=fi) +
            f'\n\nOriginal (templated) request:\n  "{original}"\n\nRewrite it ({form} form):')

def make_generator(model):
    from mlx_lm import load, generate
    m, tk = load(model)
    def run(prompt):
        out = generate(m, tk, prompt=tk.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False),
            max_tokens=80, verbose=False)
        return out.strip()
    return run

_CLEAN = re.compile(r'^["\'\s]+|["\'\s]+$')

def clean_output(raw):
    t = raw.strip().splitlines()[0] if raw.strip() else ""
    t = _CLEAN.sub("", t)
    t = re.sub(r"^(rewritten request:?\s*)", "", t, flags=re.I).strip().strip('"\'')
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--intent", default="mixed", choices=list(INTENT_FILES),
                    help="which labelled clusters supply the QUESTION-form exemplars")
    ap.add_argument("--form-mix", default="balanced", choices=list(FORM_MIX),
                    help="form blend: 'imperative' for creative task files (poems/speeches), "
                         "'question' for factual/opinion (eb_4000), 'balanced' otherwise")
    ap.add_argument("--guides-source", default="curated", choices=["curated", "mined"],
                    help="'curated' (default) = the ~30 hand-picked GUIDES; 'mined' = big raw-usage pools")
    ap.add_argument("--register", default="general", choices=["general", "factual"],
                    help="'factual' (for reference files like eb_4000): forces straightforward factual "
                         "queries, converts opinion-framings, uses factual guides")
    ap.add_argument("--exemplars-k", type=int, default=6)
    ap.add_argument("--model", default=DEFAULT_MLX)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble + print rewrite prompts for --limit rows; NO model load (no GPU/OOM)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    weights = FORM_MIX[args.form_mix]
    pools = load_exemplar_pools(args.intent, args.guides_source, args.register)
    print(f"guides '{args.guides_source}' register '{args.register}' — command: {len(pools['command'])}, "
          f"question: {len(pools['question'])} | form-mix '{args.form_mix}' | k={args.exemplars_k}")
    if not (pools["command"] or pools["question"]):
        raise SystemExit("no exemplars found — check USAGE_DIR / RAW_USAGE paths")

    rows = [json.loads(l) for l in open(args.input)]
    if args.limit:
        rows = rows[:args.limit]
    out_path = args.output or args.input.with_suffix(".natural.jsonl")

    if args.dry_run:
        import collections
        n, forms = 0, collections.Counter()
        for rec in rows:
            orig, setter = get_user_turn(rec)
            if not orig:
                continue
            n += 1
            form = pick_form(weights, rng)
            forms[form] += 1
            print("\n" + "=" * 78)
            print(f"ORIGINAL ({form}):", orig[:150])
            print("-" * 78)
            print(build_prompt(orig, pools, form, args.exemplars_k, rng, args.register)[-680:])
            if n >= (args.limit or 8):
                break
        print(f"\n[dry-run] assembled {n} rewrite prompts, no model loaded. form spread: {dict(forms)}")
        return

    run = make_generator(args.model)
    rewritten = kept = 0
    with open(out_path, "w") as f:
        for i, rec in enumerate(rows):
            orig, setter = get_user_turn(rec)
            if orig and setter:
                form = pick_form(weights, rng)
                new = clean_output(run(build_prompt(orig, pools, form, args.exemplars_k, rng, args.register)))
                if new and 2 <= len(new.split()) <= 60 and new.lower() != orig.lower():
                    if isinstance(rec, dict):
                        rec["human_original"] = orig
                        rec["prompt_style"] = "natural_rewrite"
                        rec["prompt_form"] = form
                    setter(new)
                    rewritten += 1
                else:
                    kept += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(rows)} | rewritten {rewritten} kept {kept}")
    print(f"wrote {out_path} | rewritten {rewritten}, kept-original {kept}")

if __name__ == "__main__":
    main()

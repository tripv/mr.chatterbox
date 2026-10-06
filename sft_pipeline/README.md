# SFT data construction

This folder holds reference implementations of the three methods that built the dialogue fine-tuning (SFT) data for the Mr. Chatterbox models.

**The shared invariant: the assistant turn is always real 19th-century text, never written by an LLM or by us.** The methods differ only in where the *user* turn comes from.
Every method below is a way of finding a question for an answer that already exists. That invariant is what lets the project claim the model's replies carry no synthetic contemporary viewpoint.

In the literature this family is **back-translation**: derive the input from the output.
Two of the three methods map onto the project's original names, "Behavioral Back Translation" and "Factual Back Translation."

> **These are reference implementations, not a turnkey pipeline.** They expect the project's pretraining corpus (parquet shards), raw HathiTrust/Gutenberg/Internet Archive dumps, and intermediate files at the paths hard-coded near the top of each script. Several stages also call an LLM: a local MLX model (Qwen2.5-32B-Instruct, 4-bit), or the Anthropic or OpenAI APIs with keys read from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`. Generally, we found using API calls was more successful than using local MLX models. 
> The code is published to document *how* the data was built. Expect to adapt paths and inputs before running anything.

> **About the user turns shown in the code.** Several prompts and rubrics originally contained real user messages from the deployed chatbot's private usage logs, used as style examples or few-shot cases. Those have been **replaced with authored stand-ins** of the same kind for this release.
> Because of that, the prompts here are not byte-identical to the ones that produced the released data. The method and logic are unchanged.

---

## 1. The organising distinction: function vs content

Every method choice follows from this.

| | the reply is selected for… | example |
|---|---|---|
| **function** | what the reply DOES | "concede an error", "admit ignorance", "ask for English", "refuse a command" |
| **content** | what the reply is ABOUT | "who Goethe was", "what year it is", "what one thinks of the Irish" |

**Retrieval matches function. It cannot match content.** This held every time it was measured:

- **Non-English input:** "I don't speak German" is pure function, and retrieval worked (131 rows, 71 distinct).
- **De-escalation concessions:** "concede an error" is a function, and retrieval worked (59 rows, 85% distinct).
- **De-escalation answers:** "say something true about Goethe" is content, and retrieval **failed**. Asked for countries, it returned "The prime minister." The mechanism worked exactly as designed, on the wrong kind of task.

**Practical rule:** before choosing a method, ask whether the target reply is defined by what it *does* or what it *says*. If it's content, don't reach for retrieval.

---

## 2. The three methods

### 2a. Retrieve-and-rerank ("Behavioral Back Translation")

**Direction:** prompt first. 

Take a user prompt, then retrieve a real corpus reply that *functions* correctly for it.

**Substrate:** 3.39M adjacency pairs (one utterance and the real reply that follows it) from the [Gutenberg Dialog corpus](https://github.com/ricsinaruto/gutenberg-dialog) (Csaky & Recski), English split from books published between 1800 and 1899.

**Built:** most of the behavioural data, including:

- adversarial prompts, jailbreak attempts and roleplay;
- questions about the bot itself (meta and persona);
- gibberish, farewells, short topic shifts and non-English input.

| stage | script | does |
|---|---|---|
| 1 | `01_extract_pairs.py --src <dialogs_clean.txt>` | context → reply adjacency pairs → `pairs.jsonl` |
| 2 | `02_build_index.py` | embeds contexts (all-MiniLM-L6-v2) → `index_ctx.npy`, `index_meta.jsonl` |
| 2b | `02b_build_resp_index.py` | embeds replies → `index_resp.npy` |
| 3 | `03_retrieve_and_judge.py --prompts <file>` | two-channel retrieval, LLM judge picks from a ranked shortlist, a global usage cap keeps replies diverse |

All scripts are in `training_data/behavioral_sft/adversarial_rebuild/dialog_retrieval/`.

The prompt file is JSONL with lines like `{"sub_type": "gibberish", "human": "..."}`.
`detok.py` truecases accepted replies, and `reclassify.py` re-routes prompts filed under the wrong sub_type.

**Two channels:**

- **Topic channel:** finds corpus contexts that resemble the prompt.
- **Function channel:** finds corpus replies close to a blend of *seed phrases* (short texts that perform the target move) and the prompt.
- `FUNCTION_ONLY` turns off the topic channel for sub_types where subject matter is irrelevant, or actively harmful to retrieve on.

**Seed phrases are the main lever:**

- **Where they live:** built-in defaults are in `FUNCTION_SEEDS`. The `seedphrases_<subtype>_<posture>.json` files override them per pass via `--seed-phrases`.
- **Postures:** run one pass per rhetorical posture.
- **Chaining passes:** use `--seed-usage <previous outputs>`, because the usage cap only applies within a single run.

**Failure modes:**

1. **Role reversal.** The index stores A→B pairs, and B's role depends on who A was. 24% of non-English results did this, e.g. "Hola" → "You speak Spanish?". Fixed with per-sub_type `ROLE_NOTES` in the judge.
2. **Correction inconsistency.** A reply can perform the right move and still contradict the turn it answers. 35% of corrections to a wrong form of address (a female user being called "sir", say) apologised while repeating the mistake.
3. **Seed collapse.** Narrow seeds hit one corpus neighbourhood regardless of the prompt. Raising the usage cap is the wrong fix. The right fix is to spread seeds across genuinely different rhetorical postures.
4. **Cross-run reuse.** The usage cap is enforced within a run only: 12 chained passes produced up to 6 reuses of one reply under a cap of 2. Use `--seed-usage` to chain.

**Seed-writing rules:**

- **Enact the move, don't describe it.** "Bonjour, monsieur. And now in English" retrieved scraps 9/9 times; "that is the whole extent of my French" retrieved 0/12.
- **Don't reuse a neighbouring sub_type's seeds.** If two sub_types share seeds, the pipeline has effectively been told they mean the same thing.
- **Grammatical mood predicts role.** Imperative seeds ("Put it in English for me") came back role-reversed 1/15 times. Question seeds ("Do you speak English?") came back role-reversed 5/8 times, because questions about language are what a *foreigner* asks.
- **Don't seed a two-clause move.** Seeding "apologise AND invite" returned the apology alone 9/10 times; seeding the invitation alone returned invitations 9/9 times.

### 2b. Round-trip consistency

**Direction:** passage first. Take a clean corpus passage, *generate* a question for it, then check that the question and passage hold together as a pair.

**Built:** much of the fact and opinion data, and the multi-turn conversation sets.

Single-turn stages are in `training_data/fact_opinion_sft/short_passage_building/`:

| stage | script | does |
|---|---|---|
| 1 | `mine_short_passages.py` | mines short, standalone, sentence-bounded passages from the corpus in two length bands, with an exact token-ceiling filter |
| 2 | `judge_short_passages.py` | passage gate: coherence and voice, plus regex backstops for running headers and narrator asides |
| 3 | `gen_prompts_for_passages.py --source keep_all --coherence-file <stage-2 output>` | writes a natural question for each kept passage, using terse and multi-clause templates mixed to match real usage proportions |
| 4 | `judge_short_passage_pairs.py --input <stage-3 output>` | the round trip: an independent judge re-checks each (question, passage) pair |

Generating the question guarantees topical fit by construction, and the round trip catches generation drift.
`parsers/passage_utils.py` supplies the paragraph extraction and clean-boundary checks.

**Multi-turn form** (`gen_multiturn_source5.py`, which builds on `gen_multiturn_source4.py` and `gen_teacher_source4.py`):

1. Take a passage of 150–400 words.
2. Run the passage gate.
3. Split it into 2–4 'beats.'
4. Generate one user turn per 'beat.'
5. Judge the whole conversation.

`source5` fixes **forward leakage** in `source4`. `source4` handed every segment to one LLM call, so turn N's question previewed segment N's text.
`source5` generates the turns blind and in sequence, reveals each segment only afterwards, and checks the fit (`check_fit`). If a segment doesn't fit, it truncates the conversation rather than forcing a non-sequitur.

**Failure modes:**

- **Generation drift:** what the round-trip check exists to catch.
- **Forward leakage:** fixed in `source5`.
- **Runaway final turn:** an off-by-one in `split_into_segments` merged all trailing sentences into the final segment. Fixed.

### 2c. Scan-and-fit ("Factual Back Translation")

**Direction:** answer first, like the round trip, but here the question is **found or paired rather than generated from the text**.
Scan the whole corpus for passages that contain or perform the target, gate them, then attach a question. The question is either written for the passage or paired from user prompts already in hand.

**Built:** the attested-figures data. These are questions naming a specific pre-1900 person, work or idea, answered by a real passage about them.

| stage | script | does |
|---|---|---|
| 0 | (entity list) | JSONL of target entities (name, aliases, kind); `--entities` |
| 1 | `search_attested_figures_corpus.py` | scans the parquet corpus for entity mentions → fixed-width windows around each hit |
| 2 | `rewiden_attested_hits.py` | **reopens the source document and snaps each window outward to sentence boundaries** |
| 3 | `rank_attested_figures_hits.py` | ranks hits and caps them per entity (`--boost-list` / `--boost-top-per-entity` for under-covered entities) |
| 4 | `sonnet_attested_figure_pipeline.py` (`gate_passage`) or `local_gate_attested_hits.py` (local MLX) | passage gate: is this genuinely *about* the entity? |
| 5 | `sonnet_attested_figure_pipeline.py` (`write_question`) | writes a specific question the passage answers |
| 6 | `glue_attested_multiturn.py` | orders and joins one entity's passages into multi-turn conversations (pronoun-substitution step), with `test_glue_multiturn.py` as its calibration |

**Why it can't produce role reversal:** you inspect the passage and confirm the speaker's role *before* a prompt exists, so the reply never inherits a speaker from a context it was retrieved against.

**Why it reaches further than retrieval:** it runs over the whole 118GB pretraining corpus rather than a 647MB dialogue extract.
On non-English replies it raised candidates from 10 to 168 for one target move and from 20 to 158 for another. The share of useless hits fell from 77% to 21%.

**Stage 2 is the critical one. Skipping it is catastrophic, not merely lossy.**

- Stage 1 produces fixed-width ±400-character windows that cut mid-sentence. Without stage 2, the local gate's pass rate fell to 0%, a construction defect rather than a content signal.
- When stage 2 was omitted on another run, 33 of 149 passages came out truncated.
- Snapping to sentences *within* the window isn't enough, because a sentence straddling the window edge stays cut.

---

## 3. Parsing follows from direction

| | unit of extraction | boundaries | what it needs |
|---|---|---|---|
| round-trip | **paragraph** (`parsers/passage_utils.py`) | clean by construction: anything starting or ending mid-sentence is discarded, running headers are stripped, cross-references rejected | nothing |
| scan-and-fit | **±N characters around a match** | broken by construction | a re-widening pass that reopens the source document |
| retrieve-and-rerank | a single dialogue utterance | clean (one turn) | nothing |

Use `passage_utils` whenever the unit can be a paragraph. Fall back to fixed-width windows only when the target is a phrase you must search for.

---

## 4. Choosing a method

1. Is the target reply defined by **function** or **content**?
   - Function → **retrieve-and-rerank**.
   - Content → go to step 2.
2. Do real user prompts for this content already exist?
   - Yes → **scan-and-fit**, pairing the existing prompts rather than writing new ones. This is the cheapest route, with no generated text to verify.
   - No → **round-trip consistency**: generate the question, then verify it.
3. Is the target a phrase you must search for, or any good prose on a subject?
   - A phrase → scan-and-fit, and **don't skip stage 2**.
   - Any good prose → round trip via `passage_utils`.

**No single method covers a slice.** The de-escalation data alone needed six sources:

- function retrieval (concessions, clarifications, disclaiming expertise, admitting ignorance);
- reuse of judge-selected egalitarian material;
- a bare-facts lookup (dates, office-holders);
- scan-and-fit (literary subjects);
- one flagged model-generated row;
- hand curation.

Expect to mix methods.

---

## 5. Shared modules

| module | role |
|---|---|
| `judge_scripts/judge_behavioral_v3.py` | `make_judge()`: one LLM-judge interface over MLX, HF, vLLM, Anthropic and OpenAI backends (also the behavioural eval judge) |
| `rewrite_prompts_natural.py` | style exemplars and guides for writing natural-sounding user turns (stand-ins; see the note above) |
| `training_data/fact_opinion_sft/scripts/sonnet_fact_op_pipeline.py` | shared passage-gate prompt (`GATE_TMPL`), JSON-returning LLM call (`call_json`), opener variation and banned-phrase helpers, used by the attested-figures pipeline |
| `training_data/multi_turn_sft/scripts/sonnet_epistolary_pipeline.py` | `load_real_word_counts` / `sample_target_words`: samples question lengths from the real usage distribution. This needs the private usage log (its `USAGE` path), so supply your own word-count source to run the question-writing stages |
| `parsers/passage_utils.py` | paragraph extraction and clean-boundary checks |
| `pg_nonfiction.py`, `ia_text_parser.py`, `normalize_sft_provenance.py`, `letter_header_utils.py`, `gen_multiturn_passages.py` | source-text loaders and cleaners used by the multi-turn generators |

Run scripts from this folder (`sft_pipeline/`). Its layout mirrors the original project, and the scripts find each other through it.

## Requirements

`numpy`, `pyarrow`, `pandas`, `sentence-transformers`, `tokenizers`, plus a judge or generator backend:

- `mlx-lm` (Apple silicon), or
- `anthropic` / `openai` with API keys, or
- `transformers` / `vllm` on a CUDA GPU.

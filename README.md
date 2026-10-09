# Mr. Chatterbox

Mr. Chatterbox is a family of small language models trained from scratch on 19th-century British printed material (1800–1899).
The aim is authenticity over capability: models that write in genuinely Victorian conceptual frameworks, not a modern model doing a Victorian impression.

This repository contains:

- the pretraining code;
- the tokenizer and the code that trained it;
- the hyperparameters for the four base models;
- reference code for the three methods that built the dialogue fine-tuning data ([sft_pipeline/](sft_pipeline/README.md)).

| model | parameters | training tokens | weights |
|---|---|---|---|
| 345M | 350M | 7.0B | [tventurella/Mr.Chatterbox-345M](https://huggingface.co/tventurella/Mr.Chatterbox-345M) |
| 500M | 490M | 9.8B | [tventurella/Mr.Chatterbox-500M](https://huggingface.co/tventurella/Mr.Chatterbox-500M) |
| 1.2B | 1.18B | 23.6B | [tventurella/Mr.Chatterbox-1.2B](https://huggingface.co/tventurella/Mr.Chatterbox-1.2B) |
| 2.7B | 2.71B | 54.1B | [tventurella/Mr.Chatterbox-2.7B](https://huggingface.co/tventurella/Mr.Chatterbox-2.7B) |

In the code, the sizes are called `victoriana-<size>` (e.g. `--size 1.2B`).

Full architecture, optimizer and schedule details are in [HYPERPARAMETERS.md](HYPERPARAMETERS.md).

> **A note on content.** These models learned from 19th-century text and reproduce its attitudes, including its views on race, empire, class, gender and medicine.
> Treat their output as a historical artifact, not reliable information or advice.

## Repository layout

```
victoriana_training/       pretraining
  train_victoriana.py        trainer (LlamaForCausalLM + Muon/AdamW, single- or multi-GPU)
  muon.py                    Muon + AdamW optimizer (single-GPU and distributed ZeRO-2)
  victoriana_data.py         tag-aware packed dataloader, held-out split, source round-robin
  llama_probe_configs.py     model configs for every size
  tag_schema.py              reserved [year][source][ocr] tokens, band filter, held-out hash
  smoke_pipeline.py          CPU check of the data → model interface
  fit_scaling.py             scaling-law fit from the probe logs
  setup_b200.sh              environment bootstrap (Blackwell / B200)
  run_probes.sh              trains 345M → 500M → 1.2B and fits the scaling law
tokenizer_training/
  tokenizer_sample.py        builds the balanced per-source training/eval text sample
  tokenizer_train_v2.py      trains the byte-level BPE with the reserved tokens
  tokenizer_fertility.py     measures chars/token on the held-out sample
sft_pipeline/              reference code for building the dialogue fine-tuning data (see its README)
tokenizer/
  victoriana-v2-65536/       the tokenizer the models use (tokenizer.json + CARD.md)
  source_chars.json          in-band character count per source (cache for tokenizer_fertility.py)
```

## Setup

```bash
pip install -r requirements.txt
```

Every script expects the **repository root** as its working directory, with the corpus here:

```
training_data/pretraining/<source_dir>/*.parquet
```

## Corpus format

Each parquet shard needs four columns: `text`, `year` (int), `source` and `htid` (a stable per-book ID).

- **`source`** must be one of `tag_schema.SOURCES`. It selects the `[src_*]` token.
- **`htid`** drives the held-out split.
- **Out-of-band documents** (year outside 1800–1899) are skipped at read time.
- **Shard order:** the dataloader reads source directories round-robin, in sorted order by directory name. It freezes the resulting shard list in `training_data/pretraining/_shard_manifest.json` the first time it runs.

The corpus itself is not distributed with this repository.

## Data sources and attribution

The pretraining corpus draws on text from these sources, restricted to 1800–1899:

- HathiTrust
- Institutional Books 1.0
- British Library (blbooks)
- Project Gutenberg
- Internet Archive (National Library of Scotland, UK Medical Heritage Library, Wellcome Collection, European libraries)
- Encyclopaedia Britannica
- NCSE
- Old Bailey Proceedings
- Hansard
- House of Commons "Blue Books"

*Institutional Books* provided by the Institutional Data Initiative with source material contributed by Harvard Library, available at [https://institutional.org/institutional-books](https://institutional.org/institutional-books).

## Tokenizer

The models use `tokenizer/victoriana-v2-65536/tokenizer.json`:

- byte-level BPE with a 65,536-token vocabulary and no Unicode normalisation, so period spelling and typography survive;
- 124 reserved tokens: 100 year tokens `[y1800]`–`[y1899]`, 16 source tokens `[src_*]`, 3 OCR-quality tokens `[ocr_hi|mid|lo]`, and 5 chat/structure tokens.

See its `CARD.md` for fertility numbers. To rebuild it from the corpus, run from the repo root:

```bash
export PYTHONPATH=victoriana_training          # tokenizer scripts import tag_schema
python tokenizer_training/tokenizer_sample.py  # → tokenizer/sample/{train,eval}.jsonl (~5.4B chars)
python tokenizer_training/tokenizer_train_v2.py --vocab 65536
python tokenizer_training/tokenizer_fertility.py --tokenizers tokenizer/victoriana-v2-65536
```

## Pretraining

```bash
bash victoriana_training/setup_b200.sh                 # deps + optional flash-attn build
python victoriana_training/smoke_pipeline.py           # optional CPU check of data → model interface
ATTN=sdpa NPROC=8 bash victoriana_training/run_probes.sh        # 345M, 500M, 1.2B + scaling fit
```

The 2.7B model is not part of the probe ladder. Launch it directly:

```bash
cd victoriana_training
export VICTORIANA_DATA_ROOT=../training_data/pretraining
export VICTORIANA_TOKENIZER=../tokenizer/victoriana-v2-65536/tokenizer.json
export NCCL_NVLS_ENABLE=0
torchrun --standalone --nproc_per_node=8 train_victoriana.py \
    --size 2.7B --attn sdpa --device-batch-size 8 --out ../checkpoints/victoriana-2.7B
```

Useful flags:

- `--tokens` overrides the 20 × params budget.
- `--resume <out>-resume` continues from the rolling checkpoint written every `--save-every` steps. Resuming skips shards that were already consumed.

Practical notes from training on 8×B200:

- **NCCL:** set `NCCL_NVLS_ENABLE=0`. With NVLS enabled, NCCL hung on the first collective on Blackwell, even with only two GPUs.
- **flash-attn:** it had no Blackwell (sm_100) build and the source build can hang. `--attn sdpa` with the built-in per-document mask is what the released models used.
- **Dataloader workers:** keep the DataLoader at `num_workers=0` (already hard-coded). The iterable dataset is not partitioned per worker, so more workers would feed duplicate batches.

## Using a checkpoint

The checkpoints are standard Hugging Face `LlamaForCausalLM` repositories. Each one includes the tokenizer.
You can condition generation on year, source and OCR quality by putting the tags at the start of the prompt. Any subset of the tags works, including none.

```python
import torch
from tokenizers import Tokenizer
from transformers import LlamaForCausalLM

repo = "tventurella/Mr.Chatterbox-1.2B"
tok = Tokenizer.from_pretrained(repo)
model = LlamaForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16)

prompt = "[y1851][src_hathi][ocr_hi]The Great Exhibition"
ids = torch.tensor([tok.encode(prompt).ids])
out = model.generate(ids, max_new_tokens=200, do_sample=True, temperature=0.8, top_p=0.95)
print(tok.decode(out[0].tolist()))
```

These are base (pretrained-only) models. They continue text; they are not tuned for conversation.
A conversational version of the 2.7B model is at [tventurella/Mr.Chatterbox-2.7B-Chat](https://huggingface.co/tventurella/Mr.Chatterbox-2.7B-Chat).

## Evaluation: Chronologic-EN

[Chronologic-EN 1.0](https://github.com/Historical-AI-Lab/Chronologic-EN) (Historical AI Lab, UIUC) tests whether a model can answer as a writer of a given date would.
It has 866 questions drawn from texts published between 1831 and 1930.
We scored the released 2.7B base model and the 2.7B chat model with the benchmark's own likelihood scorer. Only the model loader was adapted, for our tokenizer and chat format.
The table puts them alongside the published leaderboard ([leaderboard_1.0.md](https://github.com/Historical-AI-Lab/Chronologic-EN/blob/main/results/likelihood/leaderboard_1.0.md)).
Brier is calibrated, as on the leaderboard (median of 1,000 bootstrap draws, with the 95% interval); lower is better.

| model | overall Brier [95% CI] | cloze acc. | constrained generation acc. | knowledge & inference acc. |
|---|---|---|---|---|
| Talkie 1930 base | 0.1574 [0.1551, 0.1596] | 39.5% | 24.9% | 44.0% |
| **Mr. Chatterbox 2.7B (base)** | **0.1588 [0.1567, 0.1610]** | 34.4% | 27.1% | 44.4% |
| Talkie 1930 it | 0.1593 [0.1569, 0.1616] | 37.4% | 23.8% | 25.1% |
| Qwen 7B ft (ckpt-3500) | 0.1601 [0.1579, 0.1621] | 36.2% | 14.9% | 34.4% |
| **Mr. Chatterbox 2.7B-Chat** | **0.1603 [0.1581, 0.1626]** | 35.3% | 27.4% | 23.8% |
| Gemma 4 31B base | 0.1606 [0.1585, 0.1626] | 23.0% | 13.3% | 40.6% |
| Qwen 72B | 0.1607 [0.1586, 0.1627] | 10.8% | 8.8% | 38.0% |
| Qwen 3.5 35B-A3B base | 0.1623 [0.1603, 0.1643] | 12.6% | 8.0% | 37.1% |
| Talkie web base | 0.1627 [0.1606, 0.1646] | 18.2% | 13.6% | 34.1% |
| Qwen 7B | 0.1628 [0.1608, 0.1648] | 8.6% | 9.5% | 37.0% |
| GPT-2 (124M, our run) | 0.1635 [0.1615, 0.1656] | 16.0% | 16.8% | 34.2% |

Rows in bold are ours. The other rows are the benchmark authors' published results.

- **The 2.7B base model is second on the leaderboard,** behind Talkie 1930 base (13B). Their 95% intervals overlap.
- **Our corpus ends in 1899, but 38% of the questions (329) come from texts published 1900–1930.** Talkie 1930 was trained on text up to 1930. Even so, overall accuracy barely moves across the line (35.2% before 1900, 34.3% after), because most questions test period language and reasoning rather than knowledge of a particular text.
  The cutoff does show on the questions that need post-1899 knowledge: the base model answers none of the 16 post-1900 knowledge questions correctly.
- **Dialogue fine-tuning roughly halves knowledge-and-inference accuracy** (44.4% → 23.8%). Talkie shows a similar drop from base to instruction-tuned (44.0% → 25.1%).
  In our case the loss is almost entirely on the abstention questions, where the right answer is to decline. The base model declines on 56 of 62, the chat model on 7, because Mr. Chatterbox is trained to stay in character and answer.
  On the knowledge questions themselves, the chat model does slightly better than the base (35 of 94 correct, against 27).
- **Overlap with our corpus:** at least 19 of the benchmark's 213 source books (9%) are also in our pretraining corpus, matched by library barcode. Other editions were not checked, so this is a lower bound.

## Planned: the Great Exhibition model

A fifth model, **Great Exhibition**, is planned but not yet trained. It is about 670M parameters and trained only on text from 1850 or earlier.
Its config is already in `llama_probe_configs.py`. `train_victoriana.py --size Great-Exhibition` will run it, defaulting to `--year-max 1850` and a budget of 20 × params (about 13.3B tokens, roughly all the pre-1851 text in the corpus).
No checkpoint exists yet.

## License

- **Code (this repository):** MIT. See [LICENSE](LICENSE). `muon.py` is adapted from MIT-licensed upstream code; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **Model weights (Hugging Face):** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  Parts of the training corpus are licensed for noncommercial use only, and some carry a share-alike requirement. The weights inherit those terms; the code does not.

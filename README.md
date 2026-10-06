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

## Planned: the Great Exhibition model

A fifth model, **Great Exhibition**, is planned but not yet trained. It is about 670M parameters and trained only on text from 1850 or earlier.
Its config is already in `llama_probe_configs.py`. `train_victoriana.py --size Great-Exhibition` will run it, defaulting to `--year-max 1850` and a budget of 20 × params (about 13.3B tokens, roughly all the pre-1851 text in the corpus).
No checkpoint exists yet.

## License

MIT. See [LICENSE](LICENSE). `muon.py` is adapted from MIT-licensed upstream code; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

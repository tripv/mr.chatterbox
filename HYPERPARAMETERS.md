# Base-model hyperparameters

These settings apply to the four released base models (`victoriana-345M`, `-500M`, `-1.2B` and `-2.7B`).
All four were trained with `victoriana_training/train_victoriana.py` and share the same code and settings. Only the model shape and token budget differ between them.

## Architecture

All sizes use Hugging Face `LlamaForCausalLM` (`victoriana_training/llama_probe_configs.py`):

| | hidden | layers | attn heads | KV heads | MLP (SwiGLU) | parameters |
|---|---|---|---|---|---|---|
| 345M | 1024 | 22 | 8 | 8 | 2816 | 349.7M |
| 500M | 1280 | 21 | 10 | 10 | 3328 | 489.9M |
| 1.2B | 2048 | 21 | 16 | 16 | 5376 | 1,180.3M |
| 2.7B | 2560 | 32 | 20 | 20 | 6912 | 2,705.5M |

Shared across sizes:

| setting | value |
|---|---|
| vocabulary | 65,536 (byte-level BPE, `tokenizer/victoriana-v2-65536`) |
| context length | 2,048 |
| embeddings | tied (input embedding = output head) |
| attention | full multi-head (KV heads = attention heads), no bias |
| MLP | SwiGLU, no bias |
| normalisation | RMSNorm, eps 1e-6 |
| position encoding | RoPE, θ = 10,000 |
| EOS | `<|endoftext|>` (id 119) |
| BOS / PAD | none |
| precision | bf16 |

## Optimisation

The optimizer combines Muon and AdamW (`victoriana_training/muon.py`, adapted from nanochat / modded-nanogpt):

| parameter group | optimizer | learning rate | other |
|---|---|---|---|
| 2-D weight matrices (attention, MLP) | Muon | 0.02 | momentum 0.85→0.95 over warmup, Newton–Schulz 5 steps, β₂ 0.9, weight decay 0.1 |
| token embedding / tied head | AdamW | 0.2 × (hidden/768)^-0.5 | betas (0.8, 0.95), eps 1e-10, weight decay 0.001 |
| RMSNorm weights and other 1-D params | AdamW | 0.2 × (hidden/768)^-0.5 | betas (0.8, 0.95), eps 1e-10, weight decay 0 |

| setting | value |
|---|---|
| tokens per optimizer step | 524,288 (256 sequences × 2,048) |
| warmup | 300 steps, linear |
| schedule | cosine decay to 0 (`final_lr_frac` 0.0) |
| gradient clipping | global norm 1.0 (averaged across ranks before clipping) |
| token budget | 20 × parameters (Chinchilla-style), single pass, no repeated epochs |

The device batch size only sets how the 524,288-token step is split across GPUs and gradient-accumulation passes. Changing it does not change the training run.

## Per-size budgets

| size | tokens | optimizer steps |
|---|---|---|
| 345M | 6.99B | 13,341 |
| 500M | 9.80B | 18,689 |
| 1.2B | 23.61B | 45,023 |
| 2.7B | 54.11B | 103,206 |

## Data and conditioning

- **Corpus:** 19th-century British printed material only (`tag_schema.in_band`: 1800 ≤ year ≤ 1899).
- **Document prefix:** each document starts with up to three reserved tokens, `[yYYYY][src_*][ocr_*]` (`tag_schema.py`). The model is never trained to predict these prefix tokens; their loss is masked.
- **Tag dropout:** during training each tag is dropped independently with probability 0.15, so the model works with any subset of tags at inference, including none.
- **Held-out split:** about 0.5% of documents are held out, chosen by hashing the book ID (`tag_schema.is_holdout`). This split gives the validation loss.
- **Packing:** documents are packed into 2,048-token sequences. Position IDs restart at each document, and attention is masked at document boundaries.
- **Source mixing:** shards are read round-robin across source directories, one shard per source per round. This keeps a budget-limited run representative of every source from the start.

## Hardware and attention backend

- **Hardware:** 8×NVIDIA B200 with distributed Muon/AdamW (ZeRO-2-style sharded optimizer state).
- **Attention:** flash-attn has no Blackwell build, so the released models were trained with PyTorch SDPA plus an explicit per-document attention mask (`--attn sdpa`, document masking on).
- **NCCL:** set `NCCL_NVLS_ENABLE=0`; see `run_probes.sh`.

## How these values were established

The original pretraining logs were lost. The values above are the trainer's defaults (`train_victoriana.py`), which the base runs used. Only the device batch size was set by hand, and it does not affect training (see above).

For **2.7B**, the settings can be checked directly. A save from partway through the run (step 50,500 of 103,206) still holds the optimizer's stored settings:

| optimizer group | stored peak LR | LR at step 50,500 | other stored settings |
|---|---|---|---|
| Muon (matrices) | 0.02 | 0.010382 | momentum 0.95, ns_steps 5, β₂ 0.9, weight decay 0.1 |
| AdamW (embedding) | 0.109545 = 0.2 × (2560/768)^-0.5 | 0.056867 | betas (0.8, 0.95), eps 1e-10, weight decay 0.001 |
| AdamW (norms) | 0.109545 | 0.056867 | betas (0.8, 0.95), eps 1e-10, weight decay 0 |

At step 50,500, every group sits at 0.519122 of its peak learning rate. That matches a 300-step warmup followed by cosine decay to 0 over 103,206 steps, to within floating-point rounding (difference below 1e-15).
The step count of 103,206 is exactly 20 × 2,705.5M parameters ÷ 524,288 tokens per step, which confirms the batch size and token budget too.

For **345M, 500M and 1.2B**, no optimizer state survives, so there is no direct check like this. All three were trained on the same code revision as 2.7B.
Their architectures are confirmed by each checkpoint's `config.json`.

## Held-out perplexity

*Being measured.* All four models will be scored on one shared sample of 2,000 held-out sequences of 2,048 tokens each, drawn uniformly from the held-out split. Results, with bits per byte (BPB), will be added here.

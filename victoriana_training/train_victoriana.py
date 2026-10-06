#!/usr/bin/env python3
"""
train_victoriana.py — from-scratch pretraining of the Victoriana Llama (HF LlamaForCausalLM) with the
custom Muon optimizer, tag-aware data, and per-size-optimal token budgets. Built for the scaling probe
(345M / 500M / 1.2B) and the ~2.7B final.

Runs on GPU (single-GPU, or multi-GPU via torchrun). Not intended for CPU.

  single GPU:  python train_victoriana.py --size 345M
  multi GPU :  torchrun --nproc_per_node=8 train_victoriana.py --size 1.2B

Key pieces:
  • model   : LlamaForCausalLM(build_config(size)), bf16, flash_attention_2 (doc-boundary masking via
              the dataloader's resetting position_ids), tied embeddings.
  • optim   : Muon (2D matrix weights, grouped by shape) + AdamW (embeddings/norms) — muon.py.
  • data    : VictorianaIterableDataset — [year][src][ocr] prefix, loss-masked, per-tag dropout,
              in-band 1800-1899, holdout→val, per-rank shard sharding.
  • budget  : default Chinchilla-optimal ≈ 20 × params (overridable via --tokens).
  • eval    : val cross-entropy / perplexity on the holdout split (same tokenizer → comparable across
              sizes for the scaling fit).
"""
import argparse, json, math, os, time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from transformers import LlamaForCausalLM

import tag_schema as T
import victoriana_data as D
from llama_probe_configs import PROBES, FINAL, GREAT_EXHIBITION, build_config
from muon import MuonAdamW, DistMuonAdamW

# ── optimizer param grouping for a LlamaForCausalLM (tied embeddings) ────────────────────────────
def build_optimizer(model, world, matrix_lr, embedding_lr, weight_decay, hidden):
    seen = set(); matrix, embed, scalar = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name:
            matrix.append(p)                       # attn/mlp projection matrices → Muon
        elif "embed_tokens" in name or "lm_head" in name:
            embed.append(p)                        # (tied) token embedding / head → AdamW
        else:
            scalar.append(p)                       # RMSNorm weights & 1-D → AdamW
    dscale = (hidden / 768) ** -0.5                # AdamW LR ∝ 1/√d_model (nanochat convention)
    groups = [
        dict(kind="adamw", params=embed,  lr=embedding_lr * dscale, betas=(0.8, 0.95),
             eps=1e-10, weight_decay=0.001),
        dict(kind="adamw", params=scalar, lr=embedding_lr * dscale, betas=(0.8, 0.95),
             eps=1e-10, weight_decay=0.0),
    ]
    for shape in sorted({tuple(p.shape) for p in matrix}):   # Muon groups same-shape for stacking
        groups.append(dict(kind="muon", params=[p for p in matrix if tuple(p.shape) == shape],
                           lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay))
    opt = (DistMuonAdamW if world > 1 else MuonAdamW)(groups)
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    return opt

def _doc_ids(position_ids):
    """Per-token document index within each packed block, recovered from the dataloader's resetting
    position_ids (victoriana_data._position_ids emits 0 at every document start). Equivalent to
    cumsum over doc-start flags. A document longer than seq_len also resets at the chunk boundary,
    so its continuation counts as a new 'document' here — the conservative choice for masking."""
    return (position_ids == 0).cumsum(dim=-1)

def build_doc_mask(position_ids, attn, dtype):
    """Block-diagonal (per-document) ∧ causal attention mask, so packed documents can't attend to
    each other. WITHOUT this, `--attn sdpa` gives NO document isolation at all: resetting
    position_ids only re-bases RoPE, it does not mask attention (verified empirically 2026-07-28 —
    a doc packed after another produced logits differing by 1.17 vs. the same doc alone). That is
    doubly bad here, because with positions restarting a token can attend to a *different*
    document's token carrying the same position index, i.e. RoPE relative distance 0, as if they
    were the same position. flash_attention_2 would infer boundaries from position_ids on its own,
    but it has never built on Blackwell/sm_100 in this project, so every run so far used sdpa.

    Returns a BlockMask for flex_attention (fused, no O(L^2) materialisation) or an additive float
    mask for sdpa/eager. HF passes an already-4D mask (and BlockMask, which also reports 4 dims)
    straight through to the attention fn — see transformers.masking_utils.create_causal_mask."""
    doc = _doc_ids(position_ids)
    B, L = doc.shape
    if attn == "flex_attention":
        from torch.nn.attention.flex_attention import create_block_mask
        def mask_mod(b, h, q_idx, kv_idx):
            return (doc[b, q_idx] == doc[b, kv_idx]) & (q_idx >= kv_idx)
        return create_block_mask(mask_mod, B=B, H=None, Q_LEN=L, KV_LEN=L,
                                 device=doc.device, _compile=True)
    allow = (doc[:, :, None] == doc[:, None, :]) & \
            torch.ones(L, L, dtype=torch.bool, device=doc.device).tril()
    return torch.zeros(B, 1, L, L, dtype=dtype, device=doc.device) \
                .masked_fill_(~allow[:, None, :, :], torch.finfo(dtype).min)

def forward_batch(model, b, device, attn, doc_mask, dtype):
    ids, labels = b["input_ids"].to(device), b["labels"].to(device)
    pos = b["position_ids"].to(device)
    # flash_attention_2 infers document boundaries from position_ids itself (varlen path) and rejects
    # an arbitrary 4D mask, so never build one for it — it's the only impl that masks packing natively.
    want = doc_mask and attn != "flash_attention_2"
    # Only build the mask when this batch actually spans a document boundary. The corpus is books, so
    # documents are far longer than seq_len: measured on the real stream at seq_len=2048, only ~2% of
    # blocks contain >1 document (mean 1.02 docs/block), so ~85% of batches at device-batch 8 need no
    # mask at all — and where there's a single document per block, plain causal IS the correct mask.
    # Skipping it keeps the fast attention path (and, under sdpa, avoids materialising a B×1×L×L
    # tensor) on the large majority of steps.
    if want and int((pos == 0).cumsum(dim=-1).max()) > 1:
        return model(input_ids=ids, labels=labels, position_ids=pos,
                     attention_mask=build_doc_mask(pos, attn, dtype))
    return model(input_ids=ids, labels=labels, position_ids=pos)

def lr_mult(it, warmup, total, final_frac):
    if it < warmup:
        return (it + 1) / warmup
    frac = (it - warmup) / max(1, total - warmup)
    return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * frac))

def muon_momentum(it, warmup=300):
    f = min(1.0, it / warmup); return (1 - f) * 0.85 + f * 0.95

@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=400, attn="sdpa", doc_mask=True,
             dtype=torch.bfloat16):
    """max_batches defaults to 400, not 50: at 50 the in-training eval was noisy enough to be
    actively misleading — 1.2B's end-of-run eval reported ppl 23.88 (worse than 500M's 15.85,
    implying a broken run) while a 2000-sequence offline measurement put it at 13.63, the best of
    the ladder. Cheap insurance against chasing a phantom regression."""
    model.eval(); tot_loss = tot_tok = 0
    for i, b in enumerate(val_loader):
        if i >= max_batches: break
        out = forward_batch(model, b, device, attn, doc_mask, dtype)
        ntok = int((b["labels"] != -100).sum())
        tot_loss += out.loss.item() * ntok; tot_tok += ntok
    model.train()
    loss = tot_loss / max(1, tot_tok)
    return loss, math.exp(min(20, loss))

def save_checkpoint(ckpt_dir, model, opt, step, rank, is_main, ddp, shards_done=0):
    """Snapshot for crash/preemption recovery — a single fixed directory, overwritten each call (not
    kept per-step) to bound disk use. Model weights are replicated across ranks (post-all-gather) so
    only rank0 saves them; optimizer state is sharded per-rank under DistMuonAdamW's ZeRO-2 scheme, so
    every rank saves its own shard. shards_done (cumulative shards consumed by rank0's round-robin
    stream) is saved alongside step so a resumed run can skip forward to roughly where it left off
    instead of re-reading the stream from shard 0 every time — see VictorianaIterableDataset's
    skip_shards/progress in victoriana_data.py. Only rank0's count is saved and applied uniformly to
    every rank on resume (each rank actually reads a disjoint shard subset, so this is an
    approximation) — acceptable because all ranks train on equal-sized batches in lockstep and the
    round-robin interleave gives every rank's subset a similarly-mixed variety of sources, so their
    real positions shouldn't drift far apart. Shard-level granularity only (not exact document
    position) either way.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    if is_main:
        model.save_pretrained(ckpt_dir)
        with open(f"{ckpt_dir}/training_state.json", "w") as f:
            json.dump({"step": step, "shards_done": shards_done}, f)
    torch.save(opt.state_dict(), f"{ckpt_dir}/optimizer_rank{rank}.pt")
    if ddp:
        dist.barrier()   # avoid one rank racing ahead while others are still writing

def load_checkpoint(ckpt_dir, opt, rank, device):
    opt.load_state_dict(torch.load(f"{ckpt_dir}/optimizer_rank{rank}.pt", map_location=device))
    with open(f"{ckpt_dir}/training_state.json") as f:
        state = json.load(f)
    return state["step"], state.get("shards_done", 0)   # shards_done=0 for older checkpoints

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="345M", choices=list(PROBES) + list(FINAL) + list(GREAT_EXHIBITION))
    ap.add_argument("--year-max", type=int, default=None,
                    help="cap corpus at year<=N (default: unset, except Great-Exhibition -> 1850)")
    ap.add_argument("--tokens", type=int, default=0, help="train-token budget (0 → 20×params)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--device-batch-size", type=int, default=16)
    ap.add_argument("--total-batch-size", type=int, default=524288, help="tokens per optimizer step")
    ap.add_argument("--matrix-lr", type=float, default=0.02)
    ap.add_argument("--embedding-lr", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--final-lr-frac", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=400,
                    help="val batches per eval (was effectively 50; too noisy to trust — see evaluate())")
    ap.add_argument("--clip", type=float, default=1.0, help="global grad-norm clip (0 disables)")
    ap.add_argument("--no-doc-mask", action="store_true",
                    help="disable per-document attention masking (reproduces the pre-2026-07-28 "
                         "behaviour, where packed documents attended across boundaries under sdpa)")
    ap.add_argument("--save-every", type=int, default=500, help="checkpoint interval, 0 disables")
    ap.add_argument("--resume", default=None, help="path to a checkpoint dir saved by --save-every")
    ap.add_argument("--attn", default="flash_attention_2",
                    help="flash_attention_2 (doc-boundary masking via position_ids) | sdpa | eager")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    device = f"cuda:{local}"; torch.cuda.set_device(device)
    is_main = rank == 0
    def log(*a):
        if is_main: print(*a, flush=True)

    spec = {**PROBES, **FINAL, **GREAT_EXHIBITION}[args.size]
    year_max = args.year_max if args.year_max is not None else (1850 if args.size == "Great-Exhibition" else None)
    tok = Tokenizer.from_file(D.TOK_PATH)
    eos = tok.token_to_id(T.EOS)
    if args.resume:
        model = LlamaForCausalLM.from_pretrained(args.resume, torch_dtype=torch.bfloat16)
        model.config._attn_implementation = args.attn
        model.to(device)
    else:
        cfg = build_config(spec, eos, seq_len=args.seq_len)
        cfg._attn_implementation = args.attn    # flash_attention_2 → doc-boundary masking via position_ids
        model = LlamaForCausalLM(cfg).to(device).to(torch.bfloat16)
    model.train()
    if world > 1 and not args.resume:            # DDP replicas must start identical (no torch-DDP wrapper;
        for p in model.parameters():            # DistMuonAdamW does the grad comm itself). Skipped on
            dist.broadcast(p.data, src=0)       # resume: all ranks already loaded the same checkpoint.
    nparams = sum(p.numel() for p in model.parameters())
    budget = args.tokens or int(20 * nparams)
    per_micro = args.device_batch_size * args.seq_len * world
    if per_micro > args.total_batch_size:
        # grad_accum floors at 1, so an oversized device batch silently INFLATES tokens/step and
        # shrinks the step count (and with it the LR schedule's length) instead of erroring.
        raise SystemExit(
            f"--device-batch-size {args.device_batch_size} x seq_len {args.seq_len} x world {world} "
            f"= {per_micro:,} tokens/micro-step exceeds --total-batch-size {args.total_batch_size:,}. "
            f"grad_accum would floor at 1, making the real batch {per_micro:,} tok/step "
            f"({per_micro/args.total_batch_size:.2g}x intended) and the run "
            f"{per_micro/args.total_batch_size:.2g}x shorter. Lower --device-batch-size to "
            f"<= {args.total_batch_size // (args.seq_len * world)} or raise --total-batch-size.")
    grad_accum = max(1, args.total_batch_size // per_micro)
    total_batch = per_micro * grad_accum
    num_steps = budget // total_batch
    out = args.out or f"checkpoints/victoriana-{args.size}"
    ckpt_dir = f"{out}-resume"
    doc_mask = not args.no_doc_mask
    log(f"size {args.size} | {nparams/1e6:.0f}M params | budget {budget/1e9:.1f}B tok | "
        f"steps {num_steps:,} | grad_accum {grad_accum} | tok/step {total_batch:,} | world {world} | "
        f"year_max {year_max or 'none (full 1800-1899 band)'} | attn {args.attn} | "
        f"doc_mask {doc_mask} | eval_batches {args.eval_batches} | clip {args.clip}")

    opt = build_optimizer(model, world, args.matrix_lr, args.embedding_lr, args.weight_decay, spec["hidden"])
    start_step, start_shards_done = 0, 0
    if args.resume:
        start_step, start_shards_done = load_checkpoint(args.resume, opt, rank, device)
        log(f"resumed from {args.resume} at step {start_step} (shards_done {start_shards_done})")

    def loader(split, loop, skip_shards=0):
        ds = D.VictorianaIterableDataset(split, tok=tok, seq_len=args.seq_len, rank=rank, world=world,
                                         loop=loop, year_max=year_max, skip_shards=skip_shards)
        # num_workers=0 is required, not just a default: VictorianaIterableDataset.__iter__ has no
        # torch.utils.data.get_worker_info()-based partitioning, so num_workers>1 previously caused
        # every worker to replay the identical stream from scratch — the DataLoader interleaved two
        # exact-duplicate copies of every packed sequence (verified empirically 2026-07-28). Running
        # in-process (num_workers=0) also lets the training loop read ds.progress directly, with no
        # cross-process IPC, to persist shard-stream position into the checkpoint.
        loader_ = DataLoader(ds, batch_size=args.device_batch_size, num_workers=0, pin_memory=True)
        return loader_, ds
    train_loader, train_ds = loader("train", loop=True, skip_shards=start_shards_done)
    train_iter = iter(train_loader)
    val_loader, _ = loader("val", loop=False)

    t0 = time.time()
    for step in range(start_step, num_steps):
        # LR + Muon momentum/wd schedules
        m = lr_mult(step, args.warmup, num_steps, args.final_lr_frac)
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * m
            if g["kind"] == "muon":
                g["momentum"] = muon_momentum(step, args.warmup)
        loss_acc = 0.0
        for micro in range(grad_accum):
            b = next(train_iter)
            out_ = forward_batch(model, b, device, args.attn, doc_mask, torch.bfloat16)
            (out_.loss / grad_accum).backward()
            loss_acc += out_.loss.item() / grad_accum
        if args.clip:
            # DistMuonAdamW reduce_scatters gradients inside step() (ZeRO-2), so no rank ever holds
            # the full averaged gradient g_bar — clipping here unreduced would clip each rank's own
            # LOCAL gradient (a different vector per rank) rather than the global norm. Pre-average
            # with all_reduce(AVG) so every rank holds g_bar and computes the same, correct global
            # norm; step()'s own AVG then operates on values already identical across ranks, which
            # is a no-op for AVG, so this fixes the clip without altering the update itself.
            if ddp:
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step(); opt.zero_grad(set_to_none=True)

        if is_main and step % 20 == 0:
            toks = (step + 1) * total_batch
            log(f"step {step:>6}/{num_steps} | loss {loss_acc:.4f} | lr×{m:.3f} | "
                f"{toks/1e9:.2f}B tok | {time.time()-t0:.0f}s")
        if step > 0 and step % args.eval_every == 0:
            vl, ppl = evaluate(model, val_loader, device, args.eval_batches, args.attn, doc_mask, torch.bfloat16)
            log(f"  [eval] step {step} | val_loss {vl:.4f} | ppl {ppl:.2f}")
        if args.save_every and step > 0 and step % args.save_every == 0:
            save_checkpoint(ckpt_dir, model, opt, step, rank, is_main, ddp,
                            shards_done=train_ds.progress["shards_done"])
            log(f"  [ckpt] step {step} → {ckpt_dir} (shards_done {train_ds.progress['shards_done']})")

    vl, ppl = evaluate(model, val_loader, device, args.eval_batches, args.attn, doc_mask, torch.bfloat16)
    log(f"=== DONE {args.size} | final val_loss {vl:.4f} | ppl {ppl:.2f} | tokens {budget/1e9:.1f}B ===")
    if is_main:
        os.makedirs(out, exist_ok=True)
        model.save_pretrained(out)            # HF format → drops into transformers/vLLM
        log(f"saved → {out}")
    if ddp: dist.destroy_process_group()

if __name__ == "__main__":
    main()

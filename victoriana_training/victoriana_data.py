#!/usr/bin/env python3
"""
victoriana_data.py — tag-aware pretraining data pipeline for the Llama (HF) training stack.

Turns the tagged corpus shards into packed (input_ids, labels) sequences with:
  • [year][source][ocr] prefix prepended per document (tag_schema.make_prefix), each tag an ATOMIC
    reserved special token;
  • the prefix LOSS-MASKED (labels = -100 over the prefix span) — the model conditions on the tags
    but never predicts/emits them;
  • per-tag independent dropout (training) so the model handles any subset / unconditional generation;
  • <|endoftext|> between documents;
  • 19th-century band enforced (in_band) and held-out books (is_holdout) routed to the val split.

HF-compatible: VictorianaIterableDataset yields {"input_ids","labels"} of length seq_len, ready for
LlamaForCausalLM (which shifts internally; loss is CE(logits[:-1], labels[1:]) with -100 ignored).
Pure-Python/CPU — unit-testable without a GPU (see smoke_pipeline.py).
"""
import glob, itertools, json, random
import numpy as np
import pyarrow.parquet as pq
import os
from tokenizers import Tokenizer
import tag_schema as T

# Configurable via env so the cluster needs no code edits (bootstrap sets these).
ROOT = os.environ.get("VICTORIANA_DATA_ROOT", "training_data/pretraining")
TOK_PATH = os.environ.get("VICTORIANA_TOKENIZER", "tokenizer/victoriana-v2-65536/tokenizer.json")

def load_tokenizer(path=TOK_PATH):
    return Tokenizer.from_file(path)

def _discover_dirs_and_files(root):
    """Live filesystem scan: source dir -> sorted list of its .parquet shard paths."""
    dirs = sorted(d for d in os.listdir(root)
                  if os.path.isdir(f"{root}/{d}") and not d.startswith("_"))
    return {d: sorted(glob.glob(f"{root}/{d}/*.parquet")) for d in dirs}

def _load_or_build_manifest(root):
    """FIX for non-stationary shard order (2026-08-15): _all_shards() used to re-run
    os.listdir + glob.glob on every call. Each individual call was internally deterministic
    (both are wrapped in sorted()), but this project pulls its corpus fresh onto ephemeral
    rented instances per run (pull_from_r2.sh) and has an active corpus-cleaning workstream
    re-uploading shards -- if the discovered file SET changes between an original launch and a
    later resume (new shards added/removed/re-uploaded to R2 in between), a fresh live scan
    silently returns a DIFFERENT shard list, which breaks two things: (1) skip_shards, which
    indexes into "the shard list" by position, now points at a different physical file than the
    run that recorded it intended; (2) the round-robin coverage guarantee itself, since a
    changed per-source file count reshuffles which files land in which "round".

    Fix: freeze the discovered {dir: [files]} mapping to a manifest file
    (`{root}/_shard_manifest.json`) the FIRST time it's built, and load that manifest on every
    subsequent call instead of re-scanning the filesystem. This makes the shard list genuinely
    STATIONARY for the lifetime of a training run's manifest, independent of what changes in R2/
    local storage afterward. Regenerating is a deliberate, visible action (delete the manifest
    file, or set VICTORIANA_REBUILD_SHARD_MANIFEST=1), never a silent side effect of a live scan
    drifting between calls."""
    manifest_path = f"{root}/_shard_manifest.json"
    force_rebuild = os.environ.get("VICTORIANA_REBUILD_SHARD_MANIFEST", "") == "1"
    if os.path.exists(manifest_path) and not force_rebuild:
        with open(manifest_path) as f:
            return json.load(f)
    per_dir = _discover_dirs_and_files(root)
    with open(manifest_path, "w") as f:
        json.dump(per_dir, f, indent=2)
    print(f"[victoriana_data] {'rebuilt' if force_rebuild else 'built'} shard manifest -> "
          f"{manifest_path} ({sum(len(v) for v in per_dir.values())} shards across "
          f"{len(per_dir)} sources)")
    return per_dir

def _all_shards(root=ROOT, rank=0, world=1):
    """Round-robin shards across source directories (BL/hathi/gutenberg/hansard/...) instead of
    concatenating them in alphabetical order. Source dirs vary 2MB (supplemental) to 58GB (hathi);
    a straight concatenation means any token budget smaller than the full corpus (e.g. the 345M
    probe's ~6.9B-token Chinchilla budget vs. the ~84.5GB/15-source corpus) would exhaust the
    alphabetically-early sources and never reach sources like institutional/nls/oldbailey/wellcome
    at all. Round-robin (one shard per source per round) keeps every partial run's mix
    representative of all sources from the very first tokens, regardless of budget size.

    DDP sharding happens WITHIN each source dir (`[rank::world]`) before the interleave — NOT by
    striding the flattened round-robin list. Striding the flat list is subtly broken: with N source
    dirs the flat list has period N, so taking every `world`-th entry leaves each rank seeing only
    N/gcd(world, N) distinct sources. With N=15 that silently starves ranks at world=3 (5 sources),
    world=5 (3), world=6 (5), world=9 (5), world=10 (3), world=12 (5), world=15 (1 source!) — the
    4- and 8-GPU runs this project has actually used were only safe because gcd(4,15)=gcd(8,15)=1.
    Slicing inside each dir first guarantees every rank sees every source at any world size.

    Shard discovery (which dirs, which files, in what order) is now frozen via
    _load_or_build_manifest() rather than re-scanned live on every call -- see that function's
    docstring for why a live re-scan across a multi-resume/multi-instance training lifecycle was
    a real "non-stationary shard order" bug, not just a theoretical one."""
    per_dir_full = _load_or_build_manifest(root)
    per_dir = [files[rank::world] for files in per_dir_full.values()]
    return [s for round_ in itertools.zip_longest(*per_dir) for s in round_ if s is not None]

def iter_records(split, root=ROOT, rank=0, world=1, loop=False, year_max=None,
                 skip_shards=0, progress=None):
    """Yield in-band records for 'train' (non-holdout) or 'val' (holdout). For DDP, shards are
    assigned round-robin to ranks so each rank sees disjoint data. loop=True repeats forever (train).
    year_max, if set, additionally restricts to year<=year_max within the 1800-1899 band — used by
    the Great Exhibition model's pre-1851 corpus filter (the band itself stays locked at 1800-1899).

    skip_shards: number of shards (within THIS rank's subset) to skip on the very first pass, so a
    resumed run can continue roughly where an earlier session left off instead of re-reading the
    round-robin stream from shard 0 every time. progress, if given, is a dict this function mutates
    in place — progress["shards_done"] tracks the cumulative count of fully-consumed shards (starting
    from skip_shards), so the caller can persist it into a checkpoint for the NEXT resume."""
    want_holdout = (split == "val")
    shards = _all_shards(root, rank=rank, world=world)   # per-rank slice taken inside each source dir
    n = len(shards)
    if progress is not None:
        progress["shards_done"] = skip_shards
    first_pass = True
    while True:
        start = (skip_shards % n) if (first_pass and n) else 0
        for idx in range(start, n):
            shard = shards[idx]
            for r in pq.read_table(shard,
                    columns=["text", "year", "source", "htid"]).to_pylist():
                if not T.in_band(r["year"]) or not r["text"]:
                    continue
                if year_max is not None and r["year"] > year_max:
                    continue
                if T.is_holdout(r["htid"]) == want_holdout:
                    yield r
            if progress is not None:
                progress["shards_done"] += 1
        first_pass = False
        if not loop:
            break

def example_ids(rec, tok, eos_id, rng=None, dropout=0.0):
    """Record -> (ids, labels). Prefix span is masked (-100); text+eos carry loss."""
    prefix = T.make_prefix(rec, rng, dropout)
    pids = tok.encode(prefix, add_special_tokens=True).ids if prefix else []
    tids = tok.encode(rec["text"], add_special_tokens=False).ids
    ids = pids + tids + [eos_id]
    labels = ([-100] * len(pids)) + tids + [eos_id]
    return ids, labels

def _position_ids(starts_chunk):
    """Document-local positions: reset to 0 at each doc start (and at chunk start). With
    flash_attention_2, position_ids resetting → block-diagonal attention (no cross-document
    attention) for free; RoPE stays document-local. Caps positions at seq_len."""
    n = len(starts_chunk)
    idx = np.arange(n)
    seg_start = np.maximum.accumulate(np.where(starts_chunk, idx, 0))
    return (idx - seg_start).astype(np.int64)

def packed_stream(split, tok, seq_len=2048, dropout=T.TAG_DROPOUT, seed=0, limit_docs=None,
                  rank=0, world=1, loop=False, year_max=None, skip_shards=0, progress=None,
                  yield_source=False, root=None):
    """Stream packed (input_ids, labels, position_ids) blocks of exactly seq_len. Prefix masking
    and document boundaries are carried in lockstep with the token stream across chunk boundaries.

    yield_source: if True, each yielded tuple gets a 4th element -- the `source` of whichever
    record was most recently appended to the buffer when that block filled. Diagnostic only (a
    block can span docs from more than one source when packing); used by
    reservoir_sample_held_out() to report the source mix of an eval sample. Defaults to False so
    every existing caller (training, the old first-N eval scripts) is untouched.

    root: corpus root override, passed through to iter_records (defaults to the module-level
    ROOT, same as before this parameter existed)."""
    eos_id = tok.token_to_id(T.EOS)
    rng = random.Random(seed + rank)
    buf_i, buf_l, buf_s = [], [], []
    buf_source = None
    for n, rec in enumerate(iter_records(split, root=(root or ROOT), rank=rank, world=world,
                                         loop=loop, year_max=year_max,
                                         skip_shards=skip_shards, progress=progress)):
        if limit_docs and n >= limit_docs:
            break
        buf_source = rec.get("source")
        ids, labels = example_ids(rec, tok, eos_id, rng, dropout if split == "train" else 0.0)
        starts = [1] + [0] * (len(ids) - 1)          # 1 at this document's first token
        buf_i.extend(ids); buf_l.extend(labels); buf_s.extend(starts)
        while len(buf_i) >= seq_len:
            block = (np.array(buf_i[:seq_len], dtype=np.int64),
                     np.array(buf_l[:seq_len], dtype=np.int64),
                     _position_ids(np.array(buf_s[:seq_len], dtype=bool)))
            yield (block + (buf_source,)) if yield_source else block
            del buf_i[:seq_len]; del buf_l[:seq_len]; del buf_s[:seq_len]


def reservoir_sample_held_out(tok, n_sequences, seq_len=2048, seed=42, year_max=None, root=None,
                              report=True):
    """Uniform-random sample of n_sequences packed held-out blocks, via single-pass reservoir
    sampling (Algorithm R) over the FULL held-out stream.

    FIXES a real bug: assess_lm_v2.py and verify_held_out.py used to just take the first
    n_sequences blocks off packed_stream("val", ...) in raw shard order. _all_shards() round-
    -robins ONE shard per source directory per round regardless of that source's true size (58GB
    hathi vs. a 2MB supplemental dir get equal billing per round), so the FIRST N blocks in that
    order structurally do not reflect true corpus proportions. Measured 2026-08-03: a first-N
    held-out sample that was 4.2% hathi against the corpus's true ~65% hathi mass -- large enough
    to flip a model comparison (2.7B looked worse on that skewed sample, better corpus-weighted).

    is_holdout is a source-agnostic per-book hash, so the FULL held-out population already
    reflects true corpus proportions -- the bug was in HOW a subset got selected from it, not in
    which docs count as held-out. This reads the entire held-out stream once (CPU-only:
    tokenize + pack, no model forward pass -- held-out is ~0.5% of the corpus by construction, so
    this is a few minutes even against the full corpus) to fill a fixed-size reservoir uniformly
    at random, then returns just that reservoir for the caller to run the (expensive) model
    forward pass on."""
    rng = random.Random(seed)
    reservoir = []
    n_seen = 0
    for block in packed_stream(split="val", tok=tok, seq_len=seq_len, dropout=0.0, seed=seed,
                               root=(root or ROOT), rank=0, world=1, loop=False,
                               year_max=year_max, yield_source=True):
        n_seen += 1
        if len(reservoir) < n_sequences:
            reservoir.append(block)
        else:
            j = rng.randint(0, n_seen - 1)
            if j < n_sequences:
                reservoir[j] = block
    if report:
        import collections
        mix = collections.Counter(b[3] for b in reservoir)
        total = len(reservoir) or 1
        print(f"  reservoir_sample_held_out: sampled {len(reservoir)}/{n_seen} held-out packed "
              f"sequences seen (uniform random, seed={seed})")
        print("  sample source mix (diagnostic -- a block may span >1 source, tagged by its "
              "most recent contributing doc):")
        for src, c in mix.most_common():
            print(f"    {src:<24} {100*c/total:5.1f}%  (n={c})")
    return [(ids, labels, pos) for ids, labels, pos, _src in reservoir]

try:
    import torch
    from torch.utils.data import IterableDataset

    class VictorianaIterableDataset(IterableDataset):
        def __init__(self, split, tok=None, seq_len=2048, dropout=T.TAG_DROPOUT, seed=0,
                     limit_docs=None, rank=0, world=1, loop=None, year_max=None, skip_shards=0):
            self.split, self.seq_len, self.dropout = split, seq_len, dropout
            self.seed, self.limit_docs = seed, limit_docs
            self.rank, self.world = rank, world
            self.loop = (split == "train") if loop is None else loop
            self.year_max = year_max
            self.skip_shards = skip_shards
            # Mutated in place by packed_stream/iter_records as shards are consumed. Only meaningful
            # with num_workers=0 (dataset runs in the training process itself, no IPC needed) — see
            # train_victoriana.py's loader(), which relies on this to persist resume position.
            self.progress = {"shards_done": skip_shards}
            self.tok = tok or load_tokenizer()

        def __iter__(self):
            for ids, labels, pos in packed_stream(self.split, self.tok, self.seq_len, self.dropout,
                                                  self.seed, self.limit_docs, self.rank, self.world,
                                                  self.loop, self.year_max,
                                                  skip_shards=self.skip_shards, progress=self.progress):
                yield {"input_ids": torch.from_numpy(ids),
                       "labels": torch.from_numpy(labels),
                       "position_ids": torch.from_numpy(pos)}
except ImportError:
    pass

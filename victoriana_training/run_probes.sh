#!/usr/bin/env bash
# run_probes.sh — run the scaling probes (default: 345M / 500M / 1.2B) on the B200(s), then fit
# the scaling law and extrapolate the 2.7B point. Run after setup_b200.sh.
#
#   bash victoriana_training/run_probes.sh
#   SIZES=345M bash victoriana_training/run_probes.sh            # only the 345M probe
#   ATTN=sdpa DBS=24 bash victoriana_training/run_probes.sh      # if flash-attn unavailable / OOM
#   NPROC=2 SIZES=500M bash victoriana_training/run_probes.sh    # multi-GPU (torchrun), same instance
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# WORKSPACE = the directory holding training_data/pretraining/, tokenizer/ and checkpoints/.
# Defaults to the repo root (the parent of victoriana_training/).
WORKSPACE=${WORKSPACE:-$(dirname "$HERE")}
cd "$HERE"
export VICTORIANA_DATA_ROOT="${VICTORIANA_DATA_ROOT:-$WORKSPACE/training_data/pretraining}"
export VICTORIANA_TOKENIZER="${VICTORIANA_TOKENIZER:-$WORKSPACE/tokenizer/victoriana-v2-65536/tokenizer.json}"
export PYTHONUNBUFFERED=1
# REQUIRED on these B200 boxes: with NVLS (NVLink SHARP multicast) enabled, NCCL 2.28.9 + driver
# 595.71.05 on Blackwell HANGS on the first collective — reproduced with a bare 8-GPU all_reduce and
# no training code at all, and even at world=2. NCCL_NVLS_ENABLE=0 fixes it (verified: all_reduce
# completes, exit 0). NCCL_P2P_DISABLE=1 does NOT help, confirming NVLS specifically is the culprit.
# This is very likely what caused the 2.7B run's mid-training hang (3 of 8 GPUs pinned at 100% while
# the other 5 idled = a stalled collective), which was initially misdiagnosed as faulty hardware.
# Costs some all-reduce throughput vs NVLS, but these runs are compute-bound (~100% GPU util anyway).
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

# The interpreter must be the one that has torch + tokenizers + transformers installed. Some cloud
# images ship a second, auto-activated virtualenv with its own torch and no tokenizers; set PY= /
# TORCHRUN= explicitly if that applies to your machine.
PY=${PY:-python3}
TORCHRUN=${TORCHRUN:-torchrun}
for _bin in "$PY" "$TORCHRUN"; do
    command -v "$_bin" >/dev/null || { echo "!! $_bin not found — set PY= / TORCHRUN= explicitly"; exit 1; }
done
"$PY" -c 'import tokenizers, transformers, torch' 2>/dev/null || {
    echo "!! $PY cannot import tokenizers/transformers/torch — wrong interpreter, or setup_b200.sh did not finish"; exit 1; }
CKPT=${CKPT:-$WORKSPACE/checkpoints}
ATTN=${ATTN:-flash_attention_2}     # B200: if flash-attn didn't build, set ATTN=sdpa
DBS=${DBS:-32}                      # device batch (per GPU); B200 (180GB) handles 32 easily
SIZES=${SIZES:-"345M 500M 1.2B"}    # override to run a subset, e.g. SIZES=345M
NPROC=${NPROC:-1}                   # GPUs for this run. >1 launches via torchrun (DistMuonAdamW).
                                     # grad_accum auto-shrinks with NPROC (train_victoriana.py already
                                     # divides total_batch_size by device_batch*seq_len*world), so more
                                     # GPUs means the same token/step and step count, just split wider —
                                     # not a different training trajectory, just faster wall-clock.
mkdir -p logs "$CKPT"

echo "=== instance interface check (real data + tokenizer) ==="
"$PY" smoke_pipeline.py && echo "smoke OK" || { echo "smoke FAILED — fix before training"; exit 1; }

if [ "$NPROC" -gt 1 ]; then
    LAUNCH="$TORCHRUN --standalone --nproc_per_node=$NPROC train_victoriana.py"
else
    LAUNCH="$PY -u train_victoriana.py"
fi

for SIZE in $SIZES; do
    echo "============================================================"
    echo " TRAINING $SIZE  (attn=$ATTN, device-batch=$DBS, nproc=$NPROC)"
    echo "============================================================"
    $LAUNCH --size "$SIZE" --attn "$ATTN" \
        --device-batch-size "$DBS" --out "$CKPT/victoriana-$SIZE" \
        2>&1 | tee "logs/train_$SIZE.log"
done

if [ "$SIZES" = "345M 500M 1.2B" ]; then
    echo "=== fitting scaling law → extrapolating 2.7B ==="
    "$PY" fit_scaling.py logs/train_345M.log logs/train_500M.log logs/train_1.2B.log
else
    echo "=== skipping scaling fit (needs all 3 probe sizes; ran: $SIZES) ==="
fi

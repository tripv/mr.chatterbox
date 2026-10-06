#!/usr/bin/env bash
# setup_b200.sh — one-shot environment bootstrap for Victoriana pretraining on a vast.ai B200
# (Blackwell, compute capability sm_100). Run once on the GPU machine.
#
#   bash victoriana_training/setup_b200.sh
#
# Assumes a vast.ai PyTorch image (Ubuntu + CUDA). Picks a Blackwell-capable torch if the image's
# torch is too old, installs the python deps, and best-effort-installs flash-attn (falls back to
# SDPA attention if the build fails — the trainer's --attn flag handles that).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE=${WORKSPACE:-$(dirname "$HERE")}

echo "======================================================================"
echo " Victoriana B200 bootstrap  |  workspace=$WORKSPACE"
echo "======================================================================"

echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
    echo "!! nvidia-smi failed — is this a GPU instance?"; exit 1; }

echo "--- torch / Blackwell capability ---"
NEEDS_TORCH=0
python3 - <<'PY' || NEEDS_TORCH=1
import torch, sys
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| archs", torch.cuda.get_arch_list())
cc = torch.cuda.get_device_capability()
print("device compute capability:", cc)
# Blackwell B200 is sm_100. Require torch built with sm_100 (or a >=10.0 arch) support.
archs = torch.cuda.get_arch_list() or []
ok = any(a.startswith("sm_10") for a in archs) and torch.cuda.is_available()
sys.exit(0 if ok else 7)
PY

if [ "$NEEDS_TORCH" -ne 0 ]; then
    echo "!! current torch lacks Blackwell (sm_100) support — installing cu128 build"
    pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch
fi

echo "--- python deps ---"
pip install -q --upgrade "transformers>=4.48" tokenizers pyarrow numpy scipy ninja

echo "--- flash-attn (Blackwell) — best effort, bounded ---"
# SKIP_FLASH_ATTN=1 skips it outright (then train with --attn sdpa / ATTN=sdpa).
# The from-source build (no prebuilt wheel covers sm_100/Blackwell yet) has hung before (2026-07-13:
# CPU pegged at 96% compiling, no progress, eventually killed manually) rather than failing cleanly.
# Bound it with a timeout so an unattended run falls back to SDPA instead of hanging indefinitely.
FA_BUILD_TIMEOUT=${FA_BUILD_TIMEOUT:-1200}   # 20 min; override if you want to let it run longer
if [ "${SKIP_FLASH_ATTN:-0}" = "1" ]; then
    echo "SKIP_FLASH_ATTN=1 — skipping; train with ATTN=sdpa"
elif python3 -c "import flash_attn" 2>/dev/null; then
    echo "flash-attn already present"
else
    timeout "$FA_BUILD_TIMEOUT" pip install flash-attn --no-build-isolation \
        && echo "flash-attn installed" \
        || echo "!! flash-attn build failed or timed out after ${FA_BUILD_TIMEOUT}s — train with --attn sdpa (cross-doc masking off, OK for the probe)"
fi

echo "--- data + tokenizer presence ---"
export VICTORIANA_DATA_ROOT="${VICTORIANA_DATA_ROOT:-$WORKSPACE/training_data/pretraining}"
export VICTORIANA_TOKENIZER="${VICTORIANA_TOKENIZER:-$WORKSPACE/tokenizer/victoriana-v2-65536/tokenizer.json}"
echo "VICTORIANA_DATA_ROOT=$VICTORIANA_DATA_ROOT"
echo "VICTORIANA_TOKENIZER=$VICTORIANA_TOKENIZER"
NEED_PULL=0
[ -f "$VICTORIANA_TOKENIZER" ] && echo "  tokenizer: OK" || { echo "  tokenizer missing"; NEED_PULL=1; }
if [ -d "$VICTORIANA_DATA_ROOT" ] && [ "$(find "$VICTORIANA_DATA_ROOT" -name '*.parquet' | wc -l)" -gt 0 ]; then
    echo "  corpus shards found: $(find "$VICTORIANA_DATA_ROOT" -name '*.parquet' | wc -l)"
else
    echo "  corpus missing"; NEED_PULL=1
fi
if [ "$NEED_PULL" -eq 1 ]; then
    echo "  !! corpus and/or tokenizer not found. Place the parquet shards under \$VICTORIANA_DATA_ROOT"
    echo "     (one subdirectory per source) and the tokenizer at \$VICTORIANA_TOKENIZER — see README.md."
fi

echo "--- final import check ---"
cd "$HERE"
python3 -c "import torch,transformers,tokenizers,pyarrow,numpy,scipy; \
import tag_schema,llama_probe_configs,victoriana_data,muon,train_victoriana; print('all deps + modules import OK')" \
    && echo "READY. Next: bash run_probes.sh" \
    || echo "!! import check failed"

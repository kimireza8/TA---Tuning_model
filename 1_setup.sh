#!/bin/bash
# =============================================================================
# STEP 1: Setup environment
# Target: Vast.ai — RTX 4090
# Stack : transformers + peft + trl (tanpa unsloth, stabil di semua environment)
# =============================================================================

set -e

echo "========================================================"
echo "Setup fine-tuning environment (RTX 4090 - Vast.ai)"
echo "========================================================"

# ── System packages ──────────────────────────────────────────────────────────
apt-get update -qq && apt-get install -y git git-lfs curl wget build-essential

pip install --upgrade pip --quiet

# ── Cek torch yang tersedia ───────────────────────────────────────────────────
echo ">>> Cek torch:"
python -c "
import torch
print(f'  torch : {torch.__version__}')
print(f'  CUDA  : {torch.cuda.is_available()}')
print(f'  GPU   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"

# ── Install package fine-tuning ───────────────────────────────────────────────
echo ">>> Menginstal package fine-tuning..."
pip install \
    transformers \
    "peft>=0.13.2" \
    trl \
    accelerate \
    bitsandbytes \
    datasets \
    sentencepiece \
    protobuf \
    scipy \
    einops \
    --quiet

# Hapus torchao jika ikut terpasang
pip uninstall -y torchao 2>/dev/null || true

# ── Verifikasi ────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "Verifikasi:"
python -c "
import torch, transformers, peft, trl, datasets
print(f'  torch        : {torch.__version__}  CUDA={torch.cuda.is_available()}')
print(f'  transformers : {transformers.__version__}')
print(f'  peft         : {peft.__version__}')
print(f'  trl          : {trl.__version__}')
print(f'  datasets     : {datasets.__version__}')
print()
print('Setup selesai!')
"
echo "========================================================"
echo ""
echo "Lanjut ke:"
echo "  python 2_prepare_dataset.py"
echo "  python 3_train.py"

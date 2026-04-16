#!/bin/bash
# =============================================================================
# STEP 1: Setup environment
# Target: Vast.ai — vastai/pytorch_cuda-12.4.1-auto — RTX 4090
# Strategi: buat venv BARU yang bersih, hindari konflik base image /venv/main
# =============================================================================

set -e

VENV_PATH="/workspace/train_env"

echo "========================================================"
echo "Setup venv bersih untuk fine-tuning (RTX 4090)"
echo "========================================================"

# ── System packages ──────────────────────────────────────────────────────────
apt-get update -qq && apt-get install -y git git-lfs curl wget build-essential python3-venv

# ── Buat venv baru yang bersih ───────────────────────────────────────────────
echo ">>> Membuat virtual environment bersih di $VENV_PATH..."
python3 -m venv "$VENV_PATH" --clear
source "$VENV_PATH/bin/activate"
pip install --upgrade pip --quiet

echo ">>> Python: $(which python) — $(python --version)"

# ── Torch 2.6.0 + torchvision + CUDA 12.6 ───────────────────────────────────
echo ">>> Menginstal torch 2.6.0+cu126..."
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu126 --quiet

# Verifikasi GPU
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA tidak tersedia!'
print(f'  torch : {torch.__version__}')
print(f'  GPU   : {torch.cuda.get_device_name(0)}')
print(f'  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ── Unsloth + semua dependency-nya ──────────────────────────────────────────
# Biarkan unsloth menentukan versi transformers/trl yang cocok
echo ">>> Menginstal Unsloth..."
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" --quiet

# ── Package training tambahan ────────────────────────────────────────────────
echo ">>> Menginstal package training..."
pip install \
    bitsandbytes \
    datasets \
    sentencepiece \
    protobuf \
    scipy \
    einops \
    --quiet

# ── Hapus torchao jika ikut terpasang ────────────────────────────────────────
pip uninstall -y torchao 2>/dev/null || true

# ── Verifikasi akhir ──────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "Verifikasi instalasi:"
python -c "
import torch, transformers, peft, trl, datasets, unsloth
print(f'  torch        : {torch.__version__}  CUDA={torch.cuda.is_available()}')
print(f'  transformers : {transformers.__version__}')
print(f'  peft         : {peft.__version__}')
print(f'  trl          : {trl.__version__}')
print(f'  datasets     : {datasets.__version__}')
print(f'  unsloth      : {unsloth.__version__}')
print()
print('Semua OK!')
"
echo "========================================================"
echo ""
echo "PENTING: Aktifkan venv ini setiap kali masuk instance:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo ">>> Setup selesai! Lanjut ke:"
echo "  python 2_prepare_dataset.py"
echo "  python 3_train.py"

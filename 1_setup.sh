#!/bin/bash
# =============================================================================
# STEP 1: Setup environment
# Target: Vast.ai — vastai/pytorch_cuda-12.4.1-auto — RTX 4090
# Strategi: venv dengan --system-site-packages untuk mewarisi torch 2.11.0
#           dari base image, hindari install/reinstall torch yang konflik.
# =============================================================================

set -e

VENV_PATH="/workspace/train_env"

echo "========================================================"
echo "Setup environment fine-tuning (RTX 4090 - Vast.ai)"
echo "========================================================"

# ── System packages ──────────────────────────────────────────────────────────
apt-get update -qq && apt-get install -y git git-lfs curl wget build-essential python3-venv

# ── Verifikasi torch base image ───────────────────────────────────────────────
echo ">>> Torch di base image:"
python3 -c "
import torch
print(f'  torch  : {torch.__version__}')
print(f'  CUDA   : {torch.cuda.is_available()}')
print(f'  GPU    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"

# ── Buat venv yang mewarisi system packages (termasuk torch dari base) ────────
echo ">>> Membuat venv di $VENV_PATH (--system-site-packages)..."
python3 -m venv "$VENV_PATH" --system-site-packages --clear
source "$VENV_PATH/bin/activate"
pip install --upgrade pip --quiet

echo ">>> Python: $(which python)"
echo ">>> Torch inherited: $(python -c 'import torch; print(torch.__version__)')"

# ── Unsloth ───────────────────────────────────────────────────────────────────
echo ">>> Menginstal Unsloth..."
pip install unsloth_zoo
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"

# ── Package fine-tuning ───────────────────────────────────────────────────────
echo ">>> Menginstal package fine-tuning..."
pip install \
    "peft==0.13.2" \
    trl \
    accelerate \
    bitsandbytes \
    datasets \
    sentencepiece \
    protobuf \
    scipy \
    einops

# ── Hapus torchao — selalu konflik, tidak dibutuhkan ─────────────────────────
echo ">>> Menghapus torchao..."
pip uninstall -y torchao 2>/dev/null || true

# ── Verifikasi akhir ──────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "Verifikasi instalasi:"
python -c "
import torch, transformers, peft, trl, datasets
print(f'  torch        : {torch.__version__}  CUDA={torch.cuda.is_available()}')
print(f'  transformers : {transformers.__version__}')
print(f'  peft         : {peft.__version__}')
print(f'  trl          : {trl.__version__}')
print(f'  datasets     : {datasets.__version__}')
print()
print('Semua OK! Jalankan:')
print('  source $VENV_PATH/bin/activate')
print('  python 2_prepare_dataset.py')
print('  python 3_train.py')
"
echo "========================================================"

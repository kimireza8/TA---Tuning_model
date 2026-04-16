#!/bin/bash
# =============================================================================
# STEP 1: Setup environment
# Target: Vast.ai — vastai/pytorch_cuda-12.4.1-auto — RTX 4090
# =============================================================================

set -e

echo "========================================================"
echo "Setup untuk CUDA 12.4 + PyTorch (RTX 4090 - Vast.ai)"
echo "========================================================"

# ── System packages ──────────────────────────────────────────────────────────
apt-get update -qq && apt-get install -y git git-lfs curl wget build-essential
pip install --upgrade pip --quiet

# ── Bersihkan package yang sering konflik ────────────────────────────────────
echo ">>> Membersihkan package konflik..."
pip uninstall -y torchvision torchaudio torchao triton 2>/dev/null || true

# ── Pastikan torch pakai CUDA (bukan CPU-only) ───────────────────────────────
echo ">>> Menginstal torch 2.5.1 + CUDA 12.4..."
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --quiet

# Verifikasi GPU terdeteksi
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA tidak tersedia! Cek driver GPU.'
print(f'  torch   : {torch.__version__}')
print(f'  CUDA    : {torch.version.cuda}')
print(f'  GPU     : {torch.cuda.get_device_name(0)}')
print(f'  VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ── Triton yang kompatibel dengan torch 2.5.1 ────────────────────────────────
echo ">>> Menginstal triton yang kompatibel..."
pip install triton==3.1.0 --quiet

# ── Unsloth ──────────────────────────────────────────────────────────────────
echo ">>> Menginstal Unsloth..."
pip install \
    "unsloth @ git+https://github.com/unslothai/unsloth.git" \
    --no-deps --quiet

# ── Dependencies training (versi yang sudah diverifikasi tidak konflik) ───────
echo ">>> Menginstal dependencies training..."
pip install \
    transformers==4.44.2 \
    trl==0.11.4 \
    peft==0.13.2 \
    accelerate==0.34.2 \
    bitsandbytes==0.44.1 \
    datasets==3.1.0 \
    sentencepiece \
    protobuf \
    scipy \
    einops \
    packaging \
    ninja \
    --quiet

# Flash Attention 2 — mempercepat training secara signifikan di RTX 4090
echo ">>> Menginstal Flash Attention 2 (butuh beberapa menit)..."
pip install flash-attn==2.6.3 --no-build-isolation --quiet || \
    echo "  [WARNING] Flash Attention gagal diinstall — training tetap bisa jalan, tapi lebih lambat."

# ── Pastikan torchvision/torchao tidak ikut tertarik kembali ─────────────────
pip uninstall -y torchvision torchaudio torchao 2>/dev/null || true

# ── llama.cpp untuk konversi GGUF ────────────────────────────────────────────
echo ">>> Menyiapkan llama.cpp untuk konversi GGUF..."
if [ ! -d "/workspace/llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp /workspace/llama.cpp --depth=1
fi
cd /workspace/llama.cpp
make -j$(nproc) llama-quantize 2>/dev/null || make -j$(nproc) GGML_CUDA=1
pip install -r requirements.txt --quiet

# ── Verifikasi akhir ─────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "Verifikasi instalasi:"
python -c "
import torch
from transformers import __version__ as tv
from peft import __version__ as pv
from trl import __version__ as trlv
from datasets import __version__ as dv
print(f'  torch          : {torch.__version__}  (CUDA: {torch.cuda.is_available()})')
print(f'  transformers   : {tv}')
print(f'  peft           : {pv}')
print(f'  trl            : {trlv}')
print(f'  datasets       : {dv}')
"
echo "========================================================"
echo ""
echo ">>> Setup selesai! Lanjut ke: python 2_prepare_dataset.py"

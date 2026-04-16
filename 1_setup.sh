#!/bin/bash
# =============================================================================
# STEP 1: Setup environment di Vast.ai (RTX 4090)
# Jalankan sekali saat pertama kali masuk instance
# =============================================================================

set -e

echo ">>> Updating system..."
apt-get update -qq && apt-get install -y git git-lfs curl wget build-essential

echo ">>> Installing Python dependencies..."
pip install --upgrade pip

# Hapus torchvision & torchaudio — tidak dibutuhkan untuk text fine-tuning
# dan sering konflik dengan versi torch yang sudah ada di instance Vast.ai
echo ">>> Removing torchvision/torchaudio to prevent version conflicts..."
pip uninstall torchvision torchaudio -y 2>/dev/null || true

# Cek versi torch yang sudah terinstall di instance
TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not found")
echo ">>> Detected torch version: $TORCH_VER"

# Unsloth - framework LoRA paling efisien untuk RTX 4090
# --no-deps agar tidak menarik torchvision secara tidak langsung
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-deps

# Dependencies utama (tanpa torch/torchvision — pakai yang sudah ada di instance)
# transformers 4.44.2: versi stabil sebelum torchao ditambahkan
pip install \
    transformers==4.44.2 \
    trl==0.11.4 \
    peft==0.13.2 \
    accelerate==1.1.1 \
    bitsandbytes==0.44.1 \
    datasets==3.1.0 \
    sentencepiece \
    protobuf \
    wandb \
    scipy \
    einops \
    xformers

# Buang package yang sering konflik dengan instance Vast.ai
pip uninstall torchvision torchaudio torchao -y 2>/dev/null || true

# Verifikasi torch masih bisa diimport
python -c "import torch; print(f'torch {torch.__version__} — CUDA: {torch.cuda.is_available()}')"

# Untuk konversi GGUF
echo ">>> Cloning llama.cpp for GGUF conversion..."
if [ ! -d "/workspace/llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp /workspace/llama.cpp
else
    echo "  llama.cpp sudah ada, skip clone."
fi
cd /workspace/llama.cpp
# Build llama.cpp sekaligus (butuh untuk llama-quantize di step 4)
make -j$(nproc) llama-quantize 2>/dev/null || make -j$(nproc)
pip install -r requirements.txt

echo ""
echo ">>> Setup selesai! Lanjut ke step 2."

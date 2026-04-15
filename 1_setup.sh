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

# Unsloth - framework LoRA paling efisien untuk RTX 4090
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Dependencies utama
pip install \
    transformers==4.46.3 \
    trl==0.12.2 \
    peft==0.13.2 \
    accelerate==1.1.1 \
    bitsandbytes==0.44.1 \
    datasets==3.1.0 \
    sentencepiece \
    protobuf \
    wandb \
    scipy \
    einops

# Untuk konversi GGUF
echo ">>> Cloning llama.cpp for GGUF conversion..."
git clone https://github.com/ggerganov/llama.cpp /workspace/llama.cpp
cd /workspace/llama.cpp
pip install -r requirements.txt

echo ""
echo ">>> Setup selesai! Lanjut ke step 2."

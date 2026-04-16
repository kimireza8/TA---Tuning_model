#!/bin/bash
# =============================================================================
# STEP 4: Konversi model hasil tuning ke GGUF
# Jalankan setelah training selesai (python 3_train.py)
# =============================================================================

set -e

MERGED_DIR="/workspace/outputs/mistral-pens-lora-merged"
GGUF_DIR="/workspace/outputs/gguf"
MODEL_NAME="mistral-pens"
LLAMA_CPP="/workspace/llama.cpp"

echo "============================================"
echo "Konversi model ke GGUF"
echo "============================================"

# ── Cek merged model ada ──────────────────────────────────────────────────────
if [ ! -d "$MERGED_DIR" ]; then
    echo "ERROR: Merged model tidak ditemukan di $MERGED_DIR"
    echo "Pastikan training sudah selesai (python 3_train.py)"
    exit 1
fi
echo "Input  : $MERGED_DIR"
echo "Output : $GGUF_DIR"
mkdir -p "$GGUF_DIR"

# ── Clone & build llama.cpp jika belum ada ────────────────────────────────────
if [ ! -d "$LLAMA_CPP" ]; then
    echo ""
    echo "[0/3] Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_CPP" --depth=1
fi

# Build llama-quantize jika belum ada
QUANTIZE_BIN=""
if [ -f "$LLAMA_CPP/build/bin/llama-quantize" ]; then
    QUANTIZE_BIN="$LLAMA_CPP/build/bin/llama-quantize"
elif [ -f "$LLAMA_CPP/llama-quantize" ]; then
    QUANTIZE_BIN="$LLAMA_CPP/llama-quantize"
else
    echo ""
    echo "Building llama-quantize..."
    cd "$LLAMA_CPP"
    cmake -B build -DGGML_CUDA=ON 2>/dev/null || cmake -B build
    cmake --build build --config Release -j$(nproc) --target llama-quantize
    QUANTIZE_BIN="$LLAMA_CPP/build/bin/llama-quantize"
fi
echo "llama-quantize : $QUANTIZE_BIN"

# Install Python deps untuk convert script
pip install -r "$LLAMA_CPP/requirements.txt" --quiet

# ── Step 1: Konversi langsung ke Q4_K_M (hemat disk, skip fp16) ──────────────
# Disk terbatas — konversi langsung tanpa intermediate fp16 (~14.5 GB)
# Q4_K_M hanya ~4.1 GB, cukup untuk sisa disk
echo ""
echo "[1/2] Konversi HuggingFace → GGUF Q4_K_M langsung (~4.1 GB)..."
python "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$MERGED_DIR" \
    --outfile "$GGUF_DIR/${MODEL_NAME}-Q4_K_M.gguf" \
    --outtype q4_k_m

echo "  OK: $GGUF_DIR/${MODEL_NAME}-Q4_K_M.gguf"
ls -lh "$GGUF_DIR/${MODEL_NAME}-Q4_K_M.gguf"

# ── Step 2: Hapus merged model untuk bebaskan disk, lalu buat Q8_0 ────────────
echo ""
echo "[2/2] Membuat Q8_0 (~7.7 GB, kualitas lebih tinggi)..."
echo "  Catatan: butuh disk ~7.7 GB. Cek sisa disk dulu:"
df -h /workspace | tail -1

# Konversi ke Q8_0 dari Q4_K_M tidak bisa (lossy), jadi skip jika disk < 8 GB
AVAIL=$(df /workspace | awk 'NR==2{print $4}')
if [ "$AVAIL" -gt 8388608 ]; then   # > 8 GB dalam KB
    python "$LLAMA_CPP/convert_hf_to_gguf.py" \
        "$MERGED_DIR" \
        --outfile "$GGUF_DIR/${MODEL_NAME}-Q8_0.gguf" \
        --outtype q8_0
    echo "  OK: $GGUF_DIR/${MODEL_NAME}-Q8_0.gguf"
    ls -lh "$GGUF_DIR/${MODEL_NAME}-Q8_0.gguf"
else
    echo "  SKIP: disk tidak cukup untuk Q8_0. Gunakan Q4_K_M saja."
fi

# ── Ringkasan ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Semua file GGUF siap:"
ls -lh "$GGUF_DIR/"
echo "============================================"
echo ""
echo "Download ke lokal:"
echo "  scp -P <PORT> root@<IP>:$GGUF_DIR/${MODEL_NAME}-Q4_K_M.gguf ."
echo ""
echo "Test dengan llama.cpp lokal:"
echo "  ./llama-cli -m ${MODEL_NAME}-Q4_K_M.gguf --chat-template mistral -n 512 -i"

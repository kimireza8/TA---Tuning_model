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

# ── Step 1: Konversi HuggingFace → GGUF fp16 ─────────────────────────────────
echo ""
echo "[1/3] Konversi HuggingFace → GGUF (fp16)..."
python "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$MERGED_DIR" \
    --outfile "$GGUF_DIR/${MODEL_NAME}-fp16.gguf" \
    --outtype f16

echo "  OK: $GGUF_DIR/${MODEL_NAME}-fp16.gguf"
ls -lh "$GGUF_DIR/${MODEL_NAME}-fp16.gguf"

# ── Step 2: Quantisasi Q4_K_M (rekomendasi) ──────────────────────────────────
echo ""
echo "[2/3] Quantisasi → Q4_K_M (~4.1 GB, rekomendasi)..."
"$QUANTIZE_BIN" \
    "$GGUF_DIR/${MODEL_NAME}-fp16.gguf" \
    "$GGUF_DIR/${MODEL_NAME}-Q4_K_M.gguf" \
    Q4_K_M

echo "  OK: $GGUF_DIR/${MODEL_NAME}-Q4_K_M.gguf"
ls -lh "$GGUF_DIR/${MODEL_NAME}-Q4_K_M.gguf"

# ── Step 3: Quantisasi Q8_0 (kualitas lebih tinggi) ──────────────────────────
echo ""
echo "[3/3] Quantisasi → Q8_0 (~7.7 GB, kualitas lebih tinggi)..."
"$QUANTIZE_BIN" \
    "$GGUF_DIR/${MODEL_NAME}-fp16.gguf" \
    "$GGUF_DIR/${MODEL_NAME}-Q8_0.gguf" \
    Q8_0

echo "  OK: $GGUF_DIR/${MODEL_NAME}-Q8_0.gguf"
ls -lh "$GGUF_DIR/${MODEL_NAME}-Q8_0.gguf"

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

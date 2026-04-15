#!/bin/bash
# =============================================================================
# STEP 4: Konversi model hasil tuning ke GGUF
# Jalankan setelah training selesai
# =============================================================================

set -e

MERGED_DIR="/workspace/outputs/mistral-pens-lora-merged"
GGUF_DIR="/workspace/outputs/gguf"
MODEL_NAME="mistral-pens"
LLAMA_CPP="/workspace/llama.cpp"

mkdir -p "$GGUF_DIR"

echo "============================================"
echo "Konversi ke GGUF"
echo "============================================"
echo "Input  : $MERGED_DIR"
echo "Output : $GGUF_DIR"
echo ""

# ── Step 4a: Konversi ke GGUF (float16 dulu) ────────────────────────────────
echo "[1/3] Konversi HuggingFace → GGUF (fp16)..."
python "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$MERGED_DIR" \
    --outfile "$GGUF_DIR/${MODEL_NAME}-fp16.gguf" \
    --outtype f16

echo "  Berhasil: $GGUF_DIR/${MODEL_NAME}-fp16.gguf"
ls -lh "$GGUF_DIR/${MODEL_NAME}-fp16.gguf"

# ── Step 4b: Quantisasi ke Q4_K_M ───────────────────────────────────────────
# Q4_K_M: keseimbangan terbaik antara ukuran file dan kualitas
echo ""
echo "[2/3] Quantisasi ke Q4_K_M (rekomendasi untuk llama.cpp)..."
"$LLAMA_CPP/llama-quantize" \
    "$GGUF_DIR/${MODEL_NAME}-fp16.gguf" \
    "$GGUF_DIR/${MODEL_NAME}-Q4_K_M.gguf" \
    Q4_K_M

echo "  Berhasil: $GGUF_DIR/${MODEL_NAME}-Q4_K_M.gguf"
ls -lh "$GGUF_DIR/${MODEL_NAME}-Q4_K_M.gguf"

# ── Step 4c: (Opsional) Quantisasi ke Q8_0 — lebih akurat, file lebih besar ──
echo ""
echo "[3/3] Quantisasi ke Q8_0 (opsional, kualitas lebih tinggi)..."
"$LLAMA_CPP/llama-quantize" \
    "$GGUF_DIR/${MODEL_NAME}-fp16.gguf" \
    "$GGUF_DIR/${MODEL_NAME}-Q8_0.gguf" \
    Q8_0

echo "  Berhasil: $GGUF_DIR/${MODEL_NAME}-Q8_0.gguf"
ls -lh "$GGUF_DIR/${MODEL_NAME}-Q8_0.gguf"

# ── Ringkasan ────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Semua file GGUF siap:"
ls -lh "$GGUF_DIR/"
echo "============================================"
echo ""
echo "Download file yang Anda butuhkan:"
echo "  Q4_K_M  → ukuran ~4.1 GB, kualitas bagus (REKOMENDASI)"
echo "  Q8_0    → ukuran ~7.7 GB, kualitas sangat bagus"
echo ""
echo "Cara download dari Vast.ai ke lokal:"
echo "  scp -P <PORT> root@<IP>:$GGUF_DIR/${MODEL_NAME}-Q4_K_M.gguf ."
echo ""
echo "Cara test dengan llama.cpp lokal:"
echo "  ./llama-cli -m ${MODEL_NAME}-Q4_K_M.gguf \\"
echo "    --chat-template mistral \\"
echo "    -n 512 -i"

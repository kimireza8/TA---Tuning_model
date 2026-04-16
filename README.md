# Fine-tuning Mistral 7B untuk Jurnalistik PENS

Fine-tuning model Mistral 7B menggunakan LoRA untuk menghasilkan berita jurnalistik gaya PENS (Politeknik Elektronika Negeri Surabaya). Model hasil tuning dikonversi ke format GGUF untuk dijalankan secara lokal menggunakan llama.cpp.

## Struktur Proyek

```
.
├── README.md
├── train.json              # Dataset training (1.378 samples)
├── val.json                # Dataset validasi (154 samples)
├── 1_setup.sh              # Install dependencies di Vast.ai
├── 2_prepare_dataset.py    # Validasi & konversi dataset ke JSONL
├── 3_train.py              # Training LoRA dengan Unsloth
├── 4_convert_gguf.sh       # Konversi model ke GGUF & quantisasi
└── 5_test_inference.py     # Test inferensi sebelum konversi
```

## Dataset

| Split | Jumlah Samples | Format |
|-------|---------------|--------|
| Train | 1.378         | OpenAI chat (system / user / assistant) |
| Val   | 154           | OpenAI chat (system / user / assistant) |

**Statistik konten:**
- Panjang minimum: ~625 karakter
- Panjang maksimum: ~9.948 karakter
- Rata-rata: ~2.425 karakter

**Contoh format satu sample:**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "Anda adalah jurnalis profesional PENS..."
    },
    {
      "role": "user",
      "content": "Tulis ulang berita dengan angle \"Prestasi mahasiswa\"..."
    },
    {
      "role": "assistant",
      "content": "Surabaya, pens.ac.id – ..."
    }
  ]
}
```

## Spesifikasi Training

| Komponen        | Detail |
|-----------------|--------|
| Base model      | `mistral-7b-instruct-v0.3` |
| Framework       | Unsloth + TRL (SFTTrainer) |
| Metode          | LoRA (Low-Rank Adaptation) |
| Hardware        | RTX 4090 (24 GB VRAM) — Vast.ai |
| Presisi         | bfloat16 |

**Hyperparameter LoRA:**

| Parameter         | Nilai |
|-------------------|-------|
| LoRA rank (r)     | 16 |
| LoRA alpha        | 32 |
| LoRA dropout      | 0 |
| Target modules    | q/k/v/o_proj, gate/up/down_proj |

**Hyperparameter Training:**

| Parameter               | Nilai |
|-------------------------|-------|
| Epochs                  | 3 |
| Batch size (per device) | 4 |
| Gradient accumulation   | 4 (effective batch = 16) |
| Learning rate           | 2e-4 |
| LR scheduler            | Cosine |
| Warmup ratio            | 0.05 |
| Max sequence length     | 2.048 token |

## Cara Penggunaan

### Prasyarat

- Instance Vast.ai: image `vastai/pytorch_cuda-12.4.1-auto`, GPU RTX 4090
- SSH client di mesin lokal
- llama.cpp terinstall di mesin lokal (untuk inferensi)

> **Catatan penting:** Jangan gunakan environment bawaan instance (`/venv/main`) karena sudah berisi package lama yang konflik. Setup di bawah membuat venv bersih di `/workspace/train_env`.

### 1. Clone Repository di Vast.ai

SSH masuk ke instance, lalu jalankan:

```bash
# Clone repo (berisi semua script)
git clone https://github.com/<USERNAME>/<REPO>.git /workspace/finetune
cd /workspace/finetune

# Buat direktori data & pindahkan dataset ke sana
mkdir -p /workspace/data
cp train.json val.json /workspace/data/
```

> Ganti `<USERNAME>/<REPO>` dengan URL repo GitHub Anda.
> Dataset (`train.json` / `val.json`) harus sudah ada di repo, atau upload manual via:
> ```bash
> scp -P <PORT> train.json val.json root@<IP>:/workspace/data/
> ```

### 2. Jalankan di Vast.ai

```bash
# Step 1 — Buat venv bersih & install dependencies (~10-15 menit)
bash 1_setup.sh

# Aktifkan venv baru (wajib setiap kali buka terminal baru)
source /workspace/train_env/bin/activate

# Step 2 — Validasi & siapkan dataset
python 2_prepare_dataset.py

# Step 3 — Training LoRA (~1–2 jam)
python 3_train.py

# Step 4 — (Opsional) Test inferensi
python 5_test_inference.py

# Step 5 — Konversi ke GGUF
bash 4_convert_gguf.sh
```

> **Setiap kali buka terminal baru di instance yang sama**, jalankan dulu:
> ```bash
> source /workspace/train_env/bin/activate
> ```

### 3. Download GGUF ke Lokal

```bash
# Rekomendasi: Q4_K_M (~4.1 GB)
scp -P <PORT> root@<IP>:/workspace/outputs/gguf/mistral-pens-Q4_K_M.gguf .

# Atau Q8_0 jika butuh kualitas lebih tinggi (~7.7 GB)
scp -P <PORT> root@<IP>:/workspace/outputs/gguf/mistral-pens-Q8_0.gguf .
```

### 4. Inferensi Lokal dengan llama.cpp

**Mode interaktif (chat):**
```bash
./llama-cli -m mistral-pens-Q4_K_M.gguf \
  --chat-template mistral \
  -n 512 -i
```

**Single prompt:**
```bash
./llama-cli -m mistral-pens-Q4_K_M.gguf \
  --chat-template mistral \
  -n 512 \
  -p "[INST] Tulis berita tentang prestasi mahasiswa PENS di Surabaya. [/INST]"
```

**Dengan system prompt:**
```bash
./llama-cli -m mistral-pens-Q4_K_M.gguf \
  --chat-template mistral \
  -n 512 \
  --system-prompt "Anda adalah jurnalis profesional PENS. Tulis berita dengan struktur Lead, Body, Tail tanpa label. Gunakan gaya bahasa jurnalistik formal." \
  -i
```

## Output Files

Setelah seluruh proses selesai, struktur output di server:

```
/workspace/outputs/
├── mistral-pens-lora/          # Adapter LoRA saja
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer.*
├── mistral-pens-lora-merged/   # Model full (LoRA sudah di-merge)
│   ├── config.json
│   ├── model-*.safetensors
│   └── tokenizer.*
└── gguf/
    ├── mistral-pens-fp16.gguf      # ~14 GB — intermediate, bisa dihapus
    ├── mistral-pens-Q4_K_M.gguf   # ~4.1 GB — REKOMENDASI
    └── mistral-pens-Q8_0.gguf     # ~7.7 GB — kualitas tinggi
```

## Pilihan Quantisasi

| Format   | Ukuran  | Kualitas       | Rekomendasi |
|----------|---------|----------------|-------------|
| fp16     | ~14 GB  | Penuh          | Untuk konversi saja |
| Q8_0     | ~7.7 GB | Sangat bagus   | RAM > 10 GB |
| Q4_K_M   | ~4.1 GB | Bagus          | **Sweet spot** |
| Q4_0     | ~3.8 GB | Cukup          | RAM terbatas |

## Troubleshooting

**Import error / dependency conflict saat training:**
- Pastikan venv bersih aktif: `source /workspace/train_env/bin/activate`
- Jangan gunakan `/venv/main` (environment bawaan instance yang sudah terkontaminasi)
- Jika venv rusak, hapus dan buat ulang: `rm -rf /workspace/train_env && bash 1_setup.sh`

**CUDA tidak terdeteksi (`torch.cuda.is_available()` = False):**
```bash
source /workspace/train_env/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Harus: 2.6.0+cu126 True
```

**CUDA out of memory saat training:**
- Kurangi `BATCH_SIZE` dari 4 ke 2 di `3_train.py`
- Atau aktifkan `load_in_4bit = True` untuk QLoRA

**`convert_hf_to_gguf.py` tidak ditemukan:**
```bash
git clone https://github.com/ggerganov/llama.cpp /workspace/llama.cpp
pip install -r /workspace/llama.cpp/requirements.txt
```

**`llama-quantize` tidak ditemukan:**
```bash
cd /workspace/llama.cpp && make -j$(nproc) llama-quantize
```

**Hasil generasi terlalu pendek atau tidak relevan:**
- Naikkan `LORA_R` ke 32 dan `LORA_ALPHA` ke 64 di `3_train.py`
- Tambah epochs dari 3 ke 5

## Referensi

- [Unsloth](https://github.com/unslothai/unsloth) — Framework LoRA yang digunakan
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Runtime inferensi lokal
- [Mistral 7B Instruct v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) — Base model
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) — Training framework

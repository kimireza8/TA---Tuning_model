"""
STEP 5 (Opsional): Test inferensi model LoRA sebelum konversi ke GGUF
Berguna untuk validasi apakah training berhasil.

Jalankan: python 5_test_inference.py
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch

MODEL_PATH  = "/workspace/outputs/mistral-pens-lora"   # Atau -merged untuk full model
MAX_SEQ_LEN = 2048

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_PATH,
    max_seq_length = MAX_SEQ_LEN,
    dtype          = torch.bfloat16,
    load_in_4bit   = False,
)
tokenizer = get_chat_template(tokenizer, chat_template="mistral")
FastLanguageModel.for_inference(model)   # Mode inferensi (lebih cepat)

# ── Prompt test ───────────────────────────────────────────────────────────────
messages = [
    {
        "role": "system",
        "content": (
            "Anda adalah jurnalis profesional PENS (Politeknik Elektronika Negeri Surabaya).\n"
            "Tulis berita dengan struktur Lead, Body, Tail tanpa label.\n"
            "Gunakan gaya bahasa jurnalistik formal."
        ),
    },
    {
        "role": "user",
        "content": (
            'Tulis ulang berita dengan angle "Prestasi mahasiswa" menggunakan gaya jurnalistik '
            "piramida terbalik. Tonjolkan peran utama PENS, sertakan konteks tempat Surabaya, "
            "cantumkan waktu Senin (10/3). Awali dengan fakta paling penting."
        ),
    },
]

# Tokenize
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize              = True,
    add_generation_prompt = True,
    return_tensors        = "pt",
).to("cuda")

print("="*60)
print("INPUT PROMPT:")
print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
print("="*60)
print("OUTPUT MODEL:")

outputs = model.generate(
    input_ids       = inputs,
    max_new_tokens  = 512,
    temperature     = 0.7,
    top_p           = 0.9,
    repetition_penalty = 1.1,
    do_sample       = True,
)

# Decode hanya token baru (output saja, bukan prompt)
generated = outputs[0][inputs.shape[1]:]
print(tokenizer.decode(generated, skip_special_tokens=True))
print("="*60)

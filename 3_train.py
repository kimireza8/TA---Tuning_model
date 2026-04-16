"""
STEP 3: Fine-tuning Mistral 7B dengan LoRA (via Unsloth)
Hardware target: RTX 4090 (24 GB VRAM) di Vast.ai

Jalankan: python 3_train.py
"""

import unsloth  # harus diimport pertama sebelum trl/transformers/peft
import os
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI — ubah sesuai kebutuhan
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME   = "unsloth/mistral-7b-instruct-v0.3"   # Base model (sudah dioptimasi unsloth)
OUTPUT_DIR   = "/workspace/outputs/mistral-pens-lora"
DATA_DIR     = "/workspace/data"

# LoRA hyperparameters
LORA_R       = 16       # Rank LoRA — naikan ke 32 jika mau kapasitas lebih besar
LORA_ALPHA   = 32       # Biasanya 2x rank
LORA_DROPOUT = 0       # 0 = Unsloth full optimization, >0 = performance hit

# Training hyperparameters
MAX_SEQ_LEN  = 2048     # Panjang maksimum token per sample
BATCH_SIZE   = 4        # Per-device batch size (RTX 4090 bisa handle 4-8)
GRAD_ACCUM   = 4        # Effective batch = BATCH_SIZE * GRAD_ACCUM = 16
EPOCHS       = 3
LR           = 2e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01

# ══════════════════════════════════════════════════════════════════════════════


def main():
    print("="*60)
    print("Fine-tuning Mistral 7B dengan LoRA untuk PENS Journalism")
    print("="*60)

    # ── 1. Load model & tokenizer ─────────────────────────────────────────────
    print(f"\n[1/5] Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = MODEL_NAME,
        max_seq_length  = MAX_SEQ_LEN,
        dtype           = torch.bfloat16,   # bfloat16 lebih stabil di RTX 4090
        load_in_4bit    = False,            # Full bfloat16 — VRAM cukup di 4090
    )

    # Set chat template Mistral
    tokenizer = get_chat_template(tokenizer, chat_template="mistral")
    print("  Chat template Mistral diterapkan.")

    # ── 2. Tambahkan adapter LoRA ─────────────────────────────────────────────
    # Pakai standard PEFT langsung (hindari FastLanguageModel.get_peft_model
    # yang konflik dengan versi peft/torch tertentu)
    print(f"\n[2/5] Menambahkan LoRA adapter (r={LORA_R}, alpha={LORA_ALPHA})")
    model.enable_input_require_grads()  # diperlukan untuk gradient checkpointing
    lora_config = LoraConfig(
        r             = LORA_R,
        lora_alpha    = LORA_ALPHA,
        lora_dropout  = LORA_DROPOUT,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias          = "none",
        task_type     = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 3. Load dataset ───────────────────────────────────────────────────────
    print(f"\n[3/5] Loading dataset dari {DATA_DIR}")
    train_dataset = load_dataset(
        "json",
        data_files = os.path.join(DATA_DIR, "train.jsonl"),
        split      = "train",
    )
    val_dataset = load_dataset(
        "json",
        data_files = os.path.join(DATA_DIR, "val.jsonl"),
        split      = "train",
    )
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val  : {len(val_dataset)} samples")

    # Apply chat template ke dataset
    def apply_template(batch):
        texts = [
            tokenizer.apply_chat_template(
                msgs,
                tokenize       = False,
                add_generation_prompt = False,
            )
            for msgs in batch["messages"]
        ]
        return {"text": texts}

    train_dataset = train_dataset.map(apply_template, batched=True)
    val_dataset   = val_dataset.map(apply_template, batched=True)

    print(f"\n  Contoh formatted text (train[0] preview):")
    print(f"  {train_dataset[0]['text'][:300]}...")

    # ── 4. Konfigurasi Trainer ────────────────────────────────────────────────
    print(f"\n[4/5] Menyiapkan SFTTrainer...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = 2,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LR,
        warmup_ratio                = WARMUP_RATIO,
        weight_decay                = WEIGHT_DECAY,
        lr_scheduler_type           = "cosine",
        bf16                        = True,
        fp16                        = False,
        logging_steps               = 10,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        report_to                   = "none",    # Ganti ke "wandb" jika pakai wandb
        seed                        = 42,
        dataloader_num_workers      = 2,
    )

    trainer = SFTTrainer(
        model               = model,
        tokenizer           = tokenizer,
        train_dataset       = train_dataset,
        eval_dataset        = val_dataset,
        dataset_text_field  = "text",
        max_seq_length      = MAX_SEQ_LEN,
        packing             = False,
        args                = training_args,
    )

    # Train only on assistant responses (bukan prompt/system)
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "[INST]",
        response_part    = "[/INST]",
    )

    # ── 5. Training ───────────────────────────────────────────────────────────
    print(f"\n[5/5] Mulai training...")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Batch size   : {BATCH_SIZE} x {GRAD_ACCUM} accum = {BATCH_SIZE*GRAD_ACCUM} effective")
    print(f"  Learning rate: {LR}")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print()

    trainer_stats = trainer.train()

    # ── Simpan model ──────────────────────────────────────────────────────────
    print(f"\nMenyimpan LoRA adapter ke {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Simpan juga model yang sudah di-merge (untuk konversi ke GGUF)
    merged_dir = OUTPUT_DIR + "-merged"
    print(f"Merge LoRA ke base model → {merged_dir}")
    model.save_pretrained_merged(
        merged_dir,
        tokenizer,
        save_method = "merged_16bit",   # Simpan dalam fp16 untuk konversi GGUF
    )

    print(f"\n{'='*60}")
    print("Training selesai!")
    print(f"  LoRA adapter : {OUTPUT_DIR}")
    print(f"  Merged model : {merged_dir}")
    print(f"  Train loss   : {trainer_stats.training_loss:.4f}")
    print(f"{'='*60}")
    print("Lanjut ke step 4: bash 4_convert_gguf.sh")


if __name__ == "__main__":
    main()

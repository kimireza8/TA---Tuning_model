"""
STEP 3: Fine-tuning Mistral 7B dengan LoRA
Standard HuggingFace stack (tanpa unsloth) — stabil di semua environment.
Hardware target: RTX 4090 (24 GB VRAM) di Vast.ai

Jalankan: python 3_train.py
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME   = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR   = "/workspace/outputs/mistral-pens-lora"
DATA_DIR     = "/workspace/data"

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

MAX_SEQ_LEN  = 2048
BATCH_SIZE   = 4
GRAD_ACCUM   = 4
EPOCHS       = 3
LR           = 2e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01

# ══════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("Fine-tuning Mistral 7B dengan LoRA untuk PENS Journalism")
    print("=" * 60)
    print(f"  torch  : {torch.__version__}")
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 1. Load tokenizer & model ─────────────────────────────────────────────
    print(f"\n[1/5] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype      = torch.bfloat16,
        device_map       = "auto",
        attn_implementation = "eager",   # aman di semua environment
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    # ── 2. LoRA adapter ───────────────────────────────────────────────────────
    print(f"\n[2/5] Menambahkan LoRA adapter (r={LORA_R}, alpha={LORA_ALPHA})")
    lora_config = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias           = "none",
        task_type      = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 3. Load & format dataset ──────────────────────────────────────────────
    print(f"\n[3/5] Loading dataset dari {DATA_DIR}")
    train_dataset = load_dataset("json", data_files=os.path.join(DATA_DIR, "train.jsonl"), split="train")
    val_dataset   = load_dataset("json", data_files=os.path.join(DATA_DIR, "val.jsonl"),   split="train")
    print(f"  Train : {len(train_dataset)} samples")
    print(f"  Val   : {len(val_dataset)} samples")

    def apply_template(batch):
        texts = []
        for msgs in batch["messages"]:
            texts.append(tokenizer.apply_chat_template(
                msgs,
                tokenize              = False,
                add_generation_prompt = False,
            ))
        return {"text": texts}

    train_dataset = train_dataset.map(apply_template, batched=True)
    val_dataset   = val_dataset.map(apply_template, batched=True)

    print(f"\n  Preview train[0]:\n  {train_dataset[0]['text'][:200]}...")

    # ── 4. Training arguments ─────────────────────────────────────────────────
    print(f"\n[4/5] Menyiapkan trainer...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = SFTConfig(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = 2,
        gradient_accumulation_steps = GRAD_ACCUM,
        gradient_checkpointing      = True,
        learning_rate               = LR,
        warmup_steps                = 50,
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
        report_to                   = "none",
        seed                        = 42,
        dataloader_num_workers      = 2,
        dataset_text_field          = "text",
        max_length                  = MAX_SEQ_LEN,
        packing                     = False,
    )

    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        train_dataset    = train_dataset,
        eval_dataset     = val_dataset,
        args             = training_args,
    )

    # ── 5. Training ───────────────────────────────────────────────────────────
    print(f"\n[5/5] Mulai training...")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Batch eff.  : {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  LR          : {LR}")
    print()

    trainer_stats = trainer.train()

    # ── Simpan LoRA adapter ───────────────────────────────────────────────────
    print(f"\nMenyimpan LoRA adapter ke {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Merge LoRA ke base model untuk konversi GGUF
    merged_dir = OUTPUT_DIR + "-merged"
    print(f"Merge LoRA ke base model → {merged_dir}")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    print(f"\n{'=' * 60}")
    print("Training selesai!")
    print(f"  LoRA adapter : {OUTPUT_DIR}")
    print(f"  Merged model : {merged_dir}")
    print(f"  Train loss   : {trainer_stats.training_loss:.4f}")
    print(f"{'=' * 60}")
    print("Lanjut ke step 4: bash 4_convert_gguf.sh")


if __name__ == "__main__":
    main()

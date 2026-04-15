"""
STEP 2: Persiapan Dataset
Konversi train.json / val.json ke format JSONL yang siap dipakai training.
Dataset sudah dalam format chat (system/user/assistant) — tinggal dirapikan.

Jalankan: python 2_prepare_dataset.py
"""

import json
import os
from pathlib import Path

# ── Konfigurasi ──────────────────────────────────────────────────────────────
INPUT_DIR  = "/workspace/data"          # Tempat train.json & val.json di server
OUTPUT_DIR = "/workspace/data"
# ─────────────────────────────────────────────────────────────────────────────


def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_sample(sample: dict, idx: int) -> bool:
    """Pastikan setiap sample punya field messages dengan role yang benar."""
    if "messages" not in sample:
        print(f"  [SKIP] sample #{idx}: tidak ada field 'messages'")
        return False

    roles = [m["role"] for m in sample["messages"]]
    if "assistant" not in roles:
        print(f"  [SKIP] sample #{idx}: tidak ada role 'assistant'")
        return False

    for msg in sample["messages"]:
        if not msg.get("content", "").strip():
            print(f"  [SKIP] sample #{idx}: content kosong pada role '{msg['role']}'")
            return False

    return True


def process_split(input_path: str, output_path: str, split_name: str):
    data = load_json(input_path)
    print(f"\n[{split_name}] Loaded {len(data)} samples dari {input_path}")

    valid = []
    for i, sample in enumerate(data):
        if validate_sample(sample, i):
            # Pastikan urutan role: system (opsional) → user → assistant
            # Dataset Anda sudah benar, tapi kita normalisasi saja
            valid.append({"messages": sample["messages"]})

    print(f"[{split_name}] Valid: {len(valid)} / {len(data)} samples")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in valid:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[{split_name}] Disimpan ke: {output_path}")

    # Tampilkan 1 contoh
    print(f"\n--- Contoh sample pertama [{split_name}] ---")
    ex = valid[0]["messages"]
    for msg in ex:
        preview = msg["content"][:120].replace("\n", " ")
        print(f"  [{msg['role']}] {preview}...")
    print("---")

    return len(valid)


def main():
    train_in  = os.path.join(INPUT_DIR, "train.json")
    val_in    = os.path.join(INPUT_DIR, "val.json")
    train_out = os.path.join(OUTPUT_DIR, "train.jsonl")
    val_out   = os.path.join(OUTPUT_DIR, "val.jsonl")

    n_train = process_split(train_in, train_out, "TRAIN")
    n_val   = process_split(val_in,   val_out,   "VAL")

    print(f"\n{'='*50}")
    print(f"Dataset siap!")
    print(f"  Train : {n_train} samples  →  {train_out}")
    print(f"  Val   : {n_val} samples    →  {val_out}")
    print(f"{'='*50}")
    print("Lanjut ke step 3: python 3_train.py")


if __name__ == "__main__":
    main()

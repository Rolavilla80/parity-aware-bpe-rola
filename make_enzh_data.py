import argparse
import os
import random
from datasets import load_dataset

def safe_get_translation(ex):
    """
    Opus-100 examples usually look like:
      {"translation": {"en": "...", "zh": "..."}}
    but we'll be defensive.
    """
    if "translation" in ex and isinstance(ex["translation"], dict):
        en = ex["translation"].get("en")
        zh = ex["translation"].get("zh")
        return en, zh
    # fallback: sometimes datasets use different keys
    en = ex.get("en")
    zh = ex.get("zh")
    return en, zh

def write_split(ds, out_en, out_zh, n, seed=13, shuffle_buffer=50_000):
    """
    Stream examples and write first n valid pairs.
    We do lightweight shuffling by buffering examples (so we don't just take the dataset's first lines).
    """
    random.seed(seed)
    buffer = []
    written = 0

    def flush_buffer(buf):
        nonlocal written
        random.shuffle(buf)
        for ex in buf:
            if written >= n:
                return []
            en, zh = safe_get_translation(ex)
            if not en or not zh:
                continue
            en = en.strip().replace("\n", " ")
            zh = zh.strip().replace("\n", " ")
            if not en or not zh:
                continue
            out_en.write(en + "\n")
            out_zh.write(zh + "\n")
            written += 1
        return []

    for ex in ds:
        buffer.append(ex)
        if len(buffer) >= shuffle_buffer:
            buffer = flush_buffer(buffer)
            if written >= n:
                break

    if written < n and buffer:
        flush_buffer(buffer)

    return written

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data_enzh")
    ap.add_argument("--train_n", type=int, default=200_000)
    ap.add_argument("--dev_n", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--streaming", action="store_true", help="Use streaming to avoid downloading entire dataset")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading dataset Helsinki-NLP/opus-100 (config: en-zh) ...")
    # OPUS-100 provides splits; train is large, validation/test are smaller
    ds_train = load_dataset("Helsinki-NLP/opus-100", "en-zh", split="train", streaming=args.streaming)
    ds_dev   = load_dataset("Helsinki-NLP/opus-100", "en-zh", split="validation", streaming=args.streaming)

    train_en_path = os.path.join(args.out_dir, "train.en")
    train_zh_path = os.path.join(args.out_dir, "train.zh")
    dev_en_path   = os.path.join(args.out_dir, "dev.en")
    dev_zh_path   = os.path.join(args.out_dir, "dev.zh")

    with open(train_en_path, "w", encoding="utf-8") as f_en, open(train_zh_path, "w", encoding="utf-8") as f_zh:
        n_written = write_split(ds_train, f_en, f_zh, args.train_n, seed=args.seed)
        print(f"Wrote train pairs: {n_written} -> {train_en_path}, {train_zh_path}")

    with open(dev_en_path, "w", encoding="utf-8") as f_en, open(dev_zh_path, "w", encoding="utf-8") as f_zh:
        n_written = write_split(ds_dev, f_en, f_zh, args.dev_n, seed=args.seed + 1)
        print(f"Wrote dev pairs: {n_written} -> {dev_en_path}, {dev_zh_path}")

    print("Done.")

if __name__ == "__main__":
    main()
import csv
import os
from collections import Counter, defaultdict

from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers import pre_tokenizers

RUNS = {
    "none":   r"benchmark_outputs\merges.base.enzh.1000.none.newidea.txt",
    "max":    r"benchmark_outputs\merges.base.enzh.1000.max.newidea.txt",
    "local":  r"benchmark_outputs\merges.base.enzh.1000.local.newidea.txt",
    "rescan": r"benchmark_outputs\merges.base.enzh.1000.rescan.newidea.txt",
}

CORPORA = {
    "en_dev": r"data_enzh\dev.en",
    "zh_dev": r"data_enzh\dev.zh",
}

OUTPUT_CSV = r"benchmark_outputs\bpe_eval_summary_newidea.csv"

pre_tokenizer = pre_tokenizers.Sequence([
    Whitespace(),
    ByteLevel(use_regex=False),
])


# -----------------------------
# Byte-level UTF-8 helpers
# -----------------------------
def get_bytelevel_char_to_byte():
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))

    codepoints = byte_values[:]
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1

    return {chr(codepoint): byte for byte, codepoint in zip(byte_values, codepoints)}


BYTE_DECODER = get_bytelevel_char_to_byte()


def symbol_to_raw_bytes(symbol: str) -> bytes:
    return bytes(BYTE_DECODER[ch] for ch in symbol)


def valid_utf8_char_len(raw: bytes, i: int) -> int:
    for length in (1, 2, 3, 4):
        if i + length <= len(raw):
            try:
                raw[i:i+length].decode("utf-8")
                return length
            except UnicodeDecodeError:
                pass
    return 0


def is_valid_utf8_char_prefix(raw: bytes) -> bool:
    """
    True iff raw is a proper prefix of exactly one UTF-8 character.
    """
    n = len(raw)
    if n == 0 or n > 3:
        return False

    b0 = raw[0]

    if 0x00 <= b0 <= 0x7F:
        return False

    if n == 1:
        return (0xC2 <= b0 <= 0xDF) or (0xE0 <= b0 <= 0xEF) or (0xF0 <= b0 <= 0xF4)

    b1 = raw[1]

    def cont(x):
        return 0x80 <= x <= 0xBF

    if n == 2:
        if b0 == 0xE0:
            return 0xA0 <= b1 <= 0xBF
        if 0xE1 <= b0 <= 0xEC or 0xEE <= b0 <= 0xEF:
            return cont(b1)
        if b0 == 0xED:
            return 0x80 <= b1 <= 0x9F

        if b0 == 0xF0:
            return 0x90 <= b1 <= 0xBF
        if 0xF1 <= b0 <= 0xF3:
            return cont(b1)
        if b0 == 0xF4:
            return 0x80 <= b1 <= 0x8F

        return False

    if n == 3:
        b2 = raw[2]
        if not cont(b2):
            return False

        if b0 == 0xF0:
            return 0x90 <= b1 <= 0xBF
        if 0xF1 <= b0 <= 0xF3:
            return cont(b1)
        if b0 == 0xF4:
            return 0x80 <= b1 <= 0x8F

        return False

    return False


def utf8_symbol_state(symbol: str) -> str:
    """
    complete = valid UTF-8 string
    prefix   = full UTF-8 chars plus one unfinished char prefix at the end
    invalid  = broken structure
    """
    raw = symbol_to_raw_bytes(symbol)

    i = 0
    while i < len(raw):
        length = valid_utf8_char_len(raw, i)
        if length > 0:
            i += length
            continue

        return "prefix" if is_valid_utf8_char_prefix(raw[i:]) else "invalid"

    return "complete"


# -----------------------------
# BPE helpers
# -----------------------------
def load_bpe_ranks(merges_path: str):
    with open(merges_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    if lines and lines[0].startswith("#version"):
        lines = lines[1:]

    merges = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split(" ")
        if len(parts) != 2:
            continue
        merges.append(tuple(parts))

    return {pair: rank for rank, pair in enumerate(merges)}


def apply_bpe_to_pretoken(piece: str, bpe_ranks: dict):
    word = list(piece)
    if len(word) <= 1:
        return word

    while len(word) > 1:
        pairs = [
            (bpe_ranks[pair], i, pair)
            for i, pair in enumerate(zip(word, word[1:]))
            if pair in bpe_ranks
        ]

        if not pairs:
            break

        bigram = min(pairs)[2]
        positions = [i for _, i, pair in pairs if pair == bigram]

        i = 0
        new_word = []
        bigram_str = "".join(bigram)

        for j in positions:
            if j < i:
                continue
            new_word.extend(word[i:j])
            new_word.append(bigram_str)
            i = j + 2

        new_word.extend(word[i:])
        word = new_word

    return word


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_run_on_corpus(merges_path: str, corpus_path: str):
    bpe_ranks = load_bpe_ranks(merges_path)

    total_chars = 0
    total_final_tokens = 0

    unique_tokens = set()
    token_occurrences = Counter()

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            total_chars += len(line)

            pieces = [item[0] for item in pre_tokenizer.pre_tokenize_str(line)]

            for piece in pieces:
                if not piece:
                    continue
                final_tokens = apply_bpe_to_pretoken(piece, bpe_ranks)
                total_final_tokens += len(final_tokens)

                for tok in final_tokens:
                    unique_tokens.add(tok)
                    token_occurrences[tok] += 1

    type_counter = Counter()
    occ_counter = Counter()

    for tok in unique_tokens:
        cls = utf8_symbol_state(tok)
        type_counter[cls] += 1

    for tok, count in token_occurrences.items():
        cls = utf8_symbol_state(tok)
        occ_counter[cls] += count

    tokens_per_char = (total_final_tokens / total_chars) if total_chars else 0.0

    return {
        "total_chars": total_chars,
        "total_final_tokens": total_final_tokens,
        "tokens_per_char": tokens_per_char,

        "unique_token_types": len(unique_tokens),
        "unique_complete": type_counter["complete"],
        "unique_prefix": type_counter["prefix"],
        "unique_invalid": type_counter["invalid"],

        "token_occ_complete": occ_counter["complete"],
        "token_occ_prefix": occ_counter["prefix"],
        "token_occ_invalid": occ_counter["invalid"],
    }


def main():
    rows = []

    for run_name, merges_path in RUNS.items():
        if not os.path.exists(merges_path):
            print(f"Missing merges file: {merges_path}")
            continue

        for corpus_name, corpus_path in CORPORA.items():
            if not os.path.exists(corpus_path):
                print(f"Missing corpus file: {corpus_path}")
                continue

            result = evaluate_run_on_corpus(merges_path, corpus_path)
            row = {
                "run": run_name,
                "corpus": corpus_name,
                **result
            }
            rows.append(row)

        combined = defaultdict(float)
        combined["run"] = run_name
        combined["corpus"] = "combined"

        tmp_results = []
        for corpus_name, corpus_path in CORPORA.items():
            if os.path.exists(corpus_path):
                tmp_results.append(evaluate_run_on_corpus(merges_path, corpus_path))

        if tmp_results:
            combined["total_chars"] = sum(r["total_chars"] for r in tmp_results)
            combined["total_final_tokens"] = sum(r["total_final_tokens"] for r in tmp_results)
            combined["tokens_per_char"] = (
                combined["total_final_tokens"] / combined["total_chars"]
                if combined["total_chars"] else 0.0
            )

            combined["unique_token_types"] = ""
            combined["unique_complete"] = ""
            combined["unique_prefix"] = ""
            combined["unique_invalid"] = ""

            combined["token_occ_complete"] = sum(r["token_occ_complete"] for r in tmp_results)
            combined["token_occ_prefix"] = sum(r["token_occ_prefix"] for r in tmp_results)
            combined["token_occ_invalid"] = sum(r["token_occ_invalid"] for r in tmp_results)

            rows.append(dict(combined))

    fieldnames = [
        "run", "corpus",
        "total_chars", "total_final_tokens", "tokens_per_char",
        "unique_token_types", "unique_complete", "unique_prefix", "unique_invalid",
        "token_occ_complete", "token_occ_prefix", "token_occ_invalid",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary to {OUTPUT_CSV}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
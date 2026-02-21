#!/usr/bin/env python3
"""Run tiny parity-aware/classical BPE smoke test end-to-end."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    train_en = root / "data_tiny" / "train.en"
    train_de = root / "data_tiny" / "train.de"
    dev_en = root / "data_tiny" / "dev.en"
    dev_de = root / "data_tiny" / "dev.de"

    for p in [train_en, train_de, dev_en, dev_de]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    parity_merges = root / "merges.parity.txt"
    classic_merges = root / "merges.classic.txt"
    concat_train = root / "data_tiny" / "train.concat"

    print("[1/4] Training parity-aware BPE...")
    run([
        "python",
        "parity_aware_bpe/parity_aware_learn_bpe.py",
        "--symbols",
        "100",
        "--variant",
        "base",
        "--input",
        str(train_en),
        str(train_de),
        "--dev",
        str(dev_en),
        str(dev_de),
        "--output",
        str(parity_merges),
    ])

    print("[2/4] Creating baseline concat file...")
    concat_train.write_text(train_en.read_text(encoding="utf-8") + train_de.read_text(encoding="utf-8"), encoding="utf-8")

    print("[3/4] Training classical BPE baseline...")
    run([
        "python",
        "parity_aware_bpe/learn_bpe.py",
        "--symbols",
        "100",
        "--input",
        str(concat_train),
        "--output",
        str(classic_merges),
    ])

    print("[4/4] Checking outputs...")
    parity_lines = len(parity_merges.read_text(encoding="utf-8").splitlines())
    classic_lines = len(classic_merges.read_text(encoding="utf-8").splitlines())
    print(f"merges.parity.txt lines: {parity_lines}")
    print(f"merges.classic.txt lines: {classic_lines}")

    if parity_lines < 1 or classic_lines < 1:
        raise RuntimeError("One of the output merge files is empty.")

    print("Done. Tiny smoke test passed.")


if __name__ == "__main__":
    main()

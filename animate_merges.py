#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stepwise viewer for BPE merges: Next/Back buttons to see how dev lengths change over merges.

Run (from repo root):
  python stepwise_merges_viewer.py ^
    --dev_en .\data_tiny\dev.en ^
    --dev_de .\data_tiny\dev.de ^
    --merges .\outputs\merges_pabpe_2k_min1.txt ^
    --max_merges 250
"""

import argparse
import sys
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers import pre_tokenizers


def build_vocab(path: str, pretokenize=("whitespace", "bytelevel")) -> Counter:
    pretokenizer_list = []
    for p in pretokenize:
        if p == "whitespace":
            pretokenizer_list.append(Whitespace())
        elif p == "bytelevel":
            pretokenizer_list.append(ByteLevel(use_regex=False))
        else:
            raise ValueError(f"Unknown pretokenizer: {p}")
    pre_tok = pre_tokenizers.Sequence(pretokenizer_list)

    vocab = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            toks = [item[0] for item in pre_tok.pre_tokenize_str(line)]
            for t in toks:
                if t:
                    vocab[t] += 1
    return vocab


def to_symbol_vocab(counter: Counter) -> dict:
    # token string -> tuple of characters
    return {tuple(token): count for token, count in counter.items()}


def load_merges(path: str):
    merges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            merges.append((parts[0], parts[1]))
    return merges


def replace_pair_in_word(word_tuple, first, second):
    new = []
    i = 0
    while i < len(word_tuple):
        if i < len(word_tuple) - 1 and word_tuple[i] == first and word_tuple[i + 1] == second:
            new.append(first + second)
            i += 2
        else:
            new.append(word_tuple[i])
            i += 1
    return tuple(new)


def apply_merge_to_dev_vocab(dev_vocab: dict, pair):
    """
    dev_vocab: token_tuple -> np.array([count_en, count_de])
    returns: length_change vector np.array([delta_en, delta_de])
    """
    first, second = pair
    length_change = np.zeros(2, dtype=int)

    updates = []
    for word, freq_vec in dev_vocab.items():
        # quick adjacency test
        found = False
        for a, b in zip(word, word[1:]):
            if a == first and b == second:
                found = True
                break
        if not found:
            continue

        new_word = replace_pair_in_word(word, first, second)
        if new_word != word:
            delta_len = (len(word) - len(new_word))
            length_change += delta_len * freq_vec
            updates.append((word, new_word, freq_vec))

    for old_word, new_word, freq_vec in updates:
        del dev_vocab[old_word]
        dev_vocab[new_word] = dev_vocab.get(new_word, np.zeros(2, dtype=int)) + freq_vec

    return length_change


def precompute_history(dev_en_path, dev_de_path, merges_path, max_merges):
    v_en = build_vocab(dev_en_path)
    v_de = build_vocab(dev_de_path)

    sv_en = to_symbol_vocab(v_en)
    sv_de = to_symbol_vocab(v_de)

    all_keys = set(sv_en.keys()) | set(sv_de.keys())
    dev_vocab = {k: np.array([sv_en.get(k, 0), sv_de.get(k, 0)], dtype=int) for k in all_keys}

    lengths = np.zeros(2, dtype=int)
    for tok, freq_vec in dev_vocab.items():
        lengths += len(tok) * freq_vec

    merges = load_merges(merges_path)
    if not merges:
        raise RuntimeError("No merges found in merges file (empty or wrong path).")

    merges = merges[: min(max_merges, len(merges))]

    hist_en = [int(lengths[0])]
    hist_de = [int(lengths[1])]
    hist_ratio = [float(lengths[1] / lengths[0]) if lengths[0] else float("nan")]

    dv = dict(dev_vocab)  # working copy
    for pair in merges:
        delta = apply_merge_to_dev_vocab(dv, pair)
        lengths = lengths - delta
        hist_en.append(int(lengths[0]))
        hist_de.append(int(lengths[1]))
        hist_ratio.append(float(lengths[1] / lengths[0]) if lengths[0] else float("nan"))

    return merges, np.array(hist_en), np.array(hist_de), np.array(hist_ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_en", required=True)
    ap.add_argument("--dev_de", required=True)
    ap.add_argument("--merges", required=True)
    ap.add_argument("--max_merges", type=int, default=300)
    ap.add_argument("--play_interval_ms", type=int, default=200)
    args = ap.parse_args()

    merges, hist_en, hist_de, hist_ratio = precompute_history(
        args.dev_en, args.dev_de, args.merges, args.max_merges
    )

    # Step index: 0 = no merges applied, 1 = after first merge, etc.
    state = {"i": 0, "playing": False}
    n = len(hist_en) - 1  # number of merges we precomputed

    # --- Plot setup ---
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.22)  # make space for buttons

    ax.set_title("Stepwise BPE merge viewer (dev length)")
    ax.set_xlabel("Merge step")
    ax.set_ylabel("Total dev length (symbols)")

    x = np.arange(len(hist_en))
    line_en, = ax.plot([], [], label="EN dev length")
    line_de, = ax.plot([], [], label="DE dev length")
    ax.legend(loc="upper right")

    ax2 = ax.twinx()
    ax2.set_ylabel("DE/EN length ratio")
    line_ratio, = ax2.plot([], [], linestyle="--", label="DE/EN ratio")
    ax2.legend(loc="upper left")

    # reasonable y-limits
    ax.set_xlim(0, len(hist_en) - 1)
    ax.set_ylim(min(hist_en.min(), hist_de.min()) * 0.95, max(hist_en.max(), hist_de.max()) * 1.05)
    finite = hist_ratio[np.isfinite(hist_ratio)]
    if finite.size:
        ax2.set_ylim(finite.min() * 0.95, finite.max() * 1.05)

    info = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    def render():
        i = state["i"]
        line_en.set_data(x[: i + 1], hist_en[: i + 1])
        line_de.set_data(x[: i + 1], hist_de[: i + 1])
        line_ratio.set_data(x[: i + 1], hist_ratio[: i + 1])

        if i == 0:
            info.set_text(
                f"step=0 (no merges applied)\n"
                f"EN={hist_en[i]}  DE={hist_de[i]}  ratio={hist_ratio[i]:.3f}"
            )
        else:
            a, b = merges[i - 1]
            info.set_text(
                f"step={i}/{n}  last merge: '{a}' + '{b}' → '{a+b}'\n"
                f"EN={hist_en[i]}  DE={hist_de[i]}  ratio={hist_ratio[i]:.3f}"
            )
        fig.canvas.draw_idle()

    render()

    # --- Buttons ---
    ax_prev = plt.axes([0.10, 0.05, 0.10, 0.08])
    ax_next = plt.axes([0.21, 0.05, 0.10, 0.08])
    ax_play = plt.axes([0.34, 0.05, 0.10, 0.08])
    ax_pause = plt.axes([0.45, 0.05, 0.10, 0.08])
    ax_reset = plt.axes([0.58, 0.05, 0.10, 0.08])

    btn_prev = Button(ax_prev, "Back")
    btn_next = Button(ax_next, "Next")
    btn_play = Button(ax_play, "Play")
    btn_pause = Button(ax_pause, "Pause")
    btn_reset = Button(ax_reset, "Reset")

    def on_prev(_):
        state["playing"] = False
        state["i"] = max(0, state["i"] - 1)
        render()

    def on_next(_):
        state["playing"] = False
        state["i"] = min(n, state["i"] + 1)
        render()

    def on_play(_):
        state["playing"] = True

    def on_pause(_):
        state["playing"] = False

    def on_reset(_):
        state["playing"] = False
        state["i"] = 0
        render()

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_play.on_clicked(on_play)
    btn_pause.on_clicked(on_pause)
    btn_reset.on_clicked(on_reset)

    # Timer for Play mode
    timer = fig.canvas.new_timer(interval=args.play_interval_ms)

    def tick():
        if state["playing"]:
            if state["i"] < n:
                state["i"] += 1
                render()
            else:
                state["playing"] = False  # stop at end

    timer.add_callback(tick)
    timer.start()

    plt.show()


if __name__ == "__main__":
    main()
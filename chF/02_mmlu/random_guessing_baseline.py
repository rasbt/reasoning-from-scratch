# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import random
import statistics as stats
from collections import Counter

from datasets import load_dataset


# Gold letter is MMLU jargon for correct answer letter
def gold_letter(ans):
    if isinstance(ans, int):
        return "ABCD"[ans]
    s = str(ans).strip().upper()
    return s if s in {"A", "B", "C", "D"} else s[:1]


def main():
    parser = argparse.ArgumentParser(
        description="Show gold answer distribution for an MMLU subset and a random-guess baseline."
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="high_school_mathematics",
        help="MMLU subset name (default: 'high_school_mathematics').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the random-guess baseline (default: 42).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10_000,
        help="Number of random-guess trials (default: 10,000).",
    )
    args = parser.parse_args()

    ds = load_dataset("cais/mmlu", args.subset, split="test")

    labels = [gold_letter(ex["answer"]) for ex in ds]
    n = len(labels)
    counts = Counter(labels)

    print(f"Subset: {args.subset} | split: test | n={n}")
    print("Gold distribution provided in the dataset:")
    for letter in "ABCD":
        c = counts.get(letter, 0)
        pct = (c / n) if n else 0.0
        print(f"  {letter}: {c} ({pct:.2%})")

    if n == 0:
        print("\nNo items. Baseline undefined.")
        return

    # Repeat random guessing
    rng = random.Random(args.seed)
    accs = []
    for _ in range(args.trials):
        guesses = [rng.choice("ABCD") for _ in range(n)]
        correct = sum(1 for g, y in zip(guesses, labels) if g == y)
        accs.append(correct / n)

    mean_acc = stats.mean(accs)
    sd_acc = stats.stdev(accs) if len(accs) > 1 else 0.0

    print(f"\nRandom guessing over {args.trials:,} trials (uniform A/B/C/D, seed={args.seed}):")
    print(f"  Mean accuracy: {mean_acc:.2%}")
    print(f"  Std dev across trials: {sd_acc:.2%}")

    # Quantiles
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    accs_sorted = sorted(accs)
    print("\nSelected quantiles of accuracy:")
    for q in qs:
        idx = int(q * len(accs_sorted))
        print(f"  {q:.0%} quantile: {accs_sorted[idx]:.3%}")

    # Frequency table (rounded)
    acc_counts = Counter(round(a, 2) for a in accs)
    print("\nFull frequency table of accuracies (rounded):")
    for acc_val in sorted(acc_counts):
        freq = acc_counts[acc_val]
        pct = freq / args.trials
        print(f"  {acc_val:.3f}: {freq} times ({pct:.2%})")


if __name__ == "__main__":
    main()

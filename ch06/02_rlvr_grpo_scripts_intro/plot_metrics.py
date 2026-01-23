#!/usr/bin/env python3
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def load_metrics(path):
    steps, losses, rewards, avg_lens, eval_steps, eval_accs = [], [], [], [], [], []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return steps, losses, rewards, avg_lens, eval_steps, eval_accs
        if first and first[0].strip() == "step":
            rows = reader
        else:
            rows = [first, *reader]
        for row in rows:
            if not row:
                continue
            step = int(row[0])
            steps.append(step)
            losses.append(float(row[2]))
            rewards.append(float(row[3]))
            avg_lens.append(float(row[5]))
            if len(row) > 6 and row[6].strip():
                eval_steps.append(step)
                eval_accs.append(float(row[6]))
    return steps, losses, rewards, avg_lens, eval_steps, eval_accs


def moving_average(values, window):
    if window <= 1:
        return values
    out = []
    csum = 0.0
    for i, val in enumerate(values):
        csum += val
        if i >= window:
            csum -= values[i - window]
            out.append(csum / window)
        else:
            out.append(csum / (i + 1))
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics.")
    parser.add_argument(
        "--csv",
        required=True,
        help="Metrics CSV file to plot.",
    )
    parser.add_argument("--save_plot", help="Optional output image path.")
    parser.add_argument(
        "--moving_average",
        type=int,
        default=0,
        help="Moving average window (default: 0 disables).",
    )
    args = parser.parse_args()

    steps, losses, rewards, avg_lens, eval_steps, eval_accs = load_metrics(
        Path(args.csv)
    )
    ma_window = args.moving_average
    losses_ma = moving_average(losses, ma_window)
    rewards_ma = moving_average(rewards, ma_window)
    avg_lens_ma = moving_average(avg_lens, ma_window)
    eval_accs_ma = moving_average(eval_accs, ma_window)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7.2))
    ax_loss, ax_reward, ax_len, ax_eval = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    ax_loss.plot(steps, losses, linewidth=1.2, alpha=0.7, label="Actual")
    ax_reward.plot(steps, rewards, linewidth=1.2, alpha=0.7)
    ax_len.plot(steps, avg_lens, linewidth=1.2, alpha=0.7)
    has_eval = bool(eval_steps) and bool(eval_accs)
    if has_eval:
        ax_eval.plot(eval_steps, eval_accs, linewidth=1.2, alpha=0.7)
    if ma_window > 1:
        ax_loss.plot(
            steps, losses_ma, linewidth=1.8, linestyle="--", label="Moving average"
        )
        ax_reward.plot(steps, rewards_ma, linewidth=1.8, linestyle="--")
        ax_len.plot(steps, avg_lens_ma, linewidth=1.8, linestyle="--")
        if has_eval:
            ax_eval.plot(eval_steps, eval_accs_ma, linewidth=1.8, linestyle="--")

    ax_loss.set_ylabel("Loss")
    ax_reward.set_ylabel("Reward Average")
    ax_len.set_ylabel("Average response length")
    if has_eval:
        ax_eval.set_ylabel("Eval accuracy")
        ax_eval.set_xlabel("Step")
    else:
        ax_eval.axis("off")
    ax_len.set_xlabel("Step")
    if steps:
        min_step = min(steps)
        max_step = max(steps)
        for ax in axes.ravel():
            if ax is not ax_eval or has_eval:
                ax.set_xlim(min_step, max_step)

    for ax in axes.ravel():
        if ax is ax_eval and not has_eval:
            continue
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_loss.legend(frameon=False, loc="lower left")
    fig.suptitle("RLVR GRPO Training Metrics")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    save_path = args.save_plot
    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()

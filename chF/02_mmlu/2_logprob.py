# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import time

import torch
from datasets import load_dataset, get_dataset_config_names
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import load_model_and_tokenizer


# Same as in main notebook
def format_prompt(example):
    return (
        f"{example['question']}\n"
        f"A. {example['choices'][0]}\n"
        f"B. {example['choices'][1]}\n"
        f"C. {example['choices'][2]}\n"
        f"D. {example['choices'][3]}\n"
        "Answer: "  # trailing space encourages a single-letter next token
    )


def common_prefix_len(a, b):
    i = 0
    n = min(len(a), len(b))
    while i < n and a[i] == b[i]:
        i += 1
    return i


def first_new_token_id(tokenizer, prompt, prompt_ids, continuation):
    ids_full = tokenizer.encode(prompt + continuation)
    j = common_prefix_len(ids_full, prompt_ids)
    if j >= len(ids_full):
        raise ValueError("Continuation produced no new tokens.")
    return ids_full[j]


def predict_choice_logprobs(model, tokenizer, prompt_fmt, prompt, prompt_ids, example):
    with torch.no_grad():
        logits = model(prompt_fmt)  # [1, T, V]
        next_logp = torch.log_softmax(logits[0, -1], dim=-1)

    scores = {}
    for letter in "ABCD":
        tok_id = first_new_token_id(tokenizer, prompt, prompt_ids, letter)
        scores[letter] = next_logp[tok_id].item()

    pred = max(scores, key=scores.get)
    return pred, scores


def evaluate_mmlu_logprobs(
    model,
    tokenizer,
    device,
    subsets="high_school_mathematics",  # str, list of str, or "all"
    split="test",
    verbose_every=50,
):
    if subsets == "all":
        subset_list = get_dataset_config_names("cais/mmlu")
    elif isinstance(subsets, str):
        subset_list = [s.strip() for s in subsets.split(",")] if "," in subsets else [subsets]
    else:
        subset_list = list(subsets)

    total = 0
    correct = 0
    start = time.time()

    for subset in subset_list:
        ds = load_dataset("cais/mmlu", subset, split=split)
        for ex in ds:
            prompt = format_prompt(ex)
            prompt_ids = tokenizer.encode(prompt)
            prompt_fmt = torch.tensor(prompt_ids, device=device).unsqueeze(0)

            pred, _scores = predict_choice_logprobs(
                model, tokenizer, prompt_fmt, prompt, prompt_ids, ex
            )

            ans = ex["answer"]
            # "Gold" is the MMLU jargon for the correct answer (ground truth)
            gold = "ABCD"[ans] if isinstance(ans, int) else str(ans).strip().upper()

            total += 1
            correct += int(pred == gold)

            if verbose_every and total % verbose_every == 0:
                print(f"MMLU {total} acc={correct/total:.3f} [{subset}]")

    acc = correct / max(1, total)
    print(
        f"\nMMLU letter accuracy (log-prob): {correct}/{total} = {acc:.2%} "
        f"in {time.time()-start:.1f}s"
    )
    return {"accuracy": acc, "num_examples": total, "subsets": subset_list, "split": split}


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot MMLU evaluator via next-token log-prob scoring (A/B/C/D)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto' (default), or any torch device string like "
             "'cpu', 'cuda', 'cuda:0', 'mps'.",
    )
    parser.add_argument(
        "--which_model",
        type=str,
        default="base",
        choices=["base", "reasoning"],
        help="Model variant to load. Defaults to 'base'.",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default="high_school_mathematics",
        help="Comma-separated subset names or 'all'. "
             "Default: 'high_school_mathematics'.",
    )
    args = parser.parse_args()

    device = get_device() if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}")
    model, tokenizer = load_model_and_tokenizer(args.which_model, device, use_compile=False)
    model.eval()
    torch.set_float32_matmul_precision("high")

    metrics = evaluate_mmlu_logprobs(
        model=model,
        tokenizer=tokenizer,
        device=device,
        subsets=args.subsets,
    )
    print(metrics)


if __name__ == "__main__":
    main()

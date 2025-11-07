import argparse
import time

import torch
from datasets import load_dataset, get_dataset_config_names
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch02_ex import generate_text_basic_stream_cache
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


# Same as in main notebook
def predict_choice(
    model, tokenizer, prompt_fmt, max_new_tokens=8
):
    pred = None
    for t in generate_text_basic_stream_cache(
        model=model,
        token_ids=prompt_fmt,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    ):
        answer = tokenizer.decode(t.squeeze(0).tolist())
        for letter in answer:
            letter = letter.upper()
            if letter in "ABCD":
                pred = letter
                break
        if pred:  # stop as soon as a letter appears
            break
    return pred


def evaluate_mmlu_letter(
    model,
    tokenizer,
    device,
    subsets="high_school_mathematics",  # str, list of str, or "all"
    split="test",
    max_new_tokens=8,
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
            tok = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
            pred = predict_choice(model, tokenizer, tok, max_new_tokens)

            ans = ex["answer"]
            # "Gold" is the MMLU jargon for the correct answer (ground truth)
            gold = "ABCD"[ans] if isinstance(ans, int) else str(ans).strip().upper()

            total += 1
            correct += int(pred == gold)

            if verbose_every and total % verbose_every == 0:
                print(f"MMLU {total} acc={correct/total:.3f} [{subset}]")

    acc = correct / max(1, total)
    print(
        f"\nMMLU letter accuracy: {correct}/{total} = {acc:.2%} "
        f"in {time.time()-start:.1f}s"
    )
    return {"accuracy": acc, "num_examples": total, "subsets": subset_list, "split": split}


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot MMLU letter evaluator (A/B/C/D matching)."
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

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(args.which_model, device, use_compile=False)
    model.eval()
    torch.set_float32_matmul_precision("high")

    metrics = evaluate_mmlu_letter(
        model=model,
        tokenizer=tokenizer,
        device=device,
        subsets=args.subsets,
    )
    print(metrics)


if __name__ == "__main__":
    main()

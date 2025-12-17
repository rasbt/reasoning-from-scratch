# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import json
from pathlib import Path
import time

import torch
from collections import Counter

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    load_math500_test,
    eta_progress_message,
    render_prompt,
    grade_answer,
    load_model_and_tokenizer,
    extract_final_candidate,
)
from reasoning_from_scratch.ch04 import (
    top_p_filter,
    scale_logits_by_temperature,
)
from reasoning_from_scratch.qwen3 import KVCache


@torch.inference_mode()
def generate_text_top_p_batched(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=0.0,
    top_p=None,
    seed=None,
    seed_offset=0,
):
    """Batched variant of top-p sampling used for self-consistency."""

    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    device = token_ids.device
    batch_size = token_ids.size(0)
    generated_tokens = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    rngs = None
    if seed is not None and temperature not in (None, 1.0):
        rngs = []
        for idx in range(batch_size):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed + seed_offset + idx + 1)
            rngs.append(gen)

    out = model(token_ids, cache=cache)[:, -1]

    for _ in range(max_new_tokens):
        if temperature is None or temperature == 0.0:
            next_token = torch.argmax(out, dim=-1, keepdim=True)
            if eos_token_id is not None and torch.any(finished):
                fill = torch.full_like(next_token, eos_token_id)
                next_token = torch.where(finished.view(-1, 1), fill, next_token)
        else:
            logits = scale_logits_by_temperature(out, temperature)
            probas = torch.softmax(logits, dim=-1)
            probas = top_p_filter(probas, top_p)
            probas_cpu = probas.cpu()
            sampled = []
            for idx in range(batch_size):
                if finished[idx]:
                    fill_value = eos_token_id if eos_token_id is not None else 0
                    sampled.append(torch.tensor([fill_value], dtype=torch.long))
                    continue
                generator = rngs[idx] if rngs is not None else None
                sampled.append(
                    torch.multinomial(
                        probas_cpu[idx],
                        num_samples=1,
                        generator=generator,
                    )
                )
            next_token = torch.stack(sampled, dim=0).to(device)

        eos_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if eos_token_id is not None:
            eos_mask = next_token.squeeze(1) == eos_token_id

        for idx in range(batch_size):
            if finished[idx]:
                continue
            token_val = next_token[idx, 0].item()
            if eos_token_id is not None and token_val == eos_token_id:
                finished[idx] = True
            else:
                generated_tokens[idx].append(token_val)

        if eos_token_id is not None:
            finished = finished | eos_mask

        if torch.all(finished):
            break

        out = model(next_token, cache=cache)[:, -1]

    return generated_tokens


def self_consistency_vote(
    model,
    tokenizer,
    prompt,
    device,
    num_samples=10,
    temperature=0.8,
    top_p=0.9,
    max_new_tokens=2048,
    show_progress=True,
    show_long_answer=False,
    seed=None,
):
    full_answers, short_answers = [], []
    counts = Counter()
    groups = {}
    majority_winners, final_answer = [], None

    batch_size = max(1, num_samples)
    prompt_ids = torch.tensor(
        tokenizer.encode(prompt), device=device, dtype=torch.long
    ).unsqueeze(0)

    while len(full_answers) < num_samples:
        remaining = num_samples - len(full_answers)
        current_batch = min(batch_size, remaining)
        token_batch = prompt_ids.repeat(current_batch, 1)

        token_lists = generate_text_top_p_batched(
            model=model,
            token_ids=token_batch,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            seed_offset=len(full_answers),
        )
        answers = [tokenizer.decode(tokens) for tokens in token_lists]

        for answer in answers:
            sample_idx = len(full_answers)
            short = extract_final_candidate(answer, fallback="number_then_full")
            full_answers.append(answer)
            short_answers.append(short)
            counts[short] += 1
            groups.setdefault(short, []).append(sample_idx)

            if show_long_answer:
                print(answer)

            if show_progress:
                print(f"[Sample {sample_idx+1}/{num_samples}] → {short!r}")

    if final_answer is None:
        mc = counts.most_common()
        if mc:
            top_freq = mc[0][1]
            majority_winners = [s for s, f in mc if f == top_freq]
            final_answer = mc[0][0] if len(majority_winners) == 1 else None

    return {
        "full_answers": full_answers,
        "short_answers": short_answers,
        "counts": dict(counts),
        "groups": groups,
        "majority_winners": majority_winners,
        "final_answer": final_answer,
    }


def evaluate_math500_stream(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=2048,
    verbose=False,
    prompt_suffix="",
    temperature=1.0,
    top_p=1.0,
    seed=None,
    num_samples=10,
):

    if out_path is None:
        dev_name = str(device).replace(":", "-")
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])

            prompt += prompt_suffix
            results = self_consistency_vote(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                num_samples=num_samples,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                show_progress=False,
                show_long_answer=False,
                seed=seed,
            )

            # If final_answer was not determined (tie),
            # resolve it by first appearance
            if results["final_answer"] is None:
                extracted = results["majority_winners"][0]
            else:
                extracted = results["final_answer"]

            # extracted = extract_final_candidate(
            #     gen_text
            # )

            # Optionally, get long answer
            long_answer = None
            if extracted is not None:
                for idx, s in enumerate(results["short_answers"]):
                    if s == extracted:
                        long_answer = results["full_answers"][idx]
                        break
            gen_text = long_answer
            ###################################################################

            is_correct = grade_answer(
                extracted, row["answer"]
            )
            num_correct += int(is_correct)

            record = {
                "index": i,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": gen_text,
                "extracted": extracted,
                "correct": bool(is_correct),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            progress_msg = eta_progress_message(
                processed=i,
                total=num_examples,
                start_time=start_time,
                show_eta=True,
                label="MATH-500",
            )
            print(progress_msg, end="\r", flush=True)
            if verbose:
                print(
                    f"\n\n{'='*50}\n{progress_msg}\n"
                    f"{'='*50}\nExtracted: {extracted}\n"
                    f"Expected:  {row['answer']}\n"
                    f"Correct so far: {num_correct}\n{'-'*50}"
                )

    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(f"\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
    print(f"Total time: {seconds_elapsed/60:.1f} min")
    print(f"Logs written to: {out_path}")
    return num_correct, num_examples, acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto' (default), or any torch device string like 'cpu', 'cuda', 'cuda:0', 'mps'.",
    )
    parser.add_argument(
        "--which_model",
        type=str,
        default="base",
        choices=["base", "reasoning", "instruct"],
        help="Model variant to load. Defaults to 'base'.",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=10,
        help="Number of MATH-500 examples to evaluate. Default: 10",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max new tokens for generation. Default: 2048",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample correctness while evaluating.",
    )
    parser.add_argument(
        "--prompt_suffix",
        type=str,
        default="/n/nExplain step by step.",
        help="Adds a chain-of-thought prompt (default: '/n/nExplain step by step.')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for self-consistency sampling",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Setting for temperature scaling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Threshold for top-p filtering (nucleus sampling)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples for self-consistency sampling",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    which_model = args.which_model
    dataset_size = args.dataset_size
    max_new_tokens = args.max_new_tokens
    use_compile = args.compile

    print("Model:", which_model)
    print("Device:", device)
    dev_name = str(device).replace(":", "-")

    math_data = load_math500_test()

    if args.which_model == "instruct":
        which_model = "reasoning"
    else:
        which_model = args.which_model

    model, tokenizer = load_model_and_tokenizer(
        which_model=which_model,
        device=device,
        use_compile=args.compile
    )
    if args.which_model == "instruct":
        tokenizer.add_thinking = False

    model.eval()
    torch.set_float32_matmul_precision("high")

    num_correct, num_examples, acc = evaluate_math500_stream(
        model=model,
        out_path=f"math500_{which_model}-{dev_name}-evaluate-script.jsonl",
        tokenizer=tokenizer,
        device=device,
        math_data=math_data[:dataset_size],
        max_new_tokens=max_new_tokens,
        verbose=args.verbose,
        prompt_suffix=args.prompt_suffix,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        num_samples=args.num_samples,
    )

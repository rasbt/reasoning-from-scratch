# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import json
import time
from functools import partial
from pathlib import Path

import requests
import torch

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    render_prompt,
    extract_final_candidate,
    grade_answer,
    load_model_and_tokenizer,
    eta_progress_message,
)
from reasoning_from_scratch.ch05 import (
    heuristic_score,
    avg_logprob_answer,
    self_refinement_loop
)


@torch.inference_mode()
def avg_logprob_answer_extract(model, tokenizer, prompt, answer, device="cpu"):

    cand = extract_final_candidate(answer, fallback="none")
    if not cand:
        cand = extract_final_candidate(answer, fallback="number_only")
    if not cand:
        return -1e9

    answer = "\n\\boxed{" + cand.strip() + "}"

    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    full_ids = torch.tensor(prompt_ids + answer_ids, device=device)

    logits = model(full_ids.unsqueeze(0)).squeeze(0)  # [seq_len, vocab]
    logprobs = torch.log_softmax(logits, dim=-1)

    start = len(prompt_ids) - 1
    end = full_ids.shape[0] - 1

    # positions we score
    t_idx = torch.arange(start, end, device=device)

    # true next tokens
    next_ids = full_ids[start + 1:end + 1]

    step_logps = logprobs[t_idx, next_ids]

    if step_logps.numel() == 0:
        return -1e9

    return torch.mean(step_logps).item()


def evaluate_math500_stream(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=2048,
    verbose=False,
    iterations=2,
    critique_max_new_tokens=192,
    seed=None,
    temperature=0.7,
    top_p=0.9,
    scoring="heuristic",
    prompt_suffix="",
):
    if out_path is None:
        dev_name = str(device).replace(":", "-")
        out_path = Path(
            f"math500-{dev_name}-self-refine-{scoring}.jsonl"
        )

    num_examples = len(math_data)
    num_correct = 0
    start_time = time.time()

    if seed is not None:
        torch.manual_seed(seed)

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(math_data, start=1):
            question = row["problem"]

            if scoring == "heuristic":
                score_fn = heuristic_score
            elif scoring == "logprob":
                prompt_for_score = render_prompt(question) + prompt_suffix
                score_fn = partial(
                    avg_logprob_answer,
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_for_score,
                    device=device,
                )
            elif scoring == "logprob_extract":
                prompt_for_score = render_prompt(question) + prompt_suffix
                score_fn = partial(
                    avg_logprob_answer_extract,
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_for_score,
                    device=device,
                )
            elif scoring == "none":
                score_fn = None
            else:
                raise ValueError(f"Unknown scoring method: {scoring}")

            result = self_refinement_loop(
                model=model,
                tokenizer=tokenizer,
                raw_prompt=question,
                device=device,
                iterations=iterations,
                max_response_tokens=max_new_tokens,
                max_critique_tokens=critique_max_new_tokens,
                score_fn=score_fn,
                prompt_renderer=render_prompt,
                prompt_suffix=prompt_suffix,
                verbose=verbose,
                temperature=temperature,
                top_p=top_p,
            )

            # extracted = extract_final_candidate(
            #     best_full, fallback="number_then_full"
            # )

            # Note that the self-refinement loop already returns the extracted answer
            extracted = result["final_extracted"]

            is_correct = grade_answer(extracted, row["answer"])
            num_correct += int(is_correct)

            record = {
                "index": idx,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": result["final_full"],
                "extracted": extracted,
                "correct": bool(is_correct),
                "refinement": result,
                "scoring": scoring,
                "prompt_suffix": prompt_suffix,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            progress_msg = eta_progress_message(
                processed=idx,
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


def get_data():
    local_path = Path("math500_test.json")
    # local_path = Path("math-500-debug.json")
    url = (
        "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
        "main/ch03/01_main-chapter-code/math500_test.json"
    )

    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            math_data = json.load(f)
    else:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        math_data = r.json()

    with local_path.open("w", encoding="utf-8") as f:
        json.dump(math_data, f, ensure_ascii=False, indent=2)

    return math_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--which_model",
        type=str,
        default="base",
        choices=["base", "reasoning", "instruct"],
    )
    parser.add_argument("--dataset_size", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--critique_max_new_tokens", type=int, default=192)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scoring",
        type=str,
        default="heuristic",
        choices=["heuristic", "logprob", "logprob_extract", "none"],
        help="Scoring method to guide self-refinement.",
    )
    parser.add_argument(
        "--prompt_suffix",
        type=str,
        default="",
        help="Suffix appended to the rendered prompt, "
        "for example '\\n\\nExplain step by step.'",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = get_device() if args.device == "auto" else torch.device(args.device)
    which_model = args.which_model
    dataset_size = args.dataset_size

    print("Model:", which_model)
    print("Device:", device)
    dev_name = str(device).replace(":", "-")

    math_data = get_data()
    model, tokenizer = load_model_and_tokenizer(which_model, device, args.compile)
    model.eval()
    torch.set_float32_matmul_precision("high")

    out_path = (
        f"math500_{which_model}-{dev_name}-self-refine-"
        f"{args.scoring}.jsonl"
    )

    evaluate_math500_stream(
        model=model,
        tokenizer=tokenizer,
        device=device,
        math_data=math_data[:dataset_size],
        out_path=out_path,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
        iterations=args.iterations,
        critique_max_new_tokens=args.critique_max_new_tokens,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        scoring=args.scoring,
        prompt_suffix=args.prompt_suffix,
    )

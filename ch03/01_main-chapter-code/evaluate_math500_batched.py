
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch
#
# Batched MATH-500 evaluator

import argparse
import json
import time
from pathlib import Path
from urllib.request import urlopen

import torch

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.qwen3 import (
    download_qwen3_small,
    Qwen3Tokenizer,
    QWEN_CONFIG_06_B,
)
from reasoning_from_scratch.ch03 import (
    render_prompt,
    extract_final_candidate,
    grade_answer,
)
from reasoning_from_scratch.qwen3_batched import (
    Qwen3Model as Qwen3ModelBatched,
    generate_text_basic_batched_cache,
)


def get_data():
    local_path = Path("math500_test.json")
    url = (
        "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
        "main/ch03/01_main-chapter-code/math500_test.json"
    )

    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            math_data = json.load(f)
    else:
        with urlopen(url) as f:
            math_data = json.load(f)

    return math_data


def get_model(which_model, device, use_compile):
    if which_model == "base":

        download_qwen3_small(
            kind="base", tokenizer_only=False, out_dir="qwen3"
        )

        tokenizer_path = Path("qwen3") / "tokenizer-base.json"
        model_path = Path("qwen3") / "qwen3-0.6B-base.pth"
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

    elif which_model == "reasoning":

        download_qwen3_small(
            kind="reasoning", tokenizer_only=False, out_dir="qwen3"
        )

        tokenizer_path = Path("qwen3") / "tokenizer-reasoning.json"
        model_path = Path("qwen3") / "qwen3-0.6B-reasoning.pth"
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_path,
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=True,
        )

    else:
        raise ValueError(f"Invalid choice: WHICH_MODEL={which_model}")

    model = Qwen3ModelBatched(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)

    if use_compile:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)

    return model, tokenizer


def evaluate_math500_batched(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=512,
    verbose=False,
    batch_size=4,
):
    model.eval()

    # Default output path like the streaming variant
    if out_path is None:
        dev_name = str(device).replace(":", "-")  # Make filename compatible with Windows
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0
    print(f"MATH-500: 0/{num_examples}", end="\r", flush=True)

    start_time = time.time()

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)

    with open(out_path, "w", encoding="utf-8") as f:
        # Batched loop
        for start in range(0, num_examples, batch_size):
            batch = math_data[start:start + batch_size]

            prompts = [render_prompt(row["problem"]) for row in batch]

            # Encode and left-pad
            tokenized = [tokenizer.encode(p) for p in prompts]
            max_len = max(len(t) for t in tokenized)
            left_padded = [
                ([pad_id] * (max_len - len(t)) + t) if pad_id is not None else t
                for t in tokenized
            ]
            input_ids = torch.tensor(left_padded, device=device, dtype=torch.long)

            # Generate (batched)
            gen = generate_text_basic_batched_cache(
                model,
                token_ids=input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_id,
                pad_id=pad_id,
            )  # shape: (B, T_new)

            # Process each row
            B = gen.size(0)
            for b_idx in range(B):
                # Match the variable names used by evaluate_math500_stream
                i = start + b_idx + 1
                row = batch[b_idx]

                row_tokens = gen[b_idx]
                # Cut at first EOS if present
                if eos_id is not None:
                    eos_pos = (row_tokens == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_pos) > 0:
                        row_tokens = row_tokens[: eos_pos[0]]

                gen_text = tokenizer.decode(row_tokens.tolist())
                extracted = extract_final_candidate(gen_text)
                is_correct = grade_answer(extracted, row["answer"])
                num_correct += int(is_correct)

                record = {  # Record to be saved for inspection
                    "index": i,
                    "problem": row["problem"],
                    "gtruth_answer": row["answer"],
                    "generated_text": gen_text,
                    "extracted": extracted,
                    "correct": bool(is_correct),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if verbose:  # Print responses during the generation process
                    print(
                        f"\n\n{'='*50}\nMATH-500: {i}/{num_examples}\n"
                        f"{'='*50}\nExtracted: {extracted}\n"
                        f"Expected:  {row['answer']}\n"
                        f"Correct so far: {num_correct}\n{'-'*50}"
                    )
                else:
                    print(
                        f"MATH-500: {i}/{num_examples}",
                        end="\r", flush=True
                    )

    # Print summary information
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
        help='Device to use: "auto" (default), or any torch device string like "cpu", "cuda", "cuda:0", "mps".',
    )
    parser.add_argument(
        "--which_model",
        type=str,
        default="base",
        choices=["base", "reasoning"],
        help='Model variant to load. Defaults to "base".',
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
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for batched generation. Default: 4",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample correctness while evaluating.",
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
    batch_size = args.batch_size

    print("Model:", which_model)
    print("Device:", device)
    dev_name = str(device).replace(":", "-")
    print("Batch size:", batch_size)

    math_data = get_data()[:dataset_size]
    model, tokenizer = get_model(which_model, device, args.compile)

    out_path = f"math500_{which_model}-{dev_name}-batched-bs{batch_size}.jsonl"
    num_correct, num_examples, acc = evaluate_math500_batched(
        model=model,
        tokenizer=tokenizer,
        device=device,
        math_data=math_data,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        out_path=out_path,
        verbose=args.verbose,
    )

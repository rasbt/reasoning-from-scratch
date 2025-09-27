# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import torch

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import evaluate_math500_stream
from reasoning_from_scratch.qwen3 import (
    download_qwen3_small,
    Qwen3Tokenizer,
    Qwen3Model,
    QWEN_CONFIG_06_B
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

    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path))

    model.to(device)

    if use_compile:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)

    return model, tokenizer


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
    use_compile = args.compile

    print("Model:", which_model)
    print("Device:", device)
    dev_name = str(device).replace(":", "-")

    math_data = get_data()
    model, tokenizer = get_model(which_model, device, use_compile)

    num_correct, num_examples, acc = evaluate_math500_stream(
        model=model,
        out_path=f"math500_{which_model}-{dev_name}-evaluate-script.jsonl",
        tokenizer=tokenizer,
        device=device,
        math_data=math_data[:dataset_size],
        max_new_tokens=max_new_tokens,

        verbose=args.verbose
    )

# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import json
from pathlib import Path

import torch

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    extract_final_candidate,
    generate_text_stream_concat,
    load_math500_test,
    load_model_and_tokenizer,
    load_tokenizer_only,
    render_prompt,
)
from reasoning_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, cuda:0, mps, etc. Default: auto",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=500,
        help="Number of MATH-500 examples to evaluate. Default: 500",
    )
    parser.add_argument(
        "--which_model",
        type=str,
        default="reasoning",
        choices=["base", "reasoning", "instruct"],
        help="Model variant to generate answers with. Default: reasoning",
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
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional path to a .pth checkpoint to load model weights from.",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="math500_qwen3_answers.json",
        help="Output JSON file path. Default: math500_qwen3_answers.json",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extracted prediction for each sample.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_file = (repo_root / args.out_file).resolve()
    math_data = load_math500_test()

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    if args.which_model == "instruct":
        model_load_name = "reasoning"
    else:
        model_load_name = args.which_model

    if args.checkpoint_path:
        tokenizer = load_tokenizer_only(which_model=model_load_name)
        model = Qwen3Model(QWEN_CONFIG_06_B)
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
        if args.compile:
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
            model = torch.compile(model)
    else:
        model, tokenizer = load_model_and_tokenizer(
            which_model=model_load_name,
            device=device,
            use_compile=args.compile,
        )

    if args.which_model == "instruct":
        tokenizer.add_thinking = False

    model.eval()
    torch.set_float32_matmul_precision("high")

    selected_data = math_data[: args.dataset_size]
    num_examples = len(selected_data)
    rows = []
    for idx, row in enumerate(selected_data, start=1):
        prompt = render_prompt(row["problem"])
        gen_text = generate_text_stream_concat(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            verbose=False,
        )

        extracted = extract_final_candidate(gen_text)
        rows.append(
            {
                "answer": row["answer"],
                "prediction": f"\\boxed{{{extracted}}}",
            }
        )

        if args.verbose:
            print(f"{idx}/{num_examples} -> {rows[-1]['prediction']}")
        else:
            print(f"MATH-500: {idx}/{num_examples}", end="\r", flush=True)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nWrote {len(rows)} rows to: {out_file}")

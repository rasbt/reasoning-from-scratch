# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
from pathlib import Path

import torch

from reasoning_from_scratch.ch02 import (
    generate_text_basic_stream_cache,
    get_device,
)
from reasoning_from_scratch.ch03 import render_prompt
from reasoning_from_scratch.qwen3 import (
    download_qwen3_distill_checkpoints,
    download_qwen3_small,
    Qwen3Model,
    Qwen3Tokenizer,
    QWEN_CONFIG_06_B,
)


def build_tokenizer(local_dir):
    download_qwen3_small(kind="reasoning", tokenizer_only=True, out_dir=local_dir)
    return Qwen3Tokenizer(
        tokenizer_file_path=Path(local_dir) / "tokenizer-reasoning.json",
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True,
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Download a chapter 8 distillation checkpoint and stream a response.",
    )
    parser.add_argument(
        "--distill_type",
        type=str,
        default="deepseek_r1",
        choices=("deepseek_r1", "qwen3_235b_a22b"),
        help="Distillation checkpoint family to download.",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="06682",
        help="Training step to download.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Solve: If x + 7 = 19, what is x?",
        help="Math problem to render with the standard chapter prompt template.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="qwen3",
        help="Local directory used for checkpoint and tokenizer downloads.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (for example: cpu, cuda, mps). If omitted, auto-detects.",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    checkpoint_path = download_qwen3_distill_checkpoints(
        distill_type=args.distill_type,
        step=args.step,
        out_dir=args.local_dir,
    )
    tokenizer = build_tokenizer(args.local_dir)

    model = Qwen3Model(QWEN_CONFIG_06_B)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    prompt = render_prompt(args.prompt)
    input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

    print()
    print("=" * 60)
    print(f"torch        : {torch.__version__}")
    print(f"device       : {device}")
    print(f"distill_type : {args.distill_type}")
    print(f"step         : {args.step}")
    print("=" * 60)
    print()

    for token in generate_text_basic_stream_cache(
        model=model,
        token_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    ):
        token_id = token.squeeze(0).item()
        print(tokenizer.decode([token_id]), end="", flush=True)

    print("\n")


if __name__ == "__main__":
    main()

# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse

import torch

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import render_prompt

from hf_wrapper import build_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run local Hugging Face-style generation via a thin wrapper around "
            "the scratch Qwen3Model. This does not require exporting a model folder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tokenizer_kind",
        type=str,
        choices=("base", "reasoning"),
        default="base",
        help="Tokenizer family used for the raw .pth weights.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional local .pth file. If omitted, downloads the default model.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional tokenizer JSON file. If omitted, downloads the tokenizer.",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="qwen3",
        help="Download/cache directory for the model and tokenizer.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Solve: If x + 7 = 19, what is x?",
        help="Math problem to render with the chapter 3 prompt template.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cpu, cuda, mps). Defaults to auto-detect.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else get_device()
    model, tokenizer = build_model_and_tokenizer(
        tokenizer_kind=args.tokenizer_kind,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        local_dir=args.local_dir,
    )
    model.to(device)
    model.eval()

    prompt = render_prompt(args.prompt)
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0, input_ids.shape[1]:]

    print()
    print("=" * 60)
    print(f"device         : {device}")
    print(f"tokenizer_kind : {args.tokenizer_kind}")
    print("=" * 60)
    print()
    print(tokenizer.decode(generated_ids.tolist()))
    print()


if __name__ == "__main__":
    main()

# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

# Runs the model similar to chapter 2 and 3 in streaming mode with the least
# amount of bells and whistles. Uses KV caching by default.
# Similar to generate_simple.py but uses an interactive REPL (without memory).

import argparse
from pathlib import Path
import time
import torch

from reasoning_from_scratch.ch02 import (
    get_device,
    generate_stats
)
from reasoning_from_scratch.ch02 import (
    generate_text_basic_stream_cache
)
from reasoning_from_scratch.qwen3 import (
    download_qwen3_small,
    Qwen3Model,
    Qwen3Tokenizer,
    QWEN_CONFIG_06_B
)

parser = argparse.ArgumentParser(description="Run Qwen3 text generation (interactive REPL)")
parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Device to run on (e.g. 'cpu', 'cuda', 'mps'). "
         "If not provided, will auto-detect with get_device()."
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=2048,
    help="Maximum number of new tokens to generate (default: 2048)."
)
parser.add_argument(
    "--compile",
    action="store_true",
    help="Compile PyTorch model (default: False)."
)
parser.add_argument(
    "--reasoning",
    action="store_true",
    help="Use reasoning model variant (default: False)."
)

args = parser.parse_args()
device = torch.device(args.device) if args.device else get_device()

if args.reasoning:
    download_qwen3_small(kind="reasoning", tokenizer_only=False, out_dir="qwen3")
    tokenizer_path = Path("qwen3") / "tokenizer-reasoning.json"
    model_path = Path("qwen3") / "qwen3-0.6B-reasoning.pth"
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_path,
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True
    )
else:
    download_qwen3_small(kind="base", tokenizer_only=False, out_dir="qwen3")
    tokenizer_path = Path("qwen3") / "tokenizer-base.json"
    model_path = Path("qwen3") / "qwen3-0.6B-base.pth"
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_path,
        apply_chat_template=False,
        add_generation_prompt=False,
        add_thinking=False
    )

model = Qwen3Model(QWEN_CONFIG_06_B)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

if args.compile:
    model = torch.compile(model)

print()
print("=" * 60)
print(f"torch     : {torch.__version__}")
print(f"device    : {device}")
print("cache     : True")
print(f"compile   : {args.compile}")
print(f"reasoning : {args.reasoning}")
print("memory    : False")
print("=" * 60)
print()
print("Interactive REPL (no memory). Type '\\exit' or '\\quit' to quit.\n")


def run_once(prompt: str):
    input_ids = tokenizer.encode(prompt)
    input_token_ids_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    start_time = time.time()
    all_token_ids = []

    print("[Model]\n", end="", flush=True)
    for token in generate_text_basic_stream_cache(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id
    ):
        token_id = token.squeeze(0).item()
        print(tokenizer.decode([token_id]), end="", flush=True)

        all_token_ids.append(token_id)

    end_time = time.time()
    print("\n")

    print("[Stats]")
    generate_stats(
        torch.tensor(all_token_ids),
        tokenizer,
        start_time,
        end_time
    )
    print("-" * 60)


# REPL loop
try:
    while True:
        try:
            user_in = input(">> ").strip()
        except EOFError:
            print("")
            break
        if user_in.lower() in {r"\exit", r"\quit"}:
            break
        if not user_in:
            continue

        print("\n" + "-" * 60)
        print("[User]")
        print(user_in + "\n")
        run_once(user_in)
except KeyboardInterrupt:
    print("\nInterrupted by user.")

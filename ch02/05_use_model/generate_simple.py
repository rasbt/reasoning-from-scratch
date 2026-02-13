# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

# Runs the model similar to chapter 2 and 3 in streaming mode with the least
# amount of bells and whistles. Uses KV caching by default.

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

parser = argparse.ArgumentParser(description="Run Qwen3 text generation")
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
parser.add_argument(
    "--prompt",
    type=str,
    default=None,
    help=("Use a custom prompt. If not explicitly provided, uses the following defaults: "
          "'Explain large language models in a single sentence.' for the base model, and "
          "'Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.' for the reasoning model.")
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


if args.prompt is None:
    if args.reasoning:
        prompt = "Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field."
    else:
        prompt = "Explain large language models in a single sentence."
else:
    prompt = args.prompt


input_ids = tokenizer.encode(prompt)
input_token_ids_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

print()
print("=" * 60)
print(f"torch     : {torch.__version__}")
print(f"device    : {device}")
print("cache     : True")
print(f"compile   : {args.compile}")
print(f"reasoning : {args.reasoning}")
print("=" * 60)
print()

start_time = time.time()
all_token_ids = []

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
generate_stats(torch.tensor(all_token_ids), tokenizer, start_time, end_time)

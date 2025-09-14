# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch


import argparse
from pathlib import Path
import time
import torch

from reasoning_from_scratch.ch02 import (
    get_device,
    generate_stats
)
from reasoning_from_scratch.qwen3 import (
    download_qwen3_small,
    Qwen3Tokenizer,
    QWEN_CONFIG_06_B
)


############################
# Parse command-line args
############################
parser = argparse.ArgumentParser(description="Run Qwen3 text generation")
parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Device to run on (e.g. 'cpu', 'cuda', 'mps'). "
         "If not provided, will auto-detect with get_device()."
)
parser.add_argument(
    "--cache",
    action="store_true",
    help="Use KV cache during generation (default: False)."
)

parser.add_argument(
    "--compile",
    action="store_true",
    help="Compile PyTorch model (default: False)."
)

parser.add_argument(
    "--reasoning",
    action="store_true",
    help="Use reasoning model variant."
)

parser.add_argument(
    "--optimized",
    action="store_true",
    help="Use reasoning model variant."
)


args = parser.parse_args()

if args.optimized:
    from reasoning_from_scratch.qwen3_optimized import Qwen3Model
else:
    from reasoning_from_scratch.qwen3 import Qwen3Model


if args.cache:
    if args.optimized:
        from reasoning_from_scratch.qwen3_optimized import generate_text_basic_cache as generate_text_basic
    else:
        from reasoning_from_scratch.ch02 import generate_text_basic_cache as generate_text_basic

else:
    from reasoning_from_scratch.ch02 import generate_text_basic

device = torch.device(args.device) if args.device else get_device()

#########################
# Model + tokenizer setup
#########################

if args.reasoning:
    download_qwen3_small(kind="reasoning", tokenizer_only=False, out_dir="qwen3")
    tokenizer_file_path = Path("qwen3") / "tokenizer-reasoning.json"
    model_file = Path("qwen3") / "qwen3-0.6B-reasoning.pth"
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True
    )

else:
    download_qwen3_small(kind="base", tokenizer_only=False, out_dir="qwen3")
    tokenizer_file_path = Path("qwen3") / "tokenizer-base.json"
    model_file = Path("qwen3") / "qwen3-0.6B-base.pth"
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_file_path)

model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_file, map_location=device))

model.to(device)

if args.compile:
    major, minor = map(int, torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 8):
        # This avoids retriggering model recompilations
        # in PyTorch 2.8 and newer
        # if the model contains code like self.pos = self.pos + 1
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
    model = torch.compile(model)

#########################
# Prompt + generation
#########################

if args.reasoning:
    prompt = "Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field."
else:
    prompt = "Explain large language models in a single sentence."

input_token_ids_tensor = torch.tensor(
    tokenizer.encode(prompt),
    device=device
).unsqueeze(0)

max_new_tokens = 2048


for iteration in range(1, 4):
    print("=" * 60)
    print(f"Iteration : {iteration}")
    print(f"optimized : {args.optimized}")
    print(f"torch     : {torch.__version__}")
    print(f"device    : {device}")
    print(f"cache     : {args.cache}")
    print(f"compile   : {args.compile}")
    print(f"reasoning : {args.reasoning}")
    print("=" * 60)

    start_time = time.time()
    output_token_ids_tensor = generate_text_basic(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    end_time = time.time()

    print(f"Output length: {output_token_ids_tensor.numel()}")
    generate_stats(output_token_ids_tensor, tokenizer, start_time, end_time)

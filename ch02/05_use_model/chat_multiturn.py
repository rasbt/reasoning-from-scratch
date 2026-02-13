# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

# Runs the model similar to chapter 2 and 3 in streaming mode with the least
# amount of bells and whistles. Uses KV caching by default.
# Interactive REPL (Read, Evaluate, Print, Loop) with multiturn memory.

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
    help="Maximum number of new tokens to generate in each turn (default: 2048)."
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
else:
    download_qwen3_small(kind="base", tokenizer_only=False, out_dir="qwen3")
    tokenizer_path = Path("qwen3") / "tokenizer-base.json"
    model_path = Path("qwen3") / "qwen3-0.6B-base.pth"

# We will apply the chat template manually later
tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tokenizer_path,
    apply_chat_template=False
)

model = Qwen3Model(QWEN_CONFIG_06_B)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

if args.compile:
    model = torch.compile(model)

# The reasoning model may emit <|im_end|>; base may emit <|endoftext|>.
EOS_TOKEN_IDS = (
    tokenizer.encode("<|im_end|>")[0],
    tokenizer.encode("<|endoftext|>")[0]
)

print()
print("=" * 60)
print(f"torch     : {torch.__version__}")
print(f"device    : {device}")
print("cache     : True")
print(f"compile   : {args.compile}")
print(f"reasoning : {args.reasoning}")
print("memory    : True")
print(f"max_new_tokens (per turn): {args.max_new_tokens}")
print(f"context_length: {model.cfg['context_length']}")
print("=" * 60)
print()
print("Interactive REPL with memory. Type '\\exit' or '\\quit' to quit.")
print("Commands: \\clear (forget memory), \\history (show turn count)\n")

# Multi-turn memory as a list of role-content dicts
# Example: {"role": "system"|"user"|"assistant", "content": str}
history = [
    {"role": "system", "content": "You are a helpful assistant."}
]


def build_prompt_from_history(history, add_assistant_header=True):
    """
    history: [{"role": "system"|"user"|"assistant", "content": str}, ...]
    """
    parts = []
    for m in history:
        role = m["role"]
        content = m["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    if add_assistant_header:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def trim_input_tensor(input_ids_tensor, context_len, max_new_tokens):
    assert max_new_tokens < context_len
    keep_len = max(1, context_len - max_new_tokens)

    # If the prompt is too long, left-truncate to keep_len
    if input_ids_tensor.shape[1] > keep_len:
        input_ids_tensor = input_ids_tensor[:, -keep_len:]

    return input_ids_tensor


def run_generate(user_text):
    # Add user prompt to history
    history.append({"role": "user", "content": user_text})

    # Encode full history
    prompt = build_prompt_from_history(history, add_assistant_header=True)
    input_ids = tokenizer.encode(prompt)
    input_token_ids_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    # Left-tuncate (to make space for generation)
    input_token_ids_tensor = trim_input_tensor(
        input_ids_tensor=input_token_ids_tensor,
        context_len=model.cfg["context_length"],
        max_new_tokens=args.max_new_tokens
    )

    start_time = time.time()
    all_token_ids = []

    print("[Model]\n", end="", flush=True)
    for tok in generate_text_basic_stream_cache(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=args.max_new_tokens,
        # eos_token_id=TOKENIZER.eos_token_id
    ):
        token_id = tok.squeeze(0)
        if token_id in EOS_TOKEN_IDS:  # Manually break at stop tokens
            break
        piece = tokenizer.decode(token_id.tolist())
        print(piece, end="", flush=True)
        all_token_ids.append(token_id.item())

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

    # Add model reply to history
    assistant_text = tokenizer.decode(all_token_ids)
    history.append({"role": "assistant", "content": assistant_text})
    return assistant_text


# Interactive REPL (Read, Evaluate, Print, Loop)
try:
    while True:
        try:
            user_in = input(">> ").strip()
        except EOFError:
            print("")
            break

        low = user_in.lower()
        if low in {r"\exit", r"\quit"}:
            break
        if low == r"\clear":
            # Reset history but keep the system prompt
            system_entries = [m for m in history if m["role"] == "system"]
            history.clear()
            if system_entries:
                history.extend(system_entries)
            else:
                history.append({"role": "system", "content": "You are a helpful assistant."})
            print("(memory cleared)\n")
            continue
        if low == r"\history":
            # Count assistant turns as the number of model replies so far
            assistant_turns = sum(1 for m in history if m["role"] == "assistant")
            print(f"(stored turns: {assistant_turns})\n")
            continue
        if not user_in:
            continue

        print("\n" + "-" * 60)
        print("[User]")
        print(user_in + "\n")
        run_generate(user_in)

except KeyboardInterrupt:
    print("\nInterrupted by user.")

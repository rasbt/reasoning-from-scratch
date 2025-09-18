# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from pathlib import Path

import torch
import chainlit

from reasoning_from_scratch.ch02 import (
    get_device,
)
from reasoning_from_scratch.ch02_ex import generate_text_basic_stream_cache
from reasoning_from_scratch.qwen3 import (
    download_qwen3_small,
    Qwen3Model,
    Qwen3Tokenizer,
    QWEN_CONFIG_06_B
)

# ============================================================
# EDIT ME: Simple configuration
# ============================================================
REASONING = True          # True = "thinking" chat model, False = Base
MAX_NEW_TOKENS = 38912
LOCAL_DIR = "qwen3"
COMPILE = False
# ============================================================


def trim_input_tensor(input_ids_tensor, context_len, max_new_tokens):
    assert max_new_tokens < context_len
    keep_len = max(1, context_len - max_new_tokens)

    # If the prompt is too long, left-truncate to keep_len
    if input_ids_tensor.shape[1] > keep_len:
        input_ids_tensor = input_ids_tensor[:, -keep_len:]

    return input_ids_tensor


def get_model_and_tokenizer(qwen3_config, local_dir, device, use_compile, use_reasoning):
    if use_reasoning:
        download_qwen3_small(kind="reasoning", tokenizer_only=False, out_dir=local_dir)
        tokenizer_path = Path(local_dir) / "tokenizer-reasoning.json"
        model_path = Path(local_dir) / "qwen3-0.6B-reasoning.pth"
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_path,
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=True
        )

    else:
        download_qwen3_small(kind="base", tokenizer_only=False, out_dir=local_dir)
        tokenizer_path = Path(local_dir) / "tokenizer-base.json"
        model_path = Path(local_dir) / "qwen3-0.6B-base.pth"
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

    model = Qwen3Model(qwen3_config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    if use_compile:
        model = torch.compile(model)

    return model, tokenizer


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


DEVICE = get_device()
MODEL, TOKENIZER = get_model_and_tokenizer(
    qwen3_config=QWEN_CONFIG_06_B,
    local_dir=LOCAL_DIR,
    device=DEVICE,
    use_compile=COMPILE,
    use_reasoning=REASONING
)

# Even though the official TOKENIZER.eos_token_id is either <|im_end|> (reasoning)
# or <|endoftext|> (base), the reasoning model sometimes emits both.
EOS_TOKEN_IDS = (
    TOKENIZER.encode("<|im_end|>")[0],
    TOKENIZER.encode("<|endoftext|>")[0]
)


@chainlit.on_chat_start
async def on_start():
    chainlit.user_session.set("history", [])
    chainlit.user_session.get("history").append(
        {"role": "system", "content": "You are a helpful assistant."}
    )


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    # 0) Get and track chat history
    history = chainlit.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    # 1) Encode input
    prompt = build_prompt_from_history(history, add_assistant_header=True)
    input_ids = TOKENIZER.encode(prompt)
    input_ids_tensor = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)

    # Multi-turn can be very long, so we add this left-trimming
    input_ids_tensor = trim_input_tensor(
        input_ids_tensor=input_ids_tensor,
        context_len=MODEL.cfg["context_length"],
        max_new_tokens=MAX_NEW_TOKENS
    )

    # 2) Start an outgoing message we can stream into
    out_msg = chainlit.Message(content="")
    await out_msg.send()

    # 3) Stream generation
    for tok in generate_text_basic_stream_cache(
        model=MODEL,
        token_ids=input_ids_tensor,
        max_new_tokens=MAX_NEW_TOKENS,
        # eos_token_id=TOKENIZER.eos_token_id
    ):
        token_id = tok.squeeze(0)
        if token_id in EOS_TOKEN_IDS:
            break
        piece = TOKENIZER.decode(token_id.tolist())
        await out_msg.stream_token(piece)

    # 4) Finalize the streamed message
    await out_msg.update()

    # 5) Update chat history
    history.append({"role": "assistant", "content": out_msg.content})
    chainlit.user_session.set("history", history)

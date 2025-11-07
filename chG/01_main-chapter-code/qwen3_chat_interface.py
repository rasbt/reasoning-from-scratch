# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


import torch
import chainlit

from reasoning_from_scratch.ch02 import (
    get_device,
)
from reasoning_from_scratch.ch02_ex import generate_text_basic_stream_cache
from reasoning_from_scratch.ch03 import load_model_and_tokenizer


# ============================================================
# EDIT ME: Simple configuration
# ============================================================
WHICH_MODEL = "reasoning"  # "base" for base model
MAX_NEW_TOKENS = 38912
LOCAL_DIR = "qwen3"
COMPILE = False
# ============================================================


DEVICE = get_device()

MODEL, TOKENIZER = load_model_and_tokenizer(
    which_model=WHICH_MODEL,
    device=DEVICE,
    use_compile=COMPILE,
    local_dir=LOCAL_DIR
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
    # 1) Encode input
    input_ids = TOKENIZER.encode(message.content)
    input_ids_tensor = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)

    # 2) Start an outgoing message we can stream into
    out_msg = chainlit.Message(content="")
    await out_msg.send()

    # 3) Stream generation
    for tok in generate_text_basic_stream_cache(
        model=MODEL,
        token_ids=input_ids_tensor,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=TOKENIZER.eos_token_id
    ):
        token_id = tok.squeeze(0)
        piece = TOKENIZER.decode(token_id.tolist())
        await out_msg.stream_token(piece)

    # 4) Finalize the streamed message
    await out_msg.update()

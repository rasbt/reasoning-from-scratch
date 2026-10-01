# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import ast
from pathlib import Path

import pytest
import torch

from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache
from reasoning_from_scratch.qwen3 import Qwen3Model


@pytest.fixture(params=[
    "ch02/05_use_model/chat_multiturn.py",
    "chG/01_main-chapter-code/qwen3_chat_interface_multiturn.py",
], ids=["cli", "chainlit"])
def trim_input_tensor(request):
    path = Path(__file__).resolve().parents[1] / request.param
    # Load only the helper so the chat apps do not load full model weights.
    tree = ast.parse(path.read_text())
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "trim_input_tensor")
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["trim_input_tensor"]


@pytest.mark.parametrize("context_len, max_new_tokens, expected_prompt_length", [
    (16, 0, 16),
    (16, 1, 16),
    (16, 9, 8),
    (16, 16, 1),
    (1, 1, 1),
])
def test_trimmed_prompt_uses_full_context(
    trim_input_tensor, context_len, max_new_tokens, expected_prompt_length
):
    cfg = {
        "vocab_size": 32, "context_length": context_len, "emb_dim": 16,
        "n_heads": 4, "n_layers": 2, "hidden_dim": 32, "head_dim": 4,
        "qk_norm": True, "n_kv_groups": 2, "rope_base": 1_000_000.0,
        "dtype": torch.float32,
    }
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen3Model(cfg).to(device).eval()
    prompt = torch.arange(context_len + 4, device=device).unsqueeze(0)
    trimmed = trim_input_tensor(prompt, context_len, max_new_tokens)

    assert torch.equal(trimmed, prompt[:, -expected_prompt_length:])
    output = list(generate_text_basic_stream_cache(model, trimmed, max_new_tokens))
    assert len(output) == max_new_tokens
    # Keep as much history as possible without overflowing during generation.
    assert model.current_pos == context_len


def test_prompt_with_room_is_unchanged(trim_input_tensor):
    prompt = torch.tensor([[4, 5]])
    assert trim_input_tensor(prompt, context_len=16, max_new_tokens=9) is prompt


def test_budget_exceeding_context_is_rejected(trim_input_tensor):
    prompt = torch.tensor([[4]])
    with pytest.raises(ValueError, match="max_new_tokens must not exceed context_len"):
        trim_input_tensor(prompt, context_len=16, max_new_tokens=17)

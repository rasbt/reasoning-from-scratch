# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import pytest
import torch

from reasoning_from_scratch import qwen3, qwen3_batched, qwen3_optimized
from reasoning_from_scratch.ch02 import generate_text_basic_cache


@pytest.fixture(params=[
    (qwen3, generate_text_basic_cache),
    (qwen3_optimized, qwen3_optimized.generate_text_basic_cache),
    (qwen3_batched, qwen3_batched.generate_text_basic_batched_cache),
], ids=["standard", "optimized", "batched"])
def model_setup(request):
    module, generate = request.param
    cfg = {
        "vocab_size": 32,
        "context_length": 8,
        "emb_dim": 16,
        "n_heads": 4,
        "n_layers": 2,
        "hidden_dim": 32,
        "head_dim": 4,
        "qk_norm": True,
        "n_kv_groups": 2,
        "rope_base": 1_000_000.0,
        "dtype": torch.float32,
    }
    torch.manual_seed(123)
    model = module.Qwen3Model(cfg).eval()
    if module is qwen3_optimized:
        cache = module.KVCache(
            n_layers=cfg["n_layers"],
            max_len=cfg["context_length"],
            num_kv_groups=cfg["n_kv_groups"],
            head_dim=cfg["head_dim"],
            device="cpu",
            dtype=cfg["dtype"],
        )
    else:
        cache = module.KVCache(n_layers=cfg["n_layers"])
    return model, cache, generate


@torch.inference_mode()
@pytest.mark.parametrize("use_cache", [False, True])
def test_overlong_prompt_reports_the_context_limit(model_setup, use_cache):
    model, cache, _ = model_setup
    prompt = torch.ones((1, 9), dtype=torch.long)

    with pytest.raises(ValueError, match=r"Sequence length 9 exceeds .*8 tokens"):
        model(prompt, cache=cache if use_cache else None)


@torch.inference_mode()
def test_rejected_decode_preserves_cache_and_allows_the_last_position(model_setup):
    model, cache, _ = model_setup
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    model(prompt, cache=cache)

    with pytest.raises(ValueError, match=r"Sequence length 9 exceeds .*8 tokens"):
        model(torch.tensor([[8, 9]]), cache=cache)

    # A rejected call must leave the previous context usable.
    next_token = torch.tensor([[8]])
    actual = model(next_token, cache=cache)
    expected = model(torch.cat([prompt, next_token], dim=1))[:, -1:]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@torch.inference_mode()
def test_large_generation_budget_can_finish_early_at_eos(model_setup):
    model, _, generate = model_setup
    model.out_head.weight.zero_()  # Make the next token EOS (token 0).

    output = generate(
        model,
        torch.tensor([[1, 2]]),
        max_new_tokens=16,
        eos_token_id=0,
    )

    # Batched generation includes EOS; the other helpers omit it.
    assert output.shape[1] <= 1
    assert torch.all(output == 0)

# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import pytest
import torch

from reasoning_from_scratch import ch02, ch04, ch06
from reasoning_from_scratch import qwen3, qwen3_batched, qwen3_optimized


@pytest.fixture(params=[
    (qwen3, ch02.generate_text_basic_cache, {}),
    (qwen3, ch02.generate_text_basic_stream_cache, {}),
    (qwen3, ch04.generate_text_temp_stream_cache, {"temperature": 0.8}),
    (qwen3, ch04.generate_text_top_p_stream_cache, {"temperature": 0.8, "top_p": 0.9}),
    (qwen3, ch06.sample_response, {"temperature": 0.8, "top_p": 0.9}),
    (qwen3_optimized, qwen3_optimized.generate_text_basic_cache, {}),
    (qwen3_batched, qwen3_batched.generate_text_basic_batched_cache, {}),
    (qwen3_batched, qwen3_batched.generate_text_basic_batched_stream_cache, {}),
    (qwen3_batched, qwen3_batched.generate_text_basic_batched_cache_stop, {}),
    (qwen3_batched, qwen3_batched.generate_text_basic_batched_stream_cache_stop, {}),
], ids=["greedy", "stream", "temperature", "top_p", "rollout", "optimized",
        "batched", "batched_stream", "batched_stop", "batched_stream_stop"])
def generation_setup(request):
    module, generate, kwargs = request.param
    cfg = {
        "vocab_size": 32, "context_length": 8, "emb_dim": 16,
        "n_heads": 4, "n_layers": 2, "hidden_dim": 32, "head_dim": 4,
        "qk_norm": True, "n_kv_groups": 2, "rope_base": 1_000_000.0,
        "dtype": torch.float32,
    }
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = module.Qwen3Model(cfg).to(device).eval()
    calls = []
    model.register_forward_pre_hook(lambda _, args: calls.append(args[0].shape[1]))
    return model, generate, kwargs, calls


def run_generation(setup, prompt_length, max_new_tokens, eos_token_id=None):
    model, generate, kwargs, _ = setup
    device = model.tok_emb.weight.device
    prompt = torch.arange(1, prompt_length + 1, device=device).unsqueeze(0)

    if generate is ch06.sample_response:
        class Tokenizer:
            def encode(self, text):
                return list(range(1, prompt_length + 1))

            def decode(self, tokens):
                return str(tokens)

        tokenizer = Tokenizer()
        tokenizer.eos_token_id = eos_token_id
        output, length, _ = generate(
            model, tokenizer, "prompt", device,
            max_new_tokens=max_new_tokens, **kwargs,
        )
        return output[length:].unsqueeze(0)

    if isinstance(model, qwen3_batched.Qwen3Model):
        prompt = prompt.repeat(2, 1)
        # Exercise the padding-mask extension while decoding.
        prompt[0, 0] = 0
        kwargs = {**kwargs, "pad_id": 0}
    output = generate(model, prompt, max_new_tokens=max_new_tokens,
                      eos_token_id=eos_token_id, **kwargs)
    if torch.is_tensor(output):
        return output
    steps = list(output)
    return torch.cat(steps, dim=1) if steps else prompt[:, :0]


@pytest.mark.parametrize("max_new_tokens", [1, 3])
def test_last_token_needs_no_forward_beyond_context(generation_setup, max_new_tokens):
    # The last valid input position predicts one more token. Processing that
    # token would exceed the context limit, and its logits would never be used.
    _, _, _, calls = generation_setup
    prompt_length = 9 - max_new_tokens
    output = run_generation(generation_setup, prompt_length, max_new_tokens)
    assert output.shape[1] == max_new_tokens
    assert calls == [prompt_length] + [1] * (max_new_tokens - 1)


def test_eos_needs_no_unused_forward(generation_setup):
    model, _, _, calls = generation_setup

    def force_eos(_, args, output):
        output.fill_(-torch.inf)
        output[..., 0] = 0
        return output

    model.register_forward_hook(force_eos)
    output = run_generation(generation_setup, 8, 16, eos_token_id=0)
    # Helpers differ in whether they include EOS, but none need its logits.
    assert output.shape[1] <= 1
    assert torch.all(output == 0)
    assert calls == [8]


def test_zero_generation_budget_returns_no_tokens(generation_setup):
    output = run_generation(generation_setup, 2, 0)
    assert output.shape[1] == 0

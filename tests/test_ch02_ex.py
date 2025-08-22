# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Reasoning Model From Scratch"
#   - https://www.manning.com/books/build-a-reasoning-model-from-scratch
# Code: https://github.com/rasbt/reasoning-from-scratch

import torch

from reasoning_from_scratch.ch02 import (
    generate_text_basic,
    generate_text_basic_cache,
)
from reasoning_from_scratch.ch02_ex import (
    generate_text_basic_stream,
    generate_text_basic_stream_cache,
)


# Dummy model for generate_text_basic tests.
class DummyModel:
    def __init__(self, fixed_token, vocab_size=5):
        self.fixed_token = fixed_token
        self.vocab_size = vocab_size
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, token_ids, cache=None):
        batch_size, seq_len = token_ids.size()
        out = torch.zeros(batch_size, seq_len, self.vocab_size)
        # Set the fixed_token column to the highest value so argmax returns fixed_token
        out[..., self.fixed_token] = 1.0
        return out


class DummyModelCache(DummyModel):
    def __init__(self, fixed_token, vocab_size=5, n_layers=2):
        super().__init__(fixed_token, vocab_size)
        self.cfg = {"n_layers": n_layers}
        self.reset_called = False

    def reset_kv_cache(self):
        self.reset_called = True


class DummyTokenizer:
    def decode(self, token_list):
        return " ".join(str(t) for t in token_list)


def test_generate_text_basic_stream_equivalence():
    max_new_tokens = 10
    fixed_token = 2

    dummy_model = DummyModel(fixed_token=fixed_token)
    token_ids = torch.tensor([[1, 3, 4]])  # shape (batch, seq_len)

    # Set eos_token_id to be the fixed_token so that generation stops immediately
    output_1 = generate_text_basic(dummy_model, token_ids, max_new_tokens, eos_token_id=fixed_token)
    output_1 = output_1.squeeze(0).tolist()

    output_2 = []
    for token in generate_text_basic_stream(
        model=dummy_model,
        token_ids=token_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=fixed_token
    ):
        output_2.append(token.squeeze(0).item())

    assert output_1 == output_2


def test_generate_text_basic_stream_generates_tokens_without_eos():
    max_new_tokens = 3
    fixed_token = 1

    dummy_model = DummyModel(fixed_token=fixed_token)
    token_ids = torch.tensor([[0, 4]])
    output_1 = generate_text_basic(dummy_model, token_ids, max_new_tokens, eos_token_id=None)
    output_1 = output_1.squeeze(0).tolist()

    output_2 = []
    for token in generate_text_basic_stream(
        model=dummy_model,
        token_ids=token_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=None
    ):
        output_2.append(token.squeeze(0).item())

    assert output_1 == output_2


def test_generate_text_basic_cache_stream_equivalence():
    max_new_tokens = 10
    fixed_token = 2

    dummy_model = DummyModelCache(fixed_token=fixed_token)
    token_ids = torch.tensor([[1, 3, 4]])  # shape (batch, seq_len)

    # Set eos_token_id to be the fixed_token so that generation stops immediately
    output_1 = generate_text_basic(dummy_model, token_ids, max_new_tokens, eos_token_id=fixed_token)
    output_1 = output_1.squeeze(0).tolist()

    output_2 = []
    for token in generate_text_basic_stream_cache(
        model=dummy_model,
        token_ids=token_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=fixed_token
    ):
        output_2.append(token.squeeze(0).item())

    assert output_1 == output_2


def test_generate_text_basic_cache_stream_generates_tokens_without_eos():
    max_new_tokens = 3
    fixed_token = 1

    dummy_model = DummyModelCache(fixed_token=fixed_token)
    token_ids = torch.tensor([[0, 4]])
    output_1 = generate_text_basic_cache(dummy_model, token_ids, max_new_tokens, eos_token_id=None)
    output_1 = output_1.squeeze(0).tolist()

    output_2 = []
    for token in generate_text_basic_stream_cache(
        model=dummy_model,
        token_ids=token_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=None
    ):
        output_2.append(token.squeeze(0).item())

    assert output_1 == output_2

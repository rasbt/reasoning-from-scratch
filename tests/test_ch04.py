# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import pytest
import torch

import reasoning_from_scratch.ch04 as ch04


class DummyTokenizer:
    def __init__(self, eos_token_id=None):
        self.eos_token_id = eos_token_id
        self.decode_map = {7: "X", 8: "Y"}

    def encode(self, prompt):
        # Content of the prompt is irrelevant for these tests.
        return [1, 2]

    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(self.decode_map.get(i, str(i)) for i in ids)


class DummyModelCache:
    def __init__(self, fixed_token, vocab_size=5, n_layers=2):
        self.fixed_token = fixed_token
        self.vocab_size = vocab_size
        self.cfg = {"n_layers": n_layers}
        self.reset_called = False
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def reset_kv_cache(self):
        self.reset_called = True

    def __call__(self, token_ids, cache=None):
        batch_size, seq_len = token_ids.size()
        logits = torch.zeros(batch_size, seq_len, self.vocab_size)
        logits[..., self.fixed_token] = 1.0
        return logits


def test_generate_text_stream_concat_flex_uses_custom_generator():
    calls = []

    def fake_generate_func(**kwargs):
        calls.append(kwargs)
        for tok in (torch.tensor([7]), torch.tensor([8])):
            yield tok

    tok = DummyTokenizer(eos_token_id=0)
    out = ch04.generate_text_stream_concat_flex(
        model=None,
        tokenizer=tok,
        prompt="ignored",
        device="cpu",
        max_new_tokens=2,
        generate_func=fake_generate_func,
        temperature=0.5,
    )

    assert out == "XY"
    assert calls, "Generator should have been invoked"
    assert calls[0]["temperature"] == 0.5
    assert calls[0]["eos_token_id"] == tok.eos_token_id


def test_scale_logits_by_temperature_validates_and_scales():
    logits = torch.tensor([[2.0, 4.0]])
    scaled = ch04.scale_logits_by_temperature(logits, 2.0)
    assert torch.allclose(scaled, logits / 2.0)

    with pytest.raises(ValueError):
        ch04.scale_logits_by_temperature(logits, 0.0)


def test_top_p_filter_truncates_and_renormalizes():
    probas = torch.tensor([[0.5, 0.4, 0.1]])
    filtered = ch04.top_p_filter(probas, top_p=0.6)
    assert torch.allclose(filtered, torch.tensor([[1.0, 0.0, 0.0]]))

    # When no filtering is needed, output should match input
    unfiltered = ch04.top_p_filter(probas, top_p=1.0)
    assert torch.allclose(unfiltered, probas)


def test_generate_text_temp_stream_cache_stops_on_eos():
    model = DummyModelCache(fixed_token=3)
    token_ids = torch.tensor([[0, 1]])

    out = list(
        ch04.generate_text_temp_stream_cache(
            model,
            token_ids=token_ids,
            max_new_tokens=5,
            eos_token_id=3,
            temperature=0.0,
        )
    )

    assert out == []
    assert model.reset_called is True
    assert model.eval_called is True


def test_self_consistency_vote_majority(monkeypatch):
    answers = ["2", "2", "3"]

    def fake_generate_text_stream_concat_flex(**kwargs):
        idx = kwargs.pop("_call_idx", None)
        idx = idx if idx is not None else 0
        return answers[idx % len(answers)]

    # Wrap to inject call index so we can cycle through answers deterministically
    call_counter = {"i": 0}

    def wrapped_generate(**kwargs):
        kwargs["_call_idx"] = call_counter["i"]
        call_counter["i"] += 1
        return fake_generate_text_stream_concat_flex(**kwargs)

    monkeypatch.setattr(ch04, "generate_text_stream_concat_flex", wrapped_generate)

    result = ch04.self_consistency_vote(
        model=None,
        tokenizer=DummyTokenizer(),
        prompt="unused",
        device="cpu",
        num_samples=3,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=5,
        show_progress=False,
        show_long_answer=False,
        seed=123,
    )

    assert result["final_answer"] == "2"
    assert result["counts"]["2"] == 2
    assert result["majority_winners"] == ["2"]

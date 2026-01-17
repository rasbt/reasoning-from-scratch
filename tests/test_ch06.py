# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import json
import torch
import reasoning_from_scratch.ch06 as ch06


class DummyTokenizer:
    def __init__(self):
        self.eos_token_id = 0
        self.decode_map = {2: "X"}

    def encode(self, text):
        return [5, 6]

    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(self.decode_map.get(i, str(i)) for i in ids)


class DummySampleModel:
    def __init__(self):
        self.cfg = {"n_layers": 1}
        self.reset_called = False

    def reset_kv_cache(self):
        self.reset_called = True

    def __call__(self, token_ids, cache=None):
        batch_size, seq_len = token_ids.size()
        vocab_size = 4
        logits = torch.full(
            (batch_size, seq_len, vocab_size),
            -1e9,
            dtype=torch.float32,
        )
        if seq_len > 1:
            logits[..., 2] = 0.0
        else:
            logits[..., 0] = 0.0
        return logits


class DummyConstLogitModel:
    def __init__(self, base_logits):
        self.base_logits = torch.tensor(base_logits, dtype=torch.float32)

    def __call__(self, token_ids):
        seq_len = token_ids.size(1)
        logits = self.base_logits.repeat(seq_len, 1)
        return logits.unsqueeze(0)


class DummyTrainModel:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self


def test_load_math_train_uses_local_file(tmp_path, monkeypatch):
    data = [{"problem": "1+1", "answer": "2"}]
    local_path = tmp_path / "math_train.json"
    local_path.write_text(json.dumps(data), encoding="utf-8")

    def fake_get(*args, **kwargs):
        raise AssertionError("requests.get should not be called")

    monkeypatch.setattr(ch06.requests, "get", fake_get)

    loaded = ch06.load_math_train(local_path=local_path, save_copy=True)
    assert loaded == data


def test_sample_response_generates_until_eos():
    model = DummySampleModel()
    tokenizer = DummyTokenizer()

    token_ids, prompt_len, text = ch06.sample_response(
        model=model,
        tokenizer=tokenizer,
        prompt="ignored",
        device="cpu",
        max_new_tokens=5,
        temperature=1.0,
        top_p=0.0,
    )

    assert model.reset_called is True
    assert prompt_len == 2
    assert text == "X"
    assert token_ids.tolist() == [5, 6, 2]


def test_reward_rlvr_requires_boxed_answers():
    assert ch06.reward_rlvr("Result \\boxed{2}", "2") == 1.0
    assert ch06.reward_rlvr("Answer is 2", "2") == 0.0


def test_sequence_logprob_matches_manual_sum():
    token_ids = torch.tensor([1, 2, 0, 2], dtype=torch.long)
    model = DummyConstLogitModel([0.0, 0.2, 0.4])
    prompt_len = 2

    out = ch06.sequence_logprob(model, token_ids, prompt_len)

    logprobs = torch.log_softmax(model.base_logits, dim=-1)
    selected = logprobs[token_ids[1:]]
    expected = torch.sum(selected[prompt_len - 1:])

    assert torch.allclose(out, expected)


def test_compute_grpo_loss_restores_training_and_returns_stats(
    monkeypatch,
):
    samples = iter(
        [
            (torch.tensor([1, 2, 3]), 2, "sample-1"),
            (torch.tensor([1, 2, 4]), 2, "sample-2"),
        ]
    )
    rewards = iter([1.0, 0.0])
    logps = iter([torch.tensor(0.4), torch.tensor(0.2)])

    monkeypatch.setattr(
        ch06,
        "sample_response",
        lambda **kwargs: next(samples),
    )
    monkeypatch.setattr(
        ch06, "reward_rlvr",
        lambda *args, **kwargs: next(rewards),
    )
    monkeypatch.setattr(
        ch06, "sequence_logprob",
        lambda *args, **kwargs: next(logps),
    )

    model = DummyTrainModel()
    example = {"problem": "Q", "answer": "A"}

    result = ch06.compute_grpo_loss(
        model=model,
        tokenizer=DummyTokenizer(),
        example=example,
        device="cpu",
        num_rollouts=2,
        max_new_tokens=4,
        temperature=0.5,
        top_p=0.9,
    )

    expected_rewards = torch.tensor([1.0, 0.0])
    advantages = (
        (expected_rewards - expected_rewards.mean())
        / (expected_rewards.std() + 1e-4)
    )
    expected_logps = torch.stack(
        [torch.tensor(0.4), torch.tensor(0.2)]
    )
    expected_pg_loss = -(advantages.detach() * expected_logps).mean()

    assert model.training is True
    assert result["rewards"] == [1.0, 0.0]
    assert len(result["samples"]) == 2
    assert result["samples"][0]["text"] == "sample-1"
    assert result["samples"][0]["gen_len"] == 1
    assert result["loss"] == result["pg_loss"]
    assert torch.allclose(
        torch.tensor(result["pg_loss"]),
        expected_pg_loss,
    )


def test_save_checkpoint_writes_file(tmp_path):
    class DummyModel:
        def state_dict(self):
            return {"w": torch.tensor([1.0])}

    path = ch06.save_checkpoint(
        model=DummyModel(),
        checkpoint_dir=tmp_path,
        step=7,
        suffix="test",
    )

    assert path.exists()
    assert path.name == "qwen3-0.6B-rlvr-grpo-step00007-test.pth"

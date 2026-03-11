# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import json
import math

import matplotlib
import torch

import reasoning_from_scratch.ch08 as ch08


matplotlib.use("Agg")


class DummyJsonResponse:
    def __init__(self, payload):
        self.payload = payload
        self.raise_called = False

    def raise_for_status(self):
        self.raise_called = True

    def json(self):
        return self.payload


class DummyTokenizer:
    eos_token_id = 99

    def __init__(self):
        self.vocab = {}

    def encode(self, text, chat_wrapped=True):
        ids = []
        for token in text.split():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
            ids.append(self.vocab[token])
        return ids


class DummyConstLogitModel:
    def __init__(self, base_logits):
        self.base_logits = torch.tensor(base_logits, dtype=torch.float32)
        self.training = True

    def __call__(self, input_ids):
        batch_size, seq_len = input_ids.size()
        return self.base_logits.repeat(batch_size, seq_len, 1)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


def test_load_distill_data_uses_cached_file_without_request(
    tmp_path, monkeypatch
):
    cached = [{"problem": "cached"}]
    out = tmp_path / "cached.json"
    out.write_text(json.dumps(cached), encoding="utf-8")

    def fake_get(*args, **kwargs):
        raise AssertionError("requests.get should not be called for cached files")

    monkeypatch.setattr(ch08.requests, "get", fake_get)

    returned = ch08.load_distill_data(
        local_path=out,
        partition="deepseek-r1-math-train",
    )
    assert returned == cached


def test_load_distill_data_downloads_when_missing(tmp_path, monkeypatch):
    calls = {"url": None, "timeout": None}
    payload = [{"problem": "downloaded"}]
    response = DummyJsonResponse(payload)

    def fake_get(url, timeout):
        calls["url"] = url
        calls["timeout"] = timeout
        return response

    out = tmp_path / "downloaded.json"
    monkeypatch.setattr(ch08.requests, "get", fake_get)

    returned = ch08.load_distill_data(
        local_path=out,
        partition="deepseek-r1-math-train",
    )

    assert returned == payload
    assert calls["timeout"] == 30
    assert calls["url"].endswith("deepseek-r1-math-train.json")
    assert response.raise_called is True
    assert json.loads(out.read_text(encoding="utf-8")) == payload


def test_format_distilled_answer_wraps_thinking_and_content():
    entry = {
        "message_thinking": "Reason carefully.",
        "message_content": "The answer is 4.",
    }

    result = ch08.format_distilled_answer(entry)

    assert result == "<think>Reason carefully.</think>\n\nThe answer is 4."


def test_load_reasoning_tokenizer_downloads_and_configures_tokenizer(
    monkeypatch, tmp_path
):
    calls = {"download": None, "tokenizer": None}

    def fake_download_qwen3_small(kind, tokenizer_only, out_dir):
        calls["download"] = {
            "kind": kind,
            "tokenizer_only": tokenizer_only,
            "out_dir": out_dir,
        }

    class FakeTokenizer:
        def __init__(
            self,
            tokenizer_file_path,
            apply_chat_template,
            add_generation_prompt,
            add_thinking,
        ):
            calls["tokenizer"] = {
                "tokenizer_file_path": tokenizer_file_path,
                "apply_chat_template": apply_chat_template,
                "add_generation_prompt": add_generation_prompt,
                "add_thinking": add_thinking,
            }

    monkeypatch.setattr(ch08, "download_qwen3_small", fake_download_qwen3_small)
    monkeypatch.setattr(ch08, "Qwen3Tokenizer", FakeTokenizer)

    tokenizer = ch08.load_reasoning_tokenizer(local_dir=tmp_path)

    assert isinstance(tokenizer, FakeTokenizer)
    assert calls["download"] == {
        "kind": "reasoning",
        "tokenizer_only": True,
        "out_dir": tmp_path,
    }
    assert calls["tokenizer"] == {
        "tokenizer_file_path": tmp_path / "tokenizer-reasoning.json",
        "apply_chat_template": True,
        "add_generation_prompt": True,
        "add_thinking": True,
    }


def test_build_examples_encodes_prompt_answer_and_skips_invalid_rows(
    monkeypatch,
):
    tokenizer = DummyTokenizer()
    monkeypatch.setattr(ch08, "render_prompt", lambda prompt: f"Prompt: {prompt}")

    data = [
        {
            "problem": "1 + 1",
            "message_thinking": "Add the ones.",
            "message_content": "2",
        },
        {
            "problem": "2 + 2",
            "message_thinking": "Oops",
            "message_content": "",
        },
        {
            "problem": "3 + 3",
            "message_content": "6",
        },
    ]

    examples, skipped = ch08.build_examples(data, tokenizer)

    assert len(examples) == 1
    assert skipped == 2
    assert examples[0]["prompt_len"] == len(tokenizer.encode("Prompt: 1 + 1"))
    assert examples[0]["token_ids"][-1] == tokenizer.eos_token_id


def test_compute_length_returns_total_and_answer_only_stats(capsys):
    examples = [
        {"token_ids": [1, 2, 3], "prompt_len": 1},
        {"token_ids": [1, 2, 3, 4, 5], "prompt_len": 2},
    ]

    assert ch08.compute_length(examples) is None
    total_out = capsys.readouterr().out

    assert ch08.compute_length(examples, answer_only=True) is None
    answer_out = capsys.readouterr().out

    assert "Average: 4 tokens" in total_out
    assert "Shortest: 3 tokens (index 0)" in total_out
    assert "Longest: 5 tokens (index 1)" in total_out

    assert "Average: 2 tokens" in answer_out
    assert "Shortest: 2 tokens (index 0)" in answer_out
    assert "Longest: 3 tokens (index 1)" in answer_out


def test_filter_examples_by_max_len_keeps_only_short_examples(capsys):
    examples = [
        {"token_ids": [1, 2, 3], "prompt_len": 1},
        {"token_ids": [1, 2, 3, 4, 5], "prompt_len": 2},
    ]

    filtered = ch08.filter_examples_by_max_len(examples, max_len=3)
    _ = capsys.readouterr()

    assert filtered == [examples[0]]


def test_compute_example_loss_and_evaluate_examples_match_manual_values():
    example = {"token_ids": [0, 2, 1, 2], "prompt_len": 2}
    model = DummyConstLogitModel([0.1, 0.3, 0.6])

    loss = ch08.compute_example_loss(model, example, device="cpu")

    logits = model.base_logits.repeat(3, 1)
    target_ids = torch.tensor(example["token_ids"][1:], dtype=torch.long)
    expected = torch.nn.functional.cross_entropy(
        logits[1:],
        target_ids[1:],
    )

    assert torch.allclose(loss, expected)

    avg_loss = ch08.evaluate_examples(
        model=model,
        examples=[example, example],
        device="cpu",
    )
    assert math.isclose(avg_loss, expected.item(), rel_tol=1e-6)


def test_append_csv_metrics_writes_header_once(tmp_path):
    csv_path = tmp_path / "metrics.csv"

    ch08.append_csv_metrics(csv_path, 1, 10, 1.23, 0.91)
    ch08.append_csv_metrics(csv_path, 2, 20, 0.98, 0.77)

    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "epoch,total_steps,train_loss,val_loss",
        "1,10,1.230000,0.910000",
        "2,20,0.980000,0.770000",
    ]


def test_plot_distill_metrics_reads_csv_and_calls_show(tmp_path, monkeypatch):
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        (
            "epoch,total_steps,train_loss,val_loss\n"
            "1,10,1.2,1.1\n"
            "1,20,1.0,0.9\n"
            "2,30,0.8,0.7\n"
        ),
        encoding="utf-8",
    )

    called = {"show": False}

    def fake_show():
        called["show"] = True

    monkeypatch.setattr(ch08.plt, "show", fake_show)

    ch08.plot_distill_metrics(csv_path=csv_path)

    assert called["show"] is True


def test_train_distillation_saves_checkpoints_and_metrics(
    tmp_path, monkeypatch
):
    def fake_compute_example_loss(model, example, device):
        return (model.weight ** 2).sum()

    monkeypatch.setattr(ch08, "compute_example_loss", fake_compute_example_loss)

    model = torch.nn.Linear(1, 1, bias=False)
    train_examples = [{"token_ids": [1, 2], "prompt_len": 1}] * 2
    val_examples = [{"token_ids": [1, 2], "prompt_len": 1}]
    csv_path = tmp_path / "distill.csv"
    checkpoint_dir = tmp_path / "checkpoints"

    returned = ch08.train_distillation(
        model=model,
        train_examples=train_examples,
        val_examples=val_examples,
        device="cpu",
        epochs=1,
        lr=0.1,
        log_every=1,
        checkpoint_dir=checkpoint_dir,
        csv_log_path=csv_path,
    )

    assert returned is model
    assert csv_path.exists()
    assert any(checkpoint_dir.glob("*.pth"))

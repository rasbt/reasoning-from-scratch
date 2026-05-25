# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import math
import os

import matplotlib
import pytest
import torch
import reasoning_from_scratch.ch07 as ch07

matplotlib.use("Agg")


class DummyResponse:
    def __init__(self, content):
        self.content = content
        self.raise_called = False

    def raise_for_status(self):
        self.raise_called = True


class DummyConstLogitModel:
    def __init__(self, base_logits):
        self.base_logits = torch.tensor(base_logits, dtype=torch.float32)

    def __call__(self, token_ids):
        batch_size, seq_len = token_ids.size()
        return self.base_logits.repeat(batch_size, seq_len, 1)


class DummyTagTokenizer:
    def __init__(self):
        self._next_id = 1
        self._vocab = {
            "<think>": ch07.THINK_TOKEN_ID,
            "</think>": ch07.END_THINK_TOKEN_ID,
        }

    def encode(self, text):
        ids = []
        for token in text.split():
            if token not in self._vocab:
                while self._next_id in (
                    ch07.THINK_TOKEN_ID,
                    ch07.END_THINK_TOKEN_ID,
                ):
                    self._next_id += 1
                self._vocab[token] = self._next_id
                self._next_id += 1
            ids.append(self._vocab[token])
        return ids


run_real_download = os.environ.get("RUN_REAL_DOWNLOAD_TESTS", "0") == "1"
skip_expensive = os.environ.get("SKIP_EXPENSIVE", "0") == "1"


def test_moving_average_uses_trailing_window_and_minimum_window_size():
    values = [1.0, 3.0, 5.0, 7.0]
    smoothed = ch07.moving_average(values, window_fraction=0.5)
    assert smoothed == [1.0, 2.0, 4.0, 6.0]

    # window_fraction=0 still uses a 1-step window via max(1, ...)
    no_smoothing = ch07.moving_average(values, window_fraction=0.0)
    assert no_smoothing == values


def test_compute_advantage_stats_matches_manual_standardization():
    rewards = [1.0, 2.0, 4.0]
    advantages, adv_avg, adv_std = ch07.compute_advantage_stats(rewards)

    rewards_t = torch.tensor(rewards)
    expected = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-4)

    assert torch.allclose(advantages, expected)
    assert math.isclose(adv_avg, expected.mean().item(), rel_tol=1e-6)
    assert math.isclose(adv_std, expected.std().item(), rel_tol=1e-6)


def test_sequence_logprob_and_entropy_matches_manual_values():
    token_ids = torch.tensor([0, 2, 1, 2], dtype=torch.long)
    prompt_len = 2
    model = DummyConstLogitModel([0.1, 0.3, 0.6])

    out_logp, out_entropy = ch07.sequence_logprob_and_entropy(
        model=model,
        token_ids=token_ids,
        prompt_len=prompt_len,
    )

    logprobs = torch.log_softmax(model.base_logits, dim=-1)
    selected = logprobs[token_ids[1:]]
    expected_logp = selected[prompt_len - 1 :].sum()

    probs = torch.exp(logprobs)
    expected_step_entropy = -(probs * logprobs).sum()
    expected_entropy = expected_step_entropy

    assert torch.allclose(out_logp, expected_logp)
    assert torch.allclose(out_entropy, expected_entropy)


def test_sequence_logprob_and_entropy_handles_empty_answer_segment():
    token_ids = torch.tensor([1, 2], dtype=torch.long)
    model = DummyConstLogitModel([0.0, 0.2, 0.4])

    out_logp, out_entropy = ch07.sequence_logprob_and_entropy(
        model=model,
        token_ids=token_ids,
        prompt_len=3,
    )

    assert out_logp.item() == 0.0
    assert out_entropy.item() == 0.0


def test_download_from_github_uses_cached_file_without_request(
    tmp_path, monkeypatch
):
    out = tmp_path / "cached.txt"
    out.write_bytes(b"cached-bytes")

    def fake_get(*args, **kwargs):
        raise AssertionError("requests.get should not be called for cached files")

    monkeypatch.setattr(ch07.requests, "get", fake_get)

    returned = ch07.download_from_github("does/not/matter.txt", out=out)
    assert returned == out
    assert out.read_bytes() == b"cached-bytes"


def test_download_from_github_downloads_when_missing(tmp_path, monkeypatch):
    calls = {"url": None}
    response = DummyResponse(b"new-content")

    def fake_get(url):
        calls["url"] = url
        return response

    out = tmp_path / "downloaded.txt"
    monkeypatch.setattr(ch07.requests, "get", fake_get)

    ch07.download_from_github("ch07/README.md", out=out)

    assert calls["url"].endswith("ch07/README.md")
    assert response.raise_called is True
    assert out.read_bytes() == b"new-content"


@pytest.mark.skipif(
    skip_expensive or not run_real_download,
    reason="Set RUN_REAL_DOWNLOAD_TESTS=1 and unset SKIP_EXPENSIVE to run real download tests",
)
def test_download_from_github_real_download(tmp_path):
    out = tmp_path / "ch07-readme.md"

    ch07.download_from_github("ch07/README.md", out=out)

    text = out.read_text(encoding="utf-8")
    assert out.exists()
    assert "# Chapter 7" in text
    assert "Improving Policy Optimization" in text


def test_plot_grpo_metrics_reads_csv_and_calls_show(tmp_path, monkeypatch):
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        (
            "step,pg_loss,mean_reward,eval_acc\n"
            "10,1.2,0.1,\n"
            "20,1.0,0.2,0.5\n"
            "30,0.8,0.4,\n"
        ),
        encoding="utf-8",
    )

    called = {"show": False}

    def fake_show():
        called["show"] = True

    monkeypatch.setattr(ch07.plt, "show", fake_show)

    out = tmp_path / "plot.png"
    ch07.plot_grpo_metrics(
        csv_path=csv_path,
        columns=["pg_loss", "mean_reward", "eval_acc", "missing_metric"],
        save_as=out,
    )

    assert called["show"] is True
    assert out.exists()


def test_reward_format_cases_from_prompt_example():
    tokenizer = DummyTagTokenizer()
    prompt = "Calculate ..."

    cases = [
        ("Correct order", "Let's ... <think> ... </think> ...", 1.0),
        ("Invalid opening tag", "Let's ... <th1nk> ... </think> ...", 0.0),
        ("Reversed order", "Let's ... </think> ... <think> ...", 0.0),
        ("Missing </think>", "Let's ... <think> ...", 0.0),
    ]

    for _, rollout, expected in cases:
        token_ids = tokenizer.encode(prompt + " " + rollout)
        prompt_len = len(tokenizer.encode(prompt))

        reward = ch07.reward_format(
            token_ids=torch.tensor(token_ids),
            prompt_len=prompt_len,
        )

        assert reward == expected


###################################################
# PPO-Style clipping implementation check
###################################################

# See https://livebook.manning.com/forum?comment=584433 for discussion


CHAPTER_CLIP_EPS = 10.0
PRACTICAL_CLIP_EPS = 0.2


def _old_torch_where_objective(advantage, ratio, clip_eps):
    # Previous listing 7.8 formulation
    advantage = torch.tensor([advantage])
    ratio = torch.tensor([ratio])
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    unclipped = ratio * advantage
    clipped = clipped_ratio * advantage
    return torch.where(
        advantage >= 0,
        torch.minimum(unclipped, clipped),
        torch.maximum(unclipped, clipped),
    )


def _correct_torch_minimum_objective(advantage, ratio, clip_eps):
    # Corrected listing 7.8 formulation (PPO-style clipping)
    advantage = torch.tensor([advantage])
    ratio = torch.tensor([ratio])
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    unclipped = ratio * advantage
    clipped = clipped_ratio * advantage
    return torch.minimum(unclipped, clipped)


def test_positive_advantage_clips_ratio_above_upper_bound_in_both_versions():
    # Positive advantages should clip ratios above
    # the upper bound (correct in both versions)
    old_where_obj = _old_torch_where_objective(1.0, 20.0, CHAPTER_CLIP_EPS)
    correct_minimum_obj = _correct_torch_minimum_objective(1.0, 20.0, CHAPTER_CLIP_EPS)

    assert old_where_obj.item() == pytest.approx(11.0)
    assert correct_minimum_obj.item() == pytest.approx(11.0)

    old_where_obj = _old_torch_where_objective(1.0, 1.5, PRACTICAL_CLIP_EPS)
    correct_minimum_obj = _correct_torch_minimum_objective(1.0, 1.5, PRACTICAL_CLIP_EPS)

    assert old_where_obj.item() == pytest.approx(1.2)
    assert correct_minimum_obj.item() == pytest.approx(1.2)


def test_positive_advantage_does_not_use_lower_clipping_bound_in_both_versions():
    # Positive advantages should not use the lower
    # clipping bound (correct in both versions)
    old_where_obj = _old_torch_where_objective(1.0, 0.5, CHAPTER_CLIP_EPS)
    correct_minimum_obj = _correct_torch_minimum_objective(1.0, 0.5, CHAPTER_CLIP_EPS)

    assert old_where_obj.item() == pytest.approx(0.5)
    assert correct_minimum_obj.item() == pytest.approx(0.5)

    old_where_obj = _old_torch_where_objective(1.0, 0.5, PRACTICAL_CLIP_EPS)
    correct_minimum_obj = _correct_torch_minimum_objective(1.0, 0.5, PRACTICAL_CLIP_EPS)

    assert old_where_obj.item() == pytest.approx(0.5)
    assert correct_minimum_obj.item() == pytest.approx(0.5)


def test_negative_advantage_lower_bound_behavior_differs_by_clip_eps():
    # With clip_eps = 10.0, the lower bound is negative
    # and therefore inactive for positive policy ratios
    old_where_obj = _old_torch_where_objective(-1.0, 0.5, CHAPTER_CLIP_EPS)
    correct_minimum_obj = _correct_torch_minimum_objective(-1.0, 0.5, CHAPTER_CLIP_EPS)

    assert old_where_obj.item() == pytest.approx(-0.5)
    assert correct_minimum_obj.item() == pytest.approx(-0.5)

    # For a negative advantage, use the lower clipping bound
    # (wrong in previous version)
    old_where_obj = _old_torch_where_objective(-1.0, 0.5, PRACTICAL_CLIP_EPS)
    correct_minimum_obj = _correct_torch_minimum_objective(-1.0, 0.5, PRACTICAL_CLIP_EPS)

    assert old_where_obj.item() != pytest.approx(correct_minimum_obj.item())
    assert correct_minimum_obj.item() == pytest.approx(-0.8)


def test_negative_advantage_does_not_clip_upper_bound_in_correct_version():
    # For a negative advantage, do not clip ratios above
    # the upper bound (wrong in previous version)
    old_where_obj = _old_torch_where_objective(-1.0, 20.0, CHAPTER_CLIP_EPS)
    correct_minimum_obj = _correct_torch_minimum_objective(-1.0, 20.0, CHAPTER_CLIP_EPS)

    assert old_where_obj.item() != pytest.approx(correct_minimum_obj.item())
    assert correct_minimum_obj.item() == pytest.approx(-20.0)

    old_where_obj = _old_torch_where_objective(-1.0, 1.5, PRACTICAL_CLIP_EPS)
    correct_minimum_obj = _correct_torch_minimum_objective(-1.0, 1.5, PRACTICAL_CLIP_EPS)

    assert old_where_obj.item() != pytest.approx(correct_minimum_obj.item())
    assert correct_minimum_obj.item() == pytest.approx(-1.5)


def test_ratio_inside_clipping_interval_remains_unchanged_in_both_versions():
    # Ratios inside the clipping interval should remain unchanged
    # (correct in both versions)
    old_where_obj = _old_torch_where_objective(-1.0, 1.0, CHAPTER_CLIP_EPS)
    correct_minimum_obj = _correct_torch_minimum_objective(-1.0, 1.0, CHAPTER_CLIP_EPS)

    assert old_where_obj.item() == pytest.approx(-1.0)
    assert correct_minimum_obj.item() == pytest.approx(-1.0)

    old_where_obj = _old_torch_where_objective(-1.0, 1.0, PRACTICAL_CLIP_EPS)
    correct_minimum_obj = _correct_torch_minimum_objective(-1.0, 1.0, PRACTICAL_CLIP_EPS)

    assert old_where_obj.item() == pytest.approx(-1.0)
    assert correct_minimum_obj.item() == pytest.approx(-1.0)

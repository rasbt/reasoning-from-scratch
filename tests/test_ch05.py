# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch


import math
import torch

import reasoning_from_scratch.ch05 as ch05


class DummyTokenizer:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}
        self.eos_token_id = 0

    def encode(self, text):
        if text in self.mapping:
            return self.mapping[text]
        return [len(text) % 10]


class DummyLogitModel:
    def __init__(self, base_logits):
        self.base_logits = torch.tensor(base_logits, dtype=torch.float32)

    def __call__(self, token_ids):
        seq_len = token_ids.size(1)
        logits = self.base_logits.repeat(seq_len, 1)
        return logits.unsqueeze(0)


def test_heuristic_score_rewards_boxed_answers_more_than_numbers():
    boxed = "Result \\boxed{7}"
    number_only = "The answer is 7"

    boxed_score = ch05.heuristic_score(boxed)
    number_score = ch05.heuristic_score(number_only)

    expected_boxed = 2.0 + 1.5 * math.exp(-len(boxed) / 500.0)
    expected_number = 1.0 + 1.5 * math.exp(-len(number_only) / 500.0)

    assert math.isclose(boxed_score, expected_boxed, rel_tol=1e-6)
    assert math.isclose(number_score, expected_number, rel_tol=1e-6)
    assert boxed_score > number_score


def test_heuristic_score_adds_fulltext_bonus_when_no_number():
    response = "No numeric result here"
    score = ch05.heuristic_score(
        response, brevity_bonus=100.0, fulltext_bonus=0.3
    )

    expected = 0.3 + 1.5 * math.exp(-len(response) / 100.0)
    assert math.isclose(score, expected, rel_tol=1e-6)


def test_avg_logprob_answer_uses_answer_token_logprobs():
    tokenizer = DummyTokenizer(
        mapping={
            "prompt": [1, 2],
            "answer": [3, 4],
        }
    )
    base_logits = [0.0, 0.1, 0.2, 0.3, 0.4]
    model = DummyLogitModel(base_logits)

    expected_logprobs = torch.log_softmax(
        model.base_logits, dim=-1
    )
    expected = torch.mean(expected_logprobs[[3, 4]]).item()

    out = ch05.avg_logprob_answer(
        model=model,
        tokenizer=tokenizer,
        prompt="prompt",
        answer="answer",
        device="cpu",
    )

    assert math.isclose(out, expected, rel_tol=1e-6)


def test_prompt_builders_embed_question_and_context():
    raw_prompt = "What is 1+1?"
    draft = "It is \\boxed{2}."
    critique_text = "Looks fine."

    critique_prompt = ch05.make_critique_prompt(raw_prompt, draft)
    assert "meticulous reviewer" in critique_prompt
    assert f"Question:\n{raw_prompt}" in critique_prompt
    assert f"Draft answer:\n{draft}" in critique_prompt
    assert critique_prompt.strip().endswith("Critique:")

    refine_prompt = ch05.make_refine_prompt(
        raw_prompt, draft, critique_text
    )
    assert "Revised answer:" in refine_prompt
    assert f"Previous answer:\n{draft}" in refine_prompt
    assert f"Critique:\n{critique_text}" in refine_prompt
    assert refine_prompt.strip().endswith("Revised answer:")


def test_self_refinement_loop_accepts_improving_revisions(monkeypatch):
    responses = iter(
        [
            "initial draft",                # initial generation
            "first critique",               # critique 1
            "draft with more detail",       # refine 1 (accepted)
            "second critique",              # critique 2
            "bad",                          # refine 2 (rejected)
        ]
    )
    prompts_seen = []

    def fake_generate_text_stream_concat_flex(**kwargs):
        prompts_seen.append(kwargs.get("prompt"))
        return next(responses)

    monkeypatch.setattr(
        ch05, "generate_text_stream_concat_flex",
        fake_generate_text_stream_concat_flex
    )

    def score_fn(answer, prompt):
        return len(answer)

    result = ch05.self_refinement_loop(
        model=None,
        tokenizer=DummyTokenizer(),
        raw_prompt="Compute something",
        device="cpu",
        iterations=2,
        score_fn=score_fn,
        prompt_renderer=lambda x: f"Q: {x}",
        temperature=0.3,
        top_p=0.8,
    )

    assert result["final_extracted"] == "draft with more detail"
    assert len(result["steps"]) == 2
    assert result["steps"][0]["draft_full"] == "initial draft"
    assert result["steps"][0]["revised_full"] == "draft with more detail"
    assert result["steps"][1]["draft_full"] == "draft with more detail"
    assert result["steps"][1]["revised_full"] == "bad"
    assert result["steps"][0]["score_after"] >= result["steps"][0]["score_before"]
    assert result["steps"][1]["score_after"] < result["steps"][1]["score_before"]
    assert len(prompts_seen) == 5

# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from reasoning_from_scratch.ch03 import render_prompt


REPO_ROOT = Path(__file__).resolve().parents[1]
HF_DIR = REPO_ROOT / "ch08" / "06_use_via_huggingface" / "export_approach"
if str(HF_DIR) not in sys.path:
    sys.path.insert(0, str(HF_DIR))

from hf_export import build_hf_config, build_hf_tokenizer, export_qwen3_to_hf  # noqa: E402
import hf_trainer  # noqa: E402
from hf_qwen3 import Qwen3FromScratchForCausalLM  # noqa: E402


TINY_MODEL_CFG = {
    "vocab_size": 64,
    "context_length": 32,
    "emb_dim": 16,
    "n_heads": 4,
    "n_layers": 2,
    "hidden_dim": 48,
    "head_dim": 4,
    "qk_norm": True,
    "n_kv_groups": 2,
    "rope_base": 10_000.0,
    "dtype": torch.float32,
}


def write_test_tokenizer(path):
    vocab = {
        "[UNK]": 0,
        "<|endoftext|>": 1,
        "<|im_start|>": 2,
        "<|im_end|>": 3,
        "<think>": 4,
        "</think>": 5,
        "hello": 6,
        "world": 7,
        "2": 8,
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))


def build_export_dir(tmp_path, tokenizer_kind):
    tokenizer_path = tmp_path / f"tokenizer-{tokenizer_kind}.json"
    write_test_tokenizer(tokenizer_path)

    tokenizer = build_hf_tokenizer(
        tokenizer_path=tokenizer_path,
        tokenizer_kind=tokenizer_kind,
        model_max_length=TINY_MODEL_CFG["context_length"],
    )
    config = build_hf_config(
        tokenizer=tokenizer,
        tokenizer_kind=tokenizer_kind,
        model_cfg=TINY_MODEL_CFG,
    )

    torch.manual_seed(123)
    model = Qwen3FromScratchForCausalLM(config).eval()
    export_dir = tmp_path / f"export-{tokenizer_kind}-{tmp_path.name}"
    export_qwen3_to_hf(
        output_dir=export_dir,
        state_dict=model.state_dict(),
        tokenizer_path=tokenizer_path,
        tokenizer_kind=tokenizer_kind,
        model_cfg=TINY_MODEL_CFG,
    )
    return export_dir, model


def test_export_roundtrip_supports_auto_model_and_generate(tmp_path):
    export_dir, reference_model = build_export_dir(tmp_path, tokenizer_kind="base")

    config = AutoConfig.from_pretrained(export_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(export_dir, trust_remote_code=True)
    loaded_model = AutoModelForCausalLM.from_pretrained(
        export_dir,
        trust_remote_code=True,
    ).eval()

    assert config.tokenizer_kind == "base"
    assert (export_dir / "hf_qwen3.py").exists()

    inputs = tokenizer(
        "hello world",
        return_tensors="pt",
        add_special_tokens=False,
    )

    with torch.no_grad():
        reference_logits = reference_model(**inputs).logits
        loaded_logits = loaded_model(**inputs).logits

    torch.testing.assert_close(loaded_logits, reference_logits)

    generated = loaded_model.generate(
        **inputs,
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    assert generated.shape == (1, inputs["input_ids"].shape[1] + 2)

    generated_no_cache = loaded_model.generate(
        **inputs,
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )
    torch.testing.assert_close(generated, generated_no_cache)


def test_reasoning_export_keeps_chat_template_and_answer_only_labels(tmp_path):
    export_dir, _ = build_export_dir(tmp_path, tokenizer_kind="reasoning")

    tokenizer = AutoTokenizer.from_pretrained(export_dir, trust_remote_code=True)
    assert (export_dir / "chat_template.jinja").exists()

    prompt = hf_trainer.wrap_prompt(
        prompt=render_prompt("1 + 1"),
        tokenizer=tokenizer,
        tokenizer_kind="reasoning",
    )
    assert prompt.startswith("<|im_start|>user\n")
    assert prompt.endswith("<|im_start|>assistant\n")

    records, skipped = hf_trainer.build_records(
        data=[
            {
                "problem": "1 + 1",
                "message_thinking": "Add one and one.",
                "message_content": "2",
            }
        ],
        tokenizer=tokenizer,
        tokenizer_kind="reasoning",
        max_seq_len=128,
    )

    assert skipped == 0
    assert len(records) == 1

    prompt_len = len(
        tokenizer(
            prompt,
            add_special_tokens=False,
        ).input_ids
    )
    labels = records[0]["labels"]

    assert labels[:prompt_len] == [-100] * prompt_len
    assert any(label != -100 for label in labels[prompt_len:])

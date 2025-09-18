# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import importlib
import os
import pytest
import torch
from pathlib import Path

from reasoning_from_scratch.ch02 import (
    generate_text_basic_cache,
)
from reasoning_from_scratch.qwen3 import (
    download_qwen3_small,
    load_hf_weights_into_qwen,
    Qwen3Tokenizer,
    Qwen3Model,
    QWEN_CONFIG_06_B,
)

from reasoning_from_scratch.qwen3_optimized import (
    Qwen3Model as Qwen3ModelOptimized,
    generate_text_basic_cache as generate_text_basic_cache_optimized,
)


skip_expensive = os.environ.get("SKIP_EXPENSIVE", "0") == "1"
transformers_installed = importlib.util.find_spec("transformers") is not None


@torch.inference_mode()
@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_qwen3_base_equivalence_with_transformers():

    from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

    # Tiny config so the test is fast
    cfg = {
        "vocab_size": 257,
        "context_length": 8,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "hidden_dim": 64,
        "head_dim": 8,
        "qk_norm": True,
        "n_kv_groups": 2,
        "rope_base": 1_000_000.0,
        "dtype": torch.float32,
    }
    model = Qwen3Model(cfg)

    hf_cfg = Qwen3Config(
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["context_length"],
        hidden_size=cfg["emb_dim"],
        num_attention_heads=cfg["n_heads"],
        num_hidden_layers=cfg["n_layers"],
        intermediate_size=cfg["hidden_dim"],
        head_dim=cfg["head_dim"],
        num_key_value_heads=cfg["n_kv_groups"],
        rope_theta=cfg["rope_base"],
        tie_word_embeddings=False,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    hf_model = Qwen3ForCausalLM(hf_cfg)

    hf_state = hf_model.state_dict()
    param_config = {"n_layers": cfg["n_layers"], "hidden_dim": cfg["hidden_dim"]}
    load_hf_weights_into_qwen(model, param_config, hf_state)

    x = torch.randint(0, cfg["vocab_size"], (2, cfg["context_length"]), dtype=torch.long)
    ours_logits = model(x)
    theirs_logits = hf_model(x).logits
    torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(skip_expensive, reason="Skipping expensive test on CI")
@pytest.mark.parametrize("reasoning", [False, True])
def test_qwen3_vs_optimized_qwen3(reasoning):

    device = "cpu"  # get_device()

    # Download and init tokenizer
    kind = "reasoning" if reasoning else "base"
    download_qwen3_small(kind=kind, tokenizer_only=False, out_dir="qwen3")
    tokenizer_path = Path("qwen3") / (
        "tokenizer-reasoning.json" if reasoning else "tokenizer-base.json"
    )
    model_path = Path("qwen3") / (
        "qwen3-0.6B-reasoning.pth" if reasoning else "qwen3-0.6B-base.pth"
    )
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_path,
        apply_chat_template=True if reasoning else False,
        add_generation_prompt=True if reasoning else False,
        add_thinking=True if reasoning else False,
    )

    # Models
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    model_optimized = Qwen3ModelOptimized(QWEN_CONFIG_06_B, exact=True)
    model_optimized.load_state_dict(torch.load(model_path, map_location=device))
    model_optimized.to(device)
    model_optimized.eval()

    # Prompts
    prompts = [
        "Explain large language models in two sentences.",
        "Explain large language models in one sentence.",
        "1+1?"
    ]

    single_inputs = [
        torch.tensor(tokenizer.encode(p), device=device).unsqueeze(0)
        for p in prompts
    ]

    # Generation
    max_new_tokens = 12  # cheap but enough to check consistency
    outputs_simple = []
    for input_ids in single_inputs:
        out = generate_text_basic_cache(
            model=model,
            token_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        outputs_simple.append(out[0].tolist())

    outputs_optimized = []
    for input_ids in single_inputs:
        out = generate_text_basic_cache_optimized(
            model=model_optimized,
            token_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        outputs_optimized.append(out[0])

    # Check equivalency
    for idx, out_single in enumerate(outputs_simple):
        out_batch = outputs_optimized[idx].tolist()

        text_single = tokenizer.decode(out_single)
        text_batch = tokenizer.decode(out_batch)

        # Assert the text beyond the first token is identical
        assert text_single == text_batch, (
            f"Mismatch after first token at prompt {idx}:\n"
            f"single={text_single}\n"
            f"batched={text_batch}"
        )

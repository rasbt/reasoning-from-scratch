# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import json
import os
import shutil
import sys
import types
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import reasoning_from_scratch.appendix_c as appendix_c
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer


run_real_download = (
    os.environ.get("RUN_REAL_DOWNLOAD_TESTS", "0") == "1"
    or os.environ.get("RUN_APPENDIX_D_REAL_DOWNLOAD", "0") == "1"
)
skip_expensive = os.environ.get("SKIP_EXPENSIVE", "0") == "1"


def install_fake_hf_modules(monkeypatch, *, snapshot_download, hf_hub_download, load_file):
    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = snapshot_download
    fake_hf.hf_hub_download = hf_hub_download

    fake_safetensors = types.ModuleType("safetensors")
    fake_safetensors_torch = types.ModuleType("safetensors.torch")
    fake_safetensors_torch.load_file = load_file

    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "safetensors", fake_safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_safetensors_torch)


def write_minimal_qwen_tokenizer(path):
    vocab = {
        token: idx for idx, token in enumerate(Qwen3Tokenizer._SPECIALS)
    }
    for token in ("user", "assistant", "Explain", "large", "language", "models"):
        vocab[token] = len(vocab)

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))
    return vocab


def test_download_from_huggingface_from_snapshots_merges_multi_shard_weights(
    tmp_path, monkeypatch
):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    index = {
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.norm.weight": "model-00002-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
        }
    }
    (repo_dir / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )
    for shard_name in set(index["weight_map"].values()):
        (repo_dir / shard_name).write_text("", encoding="utf-8")

    calls = {"loaded": []}

    def fake_snapshot_download(repo_id, local_dir):
        calls["snapshot"] = {
            "repo_id": repo_id,
            "local_dir": local_dir,
        }
        return str(repo_dir)

    def fake_hf_hub_download(*args, **kwargs):
        raise AssertionError("hf_hub_download should not be used for indexed shards")

    def fake_load_file(path):
        calls["loaded"].append(Path(path).name)
        if Path(path).name == "model-00001-of-00002.safetensors":
            return {
                "model.embed_tokens.weight": torch.tensor([1.0]),
                "model.layers.0.self_attn.q_proj.weight": torch.tensor([2.0]),
            }
        if Path(path).name == "model-00002-of-00002.safetensors":
            return {"model.norm.weight": torch.tensor([3.0])}
        raise AssertionError(f"Unexpected shard path: {path}")

    install_fake_hf_modules(
        monkeypatch,
        snapshot_download=fake_snapshot_download,
        hf_hub_download=fake_hf_hub_download,
        load_file=fake_load_file,
    )

    local_dir = tmp_path / "download"
    weights = appendix_c.download_from_huggingface_from_snapshots(
        repo_id="Qwen/Qwen3-4B-Base",
        local_dir=local_dir,
    )

    assert calls["snapshot"] == {
        "repo_id": "Qwen/Qwen3-4B-Base",
        "local_dir": local_dir,
    }
    assert sorted(calls["loaded"]) == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert torch.equal(weights["model.embed_tokens.weight"], torch.tensor([1.0]))
    assert torch.equal(
        weights["model.layers.0.self_attn.q_proj.weight"],
        torch.tensor([2.0]),
    )
    assert torch.equal(weights["model.norm.weight"], torch.tensor([3.0]))


def test_download_from_huggingface_from_snapshots_loads_single_safetensor(
    tmp_path, monkeypatch
):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "model.safetensors").write_text("", encoding="utf-8")

    calls = {}

    def fake_snapshot_download(repo_id, local_dir):
        calls["snapshot"] = {
            "repo_id": repo_id,
            "local_dir": local_dir,
        }
        return str(repo_dir)

    def fake_hf_hub_download(repo_id, filename, local_dir):
        calls["download"] = {
            "repo_id": repo_id,
            "filename": filename,
            "local_dir": local_dir,
        }
        return str(repo_dir / filename)

    def fake_load_file(path):
        calls["loaded_path"] = Path(path).name
        return {"model.embed_tokens.weight": torch.tensor([4.0])}

    install_fake_hf_modules(
        monkeypatch,
        snapshot_download=fake_snapshot_download,
        hf_hub_download=fake_hf_hub_download,
        load_file=fake_load_file,
    )

    local_dir = tmp_path / "download"
    weights = appendix_c.download_from_huggingface_from_snapshots(
        repo_id="Qwen/Qwen3-4B-Base",
        local_dir=local_dir,
    )

    assert calls["snapshot"] == {
        "repo_id": "Qwen/Qwen3-4B-Base",
        "local_dir": local_dir,
    }
    assert calls["download"] == {
        "repo_id": "Qwen/Qwen3-4B-Base",
        "filename": "model.safetensors",
        "local_dir": local_dir,
    }
    assert calls["loaded_path"] == "model.safetensors"
    assert torch.equal(weights["model.embed_tokens.weight"], torch.tensor([4.0]))


def test_appendix_d_tokenizer_copy_uses_base_eos_behavior(tmp_path):
    tokenizer_src = tmp_path / "tokenizer.json"
    vocab = write_minimal_qwen_tokenizer(tokenizer_src)

    tokenizer_base = tmp_path / "tokenizer-base.json"
    if not tokenizer_base.exists():
        shutil.copyfile(tokenizer_src, tokenizer_base)

    base_tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_base)
    original_name_tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_src)

    assert base_tokenizer.eos_token == "<|endoftext|>"
    assert base_tokenizer.eos_token_id == vocab["<|endoftext|>"]
    assert original_name_tokenizer.eos_token == "<|im_end|>"
    assert original_name_tokenizer.eos_token_id == vocab["<|im_end|>"]
    assert base_tokenizer.encode("<|endoftext|>") == [vocab["<|endoftext|>"]]
    assert base_tokenizer.encode("<|im_end|>") == [vocab["<|im_end|>"]]


@pytest.mark.skipif(
    skip_expensive or not run_real_download,
    reason="Set RUN_REAL_DOWNLOAD_TESTS=1 and unset SKIP_EXPENSIVE to run real download tests",
)
def test_appendix_d_real_download_1_7b_base_snapshot(tmp_path):
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("safetensors.torch")

    local_dir = tmp_path / "qwen3-1.7b-base"
    weights = appendix_c.download_from_huggingface_from_snapshots(
        repo_id="Qwen/Qwen3-1.7B-Base",
        local_dir=local_dir,
    )

    tokenizer_src = local_dir / "tokenizer.json"
    tokenizer_base = local_dir / "tokenizer-base.json"
    shutil.copyfile(tokenizer_src, tokenizer_base)
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_base)

    cfg = appendix_c.QWEN3_CONFIG_1_7B
    assert tokenizer.eos_token == "<|endoftext|>"
    assert "model.embed_tokens.weight" in weights
    assert "model.layers.0.self_attn.q_proj.weight" in weights
    assert "model.layers.0.self_attn.k_proj.weight" in weights
    assert "model.norm.weight" in weights
    assert tuple(weights["model.embed_tokens.weight"].shape) == (
        cfg["vocab_size"],
        cfg["emb_dim"],
    )
    assert tuple(weights["model.layers.0.self_attn.q_proj.weight"].shape) == (
        cfg["n_heads"] * cfg["head_dim"],
        cfg["emb_dim"],
    )
    assert tuple(weights["model.layers.0.self_attn.k_proj.weight"].shape) == (
        cfg["n_kv_groups"] * cfg["head_dim"],
        cfg["emb_dim"],
    )
    assert tuple(weights["model.norm.weight"].shape) == (cfg["emb_dim"],)

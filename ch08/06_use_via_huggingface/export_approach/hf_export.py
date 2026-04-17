# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import json
from pathlib import Path

import torch
from transformers import GenerationConfig, PreTrainedTokenizerFast

from hf_qwen3 import (
    Qwen3FromScratchConfig,
    Qwen3FromScratchForCausalLM,
)
from reasoning_from_scratch.qwen3 import QWEN_CONFIG_06_B, download_qwen3_small


MODEL_FILENAMES = {
    "base": "qwen3-0.6B-base.pth",
    "reasoning": "qwen3-0.6B-reasoning.pth",
}

TOKENIZER_FILENAMES = {
    "base": "tokenizer-base.json",
    "reasoning": "tokenizer-reasoning.json",
}

EOS_TOKENS = {
    "base": "<|endoftext|>",
    "reasoning": "<|im_end|>",
}

PAD_TOKEN = "<|endoftext|>"

REASONING_CHAT_TEMPLATE = """{% for message in messages %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""


def torch_dtype_to_name(dtype_value):
    # The scratch config stores dtypes as torch objects such as torch.bfloat16,
    # but the Hugging Face config is saved as JSON, so we convert them to strings
    # like "bfloat16" before writing config.json.
    if isinstance(dtype_value, torch.dtype):
        return str(dtype_value).removeprefix("torch.")
    if isinstance(dtype_value, str):
        return dtype_value.removeprefix("torch.")
    raise TypeError(f"Unsupported dtype value: {dtype_value!r}")


def resolve_source_paths(
    tokenizer_kind="base",
    model_path=None,
    tokenizer_path=None,
    download_dir="qwen3",
):
    download_dir = Path(download_dir)

    if model_path is not None:
        model_path = Path(model_path)
    else:
        download_qwen3_small(
            kind=tokenizer_kind,
            tokenizer_only=False,
            out_dir=download_dir,
        )
        model_path = download_dir / MODEL_FILENAMES[tokenizer_kind]

    if tokenizer_path is not None:
        tokenizer_path = Path(tokenizer_path)
    else:
        download_qwen3_small(
            kind=tokenizer_kind,
            tokenizer_only=True,
            out_dir=download_dir,
        )
        tokenizer_path = download_dir / TOKENIZER_FILENAMES[tokenizer_kind]

    return model_path, tokenizer_path


def build_hf_tokenizer(tokenizer_path, tokenizer_kind, model_max_length):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        pad_token=PAD_TOKEN,
        eos_token=EOS_TOKENS[tokenizer_kind],
    )
    tokenizer.model_max_length = model_max_length
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    if tokenizer_kind == "reasoning":
        tokenizer.chat_template = REASONING_CHAT_TEMPLATE

    return tokenizer


def build_hf_config(tokenizer, tokenizer_kind, model_cfg=None):
    if model_cfg is None:
        model_cfg = dict(QWEN_CONFIG_06_B)

    return Qwen3FromScratchConfig(
        vocab_size=model_cfg["vocab_size"],
        hidden_size=model_cfg["emb_dim"],
        num_hidden_layers=model_cfg["n_layers"],
        num_attention_heads=model_cfg["n_heads"],
        intermediate_size=model_cfg["hidden_dim"],
        num_key_value_heads=model_cfg["n_kv_groups"],
        head_dim=model_cfg["head_dim"],
        max_position_embeddings=model_cfg["context_length"],
        rope_theta=model_cfg["rope_base"],
        use_qk_norm=model_cfg["qk_norm"],
        dtype=torch_dtype_to_name(model_cfg["dtype"]),
        tokenizer_kind=tokenizer_kind,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


def export_qwen3_to_hf(
    output_dir,
    state_dict=None,
    tokenizer_path=None,
    tokenizer_kind="base",
    model_path=None,
    download_dir="qwen3",
    model_cfg=None,
    source_path=None,
):
    if state_dict is not None and model_path is not None:
        raise ValueError("Pass either state_dict or model_path, not both.")

    if state_dict is None:
        source_path, tokenizer_path = resolve_source_paths(
            tokenizer_kind=tokenizer_kind,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            download_dir=download_dir,
        )
        try:
            state_dict = torch.load(source_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(source_path, map_location="cpu")
    else:
        if tokenizer_path is None:
            download_dir = Path(download_dir)
            download_qwen3_small(
                kind=tokenizer_kind,
                tokenizer_only=True,
                out_dir=download_dir,
            )
            tokenizer_path = download_dir / TOKENIZER_FILENAMES[tokenizer_kind]
        else:
            tokenizer_path = Path(tokenizer_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_hf_tokenizer(
        tokenizer_path=tokenizer_path,
        tokenizer_kind=tokenizer_kind,
        model_max_length=(
            model_cfg["context_length"]
            if model_cfg is not None
            else QWEN_CONFIG_06_B["context_length"]
        ),
    )
    config = build_hf_config(
        tokenizer=tokenizer,
        tokenizer_kind=tokenizer_kind,
        model_cfg=model_cfg,
    )

    Qwen3FromScratchConfig.register_for_auto_class("AutoConfig")
    Qwen3FromScratchForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    model = Qwen3FromScratchForCausalLM(config)
    state_dict = dict(state_dict)
    state_dict.setdefault("cos", model.cos)
    state_dict.setdefault("sin", model.sin)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "State dict mismatch during export. "
            f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
        )

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(output_dir)

    generation_config = GenerationConfig(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generation_config.save_pretrained(output_dir)

    export_info = {
        "tokenizer_kind": tokenizer_kind,
        "source_path": str(source_path) if source_path is not None else None,
        "safe_serialization": True,
    }
    (output_dir / "export_info.json").write_text(
        json.dumps(export_info, indent=2),
        encoding="utf-8",
    )

    return output_dir
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert the scratch Qwen3 model or a saved .pth checkpoint into "
            "a Hugging Face Transformers-compatible folder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the Hugging Face model folder will be written.",
    )
    parser.add_argument(
        "--tokenizer_kind",
        type=str,
        choices=("base", "reasoning"),
        default="base",
        help=(
            "Tokenizer family to bundle with the export. For chapter 8 "
            "distillation checkpoints, use 'reasoning'."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "Optional local .pth file to export. This can be either the "
            "original Qwen3 weights or a trained checkpoint."
        ),
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help=(
            "Optional local path to tokenizer-base.json or "
            "tokenizer-reasoning.json. If omitted, the tokenizer is downloaded."
        ),
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default="qwen3",
        help="Cache directory used when the script downloads source weights/tokenizers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    export_dir = export_qwen3_to_hf(
        output_dir=args.output_dir,
        tokenizer_kind=args.tokenizer_kind,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        download_dir=args.download_dir,
    )

    print(f"Exported Hugging Face model folder to {export_dir}")


if __name__ == "__main__":
    main()

# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

from pathlib import Path

import torch
from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from reasoning_from_scratch.qwen3 import (
    QWEN_CONFIG_06_B,
    Qwen3Model,
    Qwen3Tokenizer,
    download_qwen3_small,
)


MODEL_FILENAMES = {
    "base": "qwen3-0.6B-base.pth",
    "reasoning": "qwen3-0.6B-reasoning.pth",
}

TOKENIZER_FILENAMES = {
    "base": "tokenizer-base.json",
    "reasoning": "tokenizer-reasoning.json",
}


def _dtype_to_name(dtype_value):
    if isinstance(dtype_value, torch.dtype):
        return str(dtype_value).removeprefix("torch.")
    return dtype_value


def _qwen_cfg_to_jsonable(model_cfg):
    model_cfg = dict(model_cfg)
    model_cfg["dtype"] = _dtype_to_name(model_cfg["dtype"])
    return model_cfg


def _qwen_cfg_from_jsonable(model_cfg):
    model_cfg = dict(model_cfg)
    if isinstance(model_cfg["dtype"], str):
        model_cfg["dtype"] = getattr(torch, model_cfg["dtype"])
    return model_cfg


class Qwen3LocalWrapperConfig(PretrainedConfig):
    model_type = "qwen3_local_wrapper"

    def __init__(
        self,
        qwen_cfg=None,
        tokenizer_kind="base",
        pad_token_id=None,
        eos_token_id=None,
        **kwargs,
    ):
        if qwen_cfg is None:
            qwen_cfg = dict(QWEN_CONFIG_06_B)

        self.qwen_cfg = _qwen_cfg_to_jsonable(qwen_cfg)
        self.tokenizer_kind = tokenizer_kind
        self.vocab_size = qwen_cfg["vocab_size"]
        self.hidden_size = qwen_cfg["emb_dim"]
        self.num_attention_heads = qwen_cfg["n_heads"]
        self.num_key_value_heads = qwen_cfg["n_kv_groups"]
        self.num_hidden_layers = qwen_cfg["n_layers"]
        self.max_position_embeddings = qwen_cfg["context_length"]
        kwargs.pop("tie_word_embeddings", None)

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            dtype=self.qwen_cfg["dtype"],
            use_cache=False,
            tie_word_embeddings=False,
            **kwargs,
        )


class Qwen3LocalWrapperForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = Qwen3LocalWrapperConfig
    main_input_name = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(_qwen_cfg_from_jsonable(config.qwen_cfg))
        self.generation_config.pad_token_id = config.pad_token_id
        self.generation_config.eos_token_id = config.eos_token_id
        self.generation_config.use_cache = False

    def get_input_embeddings(self):
        return self.model.tok_emb

    def set_input_embeddings(self, value):
        self.model.tok_emb = value

    def get_output_embeddings(self):
        return self.model.out_head

    def set_output_embeddings(self, new_embeddings):
        self.model.out_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        **kwargs,
    ):
        del attention_mask, use_cache, kwargs
        if input_ids is None:
            raise ValueError("input_ids must be provided.")

        logits = self.model(input_ids)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if return_dict is False:
            outputs = (logits,)
            return ((loss,) + outputs) if loss is not None else outputs

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        **kwargs,
    ):
        del attention_mask, kwargs
        # This wrapper keeps things simple and recomputes from the full prefix
        # during generation instead of adapting the scratch KV cache to HF's
        # cache classes.
        return {"input_ids": input_ids, "use_cache": False}


def build_tokenizer(tokenizer_kind="base", tokenizer_path=None, local_dir="qwen3"):
    if tokenizer_path is None:
        download_qwen3_small(
            kind=tokenizer_kind,
            tokenizer_only=True,
            out_dir=local_dir,
        )
        tokenizer_path = Path(local_dir) / TOKENIZER_FILENAMES[tokenizer_kind]
    else:
        tokenizer_path = Path(tokenizer_path)

    if tokenizer_kind == "reasoning":
        return Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_path,
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=True,
        )

    return Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)


def build_model(
    tokenizer_kind="base",
    model_path=None,
    tokenizer=None,
    local_dir="qwen3",
):
    if tokenizer is None:
        tokenizer = build_tokenizer(
            tokenizer_kind=tokenizer_kind,
            local_dir=local_dir,
        )

    if model_path is None:
        download_qwen3_small(
            kind=tokenizer_kind,
            tokenizer_only=False,
            out_dir=local_dir,
        )
        model_path = Path(local_dir) / MODEL_FILENAMES[tokenizer_kind]
    else:
        model_path = Path(model_path)

    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location="cpu")

    config = Qwen3LocalWrapperConfig(
        qwen_cfg=QWEN_CONFIG_06_B,
        tokenizer_kind=tokenizer_kind,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = Qwen3LocalWrapperForCausalLM(config)
    model.model.load_state_dict(state_dict)
    return model


def build_model_and_tokenizer(
    tokenizer_kind="base",
    model_path=None,
    tokenizer_path=None,
    local_dir="qwen3",
):
    tokenizer = build_tokenizer(
        tokenizer_kind=tokenizer_kind,
        tokenizer_path=tokenizer_path,
        local_dir=local_dir,
    )
    model = build_model(
        tokenizer_kind=tokenizer_kind,
        model_path=model_path,
        tokenizer=tokenizer,
        local_dir=local_dir,
    )
    return model, tokenizer


def format_distilled_answer(entry, use_think_tokens=False):
    content = str(entry["message_content"]).replace("<think>", "").replace("</think>", "").strip()
    if not content:
        raise ValueError("Missing non-empty 'message_content' field.")

    thinking = str(entry.get("message_thinking", "")).replace("<think>", "").replace("</think>", "").strip()

    if use_think_tokens:
        return f"<think>{thinking}</think>\n\n{content}"

    if thinking:
        return f"{thinking}\n\n{content}"

    return content

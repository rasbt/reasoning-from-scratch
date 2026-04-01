# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

from typing import Optional

import torch
import torch.nn as nn
from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from reasoning_from_scratch.qwen3 import RMSNorm, apply_rope, compute_rope_params


class Qwen3FromScratchConfig(PretrainedConfig):
    model_type = "qwen3_from_scratch"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151_936,
        hidden_size=1024,
        num_hidden_layers=28,
        num_attention_heads=16,
        intermediate_size=3072,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=40_960,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        use_qk_norm=True,
        tokenizer_kind="base",
        pad_token_id=151_643,
        eos_token_id=151_643,
        bos_token_id=None,
        dtype="bfloat16",
        torch_dtype=None,
        initializer_range=0.02,
        emb_dim=None,
        n_layers=None,
        n_heads=None,
        hidden_dim=None,
        n_kv_groups=None,
        context_length=None,
        rope_base=None,
        qk_norm=None,
        **kwargs,
    ):
        if emb_dim is not None:
            hidden_size = emb_dim
        if n_layers is not None:
            num_hidden_layers = n_layers
        if n_heads is not None:
            num_attention_heads = n_heads
        if hidden_dim is not None:
            intermediate_size = hidden_dim
        if n_kv_groups is not None:
            num_key_value_heads = n_kv_groups
        if context_length is not None:
            max_position_embeddings = context_length
        if rope_base is not None:
            rope_theta = rope_base
        if qk_norm is not None:
            use_qk_norm = qk_norm
        if torch_dtype is not None:
            dtype = torch_dtype

        self.use_cache = kwargs.pop("use_cache", True)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.use_qk_norm = use_qk_norm
        self.tokenizer_kind = tokenizer_kind
        self.initializer_range = initializer_range

        self.emb_dim = hidden_size
        self.n_layers = num_hidden_layers
        self.n_heads = num_attention_heads
        self.hidden_dim = intermediate_size
        self.n_kv_groups = num_key_value_heads
        self.context_length = max_position_embeddings
        self.rope_base = rope_theta
        self.qk_norm = use_qk_norm

        kwargs.pop("tie_word_embeddings", None)

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            dtype=dtype,
            tie_word_embeddings=False,
            **kwargs,
        )


def _resolve_torch_dtype(dtype_value):
    if isinstance(dtype_value, torch.dtype):
        return dtype_value
    if dtype_value is None:
        return torch.float32
    if isinstance(dtype_value, str):
        return getattr(torch, dtype_value)
    raise TypeError(f"Unsupported torch dtype value: {dtype_value!r}")


class FeedForward(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.fc1 = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            dtype=dtype,
        )
        self.fc2 = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            dtype=dtype,
        )
        self.fc3 = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            dtype=dtype,
        )

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        assert (
            config.num_attention_heads % config.num_key_value_heads == 0
        ), "num_attention_heads must be divisible by num_key_value_heads"

        self.num_heads = config.num_attention_heads
        self.num_kv_groups = config.num_key_value_heads
        self.group_size = self.num_heads // self.num_kv_groups
        self.head_dim = config.head_dim
        self.d_out = self.num_heads * self.head_dim

        self.W_query = nn.Linear(
            config.hidden_size,
            self.d_out,
            bias=False,
            dtype=dtype,
        )
        self.W_key = nn.Linear(
            config.hidden_size,
            self.num_kv_groups * self.head_dim,
            bias=False,
            dtype=dtype,
        )
        self.W_value = nn.Linear(
            config.hidden_size,
            self.num_kv_groups * self.head_dim,
            bias=False,
            dtype=dtype,
        )
        self.out_proj = nn.Linear(
            self.d_out,
            config.hidden_size,
            bias=False,
            dtype=dtype,
        )

        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None, layer_idx=None):
        batch_size, num_tokens, _ = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            queries = self.q_norm(queries)
        if self.k_norm is not None:
            keys_new = self.k_norm(keys_new)

        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        if hasattr(cache, "update") and layer_idx is not None:
            keys, values = cache.update(keys_new, values_new, layer_idx)
        elif cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
        else:
            keys, values = keys_new, values_new

        next_cache = (keys, values)
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = (attn_weights @ values).transpose(1, 2).reshape(
            batch_size,
            num_tokens,
            self.d_out,
        )
        return self.out_proj(context), next_cache


class TransformerBlock(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.att = GroupedQueryAttention(config, dtype=dtype)
        self.ff = FeedForward(config, dtype=dtype)
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None, layer_idx=None):
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(
            x,
            mask,
            cos,
            sin,
            start_pos=start_pos,
            cache=cache,
            layer_idx=layer_idx,
        )
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x, next_cache


class Qwen3FromScratchForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = Qwen3FromScratchConfig
    main_input_name = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        self.param_dtype = _resolve_torch_dtype(config.dtype)

        self.tok_emb = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=self.param_dtype,
        )
        self.trf_blocks = nn.ModuleList(
            [
                TransformerBlock(config, dtype=self.param_dtype)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.out_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=self.param_dtype,
        )

        cos, sin = compute_rope_params(
            head_dim=config.head_dim,
            theta_base=config.rope_theta,
            context_length=config.max_position_embeddings,
        )
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.scale)
            if module.shift is not None:
                nn.init.zeros_(module.shift)

    def get_input_embeddings(self):
        return self.tok_emb

    def set_input_embeddings(self, value):
        self.tok_emb = value

    def get_output_embeddings(self):
        return self.out_head

    def set_output_embeddings(self, new_embeddings):
        self.out_head = new_embeddings

    def _build_causal_mask(self, attention_mask, batch_size, seq_len, past_len, device):
        total_kv_len = past_len + seq_len
        query_positions = torch.arange(
            past_len,
            past_len + seq_len,
            device=device,
        ).unsqueeze(1)
        key_positions = torch.arange(total_kv_len, device=device).unsqueeze(0)
        causal_mask = key_positions > query_positions
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is None:
            return causal_mask

        if attention_mask.size(1) < total_kv_len:
            prefix = torch.ones(
                batch_size,
                total_kv_len - attention_mask.size(1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix, attention_mask], dim=1)
        elif attention_mask.size(1) > total_kv_len:
            attention_mask = attention_mask[:, -total_kv_len:]

        key_padding_mask = attention_mask[:, None, None, :].eq(0)
        return causal_mask | key_padding_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided.")

        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        is_dynamic_cache = hasattr(past_key_values, "update") and hasattr(
            past_key_values,
            "get_seq_length",
        )

        hidden_states = self.tok_emb(input_ids)

        batch_size, seq_len = input_ids.shape
        past_len = 0
        if is_dynamic_cache:
            past_len = past_key_values.get_seq_length(0)
        elif past_key_values:
            past_len = past_key_values[0][0].shape[2]

        mask = self._build_causal_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            past_len=past_len,
            device=input_ids.device,
        )

        next_past_key_values = [] if use_cache else None
        for layer_idx, block in enumerate(self.trf_blocks):
            if is_dynamic_cache:
                hidden_states, next_cache = block(
                    hidden_states,
                    mask,
                    self.cos,
                    self.sin,
                    start_pos=past_len,
                    cache=past_key_values,
                    layer_idx=layer_idx,
                )
            else:
                layer_past = past_key_values[layer_idx] if past_key_values else None
                hidden_states, next_cache = block(
                    hidden_states,
                    mask,
                    self.cos,
                    self.sin,
                    start_pos=past_len,
                    cache=layer_past,
                    layer_idx=layer_idx,
                )
            if use_cache and not is_dynamic_cache:
                next_past_key_values.append(next_cache)

        hidden_states = self.final_norm(hidden_states)
        logits = self.out_head(hidden_states.to(self.param_dtype))

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if use_cache:
            if is_dynamic_cache:
                next_past_key_values = past_key_values
            else:
                next_past_key_values = tuple(next_past_key_values)
        else:
            next_past_key_values = None

        if not return_dict:
            outputs = (logits, next_past_key_values)
            return ((loss,) + outputs) if loss is not None else outputs

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_past_key_values,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        has_past = past_key_values is not None
        if has_past and hasattr(past_key_values, "get_seq_length"):
            has_past = past_key_values.get_seq_length(0) > 0
        elif has_past:
            has_past = len(past_key_values) > 0

        if has_past:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if past_key_values is None:
            return past_key_values
        if hasattr(past_key_values, "reorder_cache"):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values

        reordered = []
        for layer_past in past_key_values:
            reordered.append(
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            )
        return tuple(reordered)

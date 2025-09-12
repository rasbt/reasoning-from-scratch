# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

from .qwen3 import KVCache

import torch
import torch.nn as nn

# 0.6 billion parameters
QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,     # Vocabulary size
    "context_length": 40_960,  # Length originally used during training
    "emb_dim": 1024,           # Embedding dimension
    "n_heads": 16,             # Number of attention heads
    "n_layers": 28,            # Number of layers
    "hidden_dim": 3072,        # Size of intermediate dim in FeedForward
    "head_dim": 128,           # Size of the heads in GQA
    "qk_norm": True,           # Whether to normalize queries & keys in GQA
    "n_kv_groups": 8,          # Key-Value groups for GQA
    "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
    "dtype": torch.bfloat16,   # Lower-precision dtype to reduce memory
}


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusable utilities
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(self, in_idx, cache=None, attn_mask=None):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        B, num_tokens = x.shape[0], x.shape[1]

        # Derive pos_start from cache content (layer 0 K length) if present
        if cache is not None and cache.get(0) is not None:
            prev_k0, _ = cache.get(0)                 # (B, G_kv, L_prev, D)
            pos_start = prev_k0.size(2)               # L_prev
        else:
            pos_start = 0

        pos_end = pos_start + num_tokens

        # Build causal mask for [Q=num_tokens, K=pos_end]
        base = torch.triu(
            torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), diagonal=1
        )
        causal4d = base[pos_start:pos_end, :pos_end][None, None, :, :]

        has_pad = attn_mask is not None and (~attn_mask[:, :pos_end]).any().item()
        if has_pad:
            # Mask out padded keys so they don't appear in the softmax denominator
            kpm = (attn_mask[:, :pos_end] == 0).view(B, 1, 1, pos_end)
            mask = causal4d | kpm
        else:
            mask = causal4d

        pos_ids_current = torch.arange(pos_start, pos_end, device=x.device).unsqueeze(0).expand(B, -1)

        # zero-out padded query rows so their Q/K/V become zeros and don't affect cache
        if attn_mask is not None:
            qmask = attn_mask[:, pos_start:pos_end].unsqueeze(-1)
            x = x * qmask.to(x.dtype)

        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin,
                                     cache=blk_cache,
                                     pos_ids=pos_ids_current)
            if cache is not None:
                cache.update(i, new_blk_cache)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    # Keep for compatibility with regular, non-batched generate_text_basic_cache function
    def reset_kv_cache(self):
        pass


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, cache=None, pos_ids=None):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(x, mask, cos, sin, cache=cache, pos_ids=pos_ids)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x, next_cache


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, cache=None, pos_ids=None):
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        # Apply RoPE (per-token position ids)
        queries = apply_rope_with_pos_ids(queries, cos, sin, pos_ids)
        keys_new = apply_rope_with_pos_ids(keys_new, cos, sin, pos_ids)
        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
        else:
            keys, values = keys_new, values_new
        next_cache = (keys, values)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = torch.matmul(queries.to(torch.float32), keys.transpose(2, 3).to(torch.float32))
        attn_scores = attn_scores / self.head_dim**0.5

        # Apply mask with -inf so masked entries are exactly zero after softmax
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        # Stable log-sum-exp over the unmasked set
        row_max = attn_scores.amax(dim=-1, keepdim=True)
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        exp_scores = torch.exp(attn_scores - row_max)
        exp_scores = exp_scores.masked_fill(mask, 0.0)

        denom = exp_scores.sum(dim=-1, keepdim=True)
        attn_weights = exp_scores / denom.clamp(min=torch.finfo(exp_scores.dtype).tiny)

        # Back to model dtype
        attn_weights = attn_weights.to(values.dtype)

        # As before
        context = torch.matmul(attn_weights, values)
        context = context.transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context), next_cache


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope_with_pos_ids(x, cos, sin, pos_ids):
    B, H, L, D = x.shape
    cos_sel = cos[pos_ids]  # (B, L, D)
    sin_sel = sin[pos_ids]  # (B, L, D)
    cos_sel = cos_sel.unsqueeze(1)  # (B, 1, L, D)
    sin_sel = sin_sel.unsqueeze(1)  # (B, 1, L, D)
    x1 = x[..., : D // 2]
    x2 = x[..., D // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos_sel) + (rotated * sin_sel)
    return x_rotated.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)


@torch.inference_mode()
def generate_text_basic_batched_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    attn_mask=None,
    pad_id=None,
):
    device = token_ids.device
    model.eval()

    batch_size, input_length = token_ids.shape

    if attn_mask is None and pad_id is not None:
        attn_mask = (token_ids != pad_id).to(torch.bool)
    if attn_mask is not None:
        attn_mask = attn_mask.to(torch.bool).to(device)

    # Init cache and model position
    cache = KVCache(n_layers=model.cfg["n_layers"])

    # Prefill
    out = model(token_ids, cache=cache, attn_mask=attn_mask)[:, -1]

    # Track which sequences have already produced EOS
    if eos_token_id is not None:
        # If a prompt already ends with EOS, consider it finished
        finished = (token_ids[:, -1] == eos_token_id)
    else:
        finished = None

    # Decode
    cur_attn = attn_mask
    for _ in range(max_new_tokens):
        # If all sequences are already finished, stop
        if eos_token_id is not None and finished is not None and torch.all(finished):
            break

        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if eos_token_id is not None:
            # Force already finished rows to keep emitting EOS to maintain shape
            eos_tok = next_token.new_full((batch_size, 1), eos_token_id)
            next_token = torch.where(finished.view(batch_size, 1), eos_tok, next_token)

        # Extend mask to include the newly generated token
        if cur_attn is not None:
            ones = torch.ones((batch_size, 1), dtype=cur_attn.dtype, device=device)
            cur_attn = torch.cat([cur_attn, ones], dim=1)

        # Advance one token with KV cache
        out = model(next_token, cache=cache, attn_mask=cur_attn)[:, -1]
        token_ids = torch.cat([token_ids, next_token], dim=1)

        # Update finished mask after appending this step's token
        if eos_token_id is not None:
            finished = finished | (next_token.squeeze(1) == eos_token_id)

    return token_ids[:, input_length:]


@torch.inference_mode()
def generate_text_basic_batched_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    attn_mask=None,
    pad_id=None,
):
    device = token_ids.device
    model.eval()

    B, T = token_ids.shape

    if attn_mask is None and pad_id is not None:
        attn_mask = (token_ids != pad_id).to(torch.bool)
    if attn_mask is not None:
        attn_mask = attn_mask.to(torch.bool).to(device)

    # Init cache and model position
    cache = KVCache(n_layers=model.cfg["n_layers"])

    # Prefill
    out = model(token_ids, cache=cache, attn_mask=attn_mask)[:, -1]

    # Decode
    cur_attn = attn_mask
    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
            break

        yield next_token

        # Extend mask to include the newly generated token
        if cur_attn is not None:
            ones = torch.ones((B, 1), dtype=cur_attn.dtype, device=device)
            cur_attn = torch.cat([cur_attn, ones], dim=1)

        # Advance one token with KV cache
        out = model(next_token, cache=cache, attn_mask=cur_attn)[:, -1]
        token_ids = torch.cat([token_ids, next_token], dim=1)

"""TokenFormer model for the PCVR-style project layout.

This implementation follows the core structure described in the paper:

- Unified token stream ``F + <sep> + T + <sep> + V (+ decision token)``
- RoPE-enhanced decoder-only backbone
- Bottom-Full-Top-Sliding (BFTS) attention hierarchy
- Non-Linear Interacted Representation (NLIR)
- SwiGLU feed-forward network

The surrounding engineering surface is intentionally kept aligned with the
current project so that ``train.py`` / ``trainer.py`` / ``infer.py`` can keep
the same responsibilities as the existing HyFormer codebase.
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: Dict[str, torch.Tensor]
    seq_lens: Dict[str, torch.Tensor]
    seq_raw_timestamps: Dict[str, torch.Tensor]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


class RotaryEmbedding(nn.Module):
    """RoPE cache for arbitrary position ids."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids.to(self.inv_freq.dtype).unsqueeze(-1) * self.inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin


class SparseFieldTokenizer(nn.Module):
    """One token per sparse field via mean-pooled field embedding."""

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        d_model: int,
        emb_dim: int,
    ) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.emb_dim = emb_dim
        self.output_dim = d_model
        self.embs = nn.ModuleList()
        for vocab_size, _, _ in feature_specs:
            vocab = max(int(vocab_size), 1)
            emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
            nn.init.xavier_uniform_(emb.weight)
            emb.weight.data[0].zero_()
            self.embs.append(emb)
        self.proj = nn.Identity() if emb_dim == d_model else nn.Linear(emb_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens: List[torch.Tensor] = []
        for emb, (_, offset, length) in zip(self.embs, self.feature_specs):
            field_ids = x[:, offset:offset + length]
            if length == 1:
                pooled = emb(field_ids[:, 0])
            else:
                field_emb = emb(field_ids)
                valid = (field_ids > 0).unsqueeze(-1)
                denom = valid.sum(dim=1).clamp_min(1).to(field_emb.dtype)
                pooled = (field_emb * valid).sum(dim=1) / denom
            tokens.append(self.proj(pooled))
        if not tokens:
            return x.new_zeros(x.shape[0], 0, self.output_dim, dtype=torch.float)
        return torch.stack(tokens, dim=1)


class DenseFieldTokenizer(nn.Module):
    """One token per dense field."""

    def __init__(
        self,
        field_entries: List[Tuple[int, int, int]],
        d_model: int,
    ) -> None:
        super().__init__()
        self.field_entries = field_entries
        self.output_dim = d_model
        self.projs = nn.ModuleList([nn.Linear(length, d_model) for _, _, length in field_entries])
        for proj in self.projs:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens: List[torch.Tensor] = []
        for proj, (_, offset, length) in zip(self.projs, self.field_entries):
            tokens.append(proj(x[:, offset:offset + length]))
        if not tokens:
            return x.new_zeros(x.shape[0], 0, self.output_dim, dtype=x.dtype)
        return torch.stack(tokens, dim=1)


class SequenceDomainTokenizer(nn.Module):
    """Projects one behavior step into one token for a single sequence domain."""

    def __init__(
        self,
        vocab_sizes: List[int],
        d_model: int,
        emb_dim: int,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.embs = nn.ModuleList()
        for vocab_size in vocab_sizes:
            vocab = max(int(vocab_size), 1)
            emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
            nn.init.xavier_uniform_(emb.weight)
            emb.weight.data[0].zero_()
            self.embs.append(emb)
        in_dim = max(len(vocab_sizes), 1) * emb_dim
        self.proj = nn.Linear(in_dim, d_model)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for field_idx, emb in enumerate(self.embs):
            parts.append(emb(seq[:, field_idx, :]))
        if not parts:
            return seq.new_zeros(seq.shape[0], seq.shape[2], self.proj.out_features, dtype=torch.float)
        cat = torch.cat(parts, dim=-1)
        return self.proj(cat)


class TokenFormerAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(position_ids)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        sdpa_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = torch.nan_to_num(out, nan=0.0)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)


class UnifiedInteractionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_mult: int,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.attn = TokenFormerAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_base=rope_base,
        )
        self.gate_proj = nn.Linear(d_model, d_model)
        hidden_dim = d_model * ffn_mult
        self.ffn_w1 = nn.Linear(d_model, hidden_dim)
        self.ffn_w2 = nn.Linear(d_model, hidden_dim)
        self.ffn_w3 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_out = self.attn(self.attn_norm(x), position_ids, attn_mask)
        x = x + self.dropout(torch.sigmoid(self.gate_proj(x)) * attn_out)
        ffn_in = self.ffn_norm(x)
        ffn_out = self.ffn_w3(F.silu(self.ffn_w1(ffn_in)) * self.ffn_w2(ffn_in))
        x = x + self.dropout(ffn_out)
        return x * valid_mask.unsqueeze(-1).to(x.dtype)


class TokenFormer(nn.Module):
    def __init__(
        self,
        user_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        seq_domain_vocab_sizes: Dict[str, List[int]],
        max_behavior_tokens: int,
        d_model: int = 128,
        emb_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        full_attention_layers: int = 2,
        sliding_window_sizes: Optional[List[int]] = None,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
        action_dim: int = 2,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if full_attention_layers <= 0 or full_attention_layers > num_layers:
            raise ValueError("full_attention_layers must be in [1, num_layers]")

        self.seq_domains = sorted(seq_domain_vocab_sizes.keys())
        self.max_behavior_tokens = int(max_behavior_tokens)
        self.d_model = d_model
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.full_attention_layers = full_attention_layers

        num_sliding_layers = num_layers - full_attention_layers
        if sliding_window_sizes is None:
            sliding_window_sizes = self._default_window_schedule(num_sliding_layers)
        if len(sliding_window_sizes) != num_sliding_layers:
            raise ValueError(
                "sliding_window_sizes length must equal num_layers - full_attention_layers"
            )
        self.sliding_window_sizes = [int(w) for w in sliding_window_sizes]

        self.user_int_tokenizer = SparseFieldTokenizer(user_int_feature_specs, d_model, emb_dim)
        self.user_dense_tokenizer = DenseFieldTokenizer(user_dense_feature_specs, d_model)
        self.item_int_tokenizer = SparseFieldTokenizer(item_int_feature_specs, d_model, emb_dim)
        self.seq_tokenizers = nn.ModuleDict(
            {
                domain: SequenceDomainTokenizer(vocab_sizes, d_model, emb_dim)
                for domain, vocab_sizes in seq_domain_vocab_sizes.items()
            }
        )

        self.sep_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.decision_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.input_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                UnifiedInteractionBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    rope_base=rope_base,
                )
                for _ in range(num_layers)
            ]
        )
        self.head_norm = RMSNorm(d_model)
        self.classifier = nn.Linear(d_model, action_dim)

        self.num_static_tokens = len(user_int_feature_specs) + len(user_dense_feature_specs)
        self.num_target_tokens = len(item_int_feature_specs)

    def _default_window_schedule(self, num_sliding_layers: int) -> List[int]:
        if num_sliding_layers <= 0:
            return []
        if num_sliding_layers == 1:
            return [32]
        first = 32
        second = 16
        schedule = [first, second]
        while len(schedule) < num_sliding_layers:
            schedule.append(max(8, schedule[-1] // 2))
        return schedule[:num_sliding_layers]

    def _encode_static_tokens(self, inputs: ModelInput) -> torch.Tensor:
        user_int_tokens = self.user_int_tokenizer(inputs.user_int_feats)
        user_dense_tokens = self.user_dense_tokenizer(inputs.user_dense_feats)
        return torch.cat([user_int_tokens, user_dense_tokens], dim=1)

    def _encode_target_tokens(self, inputs: ModelInput) -> torch.Tensor:
        return self.item_int_tokenizer(inputs.item_int_feats)

    def _build_behavior_block(
        self,
        inputs: ModelInput,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = inputs.user_int_feats.shape[0]
        device = inputs.user_int_feats.device
        token_parts: List[torch.Tensor] = []
        ts_parts: List[torch.Tensor] = []
        valid_parts: List[torch.Tensor] = []

        for domain in self.seq_domains:
            domain_tokens = self.seq_tokenizers[domain](inputs.seq_data[domain])
            domain_lens = inputs.seq_lens[domain]
            domain_ts = inputs.seq_raw_timestamps[domain]
            max_len = domain_tokens.shape[1]
            valid = torch.arange(max_len, device=device).unsqueeze(0) < domain_lens.unsqueeze(1)
            domain_tokens = domain_tokens * valid.unsqueeze(-1).to(domain_tokens.dtype)
            token_parts.append(domain_tokens)
            ts_parts.append(domain_ts)
            valid_parts.append(valid)

        if token_parts:
            all_tokens = torch.cat(token_parts, dim=1)
            all_ts = torch.cat(ts_parts, dim=1)
            all_valid = torch.cat(valid_parts, dim=1)
        else:
            all_tokens = inputs.user_int_feats.new_zeros(batch_size, 0, self.d_model, dtype=torch.float)
            all_ts = inputs.user_int_feats.new_zeros(batch_size, 0, dtype=torch.long)
            all_valid = inputs.user_int_feats.new_zeros(batch_size, 0, dtype=torch.bool)

        if all_tokens.shape[1] == 0:
            return (
                inputs.user_int_feats.new_zeros(batch_size, self.max_behavior_tokens, self.d_model, dtype=torch.float),
                inputs.user_int_feats.new_zeros(batch_size, self.max_behavior_tokens, dtype=torch.long),
                inputs.user_int_feats.new_zeros(batch_size, self.max_behavior_tokens, dtype=torch.bool),
            )

        large_ts = torch.full_like(all_ts, fill_value=torch.iinfo(all_ts.dtype).max)
        sort_key = torch.where(all_valid, all_ts, large_ts)
        order = torch.argsort(sort_key, dim=1)
        gather_token_idx = order.unsqueeze(-1).expand(-1, -1, all_tokens.shape[-1])
        sorted_tokens = torch.gather(all_tokens, 1, gather_token_idx)
        sorted_valid = torch.gather(all_valid, 1, order)

        out_tokens = all_tokens.new_zeros(batch_size, self.max_behavior_tokens, self.d_model)
        out_pos = all_ts.new_zeros(batch_size, self.max_behavior_tokens)
        out_valid = all_valid.new_zeros(batch_size, self.max_behavior_tokens)

        for row_idx in range(batch_size):
            valid_count = int(sorted_valid[row_idx].sum().item())
            if valid_count <= 0:
                continue
            use_len = min(valid_count, self.max_behavior_tokens)
            selected = sorted_tokens[row_idx, valid_count - use_len:valid_count]
            start = self.max_behavior_tokens - use_len
            out_tokens[row_idx, start:] = selected
            out_valid[row_idx, start:] = True
            out_pos[row_idx, start:] = torch.arange(1, use_len + 1, device=device)

        return out_tokens, out_pos, out_valid

    def _build_stream(
        self,
        inputs: ModelInput,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        static_tokens = self._encode_static_tokens(inputs)
        target_tokens = self._encode_target_tokens(inputs)
        behavior_tokens, behavior_pos, behavior_valid = self._build_behavior_block(inputs)

        batch_size = static_tokens.shape[0]
        device = static_tokens.device
        sep1 = self.sep_token.expand(batch_size, -1, -1)
        sep2 = self.sep_token.expand(batch_size, -1, -1)
        decision = self.decision_token.expand(batch_size, -1, -1)

        stream = torch.cat(
            [static_tokens, sep1, behavior_tokens, sep2, target_tokens, decision],
            dim=1,
        )
        stream = self.input_dropout(stream)

        target_pos = self.max_behavior_tokens + 1
        static_pos = torch.zeros(batch_size, static_tokens.shape[1] + 1, dtype=torch.long, device=device)
        tail_pos = torch.full(
            (batch_size, 1 + target_tokens.shape[1] + 1),
            fill_value=target_pos,
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.cat([static_pos, behavior_pos, tail_pos], dim=1)

        valid_mask = torch.cat(
            [
                torch.ones(batch_size, static_tokens.shape[1] + 1, dtype=torch.bool, device=device),
                behavior_valid,
                torch.ones(batch_size, 1 + target_tokens.shape[1] + 1, dtype=torch.bool, device=device),
            ],
            dim=1,
        )
        static_prefix_len = static_tokens.shape[1] + 1
        return stream, position_ids, valid_mask, static_prefix_len

    def _build_attention_mask(
        self,
        valid_mask: torch.Tensor,
        window_size: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = valid_mask.shape
        query_idx = torch.arange(seq_len, device=valid_mask.device).view(1, seq_len, 1)
        key_idx = torch.arange(seq_len, device=valid_mask.device).view(1, 1, seq_len)
        causal = key_idx <= query_idx
        if window_size is not None:
            causal = causal & ((query_idx - key_idx) < int(window_size))
        mask = causal.expand(batch_size, -1, -1)
        mask = mask & valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
        return mask

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        x, position_ids, valid_mask, static_prefix_len = self._build_stream(inputs)

        for block_idx in range(self.full_attention_layers):
            attn_mask = self._build_attention_mask(valid_mask, window_size=None)
            x = self.blocks[block_idx](x, position_ids, attn_mask, valid_mask)

        if self.full_attention_layers < self.num_layers:
            x = x[:, static_prefix_len:, :]
            position_ids = position_ids[:, static_prefix_len:]
            valid_mask = valid_mask[:, static_prefix_len:]

        for window_size, block in zip(
            self.sliding_window_sizes,
            self.blocks[self.full_attention_layers:],
        ):
            attn_mask = self._build_attention_mask(valid_mask, window_size=window_size)
            x = block(x, position_ids, attn_mask, valid_mask)

        decision_repr = self.head_norm(x[:, -1, :])
        return self.classifier(decision_repr)

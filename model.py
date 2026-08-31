"""PCVRHyFormer: A hybrid transformer model for post-click conversion rate prediction."""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, NamedTuple, Tuple, Optional, Union

SEQ_TEMPORAL_DENSE_DIM = 2
SEQ_TEMPORAL_ID_DIM = 11
CONTEXT_TEMPORAL_DENSE_DIM = 0
CONTEXT_TEMPORAL_ID_DIM = 4


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    coarse_feats: torch.Tensor
    target_seq_feats: torch.Tensor
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}
    seq_temporal_ids: dict  # {domain: tensor [B, L, K]}
    seq_temporal_dense: dict  # {domain: tensor [B, L, T]}
    context_temporal_ids: torch.Tensor
    context_temporal_dense: torch.Tensor
    iat_hist_ins_emb: torch.Tensor
    iat_hist_labels: torch.Tensor
    iat_hist_time_buckets: torch.Tensor
    iat_hist_task_ids: torch.Tensor
    iat_hist_mask: torch.Tensor


# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin values.

    Attributes:
        dim: Rotary embedding dimension.
        max_seq_len: Maximum sequence length for cache.
        base: Base frequency for rotary encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inv_freq: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)  # (1, seq_len, dim)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)  # (1, seq_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cos/sin values for the given sequence length.

        Returns pre-computed slices from the cache. The cache is built once
        in __init__ with max_seq_len; no runtime expansion is performed so
        that the forward pass remains compatible with torch.compile().
        """
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps and negates the first and second halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to a single tensor.

    Args:
        x: (B, num_heads, L, head_dim)
        cos: (1, L_max, head_dim) or (B, L, head_dim) for batch-specific positions.
        sin: Same shape as cos.

    Returns:
        Rotated tensor of shape (B, num_heads, L, head_dim).
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Basic Components
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLU(nn.Module):
    """SwiGLU activation: x1 * SiLU(x2)."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class RMSNorm(nn.Module):
    """Root mean square normalization used by UniMixer."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding support.

    Manually projects Q/K/V and reshapes for multi-head, then injects RoPE
    after projection and before dot-product. Uses F.scaled_dot_product_attention
    for efficient computation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        """Computes multi-head attention with optional RoPE.

        Args:
            query: (B, Lq, D)
            key: (B, Lk, D)
            value: (B, Lk, D)
            key_padding_mask: (B, Lk), True indicates padding positions.
            attn_mask: (Lq, Lk) or (B*num_heads, Lq, Lk), additive mask.
            rope_cos: (1, L, head_dim), RoPE for KV side (also used for Q
                unless q_rope_* is provided).
            rope_sin: Same shape as rope_cos.
            q_rope_cos: (B, Lq, head_dim) or (1, Lq, head_dim), Q-specific
                RoPE for cross-attention with gathered positions.
            q_rope_sin: Same shape as q_rope_cos.
            need_weights: Compatibility parameter, not used.

        Returns:
            Tuple of (output, None).
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 1. Linear projection
        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)    # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        # 2. Reshape to (B, num_heads, L, head_dim)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE independently to Q and K
        if rope_cos is not None and rope_sin is not None:
            # K always uses rope_cos/rope_sin (KV-side positional encoding)
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

            if self.rope_on_q:
                # Q side: prefer dedicated q_rope_cos/sin (top_k positions in LongerEncoder cross-attn)
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        # 4. Convert key_padding_mask to SDPA format
        sdpa_attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk), True = padding
            # SDPA expects (B, 1, 1, Lk) bool mask, True = attend
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            # attn_mask: additive float mask (Lq, Lk), -inf means do not attend
            # Convert to bool: positions that are not -inf are True
            bool_attn = (attn_mask == 0)  # (Lq, Lk)
            bool_attn = bool_attn.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            if sdpa_attn_mask is not None:
                sdpa_attn_mask = sdpa_attn_mask & bool_attn
            else:
                sdpa_attn_mask = bool_attn

        # 5. Scaled Dot-Product Attention
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout_p,
        )  # (B, num_heads, Lq, head_dim)

        # Replace NaN from all-padding softmax with 0 (zero vectors preserve original input via residual)
        out = torch.nan_to_num(out, nan=0.0)

        # 6. Reshape back and output projection
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


class CrossAttention(nn.Module):
    """Cross-attention module.

    Query comes from global tokens (Q tokens), Key/Value comes from sequence
    tokens. Only applies RoPE to KV side (rope_on_q=False).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ln_mode: str = 'pre'
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
        )

        if ln_mode in ['pre', 'post']:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes cross-attention between query tokens and sequence tokens.

        Args:
            query: (B, Nq, D), query tokens.
            key_value: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), KV-side RoPE cosine values.
            rope_sin: (1, L, head_dim), KV-side RoPE sine values.

        Returns:
            Output tensor of shape (B, Nq, D).
        """
        residual = query

        if self.ln_mode == 'pre':
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)

        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        out = residual + out

        if self.ln_mode == 'post':
            out = self.norm_q(out)

        return out


class RankMixerBlock(nn.Module):
    """HyFormer Query Boosting block.

    Performs three steps:
    1. Token Mixing: Parameter-free tensor reshaping.
    2. Per-token FFN: Shared-parameter feedforward network.
    3. Residual connection: Q_boost = Q + Q_e.

    Constraint: d_model must be divisible by n_total in 'full' mode.
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,  # T = Nq + Nns
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = 'full'  # 'full' | 'ffn_only' | 'none'
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode

        if mode == 'none':
            # Pure identity mapping, no submodules created
            return

        if mode == 'full':
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total} for token mixing."
                )
            self.d_sub = d_model // n_total

        # Per-token FFN (shared parameters) — used by both 'full' and 'ffn_only'
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        # Post-LN after residual to stabilize stacked block outputs
        self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        """Performs parameter-free token mixing via reshape and transpose.

        Steps:
        1. Splits channels into T subspaces: (B, T, D) -> (B, T, T, d_sub).
        2. Swaps token and subspace axes: (B, token, h, d_sub) -> (B, h, token, d_sub).
        3. Flattens back: (B, T, D).

        Args:
            Q: (B, T, D)

        Returns:
            Mixed tensor of shape (B, T, D).
        """
        B, T, D = Q.shape

        # (B, T, D) -> (B, T, T, d_sub)
        Q_split = Q.view(B, T, self.T, self.d_sub)

        # (B, token, h, d_sub) -> (B, h, token, d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()

        # (B, T, T, d_sub) -> (B, T, D)
        Q_hat = Q_rewired.view(B, T, D)
        return Q_hat

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """Applies query boosting: token mixing, FFN, and residual connection.

        Args:
            Q: (B, T, D) where T = Nq + Nns.

        Returns:
            Boosted tensor of shape (B, T, D).
        """
        if self.mode == 'none':
            return Q

        # Token Mixing (parameter-free rewire) or identity
        if self.mode == 'full':
            Q_hat = self.token_mixing(Q)
        else:  # 'ffn_only'
            Q_hat = Q

        # Per-token FFN
        x = self.norm(Q_hat)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        Q_e = self.fc2(x)

        # Residual from original Q
        Q_boost = Q + Q_e
        Q_boost = self.post_norm(Q_boost)
        return Q_boost


def sinkhorn_normalize(
    weights: torch.Tensor,
    num_iters: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Approximate a doubly-stochastic matrix with Sinkhorn iterations."""
    mat = weights.clamp_min(eps)
    for _ in range(num_iters):
        mat = mat / mat.sum(dim=-1, keepdim=True).clamp_min(eps)
        mat = mat / mat.sum(dim=-2, keepdim=True).clamp_min(eps)
    return mat


def build_constrained_mixing_matrix(
    weights: torch.Tensor,
    temperature: float,
    sinkhorn_iters: int,
    symmetric: bool = True,
) -> torch.Tensor:
    """Apply symmetry, temperature scaling, and Sinkhorn normalization."""
    if symmetric:
        weights = 0.5 * (weights + weights.transpose(-1, -2))
    logits = (weights / max(float(temperature), 1e-4)).clamp(min=-20.0, max=20.0)
    positive = torch.exp(logits)
    return sinkhorn_normalize(positive, num_iters=sinkhorn_iters)


def resolve_unimixing_block_size(total_dim: int, requested_block_size: int) -> int:
    """Resolve the paper's block size B for flattened UniMixing."""
    if requested_block_size > 0:
        if total_dim % requested_block_size != 0:
            raise ValueError(
                f"unimixing_block_size={requested_block_size} must divide total_dim={total_dim}"
            )
        return requested_block_size

    preferred = [6, 8, 4, 12, 16, 24, 32, 48, 64]
    for candidate in preferred:
        if candidate < total_dim and total_dim % candidate == 0:
            return candidate

    for candidate in range(2, int(math.sqrt(total_dim)) + 1):
        if total_dim % candidate == 0:
            return candidate
    return total_dim


class FlatRMSNorm(nn.Module):
    """RMSNorm over the flattened hidden state used by UniMixer."""

    def __init__(self, total_dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.total_dim = total_dim
        self.norm = RMSNorm(total_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        flat = x.reshape(batch_size, self.total_dim)
        return self.norm(flat).reshape_as(x)


class PerBlockSwiGLU(nn.Module):
    """Block-specific SwiGLU on flattened UniMixing blocks."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        hidden_mult: int = 4,
    ) -> None:
        super().__init__()
        hidden_dim = block_size * hidden_mult
        self.w_up = nn.Parameter(torch.empty(num_blocks, block_size, hidden_dim))
        self.w_gate = nn.Parameter(torch.empty(num_blocks, block_size, hidden_dim))
        self.w_down = nn.Parameter(torch.empty(num_blocks, hidden_dim, block_size))
        self.b_up = nn.Parameter(torch.zeros(num_blocks, hidden_dim))
        self.b_gate = nn.Parameter(torch.zeros(num_blocks, hidden_dim))
        self.b_down = nn.Parameter(torch.zeros(num_blocks, block_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_up)
        nn.init.xavier_uniform_(self.w_gate)
        nn.init.xavier_uniform_(self.w_down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = torch.einsum('bnd,ndh->bnh', x, self.w_up) + self.b_up.unsqueeze(0)
        gate = torch.einsum('bnd,ndh->bnh', x, self.w_gate) + self.b_gate.unsqueeze(0)
        hidden = up * F.silu(gate)
        return torch.einsum('bnh,nhd->bnd', hidden, self.w_down) + self.b_down.unsqueeze(0)


class UniMixingBase(nn.Module):
    """Paper-faithful flattened UniMixing scaffold."""

    def __init__(
        self,
        d_model: int,
        n_total: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        block_size: int = 0,
        temperature: float = 1.0,
        sinkhorn_iters: int = 3,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_total = n_total
        self.total_dim = d_model * n_total
        self.block_size = resolve_unimixing_block_size(self.total_dim, block_size)
        self.num_blocks = self.total_dim // self.block_size
        self.register_buffer(
            '_temperature',
            torch.tensor(float(max(temperature, 1e-4))),
            persistent=False,
        )
        self.sinkhorn_iters = sinkhorn_iters
        self.mix_norm = FlatRMSNorm(self.total_dim)
        self.ffn_norm = FlatRMSNorm(self.total_dim)
        self.per_block_swiglu = PerBlockSwiGLU(
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            hidden_mult=hidden_mult,
        )
        self.dropout = nn.Dropout(dropout)

    @property
    def temperature(self) -> float:
        return float(self._temperature.item())

    def set_temperature(self, temperature: float) -> None:
        self._temperature.fill_(float(max(temperature, 1e-4)))

    def _constrain_square(self, weights: torch.Tensor) -> torch.Tensor:
        return build_constrained_mixing_matrix(
            weights=weights,
            temperature=self.temperature,
            sinkhorn_iters=self.sinkhorn_iters,
            symmetric=True,
        )

    def _split_blocks(self, q_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = q_tokens.shape[0]
        flat = q_tokens.reshape(batch_size, self.total_dim)
        return flat.reshape(batch_size, self.num_blocks, self.block_size)

    def _merge_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        batch_size = blocks.shape[0]
        flat = blocks.reshape(batch_size, self.total_dim)
        return flat.reshape(batch_size, self.n_total, self.d_model)

    def token_mixing(self, q_tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, q_tokens: torch.Tensor) -> torch.Tensor:
        mixed = self.token_mixing(q_tokens)
        mixed = self.mix_norm(q_tokens + mixed)
        block_tokens = self._split_blocks(mixed)
        swiglu_out = self.dropout(self.per_block_swiglu(block_tokens))
        swiglu_out = self._merge_blocks(swiglu_out)
        return self.ffn_norm(mixed + swiglu_out)


class UniMixingBlock(UniMixingBase):
    """Paper-faithful UniMixing over flattened hidden-state blocks."""

    def __init__(
        self,
        d_model: int,
        n_total: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        block_size: int = 0,
        temperature: float = 1.0,
        sinkhorn_iters: int = 3,
    ) -> None:
        super().__init__(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            block_size=block_size,
            temperature=temperature,
            sinkhorn_iters=sinkhorn_iters,
        )
        self.global_mix = nn.Parameter(torch.empty(self.num_blocks, self.num_blocks))
        self.local_mix = nn.Parameter(
            torch.empty(self.num_blocks, self.block_size, self.block_size)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.global_mix)
        nn.init.xavier_uniform_(self.local_mix)

    def token_mixing(self, q_tokens: torch.Tensor) -> torch.Tensor:
        local = self._constrain_square(self.local_mix)
        global_mix = self._constrain_square(self.global_mix)
        blocks = self._split_blocks(q_tokens)
        local_out = torch.einsum('bnd,ndh->bnh', blocks, local)
        mixed = torch.einsum('ij,bjd->bid', global_mix, local_out)
        return self._merge_blocks(mixed)


class UniMixingLiteBlock(UniMixingBase):
    """Paper-faithful UniMixing-Lite with basis local mixing and low-rank global mixing."""

    def __init__(
        self,
        d_model: int,
        n_total: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        num_basis: int = 4,
        global_rank: int = 4,
        block_size: int = 0,
        temperature: float = 1.0,
        sinkhorn_iters: int = 3,
    ) -> None:
        super().__init__(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            block_size=block_size,
            temperature=temperature,
            sinkhorn_iters=sinkhorn_iters,
        )
        self.num_basis = max(1, num_basis)
        self.global_rank = max(1, min(global_rank, self.num_blocks))
        self.global_left = nn.Parameter(torch.empty(self.num_blocks, self.global_rank))
        self.global_right = nn.Parameter(torch.empty(self.global_rank, self.num_blocks))
        self.local_basis = nn.Parameter(
            torch.empty(self.num_basis, self.block_size, self.block_size)
        )
        self.local_coeff = nn.Parameter(torch.empty(self.num_blocks, self.num_basis))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.global_left)
        nn.init.xavier_uniform_(self.global_right)
        nn.init.xavier_uniform_(self.local_basis)
        nn.init.zeros_(self.local_coeff)

    def token_mixing(self, q_tokens: torch.Tensor) -> torch.Tensor:
        local_raw = torch.einsum('nb,bdh->ndh', self.local_coeff, self.local_basis)
        local = self._constrain_square(local_raw)
        global_raw = self.global_left @ self.global_right
        global_mix = self._constrain_square(global_raw)
        blocks = self._split_blocks(q_tokens)
        local_out = torch.einsum('bnd,ndh->bnh', blocks, local)
        mixed = torch.einsum('ij,bjd->bid', global_mix, local_out)
        return self._merge_blocks(mixed)


class DINSequencePooling(nn.Module):
    """Candidate-aware attention pooling over one behavior sequence."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden_dim = max(d_model // 2, 16)
        self.score_mlp = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        seq_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        item_context: torch.Tensor,
    ) -> torch.Tensor:
        item_expanded = item_context.unsqueeze(1).expand_as(seq_tokens)
        score_input = torch.cat(
            [
                seq_tokens,
                item_expanded,
                seq_tokens * item_expanded,
                seq_tokens - item_expanded,
            ],
            dim=-1,
        )
        logits = self.score_mlp(score_input).squeeze(-1)
        logits = logits.masked_fill(~valid_mask, -1e9)
        weights = torch.softmax(logits, dim=-1) * valid_mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return (seq_tokens * weights.unsqueeze(-1)).sum(dim=1)


class MultiSeqQueryGenerator(nn.Module):
    """Multi-sequence query generation module.

    Generates Q tokens independently for each sequence:
    For each sequence i:
        GlobalInfo_i = Concat(Flatten(NS), Pool(Seq_i))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]

    Pool is masked mean pooling by default, or candidate-aware DIN pooling
    when ``use_din_query_pooling`` is enabled.
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4,
        use_din_query_pooling: bool = False,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model
        self.use_din_query_pooling = use_din_query_pooling

        global_info_dim = (num_ns + 1) * d_model

        # LayerNorm on global_info to prevent gradient explosion from large-dim concat
        self.global_info_norm = nn.LayerNorm(global_info_dim)
        if use_din_query_pooling:
            self.din_poolers = nn.ModuleList([
                DINSequencePooling(d_model) for _ in range(num_sequences)
            ])
        else:
            self.din_poolers = nn.ModuleList()

        # Each sequence has N independent FFNs
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_queries)
            ])
            for _ in range(num_sequences)
        ])

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        item_context: Optional[torch.Tensor] = None,
    ) -> list:
        """Generates query tokens for each sequence.

        Args:
            ns_tokens: (B, M, D), shared NS tokens.
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S. True
                indicates padding.
            item_context: (B, D) candidate representation required by DIN.

        Returns:
            List of (B, Nq, D) query token tensors, length S.
        """
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)
        if self.use_din_query_pooling and item_context is None:
            raise ValueError("item_context is required when DIN query pooling is enabled")

        q_tokens_list = []
        for i in range(self.num_sequences):
            valid_mask = ~seq_padding_masks[i]
            if self.use_din_query_pooling:
                assert item_context is not None
                seq_pooled = self.din_poolers[i](
                    seq_tokens_list[i],
                    valid_mask,
                    item_context,
                )
            else:
                valid_mask_expanded = valid_mask.unsqueeze(-1).float()
                seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(dim=1)
                seq_count = valid_mask_expanded.sum(dim=1).clamp(min=1)
                seq_pooled = seq_sum / seq_count

            # Keep the original HyFormer query-generation structure; DIN only
            # replaces the sequence mean used as the per-domain summary.
            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)
            global_info = self.global_info_norm(global_info)

            # Generate N query tokens
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)

        return q_tokens_list


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoders
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLUEncoder(nn.Module):
    """Efficient attention-free sequence encoder.

    Structure: x + Dropout(SwiGLU(LN(x))).
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Applies the SwiGLU encoder with residual connection.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding. Not used by
                this encoder variant.
            **kwargs: Absorbs rope_cos/rope_sin and other unused parameters.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class TransformerEncoder(nn.Module):
    """High-capacity sequence encoder with self-attention and RoPE.

    Structure: Standard Transformer Encoder Layer (Pre-LN).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies one Transformer encoder layer.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), RoPE cosine values.
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        # Self-Attention (Pre-LN) with RoPE
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = residual + x

        # FFN (Pre-LN)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, key_padding_mask

class LongerEncoder(nn.Module):
    """Top-K compressed sequence encoder.

    Adapts behavior based on input length:
    - L > top_k (first MultiSeqHyFormerBlock): Cross Attention.
      Q = latest top_k tokens, K/V = all seq tokens -> output (B, top_k, D).
    - L <= top_k (subsequent MultiSeqHyFormerBlocks): Self Attention.
      Q = K = V = top_k tokens -> output (B, top_k, D).

    Causal mask is only applied among top_k tokens (self-attention layers);
    the first cross-attention layer does not use a causal mask since Q and K
    have different lengths.

    Returns (output, new_key_padding_mask) so downstream can update the mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal

        # Pre-LN for attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # Shared RoPEMHA for both cross and self attention
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        # FFN (Pre-LN + residual)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def _gather_top_k(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects the latest top_k valid tokens from each sample.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding.

        Returns:
            top_k_tokens: (B, top_k, D)
            new_padding_mask: (B, top_k), True indicates padding.
            position_indices: (B, top_k), original position index for each
                selected token, used for Q-side RoPE.
        """
        B, L, D = x.shape
        device = x.device

        # Valid lengths per sample
        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)

        # Start position for each sample: max(valid_len - top_k, 0)
        actual_k = torch.clamp(valid_len, max=self.top_k)  # (B,)
        start_pos = valid_len - actual_k  # (B,)

        # Build gather indices: (B, top_k)
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)  # (B, top_k)
        indices = start_pos.unsqueeze(1) + offsets  # (B, top_k)

        # For samples with valid_len < top_k, early indices may exceed valid range;
        # clamp to [0, L-1] and handle via mask below
        indices = torch.clamp(indices, min=0, max=L - 1)

        # Gather: (B, top_k, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, top_k, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)

        # New padding mask: first (top_k - actual_k) positions are padding
        new_valid_len = actual_k  # (B,)
        pad_count = self.top_k - new_valid_len  # (B,)
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)  # (1, top_k)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)  # (B, top_k)

        # Zero out tokens at padding positions
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()

        # position_indices for Q-side RoPE
        position_indices = indices  # (B, top_k)

        return top_k_tokens, new_padding_mask, position_indices

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LongerEncoder with adaptive cross/self attention.

        Args:
            x: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding.
            rope_cos: (1, L, head_dim), RoPE cosine values (length must cover
                original sequence length L).
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            output: (B, top_k, D), compressed sequence.
            new_key_padding_mask: (B, top_k), updated padding mask.
        """
        B, L, D = x.shape

        if L > self.top_k:
            # === Cross Attention mode (first MultiSeqHyFormerBlock) ===
            # 1. Extract latest top_k tokens as query
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)

            # 2. Pre-LN
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)

            # 3. Build Q-side RoPE cos/sin by gathering from global cos/sin at top_k positions
            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                # rope_cos: (1, L_max, head_dim), q_pos_indices: (B, top_k)
                head_dim = rope_cos.shape[2]
                # Expand to batch dimension
                cos_expanded = rope_cos.expand(B, -1, -1)  # (B, L_max, head_dim)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(-1, -1, head_dim)  # (B, top_k, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)  # (B, top_k, head_dim)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)

            # 4. Cross Attention (no causal mask since Q and K have different lengths)
            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,  # Original (B, L) mask
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
            )
            out = q + attn_out  # Residual based on q
        else:
            # === Self Attention mode (subsequent MultiSeqHyFormerBlocks) ===
            new_mask = key_padding_mask

            # Pre-LN (Q and KV share norm_q)
            x_normed = self.norm_q(x)

            # Causal mask
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            out = x + attn_out

        # FFN (Pre-LN + residual)
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out

        return out, new_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False
) -> nn.Module:
    """Creates a sequence encoder of the specified type.

    Args:
        encoder_type: One of 'swiglu', 'transformer', or 'longer'.
        d_model: Model dimension.
        num_heads: Number of attention heads (used by transformer/longer).
        hidden_mult: FFN expansion multiplier.
        dropout: Dropout rate.
        top_k: Compression length for LongerEncoder (only used by longer).
        causal: Whether to use causal mask in LongerEncoder (only used by
            longer).

    Returns:
        A sequence encoder module.
    """
    if encoder_type == 'swiglu':
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == 'transformer':
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    elif encoder_type == 'longer':
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Blocks
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSeqHyFormerBlock(nn.Module):
    """Multi-sequence HyFormer block.

    Each of the S sequences independently performs Sequence Evolution and
    Query Decoding, then all Q tokens and shared NS tokens are merged for
    joint Query Boosting.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = 'swiglu',
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = 'full',
        token_mixer_type: str = 'rankmixer',
        unimixing_num_basis: int = 4,
        unimixing_global_rank: int = 4,
        unimixing_block_size: int = 0,
        unimixing_temperature: float = 1.0,
        unimixing_sinkhorn_iters: int = 3,
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns
        self.token_mixer_type = token_mixer_type

        # Independent sequence encoder per sequence
        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(
                encoder_type=seq_encoder_type,
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout,
                top_k=top_k,
                causal=causal
            )
            for _ in range(num_sequences)
        ])

        # Independent cross-attention per sequence
        self.cross_attns = nn.ModuleList([
            CrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                ln_mode='pre'
            )
            for _ in range(num_sequences)
        ])

        # RankMixer: input token count = Nq * S + Nns
        n_total = num_queries * num_sequences + num_ns
        if token_mixer_type == 'rankmixer':
            self.mixer = RankMixerBlock(
                d_model=d_model,
                n_total=n_total,
                hidden_mult=hidden_mult,
                dropout=dropout,
                mode=rank_mixer_mode
            )
        elif rank_mixer_mode == 'none':
            self.mixer = nn.Identity()
        elif token_mixer_type == 'unimixing':
            if rank_mixer_mode != 'full':
                logging.info(
                    "token_mixer_type=%s ignores rank_mixer_mode=%s and uses the "
                    "paper-style UniMixing block.",
                    token_mixer_type,
                    rank_mixer_mode,
                )
            self.mixer = UniMixingBlock(
                d_model=d_model,
                n_total=n_total,
                hidden_mult=hidden_mult,
                dropout=dropout,
                block_size=unimixing_block_size,
                temperature=unimixing_temperature,
                sinkhorn_iters=unimixing_sinkhorn_iters,
            )
        elif token_mixer_type == 'unimixing_lite':
            if rank_mixer_mode != 'full':
                logging.info(
                    "token_mixer_type=%s ignores rank_mixer_mode=%s and uses the "
                    "paper-style UniMixing-Lite block.",
                    token_mixer_type,
                    rank_mixer_mode,
                )
            self.mixer = UniMixingLiteBlock(
                d_model=d_model,
                n_total=n_total,
                hidden_mult=hidden_mult,
                dropout=dropout,
                num_basis=unimixing_num_basis,
                global_rank=unimixing_global_rank,
                block_size=unimixing_block_size,
                temperature=unimixing_temperature,
                sinkhorn_iters=unimixing_sinkhorn_iters,
            )
        else:
            raise ValueError(f"Unknown token_mixer_type: {token_mixer_type}")

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[list, torch.Tensor, list, list]:
        """Processes one multi-sequence HyFormer block step.

        Args:
            q_tokens_list: List of (B, Nq, D) tensors, length S.
            ns_tokens: (B, Nns, D)
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S.
            rope_cos_list: List of (1, L_i, head_dim) tensors, length S.
            rope_sin_list: List of (1, L_i, head_dim) tensors, length S.

        Returns:
            A tuple (next_q_list, next_ns, next_seq_list, next_masks), where
            next_q_list is a list of (B, Nq, D) updated query tensors,
            next_ns is (B, Nns, D) updated non-sequence tokens,
            next_seq_list is a list of (B, L_i', D) encoded sequence tensors,
            and next_masks is a list of (B, L_i') updated padding masks.
        """
        S = self.num_sequences
        Nq = self.num_queries

        # 1. Independent Sequence Evolution per sequence
        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i], seq_padding_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        # 2. Independent Query Decoding per sequence
        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i], next_seqs[i], next_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            decoded_qs.append(decoded_q_i)

        # 3. Token Fusion: concatenate all decoded_q + ns_tokens
        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)  # (B, Nq*S + Nns, D)

        # 4. Query Boosting
        boosted = self.mixer(combined)  # (B, Nq*S + Nns, D)

        # 5. Split back into per-sequence Q and NS
        next_q_list = []
        offset = 0
        for i in range(S):
            next_q_list.append(boosted[:, offset:offset + Nq, :])
            offset += Nq
        next_ns = boosted[:, offset:, :]

        return next_q_list, next_ns, next_seqs, next_masks


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer Main Model
# ═══════════════════════════════════════════════════════════════════════════════


class GroupNSTokenizer(nn.Module):
    """NS tokenizer used by ns_tokenizer_type='group'.

    Groups discrete features by fid, applies shared embedding with mean
    pooling per multi-valued feature, then projects each group to a single
    NS token (one token per group).
    """

    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Per-group projection: num_fids_in_group * emb_dim -> d_model (with LayerNorm)
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for group in groups
        ])

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds and projects grouped discrete features into NS tokens.

        Args:
            int_feats: (B, total_int_dim), concatenated integer features.

        Returns:
            Tokens of shape (B, num_groups, D).
        """
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    # Filtered high-cardinality feature: output zero vector
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        # Single-value feature: direct lookup
                        fid_emb = emb_layer(int_feats[:, offset].long())  # (B, emb_dim)
                    else:
                        # Multi-value feature: lookup then mean pooling (ignoring padding=0)
                        vals = int_feats[:, offset:offset + length].long()  # (B, length)
                        emb_all = emb_layer(vals)  # (B, length, emb_dim)
                        mask = (vals != 0).float().unsqueeze(-1)  # (B, length, 1)
                        count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count  # (B, emb_dim)
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)  # (B, num_fids*emb_dim)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))  # (B, 1, D)
        return torch.cat(tokens, dim=1)  # (B, num_groups, D)


class RankMixerNSTokenizer(nn.Module):
    """NS Tokenizer following the RankMixer paper's approach.

    All group embedding vectors are concatenated into a single long vector,
    then equally split into num_ns_tokens segments, each projected to d_model.
    This allows num_ns_tokens to be chosen freely (independent of group count).
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        """Initializes RankMixerNSTokenizer.

        Args:
            feature_specs: [(vocab_size, offset, length), ...] per feature.
            groups: List of feature index groups (defines semantic ordering).
            emb_dim: Embedding dimension per feature.
            d_model: Output token dimension.
            num_ns_tokens: Number of NS tokens to produce (T segments).
            emb_skip_threshold: Skip embedding for features with vocab > threshold.
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Compute total embedding dim: sum of all fids across all groups
        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim

        # Pad total_emb_dim to be divisible by num_ns_tokens
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        # Per-chunk projection: chunk_dim -> d_model with LayerNorm
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_ns_tokens)
        ])

        logging.info(
            f"RankMixerNSTokenizer: {total_num_fids} fids, "
            f"total_emb_dim={total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds all features, concatenates, splits, and projects.

        Args:
            int_feats: (B, total_int_dim) concatenated integer features.

        Returns:
            (B, num_ns_tokens, d_model) tensor.
        """
        # 1. Embed all fids in group order → flat cat
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                all_embs.append(fid_emb)

        cat_emb = torch.cat(all_embs, dim=-1)  # (B, total_emb_dim)

        # 2. Pad if needed
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))  # (B, padded_total_dim)

        # 3. Split into num_ns_tokens chunks and project each
        chunks = cat_emb.split(self.chunk_dim, dim=-1)  # list of (B, chunk_dim)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))  # (B, 1, d_model)

        return torch.cat(tokens, dim=1)  # (B, num_ns_tokens, d_model)


class AlignedDenseNSTokenizer(nn.Module):
    """User dense tokenizer that keeps aligned int/dense structure.

    For shared fids such as ``user_int_feats_62`` + ``user_dense_feats_62``,
    every position first fuses the id embedding with its aligned float value,
    then mean-pools to one field vector. All field vectors are concatenated
    and chunk-projected into a fixed number of NS tokens, which keeps latency
    comparable to the baseline's single dense token path.
    """

    def __init__(
        self,
        int_feature_ids: List[int],
        int_feature_specs: List[Tuple[int, int, int]],
        dense_feature_specs: List[Tuple[int, int, int]],
        emb_dim: int,
        d_model: int,
        num_dense_tokens: int = 1,
        emb_skip_threshold: int = 0,
        use_field_enhancement: bool = False,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.d_model = d_model
        self.num_dense_tokens = num_dense_tokens
        self.emb_skip_threshold = emb_skip_threshold
        self.use_field_enhancement = use_field_enhancement

        int_spec_map: Dict[int, Tuple[int, int, int]] = {
            fid: spec for fid, spec in zip(int_feature_ids, int_feature_specs)
        }
        dense_spec_map: Dict[int, Tuple[int, int]] = {
            fid: (offset, length) for fid, offset, length in dense_feature_specs
        }

        shared_fids = []
        dense_only_fids = []
        for fid, (dense_offset, dense_len) in dense_spec_map.items():
            if fid in int_spec_map and int_spec_map[fid][2] == dense_len:
                shared_fids.append(fid)
            else:
                dense_only_fids.append(fid)
        shared_fids = sorted(shared_fids)
        dense_only_fids = sorted(dense_only_fids)

        self.shared_fields: List[Tuple[int, int, int, int, int]] = []
        embs = []
        self._emb_index: List[int] = []
        real_idx = 0
        for fid in shared_fids:
            vocab_size, int_offset, field_len = int_spec_map[fid]
            dense_offset, _ = dense_spec_map[fid]
            self.shared_fields.append(
                (fid, vocab_size, int_offset, dense_offset, field_len)
            )
            skip = (
                int(vocab_size) <= 0
                or (emb_skip_threshold > 0 and int(vocab_size) > emb_skip_threshold)
            )
            if skip:
                self._emb_index.append(-1)
            else:
                embs.append(nn.Embedding(int(vocab_size) + 1, emb_dim, padding_idx=0))
                self._emb_index.append(real_idx)
                real_idx += 1
        self.embs = nn.ModuleList(embs)

        self.dense_only_fields: List[Tuple[int, int, int]] = [
            (fid, dense_spec_map[fid][0], dense_spec_map[fid][1])
            for fid in dense_only_fids
        ]

        self.value_proj = nn.Linear(1, emb_dim)
        self.value_gate = nn.Linear(1, emb_dim)
        self.shared_norm = nn.LayerNorm(emb_dim)
        self.dense_only_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(length, emb_dim),
                nn.LayerNorm(emb_dim),
            )
            for _, _, length in self.dense_only_fields
        ])

        total_fields = len(self.shared_fields) + len(self.dense_only_fields)
        total_emb_dim = max(total_fields, 1) * emb_dim
        self.chunk_dim = math.ceil(total_emb_dim / num_dense_tokens)
        self.padded_total_dim = self.chunk_dim * num_dense_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_dense_tokens)
        ])
        total_fields = max(total_fields, 1)
        if use_field_enhancement:
            self.field_embedding = nn.Embedding(total_fields, emb_dim)
            self.field_gate = nn.Sequential(
                nn.Linear(2 * emb_dim, emb_dim),
                nn.SiLU(),
                nn.Linear(emb_dim, 1),
            )
        else:
            self.field_embedding = None
            self.field_gate = None

        logging.info(
            "AlignedDenseNSTokenizer: shared=%d, dense_only=%d, "
            "num_dense_tokens=%d, chunk_dim=%d",
            len(self.shared_fields),
            len(self.dense_only_fields),
            num_dense_tokens,
            self.chunk_dim,
        )

    def forward(
        self,
        int_feats: torch.Tensor,
        dense_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse aligned user int/dense fields into a fixed number of tokens."""
        B = dense_feats.shape[0]
        field_vectors = []
        field_idx_counter = 0

        for field_idx, (_, _, int_offset, dense_offset, field_len) in enumerate(self.shared_fields):
            ids = int_feats[:, int_offset:int_offset + field_len].long()
            values = dense_feats[:, dense_offset:dense_offset + field_len].float().unsqueeze(-1)
            mask = (ids != 0).float().unsqueeze(-1)

            emb_real_idx = self._emb_index[field_idx]
            if emb_real_idx == -1:
                id_emb = dense_feats.new_zeros(B, field_len, self.emb_dim)
            else:
                id_emb = self.embs[emb_real_idx](ids)

            value_bias = self.value_proj(values)
            value_gate = torch.sigmoid(self.value_gate(values))
            fused = id_emb * (1.0 + value_gate) + value_bias
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (fused * mask).sum(dim=1) / denom
            pooled = self.shared_norm(pooled)
            if self.field_embedding is not None and self.field_gate is not None:
                field_bias = self.field_embedding.weight[field_idx_counter].view(1, -1)
                gate = torch.sigmoid(
                    self.field_gate(torch.cat([pooled, field_bias.expand_as(pooled)], dim=-1))
                )
                pooled = pooled + gate * field_bias
            field_vectors.append(pooled)
            field_idx_counter += 1

        for (fid, dense_offset, field_len), proj in zip(self.dense_only_fields, self.dense_only_projs):
            del fid
            values = dense_feats[:, dense_offset:dense_offset + field_len].float()
            pooled = F.silu(proj(values))
            if self.field_embedding is not None and self.field_gate is not None:
                field_bias = self.field_embedding.weight[field_idx_counter].view(1, -1)
                gate = torch.sigmoid(
                    self.field_gate(torch.cat([pooled, field_bias.expand_as(pooled)], dim=-1))
                )
                pooled = pooled + gate * field_bias
            field_vectors.append(pooled)
            field_idx_counter += 1

        if not field_vectors:
            return dense_feats.new_zeros(B, self.num_dense_tokens, self.d_model)

        cat_emb = torch.cat(field_vectors, dim=-1)
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))

        tokens = []
        for chunk, proj in zip(cat_emb.split(self.chunk_dim, dim=-1), self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))
        return torch.cat(tokens, dim=1)


class ContextTemporalTokenizer(nn.Module):
    """Tokenizes absolute-time features of the current sample timestamp."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        vocab_sizes = [
            25,  # hour
            8,   # weekday
            3,   # rest day
            3,   # holiday
        ]
        self.id_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model, padding_idx=0)
            for vocab_size in vocab_sizes
        ])
        self.dense_proj = None
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        temporal_ids: torch.Tensor,
        temporal_dense: torch.Tensor,
    ) -> torch.Tensor:
        if temporal_ids.shape[-1] != CONTEXT_TEMPORAL_ID_DIM:
            B = temporal_ids.shape[0]
            temporal_ids = temporal_ids.new_zeros(B, CONTEXT_TEMPORAL_ID_DIM)
        if temporal_dense.shape[-1] != CONTEXT_TEMPORAL_DENSE_DIM:
            B = temporal_dense.shape[0]
            temporal_dense = temporal_dense.new_zeros(B, CONTEXT_TEMPORAL_DENSE_DIM)

        id_emb = torch.stack(
            [
                emb(temporal_ids[:, field_idx].long())
                for field_idx, emb in enumerate(self.id_embeddings)
            ],
            dim=1,
        ).sum(dim=1)
        token = id_emb
        if self.dense_proj is not None:
            token = token + self.dense_proj(temporal_dense.float())
        return self.out_norm(token).unsqueeze(1)


class UserContextTemporalGate(nn.Module):
    """Inject current-sample timestamp features into user-side NS tokens."""

    def __init__(self, d_model: int, num_tokens: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        vocab_sizes = [
            25,  # hour
            8,   # weekday
            3,   # rest day
            3,   # holiday
        ]
        self.id_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model, padding_idx=0)
            for vocab_size in vocab_sizes
        ])
        self.dense_proj = None
        self.time_norm = nn.LayerNorm(d_model)
        self.gate_mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.token_position_bias = nn.Parameter(torch.zeros(1, num_tokens, d_model))

    def forward(
        self,
        user_tokens: torch.Tensor,
        temporal_ids: torch.Tensor,
        temporal_dense: torch.Tensor,
    ) -> torch.Tensor:
        if user_tokens.numel() == 0:
            return user_tokens
        if temporal_ids.shape[-1] != CONTEXT_TEMPORAL_ID_DIM:
            B = temporal_ids.shape[0]
            temporal_ids = temporal_ids.new_zeros(B, CONTEXT_TEMPORAL_ID_DIM)
        if temporal_dense.shape[-1] != CONTEXT_TEMPORAL_DENSE_DIM:
            B = temporal_dense.shape[0]
            temporal_dense = temporal_dense.new_zeros(B, CONTEXT_TEMPORAL_DENSE_DIM)

        id_emb = torch.stack(
            [
                emb(temporal_ids[:, field_idx].long())
                for field_idx, emb in enumerate(self.id_embeddings)
            ],
            dim=1,
        ).sum(dim=1)
        time_context = id_emb
        if self.dense_proj is not None:
            time_context = time_context + self.dense_proj(temporal_dense.float())
        time_context = self.time_norm(time_context)

        valid = (temporal_ids > 0).any(dim=-1, keepdim=True).to(user_tokens.dtype)
        time_context = time_context * valid
        time_expanded = time_context.unsqueeze(1).expand(-1, user_tokens.shape[1], -1)
        gate_input = torch.cat(
            [user_tokens, time_expanded, user_tokens * time_expanded],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_mlp(gate_input))
        position_bias = self.token_position_bias[:, :user_tokens.shape[1], :]
        return user_tokens + gate * (time_expanded + position_bias) * valid.unsqueeze(1)


class DomainGateFusion(nn.Module):
    """Lightweight domain-level gating on top of per-domain query tokens."""

    def __init__(self, d_model: int, num_sequences: int) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.gate_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 1),
            )
            for _ in range(num_sequences)
        ])

    def forward(
        self,
        q_tokens_list: List[torch.Tensor],
        ns_tokens: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        ns_summary = ns_tokens.mean(dim=1)
        gate_logits = []
        for q_tokens, mlp in zip(q_tokens_list, self.gate_mlps):
            q_summary = q_tokens.mean(dim=1)
            gate_logits.append(mlp(torch.cat([q_summary, ns_summary], dim=-1)))

        gate_logits = torch.cat(gate_logits, dim=1)
        gate_weights = torch.softmax(gate_logits, dim=1)

        gated_qs = []
        for idx, q_tokens in enumerate(q_tokens_list):
            scale = gate_weights[:, idx].view(-1, 1, 1) * self.num_sequences
            gated_qs.append(q_tokens * scale)
        return gated_qs, gate_weights


class DomainContextQueryInjection(nn.Module):
    """Injects per-domain + global context into query tokens."""

    def __init__(self, d_model: int, num_sequences: int) -> None:
        super().__init__()
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(num_sequences)
        ])

    def forward(
        self,
        q_tokens_list: List[torch.Tensor],
        domain_contexts: torch.Tensor,
        global_context: torch.Tensor,
    ) -> List[torch.Tensor]:
        injected = []
        for idx, (q_tokens, gate_mlp) in enumerate(zip(q_tokens_list, self.gates)):
            q_summary = q_tokens.mean(dim=1)
            domain_ctx = domain_contexts[:, idx, :]
            gate = torch.sigmoid(
                gate_mlp(torch.cat([q_summary, domain_ctx, global_context], dim=-1))
            )
            ctx = domain_ctx + global_context
            injected.append(q_tokens + gate.unsqueeze(1) * ctx.unsqueeze(1))
        return injected


class GlobalContextOutputInjection(nn.Module):
    """Late-fusion global context into the final pooled output."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, output: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([output, context], dim=-1)))
        return output + gate * context


class TargetSeqFeatureFusion(nn.Module):
    """Item-conditioned pooling over positive/negative target-sequence signals."""

    def __init__(
        self,
        d_model: int,
        per_sequence_dim: int,
        sideinfo_counts: List[int],
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.per_sequence_dim = per_sequence_dim
        self.sideinfo_counts = sideinfo_counts

        self.pos_proj = nn.Sequential(
            nn.Linear(per_sequence_dim // 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.neg_proj = nn.Sequential(
            nn.Linear(per_sequence_dim // 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.pos_item_proj = nn.Linear(d_model, d_model)
        self.neg_item_proj = nn.Linear(d_model, d_model)
        self.pos_score = nn.Linear(d_model, 1)
        self.neg_score = nn.Linear(d_model, 1)

    @staticmethod
    def _masked_softmax(
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if logits.numel() == 0:
            return logits
        masked_logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(masked_logits, dim=-1)
        weights = weights * mask.float()
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(
        self,
        target_seq_feats: torch.Tensor,
        item_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = target_seq_feats.shape[0]
        item_pos_bias = self.pos_item_proj(item_context).unsqueeze(1)
        item_neg_bias = self.neg_item_proj(item_context).unsqueeze(1)
        pos_domain_summaries: List[torch.Tensor] = []
        neg_domain_summaries: List[torch.Tensor] = []
        domain_valid_masks: List[torch.Tensor] = []
        offset = 0

        for count in self.sideinfo_counts:
            if count <= 0:
                pos_domain_summaries.append(item_context.new_zeros(B, self.d_model))
                neg_domain_summaries.append(item_context.new_zeros(B, self.d_model))
                domain_valid_masks.append(
                    torch.zeros(B, dtype=torch.bool, device=item_context.device)
                )
                continue

            raw = target_seq_feats[
                :, offset: offset + count * self.per_sequence_dim
            ].view(B, count, self.per_sequence_dim)
            offset += count * self.per_sequence_dim

            valid_mask = raw.abs().sum(dim=-1) > 0
            pos_raw = raw[:, :, : self.per_sequence_dim // 2]
            neg_raw = raw[:, :, self.per_sequence_dim // 2 :]
            pos_emb = self.pos_proj(pos_raw)
            neg_emb = self.neg_proj(neg_raw)

            pos_logits = self.pos_score(
                torch.tanh(pos_emb + item_pos_bias)
            ).squeeze(-1)
            neg_logits = self.neg_score(
                torch.tanh(neg_emb + item_neg_bias)
            ).squeeze(-1)
            pos_weights = self._masked_softmax(pos_logits, valid_mask)
            neg_weights = self._masked_softmax(neg_logits, valid_mask)
            pos_domain_summary = (pos_emb * pos_weights.unsqueeze(-1)).sum(dim=1)
            neg_domain_summary = (neg_emb * neg_weights.unsqueeze(-1)).sum(dim=1)

            pos_domain_summaries.append(pos_domain_summary)
            neg_domain_summaries.append(neg_domain_summary)
            domain_valid_masks.append(valid_mask.any(dim=1))

        pos_domain_stack = torch.stack(pos_domain_summaries, dim=1)
        neg_domain_stack = torch.stack(neg_domain_summaries, dim=1)
        domain_mask = torch.stack(domain_valid_masks, dim=1)
        pos_domain_logits = self.pos_score(
            torch.tanh(pos_domain_stack + item_pos_bias)
        ).squeeze(-1)
        neg_domain_logits = self.neg_score(
            torch.tanh(neg_domain_stack + item_neg_bias)
        ).squeeze(-1)
        pos_domain_weights = self._masked_softmax(pos_domain_logits, domain_mask)
        neg_domain_weights = self._masked_softmax(neg_domain_logits, domain_mask)
        pos_summary = (pos_domain_stack * pos_domain_weights.unsqueeze(-1)).sum(dim=1)
        neg_summary = (neg_domain_stack * neg_domain_weights.unsqueeze(-1)).sum(dim=1)
        return pos_summary, neg_summary


class IATSequenceBranch(nn.Module):
    """Two-stage IAT-style downstream sequence modeling over historical InsEmb."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ins_emb_dim: int,
        history_len: int = 192,
        num_layers: int = 1,
        num_time_buckets: int = 0,
        task_hash_size: int = 65536,
        hidden_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.ins_emb_dim = ins_emb_dim
        self.num_time_buckets = max(num_time_buckets, 2)
        self.label_embedding = nn.Embedding(3, 8, padding_idx=0)
        self.time_embedding = nn.Embedding(self.num_time_buckets, 8, padding_idx=0)
        self.task_embedding = nn.Embedding(task_hash_size + 1, 16, padding_idx=0)
        self.ins_emb_adapt = nn.Sequential(
            nn.Linear(ins_emb_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.token_proj = nn.Sequential(
            nn.Linear(d_model + 8 + 8 + 16, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.query_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList([
            TransformerEncoder(
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.output_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        query_context: torch.Tensor,
        hist_ins_emb: torch.Tensor,
        hist_labels: torch.Tensor,
        hist_time_buckets: torch.Tensor,
        hist_task_ids: torch.Tensor,
        hist_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hist_ins_emb.shape[1] == 0:
            return query_context

        hist_core = self.ins_emb_adapt(hist_ins_emb)
        side = torch.cat(
            [
                self.label_embedding(hist_labels),
                self.time_embedding(hist_time_buckets.clamp(max=self.num_time_buckets - 1)),
                self.task_embedding(hist_task_ids),
            ],
            dim=-1,
        )
        hist_tokens = self.token_proj(torch.cat([hist_core, side], dim=-1))

        query_token = self.query_proj(query_context).unsqueeze(1)
        x = torch.cat([query_token, hist_tokens], dim=1)
        seq_mask = torch.cat(
            [
                hist_mask.new_zeros(hist_mask.shape[0], 1),
                hist_mask,
            ],
            dim=1,
        )

        for layer in self.layers:
            x, seq_mask = layer(x, key_padding_mask=seq_mask)

        iat_context = x[:, 0, :]
        gate = torch.sigmoid(self.output_gate(torch.cat([query_context, iat_context], dim=-1)))
        return query_context + gate * iat_context


class PCVRHyFormer(nn.Module):
    """PCVRHyFormer model for post-click conversion rate prediction.

    Combines MultiSeqHyFormerBlock and MultiSeqQueryGenerator to process
    multiple input sequences with non-sequence features.
    """

    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",  # {domain: [vocab_size_per_fid, ...]}
        # NS grouping config (grouped by fid index)
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        user_int_feature_ids: Optional[List[int]] = None,
        item_int_feature_ids: Optional[List[int]] = None,
        user_dense_feature_specs: Optional[List[Tuple[int, int, int]]] = None,
        item_dense_feature_specs: Optional[List[Tuple[int, int, int]]] = None,
        coarse_feat_dim: int = 0,
        target_seq_feat_dim: int = 0,
        coarse_seq_per_domain_dim: int = 0,
        coarse_global_seq_dim: int = 0,
        coarse_global_static_dim: int = 0,
        target_seq_sideinfo_counts: Optional[List[int]] = None,
        # Model hyperparameters
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = 'full',
        token_mixer_type: str = 'rankmixer',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        # NS tokenizer variant
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        use_aligned_user_dense_tokens: bool = True,
        user_dense_tokens: int = 1,
        use_token_type_embeddings: bool = True,
        use_domain_gating: bool = True,
        use_representation_enhancements: bool = False,
        use_coarse_features: bool = True,
        use_activity_rhythm_features: bool = False,
        use_target_seq_features: bool = True,
        use_temporal_sequence_features: bool = True,
        use_temporal_id_attention_gate: bool = True,
        use_context_temporal_features: bool = False,
        use_user_context_temporal_gate: bool = False,
        use_din_query_pooling: bool = False,
        use_unimixing_lite: bool = False,
        unimixing_num_basis: int = 4,
        unimixing_global_rank: int = 4,
        unimixing_block_size: int = 0,
        unimixing_temperature: float = 1.0,
        unimixing_sinkhorn_iters: int = 3,
        use_unimixing_siamese_norm: bool = False,
        use_iat_sequence_branch: bool = False,
        iat_max_tokens: int = 192,
        iat_num_layers: int = 1,
        iat_ins_emb_dim: int = 64,
        iat_task_hash_size: int = 65536,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())  # deterministic order
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.rank_mixer_mode = rank_mixer_mode
        if use_unimixing_lite and token_mixer_type == 'rankmixer':
            token_mixer_type = 'unimixing_lite'
        self.token_mixer_type = token_mixer_type
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type
        self.use_aligned_user_dense_tokens = use_aligned_user_dense_tokens
        self.use_token_type_embeddings = use_token_type_embeddings
        self.use_domain_gating = use_domain_gating
        self.use_representation_enhancements = use_representation_enhancements
        self.use_activity_rhythm_features = bool(use_activity_rhythm_features)
        self.use_temporal_sequence_features = bool(use_temporal_sequence_features)
        self.use_temporal_id_attention_gate = bool(use_temporal_id_attention_gate)
        self.use_context_temporal_features = bool(use_context_temporal_features)
        self.use_user_context_temporal_gate = bool(use_user_context_temporal_gate)
        self.use_din_query_pooling = bool(use_din_query_pooling)
        self.use_unimixing_lite = token_mixer_type == 'unimixing_lite'
        self.use_unimixing = token_mixer_type == 'unimixing'
        self.unimixing_num_basis = unimixing_num_basis
        self.unimixing_global_rank = unimixing_global_rank
        self.unimixing_block_size = unimixing_block_size
        self.unimixing_temperature = unimixing_temperature
        self.unimixing_sinkhorn_iters = unimixing_sinkhorn_iters
        self.use_unimixing_siamese_norm = bool(use_unimixing_siamese_norm)
        self.use_iat_sequence_branch = use_iat_sequence_branch
        self.iat_history_len = iat_max_tokens
        self.iat_num_layers = iat_num_layers
        self.iat_ins_emb_dim = iat_ins_emb_dim
        self.iat_task_hash_size = iat_task_hash_size
        self.user_dense_feature_specs = user_dense_feature_specs or []
        self.item_dense_feature_specs = item_dense_feature_specs or []
        self.user_int_feature_ids = user_int_feature_ids or []
        self.item_int_feature_ids = item_int_feature_ids or []
        self.coarse_feat_dim = coarse_feat_dim
        self.use_coarse_features = use_coarse_features and coarse_feat_dim > 0
        self.coarse_seq_per_domain_dim = coarse_seq_per_domain_dim
        self.coarse_global_seq_dim = coarse_global_seq_dim
        self.coarse_global_static_dim = coarse_global_static_dim
        self.target_seq_feat_dim = target_seq_feat_dim
        self.target_seq_sideinfo_counts = target_seq_sideinfo_counts or []
        self.use_target_seq_features = use_target_seq_features and target_seq_feat_dim > 0
        self.use_seq_field_enhancement = use_representation_enhancements

        # ================== NS Tokens Construction ==================

        if ns_tokenizer_type == 'group':
            # Original: one NS token per group
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = len(user_ns_groups)

            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            # RankMixer paper style: all embeddings cat → split → project
            # 0 means auto: fall back to group count
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        # User dense feature projection (if available)
        self.has_user_dense = user_dense_dim > 0
        self.user_dense_tokenizer: Optional[AlignedDenseNSTokenizer] = None
        self.user_dense_proj: Optional[nn.Sequential] = None
        if self.has_user_dense:
            if (
                use_aligned_user_dense_tokens
                and self.user_dense_feature_specs
                and self.user_int_feature_ids
            ):
                self.user_dense_tokenizer = AlignedDenseNSTokenizer(
                    int_feature_ids=self.user_int_feature_ids,
                    int_feature_specs=user_int_feature_specs,
                    dense_feature_specs=self.user_dense_feature_specs,
                    emb_dim=emb_dim,
                    d_model=d_model,
                    num_dense_tokens=user_dense_tokens,
                    emb_skip_threshold=emb_skip_threshold,
                    use_field_enhancement=use_representation_enhancements,
                )
                num_user_dense_tokens = user_dense_tokens
            else:
                self.user_dense_proj = nn.Sequential(
                    nn.Linear(user_dense_dim, d_model),
                    nn.LayerNorm(d_model),
                )
                num_user_dense_tokens = 1
        else:
            num_user_dense_tokens = 0

        # Item dense feature projection (if available)
        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )
            num_item_dense_tokens = 1
        else:
            num_item_dense_tokens = 0

        if self.use_context_temporal_features:
            self.context_temporal_tokenizer: Optional[ContextTemporalTokenizer] = (
                ContextTemporalTokenizer(d_model=d_model)
            )
            num_context_temporal_tokens = 1
        else:
            self.context_temporal_tokenizer = None
            num_context_temporal_tokens = 0
        if self.use_user_context_temporal_gate:
            self.user_context_temporal_gate: Optional[UserContextTemporalGate] = (
                UserContextTemporalGate(
                    d_model=d_model,
                    num_tokens=num_user_ns + num_user_dense_tokens,
                )
            )
        else:
            self.user_context_temporal_gate = None

        # Total NS token count
        self.num_ns = (
            num_user_ns
            + num_user_dense_tokens
            + num_item_ns
            + num_item_dense_tokens
            + num_context_temporal_tokens
        )

        # ================== Check d_model % T == 0 constraint (full mode only) ==================
        T = num_queries * self.num_sequences + self.num_ns
        if token_mixer_type == 'rankmixer' and rank_mixer_mode == 'full' and d_model % T != 0:
            valid_T_values = [t for t in range(1, d_model + 1) if d_model % t == 0]
            raise ValueError(
                f"d_model={d_model} must be divisible by T=num_queries*num_sequences+num_ns="
                f"{num_queries}*{self.num_sequences}+{self.num_ns}={T}. "
                f"Valid T values for d_model={d_model}: {valid_T_values}"
            )

        # ================== Seq Tokens Embedding ==================
        # seq_id_threshold decides which features inside the seq tokenizer are
        # treated as id features (they receive extra dropout). It is fully
        # independent of emb_skip_threshold (which skips Embedding creation).
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            """Create embedding list, returning None for features skipped via
            emb_skip_threshold or with no vocab info (vs<=0)."""
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            # Map from position index to real index in module_list (-1 if skipped)
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        # ================== Dynamic Sequence Embeddings ==================
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}    # domain -> index_map
        self._seq_is_id = {}        # domain -> is_id list
        self._seq_vocab_sizes = {}  # domain -> vocab_sizes list
        self._seq_proj = nn.ModuleDict()
        if self.use_seq_field_enhancement:
            self._seq_field_emb = nn.ModuleDict()
            self.seq_field_gate = nn.Sequential(
                nn.Linear(2 * emb_dim, emb_dim),
                nn.SiLU(),
                nn.Linear(emb_dim, 1),
            )
        else:
            self._seq_field_emb = nn.ModuleDict()
            self.seq_field_gate = None

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            if self.use_seq_field_enhancement:
                self._seq_field_emb[domain] = nn.Embedding(len(vs), emb_dim)

        self.token_type_embedding: Optional[nn.Embedding]
        self.seq_token_type_offset = 5 if self.use_context_temporal_features else 4
        if use_token_type_embeddings:
            self.token_type_embedding = nn.Embedding(
                self.seq_token_type_offset + self.num_sequences,
                d_model,
            )
        else:
            self.token_type_embedding = None

        if self.use_coarse_features:
            self.coarse_output_head = nn.Sequential(
                nn.Linear(coarse_feat_dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Linear(d_model, action_num),
            )
        else:
            self.coarse_output_head = None

        if self.use_target_seq_features:
            self.target_seq_fusion = TargetSeqFeatureFusion(
                d_model=d_model,
                per_sequence_dim=8,
                sideinfo_counts=self.target_seq_sideinfo_counts,
            )
            self.target_output_head = nn.Sequential(
                nn.Linear(4 * d_model, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Linear(d_model, action_num),
            )
        else:
            self.target_seq_fusion = None
            self.target_output_head = None

        if self.use_iat_sequence_branch:
            self.iat_source_compress = nn.Sequential(
                nn.Linear(d_model, iat_ins_emb_dim),
                nn.LayerNorm(iat_ins_emb_dim),
                nn.SiLU(),
            )
            self.iat_source_decompress = nn.Sequential(
                nn.Linear(iat_ins_emb_dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )
            self.iat_source_norm = nn.LayerNorm(d_model)
            self.iat_sequence_branch = IATSequenceBranch(
                d_model=d_model,
                num_heads=num_heads,
                ins_emb_dim=iat_ins_emb_dim,
                history_len=iat_max_tokens,
                num_layers=iat_num_layers,
                num_time_buckets=num_time_buckets,
                task_hash_size=iat_task_hash_size,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
            )
        else:
            self.iat_source_compress = None
            self.iat_source_decompress = None
            self.iat_source_norm = None
            self.iat_sequence_branch = None

        # ================== Time Interval Bucket Embedding (optional) ==================
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)
        if self.use_temporal_sequence_features:
            temporal_id_vocab_sizes = [
                25,  # hour
                8,   # weekday
                3,   # holiday-adjusted rest day
                3,   # holiday
                max(num_time_buckets, 2),  # inter-event gap bucket
                33,  # request position
                33,  # session position
                33,  # visit position
                3,   # request boundary
                3,   # session boundary
                3,   # visit boundary
            ]
            self.seq_temporal_id_embeddings = nn.ModuleList([
                nn.Embedding(vocab_size, d_model, padding_idx=0)
                for vocab_size in temporal_id_vocab_sizes
            ])
            if self.use_temporal_id_attention_gate:
                self.seq_temporal_id_gate: Optional[nn.Sequential] = nn.Sequential(
                    nn.Linear(2 * d_model, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, 1),
                )
            else:
                self.seq_temporal_id_gate = None
        else:
            self.seq_temporal_id_embeddings = nn.ModuleList()
            self.seq_temporal_id_gate = None
        if self.use_temporal_sequence_features:
            self.seq_temporal_proj: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(SEQ_TEMPORAL_DENSE_DIM, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.seq_temporal_proj = None

        # ================== HyFormer Components ==================
        # MultiSeqQueryGenerator
        self.query_generator = MultiSeqQueryGenerator(
            d_model=d_model,
            num_ns=self.num_ns,
            num_queries=num_queries,
            num_sequences=self.num_sequences,
            hidden_mult=hidden_mult,
            use_din_query_pooling=self.use_din_query_pooling,
        )

        # MultiSeqHyFormerBlock stack
        self.blocks = nn.ModuleList([
            MultiSeqHyFormerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_queries=num_queries,
                num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                seq_encoder_type=seq_encoder_type,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                top_k=seq_top_k,
                causal=seq_causal,
                rank_mixer_mode=rank_mixer_mode,
                token_mixer_type=token_mixer_type,
                unimixing_num_basis=unimixing_num_basis,
                unimixing_global_rank=unimixing_global_rank,
                unimixing_block_size=unimixing_block_size,
                unimixing_temperature=unimixing_temperature,
                unimixing_sinkhorn_iters=unimixing_sinkhorn_iters,
            )
            for _ in range(num_hyformer_blocks)
        ])
        if self.use_unimixing_siamese_norm and token_mixer_type not in {'unimixing', 'unimixing_lite'}:
            logging.info(
                "use_unimixing_siamese_norm=True is ignored because token_mixer_type=%s",
                token_mixer_type,
            )
            self.use_unimixing_siamese_norm = False
        if self.use_unimixing_siamese_norm:
            unimixing_total_dim = T * d_model
            self.unimixing_siamese_y_norms = nn.ModuleList([
                FlatRMSNorm(unimixing_total_dim) for _ in range(num_hyformer_blocks)
            ])
            self.unimixing_siamese_x_norms = nn.ModuleList([
                FlatRMSNorm(unimixing_total_dim) for _ in range(num_hyformer_blocks)
            ])
            self.unimixing_siamese_output_norm = FlatRMSNorm(unimixing_total_dim)
        else:
            self.unimixing_siamese_y_norms = nn.ModuleList()
            self.unimixing_siamese_x_norms = nn.ModuleList()
            self.unimixing_siamese_output_norm = None
        self.domain_gate_fusion = (
            DomainGateFusion(d_model=d_model, num_sequences=self.num_sequences)
            if use_domain_gating else None
        )

        # ================== RoPE ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Dropout
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Classifier
        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num)
        )

        # Initialize parameters
        self._init_params()

        # Log emb_skip_threshold filtering stats
        if emb_skip_threshold > 0:
            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)
            for domain in self.seq_domains:
                f, t = _count_filtered(self._seq_vocab_sizes[domain], self._seq_emb_index[domain])
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features")
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features")

    def _init_params(self) -> None:
        """Applies Xavier initialization to all embedding weights."""
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.user_dense_tokenizer is not None:
            for emb in self.user_dense_tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.token_type_embedding is not None:
            nn.init.xavier_normal_(self.token_type_embedding.weight.data)
        if self.user_dense_tokenizer is not None and self.user_dense_tokenizer.field_embedding is not None:
            nn.init.xavier_normal_(self.user_dense_tokenizer.field_embedding.weight.data)
        if self.use_seq_field_enhancement:
            for emb in self._seq_field_emb.values():
                nn.init.xavier_normal_(emb.weight.data)

        for head in [self.coarse_output_head, self.target_output_head]:
            if head is not None:
                nn.init.zeros_(head[-1].weight.data)
                nn.init.zeros_(head[-1].bias.data)

        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0
        for emb in self.seq_temporal_id_embeddings:
            nn.init.xavier_normal_(emb.weight.data)
            emb.weight.data[0, :] = 0
        if self.seq_temporal_id_gate is not None:
            nn.init.zeros_(self.seq_temporal_id_gate[-1].weight.data)
            nn.init.zeros_(self.seq_temporal_id_gate[-1].bias.data)
        if self.context_temporal_tokenizer is not None:
            for emb in self.context_temporal_tokenizer.id_embeddings:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0
        if self.user_context_temporal_gate is not None:
            for emb in self.user_context_temporal_gate.id_embeddings:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0
            nn.init.zeros_(self.user_context_temporal_gate.gate_mlp[-1].weight.data)
            nn.init.zeros_(self.user_context_temporal_gate.gate_mlp[-1].bias.data)

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000
    ) -> "set[int]":
        """Reinitializes only high-cardinality embeddings.

        Preserves low-cardinality and time feature embeddings.

        Args:
            cardinality_threshold: Only embeddings with vocab_size exceeding
                this value are reinitialized.

        Returns:
            A set of data_ptr() values for reinitialized parameters.
        """
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    # Skipped by emb_skip_threshold, no embedding to reinit
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        if self.user_dense_tokenizer is not None:
            for i, (_, vocab_size, _, _, _) in enumerate(self.user_dense_tokenizer.shared_fields):
                real_idx = self.user_dense_tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = self.user_dense_tokenizer.embs[real_idx]
                if int(vocab_size) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        # Low-cardinality temporal embeddings are always preserved.
        if self.num_time_buckets > 0:
            skip_count += 1
        skip_count += len(self.seq_temporal_id_embeddings)
        if self.context_temporal_tokenizer is not None:
            skip_count += len(self.context_temporal_tokenizer.id_embeddings)
        if self.user_context_temporal_gate is not None:
            skip_count += len(self.user_context_temporal_gate.id_embeddings)

        logging.info(f"Re-initialized {reinit_count} high-cardinality Embeddings "
                     f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all embedding table parameters (optimized with Adagrad)."""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-embedding parameters (optimized with AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def set_unimixing_temperature(self, temperature: float) -> None:
        """Updates the shared temperature used by all UniMixing blocks."""
        for module in self.modules():
            if isinstance(module, UniMixingBase):
                module.set_temperature(temperature)

    def get_unimixing_temperature(self) -> Optional[float]:
        """Returns the current UniMixing temperature if enabled."""
        for module in self.modules():
            if isinstance(module, UniMixingBase):
                return module.temperature
        return None

    def _embed_seq_domain(
        self,
        domain: str,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
        temporal_ids: torch.Tensor,
        temporal_dense: torch.Tensor,
    ) -> torch.Tensor:
        """Embeds a sequence domain by concatenating sideinfo embeddings and projecting to d_model."""
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                # Feature skipped by emb_skip_threshold: output zero vector
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])  # (B, L, emb_dim)
                if self.use_seq_field_enhancement and self.seq_field_gate is not None:
                    field_bias = self._seq_field_emb[domain].weight[i].view(1, 1, -1)
                    gate = torch.sigmoid(
                        self.seq_field_gate(
                            torch.cat([e, field_bias.expand_as(e)], dim=-1)
                        )
                    )
                    e = e + gate * field_bias
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, S*emb_dim)
        token_emb = F.gelu(proj(cat_emb))  # (B, L, D)

        # Add time bucket embedding (all-zero ids produce zero vectors via padding_idx=0)
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)
        if self.use_temporal_sequence_features and temporal_ids.shape[-1] == SEQ_TEMPORAL_ID_DIM:
            temporal_id_embs = [
                emb(temporal_ids[:, :, field_idx].long())
                for field_idx, emb in enumerate(self.seq_temporal_id_embeddings)
            ]
            temporal_id_stack = torch.stack(temporal_id_embs, dim=2)
            if self.seq_temporal_id_gate is not None:
                gate_context = token_emb.unsqueeze(2).expand_as(temporal_id_stack)
                gate_input = torch.cat([temporal_id_stack, gate_context], dim=-1)
                gate_logits = self.seq_temporal_id_gate(gate_input).squeeze(-1)
                valid_temporal_ids = temporal_ids.long() > 0
                gate_logits = gate_logits.masked_fill(~valid_temporal_ids, -1e4)
                gate_weights = torch.softmax(gate_logits, dim=-1)
                token_emb = token_emb + (
                    gate_weights.unsqueeze(-1) * temporal_id_stack
                ).sum(dim=2)
            else:
                token_emb = token_emb + temporal_id_stack.sum(dim=2)
        if (
            self.use_temporal_sequence_features
            and self.seq_temporal_proj is not None
            and temporal_dense.shape[-1] == SEQ_TEMPORAL_DENSE_DIM
        ):
            token_emb = token_emb + self.seq_temporal_proj(temporal_dense)

        return token_emb

    def _make_padding_mask(
        self, seq_len: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Generates a padding mask from sequence lengths."""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _add_token_type(self, tokens: torch.Tensor, type_id: int) -> torch.Tensor:
        if self.token_type_embedding is None or tokens.numel() == 0:
            return tokens
        type_bias = self.token_type_embedding.weight[type_id].view(1, 1, -1)
        return tokens + type_bias

    def _get_coarse_feats(self, coarse_feats: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_coarse_features or coarse_feats.shape[1] == 0:
            return None
        return coarse_feats

    def _get_seq_time_bucket_list(self, inputs: ModelInput) -> List[torch.Tensor]:
        return [inputs.seq_time_buckets[domain] for domain in self.seq_domains]

    def _get_iat_history(
        self,
        inputs: ModelInput,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (
            not self.use_iat_sequence_branch
            or inputs.iat_hist_ins_emb.shape[1] == 0
        ):
            return None, None, None, None, None
        return (
            inputs.iat_hist_ins_emb,
            inputs.iat_hist_labels,
            inputs.iat_hist_time_buckets,
            inputs.iat_hist_task_ids,
            inputs.iat_hist_mask,
        )

    def _apply_iat_two_stage(
        self,
        output: torch.Tensor,
        inputs: ModelInput,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (
            not self.use_iat_sequence_branch
            or self.iat_source_compress is None
            or self.iat_source_decompress is None
            or self.iat_source_norm is None
            or self.iat_sequence_branch is None
        ):
            return output, None, None

        current_ins_emb = self.iat_source_compress(output)
        source_output = self.iat_source_norm(
            output + self.iat_source_decompress(current_ins_emb)
        )
        hist = self._get_iat_history(inputs)
        if hist[0] is None:
            return source_output, source_output, current_ins_emb

        final_output = self.iat_sequence_branch(
            source_output,
            hist[0],
            hist[1],
            hist[2],
            hist[3],
            hist[4],
        )
        return final_output, source_output, current_ins_emb

    def _build_ns_tokens(
        self,
        inputs: ModelInput,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_ns = self._add_token_type(
            self.user_ns_tokenizer(inputs.user_int_feats), type_id=0
        )
        item_ns = self._add_token_type(
            self.item_ns_tokenizer(inputs.item_int_feats), type_id=2
        )

        user_parts = [user_ns]
        if self.has_user_dense:
            if self.user_dense_tokenizer is not None:
                user_dense_tokens = self.user_dense_tokenizer(
                    inputs.user_int_feats, inputs.user_dense_feats
                )
            else:
                user_dense_tokens = F.silu(
                    self.user_dense_proj(inputs.user_dense_feats)
                ).unsqueeze(1)
            user_parts.append(self._add_token_type(user_dense_tokens, type_id=1))

        user_side_tokens = torch.cat(user_parts, dim=1)
        if self.user_context_temporal_gate is not None:
            user_side_tokens = self.user_context_temporal_gate(
                user_side_tokens,
                inputs.context_temporal_ids,
                inputs.context_temporal_dense,
            )

        ns_parts = [user_side_tokens, item_ns]
        if self.has_item_dense:
            item_dense_tokens = F.silu(
                self.item_dense_proj(inputs.item_dense_feats)
            ).unsqueeze(1)
            ns_parts.append(self._add_token_type(item_dense_tokens, type_id=3))
        if self.context_temporal_tokenizer is not None:
            context_temporal_token = self.context_temporal_tokenizer(
                inputs.context_temporal_ids,
                inputs.context_temporal_dense,
            )
            ns_parts.append(self._add_token_type(context_temporal_token, type_id=4))

        ns_tokens = torch.cat(ns_parts, dim=1)

        return ns_tokens, user_side_tokens, item_ns

    def _build_seq_tokens(
        self,
        inputs: ModelInput,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        seq_tokens_list = []
        seq_masks_list = []
        for domain_idx, domain in enumerate(self.seq_domains):
            tokens = self._embed_seq_domain(
                domain,
                inputs.seq_data[domain],
                self._seq_embs[domain],
                self._seq_proj[domain],
                self._seq_is_id[domain],
                self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                inputs.seq_temporal_ids[domain],
                inputs.seq_temporal_dense[domain],
            )
            tokens = self._add_token_type(
                tokens,
                type_id=self.seq_token_type_offset + domain_idx,
            )
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(
                inputs.seq_lens[domain],
                inputs.seq_data[domain].shape[2],
            )
            seq_masks_list.append(mask)
        return seq_tokens_list, seq_masks_list

    def _combine_q_ns_tokens(
        self,
        q_tokens_list: List[torch.Tensor],
        ns_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate per-domain queries with shared NS tokens."""
        return torch.cat(q_tokens_list + [ns_tokens], dim=1)

    def _split_combined_q_ns_tokens(
        self,
        combined: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Split the combined query/NS tensor back into its two parts."""
        next_q_list: List[torch.Tensor] = []
        offset = 0
        for _ in range(self.num_sequences):
            next_q_list.append(combined[:, offset:offset + self.num_queries, :])
            offset += self.num_queries
        next_ns = combined[:, offset:, :]
        return next_q_list, next_ns

    def _build_rope_lists(
        self,
        seq_tokens_list: List[torch.Tensor],
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """Prepare per-sequence RoPE tensors when RoPE is enabled."""
        if self.rotary_emb is None:
            return None, None

        rope_cos_list: List[torch.Tensor] = []
        rope_sin_list: List[torch.Tensor] = []
        device = seq_tokens_list[0].device
        for seq_tokens in seq_tokens_list:
            seq_len = seq_tokens.shape[1]
            cos, sin = self.rotary_emb(seq_len, device)
            rope_cos_list.append(cos)
            rope_sin_list.append(sin)
        return rope_cos_list, rope_sin_list

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        apply_dropout: bool = True
    ) -> torch.Tensor:
        """Runs the multi-sequence block stack with dropout and output projection."""
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list

        if self.use_unimixing_siamese_norm and self.unimixing_siamese_output_norm is not None:
            x_stream = self._combine_q_ns_tokens(curr_qs, curr_ns)
            y_stream = x_stream

            for block_idx, block in enumerate(self.blocks):
                rope_cos_list, rope_sin_list = self._build_rope_lists(curr_seqs)
                siamese_input = x_stream + self.unimixing_siamese_y_norms[block_idx](y_stream)
                block_qs, block_ns = self._split_combined_q_ns_tokens(siamese_input)
                next_qs, next_ns, curr_seqs, curr_masks = block(
                    q_tokens_list=block_qs,
                    ns_tokens=block_ns,
                    seq_tokens_list=curr_seqs,
                    seq_padding_masks=curr_masks,
                    rope_cos_list=rope_cos_list,
                    rope_sin_list=rope_sin_list,
                )
                block_output = self._combine_q_ns_tokens(next_qs, next_ns)
                delta = block_output - siamese_input
                x_stream = self.unimixing_siamese_x_norms[block_idx](x_stream + delta)
                y_stream = y_stream + delta
                curr_qs, curr_ns = self._split_combined_q_ns_tokens(x_stream)

            fused = x_stream + self.unimixing_siamese_output_norm(y_stream)
            curr_qs, curr_ns = self._split_combined_q_ns_tokens(fused)
        else:
            for block in self.blocks:
                rope_cos_list, rope_sin_list = self._build_rope_lists(curr_seqs)
                curr_qs, curr_ns, curr_seqs, curr_masks = block(
                    q_tokens_list=curr_qs,
                    ns_tokens=curr_ns,
                    seq_tokens_list=curr_seqs,
                    seq_padding_masks=curr_masks,
                    rope_cos_list=rope_cos_list,
                    rope_sin_list=rope_sin_list,
                )

        if self.domain_gate_fusion is not None:
            curr_qs, _ = self.domain_gate_fusion(curr_qs, curr_ns)

        # Output: concatenate all sequences' Q tokens then project via MLP
        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)  # (B, Nq*S, D)
        output = all_q.view(B, -1)  # (B, Nq*S*D)
        output = self.output_proj(output)  # (B, D)

        return output

    def _forward_impl(
        self,
        inputs: ModelInput,
        apply_dropout: bool,
    ) -> Dict[str, Optional[torch.Tensor]]:
        ns_tokens, user_ns, item_ns = self._build_ns_tokens(inputs)
        seq_tokens_list, seq_masks_list = self._build_seq_tokens(inputs)
        item_context = item_ns.mean(dim=1)
        q_tokens_list = self.query_generator(
            ns_tokens,
            seq_tokens_list,
            seq_masks_list,
            item_context=item_context,
        )

        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=apply_dropout
        )
        output, source_output, current_ins_emb = self._apply_iat_two_stage(output, inputs)

        logits = self.clsfier(output)  # (B, action_num)
        coarse_feats = self._get_coarse_feats(inputs.coarse_feats)
        if self.coarse_output_head is not None and coarse_feats is not None:
            logits = logits + self.coarse_output_head(coarse_feats)

        if self.target_seq_fusion is not None and self.target_output_head is not None:
            pos_target_ctx, neg_target_ctx = self.target_seq_fusion(
                inputs.target_seq_feats,
                item_context,
            )
            target_branch_input = torch.cat(
                [
                    pos_target_ctx,
                    neg_target_ctx,
                    pos_target_ctx - neg_target_ctx,
                    item_context,
                ],
                dim=-1,
            )
            logits = logits + self.target_output_head(target_branch_input)

        source_logits = None
        if source_output is not None:
            source_logits = self.clsfier(source_output)
        return {
            'logits': logits,
            'embedding': output,
            'source_logits': source_logits,
            'current_ins_emb': current_ins_emb,
        }

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """Runs the forward pass of the PCVRHyFormer model."""
        outputs = self._forward_impl(inputs, apply_dropout=self.training)
        assert outputs['logits'] is not None
        return outputs['logits']

    def forward_with_aux(self, inputs: ModelInput) -> Dict[str, Optional[torch.Tensor]]:
        """Runs the forward pass and returns IAT auxiliary outputs when enabled."""
        return self._forward_impl(inputs, apply_dropout=self.training)

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs inference without dropout, returning both logits and embeddings."""
        outputs = self._forward_impl(inputs, apply_dropout=False)
        assert outputs['logits'] is not None
        assert outputs['embedding'] is not None
        return outputs['logits'], outputs['embedding']

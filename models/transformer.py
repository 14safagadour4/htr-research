"""
Week 4 — Reference implementation of the four components that turn
attention into a complete Transformer:

    PositionalEncoding   — order
    TransformerBlock     — self-attention + (cross-attention) + FFN,
                           wrapped in residuals and pre-norm LayerNorm
    MiniTransformer      — full encoder-decoder wiring

This file is the reference cited in the Week 4 report (Section "How
positional encoding, residuals, LayerNorm, and FFN form a complete
Transformer"). It is also imported by experiments/ablation.py.

Design notes :
* Pre-norm: LayerNorm sits *inside* the residual branch, so the residual
  stream itself is never normalized. This is what makes deep stacks train
  stably without LR warmup.
* Expansion ratio d_ff = 4 * d_model is the standard from Vaswani et al.
* Cross-attention appears only in the decoder; Q comes from the decoder,
  K and V from the encoder output.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F



# Attention (kept here so the file is importable on its own; can be
# replaced with `from .attention import MultiHeadAttention`)
# ─────────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """Multi-head self/cross-attention.

    Accepts query [B, T_q, D], key/value [B, T_k, D].
    Returns output [B, T_q, D] and attention weights [B, H, T_q, T_k].

    Mask convention: `mask` is a boolean tensor with True = *masked*
    (position is hidden). It is applied with masked_fill(-inf) before
    softmax.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, T_q, _ = query.shape
        T_k = key.shape[1]
        H, d_k = self.n_heads, self.d_k

        # Split into heads: [B, T, D] -> [B, H, T, d_k]
        Q = self.W_q(query).view(B, T_q, H, d_k).transpose(1, 2)
        K = self.W_k(key).view(B, T_k, H, d_k).transpose(1, 2)
        V = self.W_v(value).view(B, T_k, H, d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)                         # [B, H, T_q, d_k]
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.W_o(out), attn


# ─────────────────────────────────────────────────────────────────────────
# Component 1 — positional encoding
# ─────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017, §3.5).

    Added to the token embedding *before* the first attention layer so
    that Q, K, V already carry position information. Without this,
    attention is permutation-equivariant and the model cannot tell
    [3, 7] from [7, 3].
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))         # [1, max_len, D]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return self.dropout(x + self.pe[:, : x.size(1)])


# ─────────────────────────────────────────────────────────────────────────
# Component 2–4 — block: attention + FFN, with residuals and LayerNorm
# ─────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """One encoder or decoder block.

    Encoder block:   self-attention -> FFN
    Decoder block:   masked self-attention -> cross-attention -> FFN

    Every sub-layer is wrapped as  x = x + dropout(sublayer(norm(x))).
    Pre-norm keeps the residual stream clean; the FFN is the only
    per-token non-linearity and is what makes depth useful.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
    ):
        super().__init__()
        self.is_decoder = is_decoder

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        if is_decoder:
            self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if is_decoder:
            self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor | None = None,
        self_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Sub-layer 1: self-attention (encoder or decoder)
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, mask=self_mask)
        x = x + self.dropout(attn_out)

        # Sub-layer 2: cross-attention (decoder only)
        if self.is_decoder and encoder_out is not None:
            h = self.norm2(x)
            cross_out, _ = self.cross_attn(
                h, encoder_out, encoder_out, mask=cross_mask
            )
            x = x + self.dropout(cross_out)

        # Sub-layer 3: feed-forward
        h = self.norm3(x) if self.is_decoder else self.norm2(x)
        ffn_out = self.ffn(h)
        x = x + self.dropout(ffn_out)
        return x


# ─────────────────────────────────────────────────────────────────────────
# Full wiring — shows how the components collaborate
# ─────────────────────────────────────────────────────────────────────────

class MiniTransformer(nn.Module):
    """Full encoder-decoder Transformer.

    Forward pass (the entire design in one sequence):
        embed -> add position -> encode
        embed -> add position -> decode (self-attn, cross-attn, FFN)
        -> project to vocab logits
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        max_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = TransformerBlock(
            d_model, n_heads, d_ff, dropout, is_decoder=False
        )
        self.decoder = TransformerBlock(
            d_model, n_heads, d_ff, dropout, is_decoder=True
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src_ids: torch.Tensor,          # [B, T_src]
        tgt_ids: torch.Tensor,          # [B, T_tgt]
        src_pad_mask: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Encoder path
        src = self.pos_enc(self.tok_emb(src_ids))
        enc = self.encoder(src, self_mask=src_pad_mask)

        # Decoder path
        tgt = self.pos_enc(self.tok_emb(tgt_ids))
        dec = self.decoder(
            tgt,
            encoder_out=enc,
            self_mask=causal_mask,
            cross_mask=src_pad_mask,
        )

        # Output projection: decoder hidden -> vocab logits
        return self.lm_head(dec)        # [B, T_tgt, V]
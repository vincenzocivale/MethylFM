"""Learned attention pooling over the latent token set.

Produces the canonical sample-level representation h_s from the latent
tokens Z_s. This deliberately does NOT reuse an arbitrary first latent token
as a CLS token: instead a single learned query attends over the full latent
set, so every latent contributes to h_s.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class AttentionPool(nn.Module):
    """Single learned-query attention pooling: Z_s [B, M, d] -> h_s [B, d]."""

    def __init__(self, model_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.query = nn.Parameter(torch.randn(1, model_dim) * (model_dim**-0.5))

        self.ln_q = nn.LayerNorm(model_dim)
        self.ln_kv = nn.LayerNorm(model_dim)
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.ln_out = nn.LayerNorm(model_dim)

    def _split_heads(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        b, h, n, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, n, h * dh)

    def forward(self, latents: Tensor) -> Tensor:
        """
        Args:
            latents: [B, M, d] latent tokens (dense, no padding/mask needed).

        Returns:
            [B, d] pooled sample representation h_s.
        """
        if latents.dim() != 3 or latents.shape[-1] != self.model_dim:
            raise ValueError(f"expected latents of shape [B, M, {self.model_dim}], got {tuple(latents.shape)}")

        b = latents.shape[0]
        q_in = self.ln_q(self.query).unsqueeze(0).expand(b, -1, -1)  # [B, 1, d]
        q = self._split_heads(self.q_proj(q_in))  # [B, H, 1, Dh]

        kv_in = self.ln_kv(latents)
        k = self._split_heads(self.k_proj(kv_in))
        v = self._split_heads(self.v_proj(latents))

        attn_out = F.scaled_dot_product_attention(q, k, v)  # [B, H, 1, Dh]
        merged = self._merge_heads(attn_out)  # [B, 1, d]
        out = self.out_proj(merged)
        out = q_in + out
        out = self.ln_out(out)
        return out.squeeze(1)

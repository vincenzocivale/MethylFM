"""mHC latent Transformer backbone (spec section 5).

Processes only the compressed latent tokens produced by the set-to-latent
cross-attention bridge. Residual mixing between layers follows the
manifold-constrained hyper-connections mechanism in `mhc.py`, not ordinary
residual addition.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from methylfm.model.mhc import MHCLayer, collapse_streams, init_streams


class SelfAttentionLayerFn(nn.Module):
    """Pre-LN full self-attention sublayer, output only (no internal residual).

    The mHC wrapper (`MHCLayer`) owns all residual mixing; this module must
    return the sublayer's output, not `x + attn(x)`.
    """

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, M, d] -> [B, M, d]."""
        x = self.norm(x)
        out, _ = self.attn(x, x, x, need_weights=False)
        return out


class MLPLayerFn(nn.Module):
    """Pre-LN MLP sublayer, output only (no internal residual)."""

    def __init__(self, model_dim: int, mlp_ratio: int):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        hidden = model_dim * mlp_ratio
        self.fc1 = nn.Linear(model_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, model_dim)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, M, d] -> [B, M, d]."""
        x = self.norm(x)
        return self.fc2(self.act(self.fc1(x)))


class MHCTransformerBlock(nn.Module):
    """One Transformer block: mHC-wrapped self-attention followed by mHC-wrapped MLP."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        mlp_ratio: int,
        num_streams: int,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.attn_mhc = MHCLayer(
            model_dim,
            num_streams,
            SelfAttentionLayerFn(model_dim, num_heads),
            sinkhorn_iters=sinkhorn_iters,
        )
        self.mlp_mhc = MHCLayer(
            model_dim,
            num_streams,
            MLPLayerFn(model_dim, mlp_ratio),
            sinkhorn_iters=sinkhorn_iters,
        )

    def forward(self, x_l: Tensor) -> Tensor:
        """x_l: [B, n, M, d] -> [B, n, M, d]."""
        x_l = self.attn_mhc(x_l)
        x_l = self.mlp_mhc(x_l)
        return x_l


class MHCLatentBackbone(nn.Module):
    """Stack of MHCTransformerBlocks operating on the compressed latent set.

    Takes a single-stream latent tensor [B, num_latents, model_dim], expands
    it into `mhc_expansion_rate` hyper-connection streams, processes it
    through `num_layers` mHC Transformer blocks, then collapses back to a
    single-stream tensor for downstream consumers (attention pooling).
    """

    def __init__(
        self,
        model_dim: int = 512,
        num_latents: int = 128,
        num_layers: int = 16,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        mhc_expansion_rate: int = 4,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_latents = num_latents
        self.mhc_expansion_rate = mhc_expansion_rate
        self.blocks = nn.ModuleList(
            [
                MHCTransformerBlock(
                    model_dim, num_heads, mlp_ratio, mhc_expansion_rate, sinkhorn_iters
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, latents: Tensor) -> Tensor:
        """latents: [B, num_latents, model_dim] -> [B, num_latents, model_dim]."""
        x_l = init_streams(latents, self.mhc_expansion_rate)
        for block in self.blocks:
            x_l = block(x_l)
        return collapse_streams(x_l)

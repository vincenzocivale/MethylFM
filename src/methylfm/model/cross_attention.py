"""Set-to-latent cross-attention bridge.

Compresses an arbitrary, variable-size, unordered set of CpG tokens into a
fixed-size array of learned latent tokens via cross-attention:

    Z_1 = CrossAttention(Q=Z_0, K=T_s, V=T_s)

where Z_0 is a learned latent array and T_s is the observed sample's CpG
token set. This module performs no genomic region pooling and no binning of
CpGs: the key/value side is the raw, variable-length observed set.

To support large CpG sets (e.g. full EPIC arrays with ~850k probes) without
materializing a dense [B, H, M, N] attention score tensor, the key/value side
can optionally be processed in chunks along N using a numerically stable
online-softmax (flash-attention-style) accumulation. This trades a small
amount of extra arithmetic for bounded peak memory that no longer scales
with N.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_NEG_INF = -1e9


class SetToLatentCrossAttention(nn.Module):
    """Cross-attention from a fixed learned latent array onto a CpG token set."""

    def __init__(
        self,
        model_dim: int,
        num_latents: int,
        num_heads: int = 8,
        chunk_size: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.model_dim = model_dim
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.chunk_size = chunk_size
        self.dropout = dropout

        self.latents = nn.Parameter(torch.randn(num_latents, model_dim) * (model_dim**-0.5))

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

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            tokens: [B, N, d] CpG tokens for the batch (may include padding).
            mask: [B, N] bool, True where the token is a real observed CpG.

        Returns:
            [B, M, d] latent tokens, M = num_latents.
        """
        if tokens.dim() != 3 or tokens.shape[-1] != self.model_dim:
            raise ValueError(f"expected tokens of shape [B, N, {self.model_dim}], got {tuple(tokens.shape)}")
        if mask.shape != tokens.shape[:2]:
            raise ValueError(f"mask shape {tuple(mask.shape)} must match tokens[:, :, 0] shape {tuple(tokens.shape[:2])}")

        b, n, _ = tokens.shape
        q_in = self.ln_q(self.latents).unsqueeze(0).expand(b, -1, -1)
        q = self._split_heads(self.q_proj(q_in))  # [B, H, M, Dh]

        kv_in = self.ln_kv(tokens)
        k = self._split_heads(self.k_proj(kv_in))  # [B, H, N, Dh]
        v = self._split_heads(self.v_proj(tokens))  # [B, H, N, Dh]

        use_chunking = self.chunk_size is not None and self.chunk_size < n
        if not use_chunking:
            attn_mask = torch.where(
                mask[:, None, None, :],
                torch.zeros((), device=tokens.device, dtype=q.dtype),
                torch.full((), _NEG_INF, device=tokens.device, dtype=q.dtype),
            )
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            attn_out = self._chunked_attention(q, k, v, mask)

        merged = self._merge_heads(attn_out)  # [B, M, d]
        out = self.out_proj(merged)
        out = q_in + out
        return self.ln_out(out)

    def _chunked_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        b, h, m, dh = q.shape
        n = k.shape[2]
        scale = 1.0 / math.sqrt(dh)
        chunk = self.chunk_size

        running_max = torch.full((b, h, m), _NEG_INF, device=q.device, dtype=q.dtype)
        running_sum = torch.zeros((b, h, m), device=q.device, dtype=q.dtype)
        acc = torch.zeros((b, h, m, dh), device=q.device, dtype=q.dtype)

        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            k_c = k[:, :, start:end, :]
            v_c = v[:, :, start:end, :]
            mask_c = mask[:, start:end]  # [B, chunk]

            scores_c = torch.einsum("bhmd,bhnd->bhmn", q, k_c) * scale  # [B,H,M,chunk]
            neg_inf = torch.full((), _NEG_INF, device=scores_c.device, dtype=scores_c.dtype)
            scores_c = torch.where(mask_c[:, None, None, :], scores_c, neg_inf)

            chunk_max = scores_c.max(dim=-1).values  # [B,H,M]
            new_max = torch.maximum(running_max, chunk_max)

            correction = torch.exp(running_max - new_max)
            p_c = torch.exp(scores_c - new_max.unsqueeze(-1))  # [B,H,M,chunk]

            running_sum = running_sum * correction + p_c.sum(dim=-1)
            acc = acc * correction.unsqueeze(-1) + torch.einsum("bhmn,bhnd->bhmd", p_c, v_c)
            running_max = new_max

        out = acc / running_sum.clamp_min(1e-9).unsqueeze(-1)
        return out

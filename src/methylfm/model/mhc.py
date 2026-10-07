"""Manifold-Constrained Hyper-Connections (mHC).

Faithful implementation of mHC (arXiv:2512.24880), extending the base
Hyper-Connections formulation (arXiv:2409.19606). The residual state is
expanded into ``n`` parallel streams. Each layer computes three dynamic,
input-conditioned gates per token position:

- ``H_pre``  [B, M, n]    : aggregates the n streams into one vector fed to
                            the wrapped layer function.
- ``H_post`` [B, M, n]    : re-expands the layer's output back across the
                            n streams.
- ``H_res``  [B, M, n, n] : mixes the residual streams themselves. This
                            matrix is projected onto the Birkhoff polytope
                            (doubly stochastic matrices) via Sinkhorn-Knopp
                            iteration, which is what keeps deep stacks of
                            these layers numerically stable (a product of
                            doubly-stochastic matrices stays non-expansive).

All three gates are produced dynamically as ``alpha * tanh(theta(x)) + bias``
from RMSNorm'd, stream-flattened input, matching the paper's dynamic gating
parameterization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    """RMSNorm over the last dimension, no learnable scale.

    Used only to condition the hyper-connection gates, not as a
    general-purpose normalization layer.
    """
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def sinkhorn_doubly_stochastic(logits: Tensor, num_iters: int = 20) -> Tensor:
    """Project logits onto the Birkhoff polytope (doubly stochastic matrices).

    Operates on the last two dimensions of ``logits`` (assumed square, n x n);
    any leading dimensions are treated as batch dimensions.

    Args:
        logits: [..., n, n] unconstrained residual-mixing logits.
        num_iters: number of alternating row/column normalization sweeps.

    Returns:
        [..., n, n] tensor whose rows and columns each sum to ~1 and whose
        entries are non-negative.
    """
    # Subtract the row-max before exponentiating: a numerical safeguard not
    # specified in the paper, added here to prevent overflow for large logits.
    stabilized = logits - logits.max(dim=-1, keepdim=True).values
    mat = torch.exp(stabilized)
    for _ in range(num_iters):
        mat = mat / mat.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        mat = mat / mat.sum(dim=-2, keepdim=True).clamp_min(1e-9)
    return mat


class HyperConnectionGate(nn.Module):
    """Computes the dynamic H_pre, H_post, H_res gates for one mHC layer."""

    def __init__(self, model_dim: int, num_streams: int, sinkhorn_iters: int = 20):
        super().__init__()
        self.model_dim = model_dim
        self.num_streams = num_streams
        self.sinkhorn_iters = sinkhorn_iters

        in_dim = num_streams * model_dim
        self.pre_proj = nn.Linear(in_dim, num_streams)
        self.post_proj = nn.Linear(in_dim, num_streams)
        self.res_proj = nn.Linear(in_dim, num_streams * num_streams)

        self.pre_alpha = nn.Parameter(torch.tensor(0.01))
        self.post_alpha = nn.Parameter(torch.tensor(0.01))
        self.res_alpha = nn.Parameter(torch.tensor(0.01))

        self.pre_bias = nn.Parameter(torch.zeros(num_streams))
        self.post_bias = nn.Parameter(torch.zeros(num_streams))
        self.res_bias = nn.Parameter(torch.zeros(num_streams, num_streams))

    def forward(self, x_l: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """x_l: [B, n, M, d] -> (H_pre [B,M,n], H_post [B,M,n], H_res [B,M,n,n])."""
        b, n, m, d = x_l.shape
        x_norm = rms_norm(x_l)
        # "b n m d -> b m (n d)"
        x_flat = x_norm.permute(0, 2, 1, 3).reshape(b, m, n * d)

        pre_logit = self.pre_proj(x_flat)
        post_logit = self.post_proj(x_flat)
        res_logit = self.res_proj(x_flat).reshape(b, m, n, n)

        pre_gated = self.pre_alpha * torch.tanh(pre_logit) + self.pre_bias
        post_gated = self.post_alpha * torch.tanh(post_logit) + self.post_bias
        res_gated = self.res_alpha * torch.tanh(res_logit) + self.res_bias

        h_pre = torch.sigmoid(pre_gated)
        h_post = 2.0 * torch.sigmoid(post_gated)
        h_res = sinkhorn_doubly_stochastic(res_gated, num_iters=self.sinkhorn_iters)

        return h_pre, h_post, h_res


class MHCLayer(nn.Module):
    """Wraps an arbitrary layer function with manifold-constrained hyper-connections."""

    def __init__(
        self,
        model_dim: int,
        num_streams: int,
        layer_fn: nn.Module,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_streams = num_streams
        self.layer_fn = layer_fn
        self.gate = HyperConnectionGate(model_dim, num_streams, sinkhorn_iters)

    def forward(self, x_l: Tensor) -> Tensor:
        """x_l: [B, n, M, d] -> [B, n, M, d]."""
        h_pre, h_post, h_res = self.gate(x_l)

        # Aggregate streams -> single vector per token.
        agg = torch.einsum("bmk,bkmd->bmd", h_pre, x_l)

        layer_out = self.layer_fn(agg)

        # Re-expand the layer output across streams.
        expanded = torch.einsum("bmk,bmd->bkmd", h_post, layer_out)

        # Mix residual streams: H_res[b,m,k,j] is the weight from source
        # stream j to destination stream k (row k, column j convention,
        # matching H_res @ x_l with x_l's stream axis as the contracted one).
        mixed = torch.einsum("bmkj,bjmd->bkmd", h_res, x_l)

        return mixed + expanded


def init_streams(x0: Tensor, num_streams: int) -> Tensor:
    """Replicate a single-stream state [B, M, d] into [B, n, M, d]."""
    return x0.unsqueeze(1).expand(-1, num_streams, -1, -1).clone()


def collapse_streams(x_l: Tensor) -> Tensor:
    """Final readout: collapse [B, n, M, d] -> [B, M, d] by averaging streams.

    This is distinct from the per-layer H_pre aggregation; it is only used
    once, after the final mHC layer, to produce a single-stream tensor for
    downstream (non-mHC) consumers.
    """
    return x_l.mean(dim=1)

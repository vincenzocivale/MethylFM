"""FT-Transformer-style CpG/beta tokenizer.

Encodes a sample-specific CpG token from a locus representation `l_i` and a
scalar beta value in [0, 1]: `t = LayerNorm(f_b(l_i) + beta * f_w(l_i))`.
This is the only tokenizer in the initial production model: no beta
binning, no piecewise-linear encoding, no Fourier encoding, no beta MLP, no
CpG-specific z-scoring, no quantile normalization, and no M-value
conversion.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FTCpGTokenizer(nn.Module):
    """Builds a CpG token from a locus representation and a raw beta value.

        b_i = f_b(l_i)
        w_i = f_w(l_i)
        t_{s,i} = LayerNorm(b_i + beta_{s,i} * w_i)
    """

    def __init__(self, locus_dim: int, model_dim: int):
        super().__init__()
        self.locus_dim = locus_dim
        self.model_dim = model_dim
        self.f_b = nn.Linear(locus_dim, model_dim)
        self.f_w = nn.Linear(locus_dim, model_dim)
        self.ln = nn.LayerNorm(model_dim)

    def forward(self, locus_repr: Tensor, beta: Tensor) -> Tensor:
        """
        Args:
            locus_repr: [..., locus_dim]
            beta: [...] scalar beta values in [0, 1], same leading shape as
                `locus_repr` minus its last dimension.

        Returns:
            CpG tokens: [..., model_dim]
        """
        nan_count = torch.isnan(beta).sum().item()
        if nan_count or (beta < 0).any() or (beta > 1).any():
            finite = beta[torch.isfinite(beta)]
            min_val = finite.min().item() if finite.numel() else float("nan")
            max_val = finite.max().item() if finite.numel() else float("nan")
            raise ValueError(
                "beta values must lie in [0, 1] with no NaNs; got "
                f"min={min_val}, max={max_val}, nan_count={nan_count}"
            )

        b = self.f_b(locus_repr)
        w = self.f_w(locus_repr)
        t = self.ln(b + beta.unsqueeze(-1) * w)
        return t

"""Shared validation helpers for tensors at module/data boundaries."""

import torch


def validate_beta_range(beta: torch.Tensor, name: str = "beta_values") -> None:
    """Raise loudly if any value falls outside [0, 1]. NaNs are considered
    invalid here -- callers that intentionally carry NaN for missing loci
    must filter them out before calling this (see data.sample), since the
    model must never silently treat missing CpGs as zero."""
    if torch.isnan(beta).any():
        raise ValueError(f"{name} contains NaN values; missing CpGs must be removed, not passed as NaN")
    if beta.numel() > 0 and (beta.min() < 0.0 or beta.max() > 1.0):
        raise ValueError(
            f"{name} must lie in [0, 1], got range "
            f"[{beta.min().item():.6f}, {beta.max().item():.6f}]"
        )


def validate_shape(tensor: torch.Tensor, expected_shape: tuple, name: str = "tensor") -> None:
    """expected_shape entries may be None to accept any size in that dim."""
    if tensor.dim() != len(expected_shape):
        raise ValueError(
            f"{name} expected {len(expected_shape)}D tensor, got {tensor.dim()}D with shape {tuple(tensor.shape)}"
        )
    for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
        if expected is not None and actual != expected:
            raise ValueError(
                f"{name} dim {i} expected size {expected}, got {actual} (full shape {tuple(tensor.shape)})"
            )

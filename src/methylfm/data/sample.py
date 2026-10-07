"""Normalized internal representation of a single methylation sample."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MethylationSample:
    sample_id: str
    cpg_ids: torch.Tensor  # [N], int64
    beta_values: torch.Tensor  # [N], float32
    dataset_id: str
    platform: str | None


def validate_sample(sample: MethylationSample) -> MethylationSample:
    """Drop NaN/missing loci and enforce beta in [0, 1]. Returns a new sample; never mutates input."""
    cpg_ids = sample.cpg_ids
    beta_values = sample.beta_values

    if cpg_ids.ndim != 1 or beta_values.ndim != 1 or cpg_ids.shape != beta_values.shape:
        raise ValueError(
            "cpg_ids and beta_values must be 1-D tensors of equal shape, got "
            f"cpg_ids.shape={tuple(cpg_ids.shape)} beta_values.shape={tuple(beta_values.shape)}"
        )

    keep_mask = ~torch.isnan(beta_values)
    filtered_cpg_ids = cpg_ids[keep_mask]
    filtered_beta_values = beta_values[keep_mask]

    if filtered_beta_values.numel() > 0:
        beta_min = filtered_beta_values.min().item()
        beta_max = filtered_beta_values.max().item()
        if beta_min < 0.0 or beta_max > 1.0:
            raise ValueError(
                f"beta values out of [0, 1] range for sample {sample.sample_id!r}: "
                f"min={beta_min}, max={beta_max}"
            )

    return MethylationSample(
        sample_id=sample.sample_id,
        cpg_ids=filtered_cpg_ids,
        beta_values=filtered_beta_values,
        dataset_id=sample.dataset_id,
        platform=sample.platform,
    )

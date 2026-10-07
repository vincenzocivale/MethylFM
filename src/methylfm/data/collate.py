"""Collator for variable-length CpG sets.

A batch is defined by samples/patients, never by CpGs: each sample keeps only
its own observed CpGs, padded within the minibatch (never to a global CpG
universe) with an accompanying boolean mask.

For multi-view training, this function collates a single flat list of
same-view samples (e.g. all `reference_view` samples, or all `broad_view`
samples, or a single `sparse_views[k]` slot across the sample batch). The
caller is responsible for invoking it once per view slot.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from methylfm.data.sample import MethylationSample


@dataclass
class CollatedBatch:
    cpg_ids: torch.Tensor  # [B, N_max], int64
    beta_values: torch.Tensor  # [B, N_max], float32
    mask: torch.Tensor  # [B, N_max], bool
    sample_ids: list[str]
    dataset_ids: list[str]
    platforms: list[str | None]
    num_cpgs: torch.Tensor  # [B], int64


def collate_methylation_batch(samples: list[MethylationSample]) -> CollatedBatch:
    batch_size = len(samples)
    n_max = max(s.cpg_ids.shape[0] for s in samples)

    cpg_ids = torch.zeros((batch_size, n_max), dtype=torch.int64)
    beta_values = torch.zeros((batch_size, n_max), dtype=torch.float32)
    mask = torch.zeros((batch_size, n_max), dtype=torch.bool)
    num_cpgs = torch.zeros((batch_size,), dtype=torch.int64)

    for i, sample in enumerate(samples):
        n = sample.cpg_ids.shape[0]
        cpg_ids[i, :n] = sample.cpg_ids
        beta_values[i, :n] = sample.beta_values
        mask[i, :n] = True
        num_cpgs[i] = n

    return CollatedBatch(
        cpg_ids=cpg_ids,
        beta_values=beta_values,
        mask=mask,
        sample_ids=[s.sample_id for s in samples],
        dataset_ids=[s.dataset_id for s in samples],
        platforms=[s.platform for s in samples],
        num_cpgs=num_cpgs,
    )

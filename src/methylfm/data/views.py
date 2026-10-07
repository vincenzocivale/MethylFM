"""Initial conservative view-construction policy (spec section 8).

This module isolates view construction from the model so the policy can later be
replaced by platform-aware or biologically stratified augmentations. Views are
unordered random subsets of the *observed* CpG index set for a sample - never
contiguous genomic crops, never beta perturbation/jitter.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from methylfm.data.sample import MethylationSample


@dataclass
class ViewConfig:
    global_keep_fraction: float = 0.8
    num_local_views: int = 4
    local_keep_fraction: float = 0.25


def _random_subset(sample: MethylationSample, keep_fraction: float, generator: torch.Generator) -> MethylationSample:
    """Select a random subset of observed CpGs via a random permutation of indices.

    Using `torch.randperm(N, generator=generator)` (rather than a Bernoulli mask)
    gives an exact, reproducible subset size for a given N and keep_fraction, and
    consumes the generator's state deterministically so repeated calls with
    identically-seeded generators reproduce identical views.
    """
    n = sample.cpg_ids.shape[0]
    k = round(n * keep_fraction)
    perm = torch.randperm(n, generator=generator)
    idx = perm[:k]
    return MethylationSample(
        sample_id=sample.sample_id,
        cpg_ids=sample.cpg_ids[idx].clone(),
        beta_values=sample.beta_values[idx].clone(),
        dataset_id=sample.dataset_id,
        platform=sample.platform,
    )


class ViewBuilder:
    def __init__(self, config: ViewConfig):
        self.config = config

    def build_views(self, sample: MethylationSample, generator: torch.Generator) -> dict:
        reference_view = replace(
            sample,
            cpg_ids=sample.cpg_ids.clone(),
            beta_values=sample.beta_values.clone(),
        )

        broad_view = _random_subset(sample, self.config.global_keep_fraction, generator)

        sparse_views = [
            _random_subset(sample, self.config.local_keep_fraction, generator)
            for _ in range(self.config.num_local_views)
        ]

        return {
            "reference_view": reference_view,
            "broad_view": broad_view,
            "sparse_views": sparse_views,
        }

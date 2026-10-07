"""Combined LeJEPA pretraining objective.

This implements:

    L = L_inv + lambda * L_SIGReg

Mapping onto the paper's (Balestriero & LeCun, arXiv:2511.08544) raw Eq. 9:

    L_paper = (lambda / V) * sum_v SIGReg({z_n,v}) + (1/B) * sum_n L_pred(...)

`sigreg_loss` (losses/sigreg.py) is computed independently per view and here
averaged over views -- that mean over V views already implements the
paper's `(1/V) * sum_v` factor. `invariance_loss` (losses/invariance.py)
already averages over the batch B -- that implements the paper's `(1/B) *
sum_n` factor. Therefore:

    sigreg_lambda * sigreg_term  ==  (lambda / V) * sum_v SIGReg({z_n,v})
    invariance_term              ==  (1/B) * sum_n L_pred(...)

and `total = invariance_term + sigreg_lambda * sigreg_term` is numerically
equivalent to the paper's Eq. 9 -- just factored into separately-named,
separately-testable terms rather than one fused expression.

Note: the paper's own recommended default is lambda=0.05; MethylFM's
repo-wide default (configs/model/methylfm.yaml) is the more conservative
0.02. This is a deliberate, documented deviation, not an oversight.
"""

from typing import Callable

import torch

from methylfm.pretraining.invariance import invariance_loss
from methylfm.pretraining.sigreg import sigreg_loss


def lejepa_loss(
    z_views: list[torch.Tensor],
    global_view_indices: list[int],
    sigreg_lambda: float = 0.02,
    num_slices: int = 1024,
    num_points: int = 17,
    generator: torch.Generator | None = None,
    gather_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Args:
        z_views: list of V tensors, each [B, d] -- projector outputs per view
            (the LOCAL batch, i.e. this process's shard under DDP/Accelerate).
        global_view_indices: indices into z_views used to compute the
            global-view mean mu_n for the invariance loss.
        sigreg_lambda: weight on the SIGReg term.
        num_slices, num_points: SIGReg hyperparameters.
        generator: optional torch.Generator for reproducible SIGReg slicing.
        gather_fn: optional differentiable all-gather, applied to each view's
            z BEFORE computing SIGReg. SIGReg's Epps-Pulley statistic is a
            nonlinear function of the empirical sample distribution, so
            averaging per-process-local SIGReg gradients under DDP is NOT
            equivalent to computing SIGReg over the true global batch --
            unlike `invariance_loss`, whose per-sample term depends only on
            that sample's own views and is therefore exactly correct to
            compute locally-per-process (DDP's gradient averaging already
            reproduces the global-batch gradient for it). Pass a
            differentiable all-gather (e.g. `torch.distributed.nn.functional
            .all_gather`-based) here under multi-process training; leave
            `None` for single-process runs, where it would be a no-op anyway.

    Returns:
        dict with keys "total", "invariance", "sigreg" (each a scalar tensor).
    """
    sigreg_views = [gather_fn(z) if gather_fn is not None else z for z in z_views]
    sigreg_term = torch.stack(
        [
            sigreg_loss(sigreg_views[v], num_slices=num_slices, num_points=num_points, generator=generator)
            for v in range(len(z_views))
        ]
    ).mean()
    invariance_term = invariance_loss(z_views, global_view_indices)
    total = invariance_term + sigreg_lambda * sigreg_term
    return {"total": total, "invariance": invariance_term, "sigreg": sigreg_term}

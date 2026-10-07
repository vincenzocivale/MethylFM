"""LeJEPA invariance / prediction loss (paper Eq. 8).

Paper (Balestriero & LeCun, arXiv:2511.08544), Eq. 8:

    L_pred = (1/V) * sum_{v'=1}^{V} || mu_n - z_{n,v'} ||_2^2

where mu_n is the mean of the GLOBAL views' embeddings for sample n
(DINO-style global/local asymmetry -- local/sparse views are pulled
toward the global-view mean, but do not themselves contribute to mu_n),
and V is the total number of views (global + local).

Eq. 8 is stated per-sample n. To obtain a scalar training loss we
additionally average over the batch dimension B; this batch-mean is a
MethylFM implementation detail layered on top of Eq. 8, not part of the
paper's notation itself.

In MethylFM's view set, `global_view_indices` should point at the
`reference_view` and `broad_view` slots; `sparse_views` are local views
and must NOT be included in `global_view_indices`.
"""

import torch


def invariance_loss(
    z_views: list[torch.Tensor],
    global_view_indices: list[int],
) -> torch.Tensor:
    """
    Args:
        z_views: list of V tensors, each [B, d] -- projector outputs per view.
        global_view_indices: indices into z_views identifying the global views
            used to compute mu_n. Must be non-empty.

    Returns:
        Scalar tensor: mean over views of mean-over-batch squared L2 distance
        to the global-view mean mu_n.
    """
    if len(z_views) == 0:
        raise ValueError("z_views must contain at least one view")
    if len(global_view_indices) == 0:
        raise ValueError("global_view_indices must be non-empty")
    v_total = len(z_views)
    for idx in global_view_indices:
        if not (0 <= idx < v_total):
            raise ValueError(
                f"global_view_indices entry {idx} out of range for {v_total} views"
            )

    mu = torch.stack([z_views[i] for i in global_view_indices], dim=0).mean(dim=0)  # [B, d]

    per_view_sq = [
        ((mu - z_views[v]) ** 2).sum(dim=-1)  # [B]
        for v in range(v_total)
    ]
    loss = torch.stack(per_view_sq, dim=0).mean(dim=0).mean()  # mean over V, then over B
    return loss

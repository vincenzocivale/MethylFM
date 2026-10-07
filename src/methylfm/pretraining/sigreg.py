"""SIGReg: sliced Epps-Pulley Gaussianity regularizer (paper Section 4.2).

Paper (Balestriero & LeCun, arXiv:2511.08544), Definition 2:

    SIGReg_T(A, {f_theta(x_n)}) = (1/|A|) * sum_{a in A} T({a^T f_theta(x_n)})

where T is the Epps-Pulley characteristic-function goodness-of-fit
statistic against a standard normal target, and A is a set of random
slicing directions.

Epps-Pulley statistic (paper Section 4.2.3), quoted verbatim pseudocode:

    t = linspace(-5, 5, 17)
    exp_f = exp(-0.5 * t**2)              # target standard-normal CF, also
                                           # used as the weight w(t); these are
                                           # textually separate quantities in
                                           # the paper (phi(t) and w(t)) that
                                           # happen to coincide numerically
                                           # only because sigma=1 -- kept as
                                           # two named concepts in code/docs.
    x_t = x.unsqueeze(2) * t
    ecf = (1j * x_t).exp().mean(0)        # empirical characteristic function
    err = (ecf - exp_f).abs().square().mul(exp_f)
    T = torch.trapz(err, t, dim=1) * N

The paper's quoted snippet does not mandate unit-norm slicing directions;
unit-norm is the standard sliced-projection convention and is used here
as a deliberate, documented choice (not a silent guess).
"""

import torch


def epps_pulley_statistic(
    x_proj: torch.Tensor,
    num_points: int = 17,
    t_min: float = -5.0,
    t_max: float = 5.0,
) -> torch.Tensor:
    """
    Args:
        x_proj: [N] real 1D projected samples.
        num_points: number of integration grid points.
        t_min, t_max: integration domain bounds.

    Returns:
        Scalar tensor: the Epps-Pulley statistic T for this 1D sample.
    """
    if x_proj.dim() != 1:
        raise ValueError(f"x_proj must be 1D, got shape {tuple(x_proj.shape)}")
    n = x_proj.shape[0]
    t = torch.linspace(t_min, t_max, num_points, device=x_proj.device, dtype=x_proj.dtype)
    exp_f = torch.exp(-0.5 * t**2)  # [T], target CF phi(t) == weight w(t) at sigma=1

    complex_dtype = torch.complex64 if x_proj.dtype == torch.float32 else torch.complex128
    x_t = x_proj.unsqueeze(-1) * t  # [N, T]
    ecf = torch.exp(1j * x_t.to(complex_dtype)).mean(dim=0)  # [T] complex empirical CF

    err = (ecf - exp_f.to(complex_dtype)).abs().square() * exp_f  # [T], real
    stat = torch.trapz(err, t) * n
    return stat


def _epps_pulley_statistic_batched(
    x_proj: torch.Tensor,
    num_points: int = 17,
    t_min: float = -5.0,
    t_max: float = 5.0,
) -> torch.Tensor:
    """Vectorized Epps-Pulley statistic over a slice dimension.

    Args:
        x_proj: [N, S] real samples, S independent slices.

    Returns:
        [S] tensor of per-slice Epps-Pulley statistics.
    """
    if x_proj.dim() != 2:
        raise ValueError(f"x_proj must be 2D [N, S], got shape {tuple(x_proj.shape)}")
    n = x_proj.shape[0]
    t = torch.linspace(t_min, t_max, num_points, device=x_proj.device, dtype=x_proj.dtype)
    exp_f = torch.exp(-0.5 * t**2)  # [T]

    complex_dtype = torch.complex64 if x_proj.dtype == torch.float32 else torch.complex128
    x_t = x_proj.unsqueeze(-1) * t  # [N, S, T]
    ecf = torch.exp(1j * x_t.to(complex_dtype)).mean(dim=0)  # [S, T] complex

    err = (ecf - exp_f.to(complex_dtype)).abs().square() * exp_f  # [S, T]
    stat = torch.trapz(err, t, dim=-1) * n  # [S]
    return stat


def sigreg_loss(
    z: torch.Tensor,
    num_slices: int = 1024,
    num_points: int = 17,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Args:
        z: [N, d] embeddings for a single view across the batch.
        num_slices: number of random unit-norm slicing directions |A|.
        num_points: Epps-Pulley integration grid size.
        generator: optional torch.Generator for reproducible slice sampling.

    Returns:
        Scalar tensor: (1/|A|) * sum_a T(a^T z), i.e. SIGReg_T(A, z).
    """
    if z.dim() != 2:
        raise ValueError(f"z must be 2D [N, d], got shape {tuple(z.shape)}")
    n, d = z.shape

    if generator is not None:
        a = torch.randn(num_slices, d, device=z.device, dtype=z.dtype, generator=generator)
    else:
        a = torch.randn(num_slices, d, device=z.device, dtype=z.dtype)
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # unit-norm slicing directions

    x_proj = z @ a.t()  # [N, num_slices]
    per_slice_stat = _epps_pulley_statistic_batched(x_proj, num_points=num_points)  # [num_slices]
    return per_slice_stat.mean()

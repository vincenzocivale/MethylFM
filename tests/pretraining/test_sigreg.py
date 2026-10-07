import cmath
import math

import torch

from methylfm.pretraining.sigreg import epps_pulley_statistic, sigreg_loss


def _naive_epps_pulley(x_list, t_min=-5.0, t_max=5.0, num_points=17):
    """Deliberately naive, unvectorized reference implementation: explicit
    Python double-loop over samples and t-grid points, computing the
    complex exponential sum by hand."""
    n = len(x_list)
    step = (t_max - t_min) / (num_points - 1)
    t_grid = [t_min + i * step for i in range(num_points)]

    err_vals = []
    for t in t_grid:
        acc = complex(0.0, 0.0)
        for x in x_list:
            acc += cmath.exp(complex(0.0, 1.0) * t * x)
        ecf = acc / n
        phi = math.exp(-0.5 * t * t)
        diff = ecf - complex(phi, 0.0)
        err_vals.append((abs(diff) ** 2) * phi)

    # trapezoidal integration by hand
    total = 0.0
    for i in range(num_points - 1):
        total += (err_vals[i] + err_vals[i + 1]) * step / 2.0
    return total * n


def test_epps_pulley_matches_naive_reference():
    torch.manual_seed(42)
    x = torch.randn(20, dtype=torch.float64)
    got = epps_pulley_statistic(x, num_points=17, t_min=-5.0, t_max=5.0)
    expected = _naive_epps_pulley(x.tolist(), num_points=17)
    assert abs(got.item() - expected) < 1e-4


def test_collapsed_embeddings_penalized_more_than_gaussian():
    torch.manual_seed(0)
    n, d = 512, 32
    collapsed = torch.zeros(n, d)
    gaussian = torch.randn(n, d)

    gen = torch.Generator().manual_seed(0)
    collapsed_score = sigreg_loss(collapsed, num_slices=256, num_points=17, generator=gen)
    gen = torch.Generator().manual_seed(0)
    gaussian_score = sigreg_loss(gaussian, num_slices=256, num_points=17, generator=gen)

    assert collapsed_score.item() > gaussian_score.item() * 2.0


def test_sigreg_gradients_flow():
    z = torch.randn(64, 16, requires_grad=True)
    loss = sigreg_loss(z, num_slices=32, num_points=17)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_sigreg_rejects_non_2d_input():
    try:
        sigreg_loss(torch.randn(10))
        assert False, "expected ValueError"
    except ValueError:
        pass

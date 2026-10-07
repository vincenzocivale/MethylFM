import torch
import torch.nn as nn

from methylfm.pretraining.lejepa import lejepa_loss
from methylfm.pretraining.projector import LeJEPAProjector


def test_lejepa_loss_keys_and_scalar():
    z_views = [torch.randn(8, 16) for _ in range(6)]
    out = lejepa_loss(z_views, global_view_indices=[0, 1], num_slices=64, num_points=17)
    assert set(out.keys()) == {"total", "invariance", "sigreg"}
    for v in out.values():
        assert v.dim() == 0
        assert torch.isfinite(v)


def test_lejepa_total_equals_inv_plus_lambda_sigreg():
    z_views = [torch.randn(8, 16) for _ in range(6)]
    lam = 0.02
    out = lejepa_loss(z_views, global_view_indices=[0, 1], sigreg_lambda=lam, num_slices=64, num_points=17)
    expected_total = out["invariance"] + lam * out["sigreg"]
    assert torch.allclose(out["total"], expected_total, atol=1e-6)


def test_gradients_propagate_through_encoder_and_projector():
    torch.manual_seed(0)
    encoder = nn.Linear(10, 16)
    projector = LeJEPAProjector(input_dim=16, hidden_dim=32, output_dim=8)

    num_views = 4
    inputs = [torch.randn(5, 10) for _ in range(num_views)]
    z_views = [projector(encoder(x)) for x in inputs]

    out = lejepa_loss(z_views, global_view_indices=[0, 1], num_slices=32, num_points=17)
    out["total"].backward()

    for p in list(encoder.parameters()) + list(projector.parameters()):
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum().item() > 0.0

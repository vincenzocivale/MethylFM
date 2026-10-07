import torch

from methylfm.pretraining.projector import LeJEPAProjector


def test_projector_shape():
    proj = LeJEPAProjector(input_dim=16, hidden_dim=32, output_dim=8)
    h = torch.randn(4, 16)
    z = proj(h)
    assert z.shape == (4, 8)


def test_projector_gradient_flow():
    proj = LeJEPAProjector(input_dim=16, hidden_dim=32, output_dim=8)
    h = torch.randn(4, 16, requires_grad=True)
    z = proj(h)
    z.sum().backward()
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
    for p in proj.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()

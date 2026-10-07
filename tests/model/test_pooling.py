import torch

from methylfm.model.pooling import AttentionPool


def test_output_shape_and_finite():
    torch.manual_seed(0)
    pool = AttentionPool(model_dim=32, num_heads=4)
    b, m, d = 3, 8, 32
    latents = torch.randn(b, m, d)
    h = pool(latents)
    assert h.shape == (b, d)
    assert torch.isfinite(h).all()


def test_permutation_invariant_over_latent_set():
    # Single-query attention pooling is a (masked) weighted sum over the
    # latent set, so it is mathematically invariant to the order of the M
    # latent tokens -- this is expected, not a bug.
    torch.manual_seed(1)
    pool = AttentionPool(model_dim=32, num_heads=4)
    b, m, d = 2, 8, 32
    latents = torch.randn(b, m, d)
    h1 = pool(latents)

    perm = torch.randperm(m)
    h2 = pool(latents[:, perm, :])

    assert torch.allclose(h1, h2, atol=1e-4)


def test_gradients_flow():
    torch.manual_seed(2)
    pool = AttentionPool(model_dim=32, num_heads=4)
    b, m, d = 2, 8, 32
    latents = torch.randn(b, m, d, requires_grad=True)
    h = pool(latents)
    h.sum().backward()
    assert latents.grad is not None
    assert torch.isfinite(latents.grad).all()
    assert pool.query.grad is not None

import torch

from methylfm.model.cross_attention import SetToLatentCrossAttention


def _make_module(seed: int = 0, chunk_size: int | None = None, model_dim: int = 32, num_latents: int = 8, num_heads: int = 4):
    torch.manual_seed(seed)
    return SetToLatentCrossAttention(
        model_dim=model_dim, num_latents=num_latents, num_heads=num_heads, chunk_size=chunk_size
    )


def test_permutation_invariance():
    torch.manual_seed(0)
    module = _make_module()
    b, n, d = 2, 37, 32
    tokens = torch.randn(b, n, d)
    mask = torch.ones(b, n, dtype=torch.bool)

    out1 = module(tokens, mask)

    perm = torch.randperm(n)
    tokens_perm = tokens[:, perm, :]
    mask_perm = mask[:, perm]
    out2 = module(tokens_perm, mask_perm)

    assert torch.allclose(out1, out2, atol=1e-4)


def test_masking_does_not_affect_valid_elements():
    torch.manual_seed(1)
    module = _make_module()
    b, n, d = 2, 20, 32
    tokens_real = torch.randn(b, n, d)
    mask_real = torch.ones(b, n, dtype=torch.bool)

    out_real = module(tokens_real, mask_real)

    pad_len = 15
    pad_tokens = torch.randn(b, pad_len, d) * 100.0
    tokens_padded = torch.cat([tokens_real, pad_tokens], dim=1)
    mask_padded = torch.cat([mask_real, torch.zeros(b, pad_len, dtype=torch.bool)], dim=1)

    out_padded = module(tokens_padded, mask_padded)

    assert torch.allclose(out_real, out_padded, atol=1e-4)


def test_variable_set_sizes():
    torch.manual_seed(2)
    module = _make_module(num_latents=8, model_dim=32, num_heads=4)
    b, d = 3, 32
    for n in (5, 200, 4000):
        tokens = torch.randn(b, n, d)
        mask = torch.ones(b, n, dtype=torch.bool)
        out = module(tokens, mask)
        assert out.shape == (b, 8, d)
        assert torch.isfinite(out).all()


def test_chunked_matches_non_chunked():
    torch.manual_seed(3)
    module_full = _make_module(seed=3, chunk_size=None)
    module_chunked = _make_module(seed=3, chunk_size=7)
    module_chunked.load_state_dict(module_full.state_dict())

    b, n, d = 2, 50, 32
    tokens = torch.randn(b, n, d)
    # irregular mask to exercise partially-masked chunks
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[0, 45:] = False
    mask[1, :3] = False

    out_full = module_full(tokens, mask)
    out_chunked = module_chunked(tokens, mask)

    assert torch.allclose(out_full, out_chunked, atol=1e-3)


def test_gradients_flow():
    torch.manual_seed(4)
    module = _make_module()
    b, n, d = 2, 10, 32
    tokens = torch.randn(b, n, d, requires_grad=True)
    mask = torch.ones(b, n, dtype=torch.bool)
    out = module(tokens, mask)
    out.sum().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert module.latents.grad is not None

import torch

from methylfm.model.latent_block import MHCLatentBackbone, MHCTransformerBlock
from methylfm.model.mhc import (
    HyperConnectionGate,
    MHCLayer,
    collapse_streams,
    init_streams,
    sinkhorn_doubly_stochastic,
)


def test_init_and_collapse_streams_shapes():
    b, n, m, d = 2, 4, 8, 16
    x0 = torch.randn(b, m, d)
    x_l = init_streams(x0, n)
    assert x_l.shape == (b, n, m, d)
    assert torch.allclose(x_l[:, 0], x0)
    assert torch.allclose(x_l[:, 1], x0)

    collapsed = collapse_streams(x_l)
    assert collapsed.shape == (b, m, d)


def test_mhc_layer_forward_shape():
    b, n, m, d = 2, 4, 8, 16
    x_l = init_streams(torch.randn(b, m, d), n)
    layer_fn = torch.nn.Linear(d, d)
    layer = MHCLayer(model_dim=d, num_streams=n, layer_fn=layer_fn, sinkhorn_iters=5)
    out = layer(x_l)
    assert out.shape == (b, n, m, d)
    assert torch.isfinite(out).all()


def test_sinkhorn_doubly_stochastic_nonnegative_and_row_col_sums():
    b, m, n = 3, 5, 4
    logits = torch.randn(b, m, n, n) * 3.0
    mat = sinkhorn_doubly_stochastic(logits, num_iters=100)

    assert (mat >= -1e-6).all()

    row_sums = mat.sum(dim=-1)
    col_sums = mat.sum(dim=-2)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-2)


def test_hyper_connection_gate_res_is_doubly_stochastic():
    b, n, m, d = 2, 4, 6, 8
    gate = HyperConnectionGate(model_dim=d, num_streams=n, sinkhorn_iters=50)
    x_l = torch.randn(b, n, m, d)
    h_pre, h_post, h_res = gate(x_l)

    assert h_pre.shape == (b, m, n)
    assert h_post.shape == (b, m, n)
    assert h_res.shape == (b, m, n, n)

    assert (h_res >= -1e-6).all()
    row_sums = h_res.sum(dim=-1)
    col_sums = h_res.sum(dim=-2)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-2)


def test_mhc_transformer_block_backward_finite():
    b, n, m, d = 2, 4, 6, 16
    block = MHCTransformerBlock(model_dim=d, num_heads=4, mlp_ratio=4, num_streams=n)
    x_l = init_streams(torch.randn(b, m, d, requires_grad=True), n)
    x_l.retain_grad()
    out = block(x_l)
    out.sum().backward()

    for name, p in block.named_parameters():
        assert p.grad is not None, f"missing grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_deep_mhc_latent_backbone_smoke():
    b, m, d = 2, 8, 32
    backbone = MHCLatentBackbone(
        model_dim=d,
        num_latents=m,
        num_layers=16,
        num_heads=4,
        mlp_ratio=4,
        mhc_expansion_rate=4,
    )
    x = torch.randn(b, m, d)
    out = backbone(x)
    assert out.shape == (b, m, d)
    assert torch.isfinite(out).all()

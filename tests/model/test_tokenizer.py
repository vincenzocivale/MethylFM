import pytest
import torch

from methylfm.model.cpg_encoder import MockCpGEncoder
from methylfm.model.tokenizer import FTCpGTokenizer


def _build(locus_dim=8, model_dim=8, seed=0):
    torch.manual_seed(seed)
    encoder = MockCpGEncoder(locus_dim=locus_dim)
    tokenizer = FTCpGTokenizer(locus_dim=locus_dim, model_dim=model_dim)
    return encoder, tokenizer


def test_beta_zero_gives_baseline():
    encoder, tokenizer = _build()
    cpg_ids = torch.randint(0, 1000, (4, 6), dtype=torch.long)
    l = encoder(cpg_ids)
    beta = torch.zeros(cpg_ids.shape)

    out = tokenizer(l, beta)
    expected = tokenizer.ln(tokenizer.f_b(l))
    assert torch.allclose(out, expected, atol=1e-6)


def test_beta_one_gives_baseline_plus_direction():
    encoder, tokenizer = _build()
    cpg_ids = torch.randint(0, 1000, (4, 6), dtype=torch.long)
    l = encoder(cpg_ids)
    beta = torch.ones(cpg_ids.shape)

    out = tokenizer(l, beta)
    expected = tokenizer.ln(tokenizer.f_b(l) + tokenizer.f_w(l))
    assert torch.allclose(out, expected, atol=1e-6)


def test_gradients_reach_encoder_and_tokenizer_projections():
    encoder, tokenizer = _build()
    cpg_ids = torch.randint(0, 1000, (4, 6), dtype=torch.long)
    l = encoder(cpg_ids)
    beta = torch.rand(cpg_ids.shape)

    out = tokenizer(l, beta)
    out.sum().backward()

    assert encoder.embedding.weight.grad is not None
    assert torch.any(encoder.embedding.weight.grad != 0)
    assert tokenizer.f_b.weight.grad is not None
    assert torch.any(tokenizer.f_b.weight.grad != 0)
    assert tokenizer.f_w.weight.grad is not None
    assert torch.any(tokenizer.f_w.weight.grad != 0)


@pytest.mark.parametrize(
    "bad_beta",
    [
        torch.tensor([[1.5, 0.2]]),
        torch.tensor([[-0.1, 0.2]]),
        torch.tensor([[float("nan"), 0.2]]),
    ],
)
def test_beta_out_of_range_raises(bad_beta):
    encoder, tokenizer = _build(locus_dim=4, model_dim=4)
    cpg_ids = torch.randint(0, 1000, bad_beta.shape, dtype=torch.long)
    l = encoder(cpg_ids)
    with pytest.raises(ValueError):
        tokenizer(l, bad_beta)

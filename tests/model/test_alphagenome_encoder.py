import json

import numpy as np
import pytest
import torch
from torch import nn

from methylfm import MethylFMConfig, MethylFMForPreTraining, MethylFMModel
from methylfm.cpg_embeddings.store import StoreIncompatibleError, write_synthetic_store
from methylfm.model.cpg_encoder import AlphaGenomeCpGEncoder
from methylfm.pretraining.lejepa import lejepa_loss

D1, D128, LOCUS, N = 16, 32, 24, 64


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return write_synthetic_store(tmp_path_factory.mktemp("agstore"), N, D1, D128, seed=3)


def cfg(**kw):
    base = dict(
        hidden_size=32, num_latents=8, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
        mhc_expansion_rate=2, mhc_sinkhorn_iters=3, locus_dim=LOCUS, cpg_encoder_type="alphagenome",
        cpg_embedding_1bp_dim=D1, cpg_embedding_128bp_dim=D128, cpg_adapter_hidden_size=LOCUS,
        cpg_adapter_bottleneck_size=6, projector_hidden_size=32, projector_output_size=16,
    )
    base.update(kw)
    return MethylFMConfig(**base)


def encoder(store, **kw):
    return AlphaGenomeCpGEncoder(store, locus_dim=LOCUS, bottleneck_dim=6, **kw)


def test_output_shape_and_padding(store):
    enc = encoder(store)
    ids = torch.tensor([[0, 5, 5, -1], [63, 1, -1, -1]])
    out = enc(ids)
    assert out.shape == (2, 4, LOCUS)
    assert (out[ids < 0] == 0).all() and (out[ids >= 0].abs().sum(-1) > 0).all()
    assert enc(torch.tensor([[0, 1, 2]])).shape[-1] == LOCUS  # [..., locus_dim] for any leading shape
    assert enc(torch.tensor([3])).shape == (1, LOCUS)


def test_identity_like_initialisation(store):
    enc = encoder(store).eval()
    ids = torch.arange(N)
    h = enc._fetch(ids)
    u = enc.input_norm(enc.input_proj(h.float()))
    assert (enc.adapter_up.weight == 0).all() and (enc.adapter_up.bias == 0).all()
    delta = enc.adapter_up(torch.nn.functional.gelu(enc.adapter_down(u)))
    assert delta.abs().max() == 0
    l = enc(ids)
    assert torch.allclose(l, u, atol=1e-4), (l - u).abs().max()  # l_i ~= u_i at init


def test_zero_init_survives_hf_post_init(store):
    model = MethylFMModel(cfg(), cpg_embedding_store=store)
    enc = model.cpg_encoder
    assert (enc.adapter_up.weight == 0).all() and (enc.adapter_up.bias == 0).all()
    assert enc.input_proj.weight.std() > 0  # the rest is randomly initialised


def test_forward_matches_naive_per_token_computation(store):
    enc = encoder(store).eval()
    with torch.no_grad():
        enc.adapter_up.weight.normal_(0, 0.1)
        ids = torch.randint(0, N, (3, 11))
        naive = enc._transform(enc._fetch(ids.reshape(-1))).reshape(3, 11, LOCUS)  # no dedup, no chunking
        assert torch.allclose(enc(ids), naive, atol=1e-5)
        small = encoder(store, chunk_size=7).eval(); small.load_state_dict(enc.state_dict())
        assert torch.allclose(small(ids), naive, atol=1e-5)


def test_resident_and_gather_modes_agree(store):
    a, b = encoder(store, lookup_mode="gather"), encoder(store, lookup_mode="resident")
    b.load_state_dict(a.state_dict())
    ids = torch.randint(0, N, (4, 9))
    assert torch.allclose(a(ids), b(ids), atol=1e-6)


def test_checkpointed_and_plain_gradients_agree(store):
    ids = torch.randint(0, N, (5, 20))
    grads = []
    for ckpt in (True, False):
        torch.manual_seed(0)
        enc = encoder(store, chunk_size=10, activation_checkpointing=ckpt).train()
        enc.adapter_up.weight.data.normal_(0, 0.1)
        (enc(ids) ** 3).sum().backward()
        grads.append(torch.cat([p.grad.flatten() for p in enc.parameters()]))
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


def test_no_per_cpg_parameters_and_frozen_store_is_not_state(store):
    enc = encoder(store)
    assert not any(isinstance(m, nn.Embedding) for m in enc.modules())
    assert all(p.shape[0] != N for p in enc.parameters())  # no [num_cpgs, ...] table
    assert list(enc.buffers()) == []
    keys = set(enc.state_dict())
    assert keys == {f"{m}.{p}" for m in ("input_proj", "adapter_down", "adapter_up") for p in ("weight", "bias")} | {
        f"{m}.{p}" for m in ("input_norm", "output_norm") for p in ("weight", "bias")}
    expected = (D1 + D128) * LOCUS + LOCUS + 2 * LOCUS + LOCUS * 6 + 6 + 6 * LOCUS + LOCUS + 2 * LOCUS
    assert sum(p.numel() for p in enc.parameters()) == expected


def test_unseen_locus_is_encoded_by_the_same_shared_transform(store):
    enc = encoder(store)
    optim = torch.optim.SGD(enc.parameters(), lr=0.1)
    for _ in range(3):  # "train" on loci 0..9 only
        optim.zero_grad(); (enc(torch.arange(10)) ** 2).sum().backward(); optim.step()
    never_seen = enc(torch.tensor([N - 1]))
    assert never_seen.shape == (1, LOCUS) and torch.isfinite(never_seen).all()


def test_gradients_reach_adapter_projection_tokenizer_and_nothing_else_is_trainable(store):
    model = MethylFMForPreTraining(cfg(), cpg_embedding_store=store)
    enc, tok = model.methylfm.cpg_encoder, model.methylfm.tokenizer
    ids = torch.randint(0, N, (4, 30)); beta = torch.rand(4, 30)

    def step():
        model.zero_grad()
        z = [model(ids[:, :20 + 5 * k], beta[:, :20 + 5 * k]).embeddings for k in range(3)]
        lejepa_loss(z, global_view_indices=[0, 1], num_slices=16, num_points=5)["total"].backward()

    step()
    for name, p in [("W_in", enc.input_proj.weight), ("W_up", enc.adapter_up.weight),
                    ("f_b", tok.f_b.weight), ("f_w", tok.f_w.weight)]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    # exactly at the zero-initialised residual branch, dL/dW_down is exactly 0 (W_up = 0) -- expected
    assert enc.adapter_down.weight.grad.abs().sum() == 0
    with torch.no_grad():
        for p in model.parameters():
            p -= 0.05 * p.grad
    step()  # after one update W_up != 0 -> the whole adapter gets gradient
    for p in (enc.adapter_down.weight, enc.adapter_up.weight, enc.input_proj.weight):
        assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    # the frozen embeddings are constants: not parameters of the model, no grad anywhere
    assert not any(p.shape[0] == N for p in enc.parameters())


def test_config_carries_asset_id_but_no_path_and_checks_it(store, tmp_path, monkeypatch):
    c = cfg()
    model = MethylFMModel(c, cpg_embedding_store=store)
    assert c.cpg_embedding_asset_id == store.asset_id
    model.save_pretrained(tmp_path / "m")
    text = (tmp_path / "m" / "config.json").read_text()
    assert str(store.path) not in text and "store_path" not in text
    assert json.loads(text)["cpg_embedding_asset_id"] == store.asset_id

    ids, beta = torch.randint(0, N, (2, 12)), torch.rand(2, 12)
    monkeypatch.setenv("METHYLFM_CPG_EMBEDDING_STORE", str(store.path))
    reloaded = MethylFMModel.from_pretrained(tmp_path / "m").eval()  # resolved through the env var
    assert torch.allclose(reloaded(ids, beta).pooler_output, model.eval()(ids, beta).pooler_output, atol=1e-5)

    other = write_synthetic_store(tmp_path / "other", N, D1, D128, seed=99)
    with pytest.raises(StoreIncompatibleError, match="asset id"):
        MethylFMModel.from_pretrained(tmp_path / "m", cpg_embedding_store=other)
    monkeypatch.delenv("METHYLFM_CPG_EMBEDDING_STORE")
    with pytest.raises(FileNotFoundError, match="CpG embedding store was not found"):
        MethylFMModel.from_pretrained(tmp_path / "m")


def test_dim_mismatch_between_config_and_store_is_rejected(store):
    with pytest.raises(StoreIncompatibleError, match="dims"):
        MethylFMModel(cfg(cpg_embedding_1bp_dim=1536), cpg_embedding_store=store)


def test_config_validation():
    with pytest.raises(ValueError, match="cpg_adapter_hidden_size"):
        cfg(cpg_adapter_hidden_size=999)
    with pytest.raises(ValueError, match="Unknown cpg_encoder_type"):
        cfg(cpg_encoder_type="nope")


def test_encoder_is_patient_independent_and_the_tokenizer_still_owns_beta(store):
    model = MethylFMModel(cfg(), cpg_embedding_store=store).eval()
    ids = torch.randint(0, N, (1, 10))
    # same loci, different patients -> identical locus representation l_i, different tokens t_si
    l = model.cpg_encoder(ids)
    t1 = model.tokenizer(l, torch.rand(1, 10)); t2 = model.tokenizer(l, torch.rand(1, 10))
    assert torch.equal(l, model.cpg_encoder(ids)) and not torch.allclose(t1, t2)

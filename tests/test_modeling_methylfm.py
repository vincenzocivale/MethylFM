import torch
from safetensors.torch import load_file

from methylfm import MethylFMConfig, MethylFMModel, MethylFMProcessor
from methylfm.modeling_methylfm import MethylFMForPreTraining


def _small_config(**overrides) -> MethylFMConfig:
    kwargs = dict(
        hidden_size=32,
        num_latents=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        mhc_expansion_rate=2,
        mhc_sinkhorn_iters=3,
        locus_dim=32,
        cpg_encoder_vocab_size=512,
        projector_hidden_size=32,
        projector_output_size=16,
    )
    kwargs.update(overrides)
    return MethylFMConfig(**kwargs)


def _random_inputs(batch_size=3, n=20, vocab_size=512, seed=0):
    gen = torch.Generator().manual_seed(seed)
    cpg_ids = torch.randint(0, vocab_size, (batch_size, n), generator=gen, dtype=torch.int64)
    beta_values = torch.rand(batch_size, n, generator=gen)
    attention_mask = torch.ones(batch_size, n, dtype=torch.bool)
    return cpg_ids, beta_values, attention_mask


def test_forward_shapes():
    config = _small_config()
    model = MethylFMModel(config)
    model.eval()
    cpg_ids, beta_values, attention_mask = _random_inputs()
    with torch.no_grad():
        out = model(cpg_ids, beta_values, attention_mask)
    assert out.last_hidden_state.shape == (3, 8, 32)
    assert out.pooler_output.shape == (3, 32)


def test_return_dict_false_matches_return_dict_true():
    config = _small_config()
    model = MethylFMModel(config)
    model.eval()
    cpg_ids, beta_values, attention_mask = _random_inputs()
    with torch.no_grad():
        out_dict = model(cpg_ids, beta_values, attention_mask, return_dict=True)
        out_tuple = model(cpg_ids, beta_values, attention_mask, return_dict=False)
    assert isinstance(out_tuple, tuple)
    last_hidden_state, pooler_output = out_tuple
    assert torch.equal(last_hidden_state, out_dict.last_hidden_state)
    assert torch.equal(pooler_output, out_dict.pooler_output)


def test_save_load_roundtrip(tmp_path):
    config = _small_config()
    model = MethylFMModel(config)
    model.eval()
    cpg_ids, beta_values, attention_mask = _random_inputs()
    with torch.no_grad():
        out = model(cpg_ids, beta_values, attention_mask)

    model.save_pretrained(tmp_path, safe_serialization=True)
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "model.safetensors").exists()

    reloaded = MethylFMModel.from_pretrained(tmp_path)
    reloaded.eval()
    with torch.no_grad():
        out_reloaded = reloaded(cpg_ids, beta_values, attention_mask)

    assert torch.allclose(out.pooler_output, out_reloaded.pooler_output, atol=1e-6)
    assert torch.allclose(out.last_hidden_state, out_reloaded.last_hidden_state, atol=1e-6)

    # Explicit safetensors check (not an accidental pickle fallback).
    raw_state = load_file(str(tmp_path / "model.safetensors"))
    model_state = model.state_dict()
    for key in ("tokenizer.f_b.weight", "pooler.query"):
        assert torch.allclose(raw_state[key], model_state[key])


def test_permutation_invariance():
    config = _small_config()
    model = MethylFMModel(config)
    model.eval()
    cpg_ids, beta_values, attention_mask = _random_inputs(batch_size=1, n=15)

    with torch.no_grad():
        out = model(cpg_ids, beta_values, attention_mask)

    perm = torch.randperm(cpg_ids.shape[1])
    cpg_ids_p = cpg_ids[:, perm]
    beta_values_p = beta_values[:, perm]
    attention_mask_p = attention_mask[:, perm]

    with torch.no_grad():
        out_p = model(cpg_ids_p, beta_values_p, attention_mask_p)

    assert torch.allclose(out.pooler_output, out_p.pooler_output, atol=1e-4)


def test_for_pretraining_backbone_is_real_methylfm_model():
    config = _small_config()
    pretraining_model = MethylFMForPreTraining(config)
    pretraining_model.eval()

    standalone = MethylFMModel(config)
    standalone.load_state_dict(pretraining_model.methylfm.state_dict())
    standalone.eval()

    cpg_ids, beta_values, attention_mask = _random_inputs()
    with torch.no_grad():
        out_backbone = pretraining_model.methylfm(cpg_ids, beta_values, attention_mask)
        out_standalone = standalone(cpg_ids, beta_values, attention_mask)

    assert torch.allclose(out_backbone.pooler_output, out_standalone.pooler_output, atol=1e-6)

    with torch.no_grad():
        out = pretraining_model(cpg_ids, beta_values, attention_mask)
    assert out.embeddings.shape == (3, 16)
    assert out.loss is None
    assert out.invariance_loss is None
    assert out.sigreg_loss is None


def test_compute_lejepa_loss():
    config = _small_config()
    model = MethylFMForPreTraining(config)

    view_batches = []
    for n in (10, 15, 8):
        cpg_ids, beta_values, attention_mask = _random_inputs(batch_size=4, n=n, seed=n)
        view_batches.append({"cpg_ids": cpg_ids, "beta_values": beta_values, "attention_mask": attention_mask})

    out = model.compute_lejepa_loss(view_batches, global_view_indices=[0, 1], num_slices=32, num_points=9)

    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.invariance_loss)
    assert torch.isfinite(out.sigreg_loss)

    out.loss.backward()
    backbone_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.methylfm.parameters())
    projector_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.projector.parameters())
    assert backbone_grad
    assert projector_grad


def test_end_to_end_processor_model():
    config = _small_config()
    model = MethylFMModel(config)
    model.eval()
    processor = MethylFMProcessor(vocab_size=512)

    inputs = processor(
        cpg_ids=["cg00000029", "cg00000108", "cg00000133"],
        beta_values=[0.1, 0.5, 0.9],
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)

    assert outputs.pooler_output.shape == (1, 32)

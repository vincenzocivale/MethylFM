import pytest

from methylfm.configuration_methylfm import MethylFMConfig


def _full_config(**overrides) -> MethylFMConfig:
    kwargs = dict(
        hidden_size=64,
        num_latents=12,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=256,
        mhc_expansion_rate=3,
        mhc_sinkhorn_iters=7,
        attention_dropout=0.1,
        hidden_dropout=0.2,
        initializer_range=0.03,
        locus_dim=48,
        cross_attention_chunk_size=128,
        cpg_encoder_type="mock",
        cpg_encoder_vocab_size=777,
        projector_hidden_size=96,
        projector_output_size=40,
    )
    kwargs.update(overrides)
    return MethylFMConfig(**kwargs)


def test_config_save_load_roundtrip(tmp_path):
    config = _full_config()
    config.save_pretrained(tmp_path)
    reloaded = MethylFMConfig.from_pretrained(tmp_path)

    for field in (
        "hidden_size", "num_latents", "num_hidden_layers", "num_attention_heads",
        "intermediate_size", "mhc_expansion_rate", "mhc_sinkhorn_iters",
        "attention_dropout", "hidden_dropout", "initializer_range", "locus_dim",
        "cross_attention_chunk_size", "cpg_encoder_type", "cpg_encoder_vocab_size",
        "projector_hidden_size", "projector_output_size",
    ):
        assert getattr(reloaded, field) == getattr(config, field), f"field {field} did not round-trip"


def test_mlp_ratio_property():
    config = _full_config(hidden_size=64, intermediate_size=256)
    assert config.mlp_ratio == 4

    config2 = _full_config(hidden_size=32, intermediate_size=64)
    assert config2.mlp_ratio == 2


def test_hidden_size_must_be_divisible_by_num_attention_heads():
    with pytest.raises(ValueError):
        _full_config(hidden_size=65, num_attention_heads=4)


def test_unknown_cpg_encoder_type_raises():
    with pytest.raises(ValueError):
        _full_config(cpg_encoder_type="sequence")


def test_model_type_is_methylfm():
    assert MethylFMConfig.model_type == "methylfm"
    config = _full_config()
    assert config.model_type == "methylfm"

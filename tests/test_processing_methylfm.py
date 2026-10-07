import math

import pytest
import torch

from methylfm.configuration_methylfm import MethylFMConfig
from methylfm.modeling_methylfm import MethylFMModel
from methylfm.processing_methylfm import MethylFMProcessor
from methylfm.utils.cpg_ids import probe_id_to_int


def test_single_sample():
    processor = MethylFMProcessor(vocab_size=2**20)
    inputs = processor(cpg_ids=["cg00000029", "cg00000108"], beta_values=[0.82, 0.14], return_tensors="pt")

    assert inputs["cpg_ids"].shape == (1, 2)
    assert inputs["beta_values"].shape == (1, 2)
    assert inputs["attention_mask"].shape == (1, 2)
    assert inputs["cpg_ids"].dtype == torch.long
    assert inputs["beta_values"].dtype == torch.float32
    assert inputs["attention_mask"].dtype == torch.bool
    assert inputs["attention_mask"].all()
    assert torch.allclose(inputs["beta_values"][0], torch.tensor([0.82, 0.14]), atol=1e-6)


def test_batched_samples_variable_length():
    processor = MethylFMProcessor()
    inputs = processor(
        cpg_ids=[["cg00000029", "cg00000108"], ["cg00000165", "cg00000236", "cg00000289"]],
        beta_values=[[0.1, 0.9], [0.2, 0.3, 0.4]],
        return_tensors="pt",
    )

    assert inputs["cpg_ids"].shape == (2, 3)
    assert inputs["attention_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert torch.allclose(inputs["beta_values"][0, :2], torch.tensor([0.1, 0.9]), atol=1e-6)
    assert torch.allclose(inputs["beta_values"][1], torch.tensor([0.2, 0.3, 0.4]), atol=1e-6)
    # padded positions are zero-filled
    assert inputs["beta_values"][0, 2].item() == 0.0
    assert inputs["cpg_ids"][0, 2].item() == 0


def test_unordered_cpgs_map_to_same_ids_regardless_of_order():
    vocab_size = 2**20
    processor = MethylFMProcessor(vocab_size=vocab_size)

    forward = processor(cpg_ids=["cg00000029", "cg00000108"], beta_values=[0.1, 0.2], return_tensors="pt")
    reversed_ = processor(cpg_ids=["cg00000108", "cg00000029"], beta_values=[0.2, 0.1], return_tensors="pt")

    expected_29 = probe_id_to_int("cg00000029") % vocab_size
    expected_108 = probe_id_to_int("cg00000108") % vocab_size

    assert forward["cpg_ids"][0].tolist() == [expected_29, expected_108]
    assert reversed_["cpg_ids"][0].tolist() == [expected_108, expected_29]
    # same ids present in both, just reordered -- the mapping itself is order-independent
    assert set(forward["cpg_ids"][0].tolist()) == set(reversed_["cpg_ids"][0].tolist())


def test_nan_beta_is_dropped_not_zero_filled():
    processor = MethylFMProcessor()
    inputs = processor(
        cpg_ids=["cg00000029", "cg00000108", "cg00000165"],
        beta_values=[0.5, float("nan"), 0.7],
        return_tensors="pt",
    )

    assert inputs.dropped_counts == [1]
    assert int(inputs["attention_mask"].sum().item()) == 2
    kept_betas = inputs["beta_values"][0][inputs["attention_mask"][0]]
    assert torch.allclose(torch.sort(kept_betas).values, torch.tensor([0.5, 0.7]), atol=1e-6)


@pytest.mark.parametrize("bad_beta", [1.5, -0.3])
def test_invalid_beta_raises(bad_beta):
    processor = MethylFMProcessor()
    with pytest.raises(ValueError):
        processor(cpg_ids=["cg00000029"], beta_values=[bad_beta], return_tensors="pt")


def test_duplicate_cpg_id_raises():
    processor = MethylFMProcessor()
    with pytest.raises(ValueError, match="duplicate"):
        processor(cpg_ids=["cg00000029", "cg00000029"], beta_values=[0.1, 0.2], return_tensors="pt")


def test_variable_sample_sizes_in_one_batch():
    processor = MethylFMProcessor()
    inputs = processor(
        cpg_ids=[["cg00000029", "cg00000108"], ["cg00000165", "cg00000236", "cg00000289", "cg00000363", "cg00000384"]],
        beta_values=[[0.1, 0.2], [0.1, 0.2, 0.3, 0.4, 0.5]],
        return_tensors="pt",
    )
    assert inputs["cpg_ids"].shape == (2, 5)
    assert inputs["attention_mask"][0].sum().item() == 2
    assert inputs["attention_mask"][1].sum().item() == 5


def test_save_and_load_round_trip(tmp_path):
    processor = MethylFMProcessor(vocab_size=1234)
    save_dir = tmp_path / "processor"
    processor.save_pretrained(str(save_dir))

    reloaded = MethylFMProcessor.from_pretrained(str(save_dir))
    assert reloaded.vocab_size == 1234

    original_out = processor(cpg_ids=["cg00000029", "cg00000108"], beta_values=[0.1, 0.2], return_tensors="pt")
    reloaded_out = reloaded(cpg_ids=["cg00000029", "cg00000108"], beta_values=[0.1, 0.2], return_tensors="pt")
    assert torch.equal(original_out["cpg_ids"], reloaded_out["cpg_ids"])


def test_from_pretrained_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        MethylFMProcessor.from_pretrained(str(tmp_path / "does_not_exist"))


def test_end_to_end_processor_to_model_pooler_output():
    config = MethylFMConfig(
        hidden_size=32,
        num_latents=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        mhc_expansion_rate=2,
        mhc_sinkhorn_iters=3,
        locus_dim=32,
        cpg_encoder_vocab_size=512,
    )
    model = MethylFMModel(config)
    processor = MethylFMProcessor(vocab_size=512)

    inputs = processor(
        cpg_ids=[["cg00000029", "cg00000108", "cg00000165"], ["cg00000236", "cg00000289"]],
        beta_values=[[0.1, 0.5, 0.9], [0.3, 0.7]],
        return_tensors="pt",
    )
    outputs = model(**inputs)

    assert outputs.pooler_output.shape == (2, 32)
    assert torch.isfinite(outputs.pooler_output).all()

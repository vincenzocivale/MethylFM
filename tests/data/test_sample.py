import torch
import pytest

from methylfm.data.sample import MethylationSample, validate_sample


def _make_sample(cpg_ids, betas, sample_id="s1", dataset_id="GSE_TEST", platform="EPIC"):
    return MethylationSample(
        sample_id=sample_id,
        cpg_ids=torch.tensor(cpg_ids, dtype=torch.int64),
        beta_values=torch.tensor(betas, dtype=torch.float32),
        dataset_id=dataset_id,
        platform=platform,
    )


def test_removes_nan_entries():
    sample = _make_sample([1, 2, 3, 4], [0.1, float("nan"), 0.5, float("nan")])
    out = validate_sample(sample)
    assert out.cpg_ids.tolist() == [1, 3]
    assert out.beta_values.tolist() == pytest.approx([0.1, 0.5])


def test_beta_out_of_range_raises():
    sample = _make_sample([1, 2], [0.1, 1.5])
    with pytest.raises(ValueError):
        validate_sample(sample)

    sample_neg = _make_sample([1, 2], [0.1, -0.2])
    with pytest.raises(ValueError):
        validate_sample(sample_neg)


def test_shape_mismatch_raises():
    sample = MethylationSample(
        sample_id="s1",
        cpg_ids=torch.tensor([1, 2, 3], dtype=torch.int64),
        beta_values=torch.tensor([0.1, 0.2], dtype=torch.float32),
        dataset_id="GSE_TEST",
        platform="EPIC",
    )
    with pytest.raises(ValueError):
        validate_sample(sample)


def test_does_not_mutate_input():
    sample = _make_sample([1, 2, 3, 4], [0.1, float("nan"), 0.5, 0.9])
    cpg_ids_before = sample.cpg_ids.clone()
    beta_before = sample.beta_values.clone()

    _ = validate_sample(sample)

    assert torch.equal(sample.cpg_ids, cpg_ids_before)
    # NaN != NaN under ==, so compare bit patterns instead
    assert torch.equal(sample.beta_values.isnan(), beta_before.isnan())
    valid = ~beta_before.isnan()
    assert torch.equal(sample.beta_values[valid], beta_before[valid])

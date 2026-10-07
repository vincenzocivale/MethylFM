import torch

from methylfm.data.sample import MethylationSample
from methylfm.data.collate import collate_methylation_batch


def _make_sample(n, sample_id):
    return MethylationSample(
        sample_id=sample_id,
        cpg_ids=torch.arange(n, dtype=torch.int64),
        beta_values=torch.rand(n, dtype=torch.float32),
        dataset_id="GSE_TEST",
        platform="EPIC",
    )


def test_heterogeneous_cpg_counts():
    samples = [_make_sample(5, "a"), _make_sample(50, "b"), _make_sample(500, "c")]
    batch = collate_methylation_batch(samples)

    assert batch.cpg_ids.shape == (3, 500)
    assert batch.beta_values.shape == (3, 500)
    assert batch.mask.shape == (3, 500)

    assert batch.cpg_ids.dtype == torch.int64
    assert batch.beta_values.dtype == torch.float32
    assert batch.mask.dtype == torch.bool

    assert batch.num_cpgs.tolist() == [5, 50, 500]

    assert batch.mask[0, :5].all() and not batch.mask[0, 5:].any()
    assert batch.mask[1, :50].all() and not batch.mask[1, 50:].any()
    assert batch.mask[2, :500].all()

    # padded region is zero and does not corrupt valid entries
    assert torch.equal(batch.cpg_ids[0, 5:], torch.zeros(495, dtype=torch.int64))
    assert torch.equal(batch.beta_values[0, 5:], torch.zeros(495, dtype=torch.float32))
    assert torch.equal(batch.cpg_ids[0, :5], samples[0].cpg_ids)
    assert torch.equal(batch.beta_values[0, :5], samples[0].beta_values)

    assert batch.sample_ids == ["a", "b", "c"]
    assert batch.dataset_ids == ["GSE_TEST", "GSE_TEST", "GSE_TEST"]
    assert batch.platforms == ["EPIC", "EPIC", "EPIC"]


def test_single_sample_batch():
    samples = [_make_sample(10, "only")]
    batch = collate_methylation_batch(samples)
    assert batch.cpg_ids.shape == (1, 10)
    assert batch.mask.all()

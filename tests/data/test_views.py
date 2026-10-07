import torch

from methylfm.data.sample import MethylationSample
from methylfm.data.views import ViewBuilder, ViewConfig


def _make_sample(n, sample_id="s1"):
    return MethylationSample(
        sample_id=sample_id,
        cpg_ids=torch.arange(n, dtype=torch.int64),
        beta_values=torch.rand(n, dtype=torch.float32),
        dataset_id="GSE_TEST",
        platform="EPIC",
    )


def test_reference_view_is_full_sample():
    sample = _make_sample(1000)
    builder = ViewBuilder(ViewConfig())
    gen = torch.Generator().manual_seed(0)
    views = builder.build_views(sample, gen)
    ref = views["reference_view"]
    assert torch.equal(ref.cpg_ids, sample.cpg_ids)
    assert torch.equal(ref.beta_values, sample.beta_values)


def test_keep_fraction_approximate():
    n = 1000
    sample = _make_sample(n)
    config = ViewConfig(global_keep_fraction=0.8, num_local_views=4, local_keep_fraction=0.25)
    builder = ViewBuilder(config)
    gen = torch.Generator().manual_seed(42)
    views = builder.build_views(sample, gen)

    broad_n = views["broad_view"].cpg_ids.shape[0]
    assert abs(broad_n - n * 0.8) < 0.1 * n * 0.8

    for sv in views["sparse_views"]:
        sv_n = sv.cpg_ids.shape[0]
        assert abs(sv_n - n * 0.25) < 0.1 * n * 0.25

    assert len(views["sparse_views"]) == 4


def test_deterministic_under_fixed_seed():
    sample = _make_sample(1000)
    builder = ViewBuilder(ViewConfig())

    gen1 = torch.Generator().manual_seed(1337)
    views1 = builder.build_views(sample, gen1)

    gen2 = torch.Generator().manual_seed(1337)
    views2 = builder.build_views(sample, gen2)

    assert torch.equal(views1["broad_view"].cpg_ids, views2["broad_view"].cpg_ids)
    for sv1, sv2 in zip(views1["sparse_views"], views2["sparse_views"]):
        assert torch.equal(sv1.cpg_ids, sv2.cpg_ids)


def test_different_seed_differs():
    sample = _make_sample(1000)
    builder = ViewBuilder(ViewConfig())

    gen1 = torch.Generator().manual_seed(1)
    views1 = builder.build_views(sample, gen1)

    gen2 = torch.Generator().manual_seed(2)
    views2 = builder.build_views(sample, gen2)

    assert not torch.equal(views1["broad_view"].cpg_ids, views2["broad_view"].cpg_ids)


def test_does_not_mutate_input_sample():
    sample = _make_sample(500)
    cpg_before = sample.cpg_ids.clone()
    beta_before = sample.beta_values.clone()

    builder = ViewBuilder(ViewConfig())
    gen = torch.Generator().manual_seed(7)
    _ = builder.build_views(sample, gen)

    assert torch.equal(sample.cpg_ids, cpg_before)
    assert torch.equal(sample.beta_values, beta_before)


def test_variable_set_sizes():
    for n in (1, 5, 37, 999):
        sample = _make_sample(n)
        builder = ViewBuilder(ViewConfig())
        gen = torch.Generator().manual_seed(0)
        views = builder.build_views(sample, gen)
        assert views["reference_view"].cpg_ids.shape[0] == n
        assert views["broad_view"].cpg_ids.shape[0] <= n
        for sv in views["sparse_views"]:
            assert sv.cpg_ids.shape[0] <= n

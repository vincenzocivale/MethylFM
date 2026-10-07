import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pytest

from methylfm.data.cpgcorpus import (
    CpGCorpusConfig,
    CpGCorpusDataset,
    CpGCorpusNotFoundError,
)

# Test-authored fixture schema (NOT a claim about the real CpGCorpus
# schema, which is unresolved - see cpgcorpus.py module docstring).
COLUMN_MAPPING = {"sample_id": "sample_id", "probe_id": "probe_id", "beta": "beta"}


def _write_arrow_shard(path, sample_ids, probe_ids, betas):
    table = pa.table(
        {
            "sample_id": sample_ids,
            "probe_id": probe_ids,
            "beta": betas,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        with pa_ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def _make_corpus(root, gse_id="GSE_TEST", gpl_id="GPL_TEST", extra_malformed=False):
    betas_path = root / gse_id / gpl_id / "betas" / "QCDPB.arrow"
    metadata_path = root / gse_id / gpl_id / "metadata" / "metadata.arrow"

    sample_ids = (
        ["sampleA"] * 3
        + ["sampleB"] * 3
        + ["sampleA"] * 2  # duplicate sample id (split across rows is fine,
        # grouped by read_samples; this creates a *repeated* sampleA group
        # when combined with the first - but to get a true "duplicate
        # sample_id" at the MethylationSample level we add a second,
        # separate shard-level repeat below instead.
    )
    probe_ids = ["cg00000001", "cg00000002", "cg00000003"] * 2 + ["cg00000001", "cg00000002"]
    betas = [0.1, 0.5, float("nan")] + [0.2, 0.9, 0.3] + [0.4, 0.6]

    if extra_malformed:
        sample_ids += ["sampleMalformed"] * 2
        probe_ids += ["cg00000001", "cg00000002"]
        betas += [1.5, 0.2]  # out-of-range beta -> malformed

    _write_arrow_shard(betas_path, sample_ids, probe_ids, betas)

    meta_table = pa.table({"sample_id": sorted(set(sample_ids))})
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(metadata_path), "wb") as sink:
        with pa_ipc.new_file(sink, meta_table.schema) as writer:
            writer.write_table(meta_table)

    return betas_path


def test_missing_root_raises_actionable_error(tmp_path):
    config = CpGCorpusConfig(root=None)
    with pytest.raises(CpGCorpusNotFoundError, match="Set data.root"):
        CpGCorpusDataset(config)

    config2 = CpGCorpusConfig(root=str(tmp_path / "does_not_exist"))
    with pytest.raises(CpGCorpusNotFoundError):
        CpGCorpusDataset(config2)


def test_real_schema_without_mapping_raises_not_implemented(tmp_path):
    _make_corpus(tmp_path)
    config = CpGCorpusConfig(root=str(tmp_path))
    ds = CpGCorpusDataset(config, split="all", column_mapping=None)
    with pytest.raises(NotImplementedError, match="schema not yet confirmed"):
        len(ds)


def test_synthetic_fixture_loads_and_filters_nan(tmp_path):
    _make_corpus(tmp_path)
    config = CpGCorpusConfig(root=str(tmp_path), min_cpgs_per_sample=1)
    ds = CpGCorpusDataset(config, split="all", column_mapping=COLUMN_MAPPING)

    samples = [ds[i] for i in range(len(ds))]
    by_id = {s.sample_id: s for s in samples}

    assert "sampleA" in by_id
    sample_a = by_id["sampleA"]
    assert sample_a.dataset_id == "GSE_TEST"
    assert sample_a.platform == "GPL_TEST"
    # sampleA rows: betas [0.1, 0.5, nan, 0.4, 0.6] -> nan filtered -> 4 remain
    assert sample_a.beta_values.numel() == 4
    assert bool((sample_a.beta_values == sample_a.beta_values).all())  # no NaNs

    sample_b = by_id["sampleB"]
    assert sample_b.beta_values.numel() == 3


def test_train_val_split_determinism(tmp_path):
    for i in range(10):
        _make_corpus(tmp_path, gse_id=f"GSE_{i}", gpl_id="GPL_TEST")

    config = CpGCorpusConfig(root=str(tmp_path), split_seed=1337, min_cpgs_per_sample=1)
    train_ds_1 = CpGCorpusDataset(config, split="train", column_mapping=COLUMN_MAPPING)
    val_ds_1 = CpGCorpusDataset(config, split="val", column_mapping=COLUMN_MAPPING)
    train_ds_2 = CpGCorpusDataset(config, split="train", column_mapping=COLUMN_MAPPING)

    train_gses_1 = {e.gse_id for e in train_ds_1.datasets}
    val_gses_1 = {e.gse_id for e in val_ds_1.datasets}
    train_gses_2 = {e.gse_id for e in train_ds_2.datasets}

    assert train_gses_1 == train_gses_2  # deterministic given same seed
    assert train_gses_1.isdisjoint(val_gses_1)

    all_ds = CpGCorpusDataset(config, split="all", column_mapping=COLUMN_MAPPING)
    all_gses = {e.gse_id for e in all_ds.datasets}
    assert train_gses_1 | val_gses_1 == all_gses

    different_seed_config = CpGCorpusConfig(root=str(tmp_path), split_seed=99, min_cpgs_per_sample=1)
    train_ds_other_seed = CpGCorpusDataset(
        different_seed_config, split="train", column_mapping=COLUMN_MAPPING
    )
    # not asserting inequality (could coincidentally match), just that it runs
    assert isinstance({e.gse_id for e in train_ds_other_seed.datasets}, set)

import pyarrow as pa
import pyarrow.ipc as pa_ipc

from methylfm.data.audit import run_audit, render_report
from methylfm.data.cpgcorpus import CpGCorpusConfig

COLUMN_MAPPING = {"sample_id": "sample_id", "probe_id": "probe_id", "beta": "beta"}


def _write_shard(path, sample_ids, probe_ids, betas):
    table = pa.table({"sample_id": sample_ids, "probe_id": probe_ids, "beta": betas})
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        with pa_ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def _build_fixture_corpus(root):
    # Dataset A: two unique samples, one with an out-of-range (malformed) beta.
    _write_shard(
        root / "GSE_A" / "GPL_X" / "betas" / "QCDPB.arrow",
        sample_ids=["s1", "s1", "shared_sample", "shared_sample", "bad_sample", "bad_sample"],
        probe_ids=["cg00000001", "cg00000002", "cg00000001", "cg00000002", "cg00000001", "cg00000002"],
        betas=[0.1, 0.2, 0.3, float("nan"), 1.7, 0.5],
    )
    # Dataset B: reuses "shared_sample" id (duplicate across datasets) plus
    # one unique sample.
    _write_shard(
        root / "GSE_B" / "GPL_Y" / "betas" / "QCDPB.arrow",
        sample_ids=["shared_sample", "shared_sample", "s2", "s2", "s2"],
        probe_ids=["cg00000001", "cg00000002", "cg00000001", "cg00000002", "cg00000003"],
        betas=[0.3, 0.4, 0.6, 0.6, 0.1],
    )


def test_run_audit_on_synthetic_fixture(tmp_path):
    _build_fixture_corpus(tmp_path)
    config = CpGCorpusConfig(root=str(tmp_path), min_cpgs_per_sample=1)

    report = run_audit(config, column_mapping=COLUMN_MAPPING)

    assert report.blockers == []
    assert report.num_datasets == 2
    assert report.platform_distribution == {"GPL_X": 1, "GPL_Y": 1}
    assert set(report.samples_per_dataset.keys()) == {"GSE_A", "GSE_B"}

    assert report.duplicate_sample_ids == ["shared_sample"]

    assert len(report.malformed_samples) == 1
    assert report.malformed_samples[0]["sample_id"] == "bad_sample"

    assert report.beta_min >= 0.0
    assert report.beta_max <= 1.71  # malformed sample's out-of-range value is still observed as a max

    assert report.cpg_count_distribution["max"] is not None

    text = render_report(report)
    assert "CpGCorpus Audit Report" in text
    assert "bad_sample" in text


def test_run_audit_handles_missing_root_gracefully(tmp_path):
    config = CpGCorpusConfig(root=str(tmp_path / "missing"))
    report = run_audit(config, column_mapping=COLUMN_MAPPING)
    assert report.num_datasets == 0
    assert len(report.blockers) == 1
    assert "Set data.root" in report.blockers[0]


def test_run_audit_real_schema_unresolved_is_a_blocker_not_a_crash(tmp_path):
    _build_fixture_corpus(tmp_path)
    config = CpGCorpusConfig(root=str(tmp_path), min_cpgs_per_sample=1)

    report = run_audit(config, column_mapping=None)

    assert report.num_datasets == 2  # directory-level stats still populated
    assert report.platform_distribution == {"GPL_X": 1, "GPL_Y": 1}
    assert any("schema not yet confirmed" in b for b in report.blockers)
    # row-level stats were skipped, not fabricated
    assert report.samples_per_dataset == {}

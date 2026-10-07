import json

import numpy as np
import pytest

from methylfm.cpg_embeddings.audit import CoordinateAuditError, require_passed_audit, run_coordinate_audit
from methylfm.cpg_embeddings.config import load_cpg_encoder_config
from methylfm.cpg_embeddings.manifest import build_universe

from .conftest import write_manifest


def rows(spec, platform_suffix=""):
    return [(c, p, p + 2, f"{name}{platform_suffix}", "CG") for name, c, p in spec]


def make_cfg(tmp_path, fa, manifests, **audit):
    import yaml

    cfg = {
        "reference_fasta": str(fa),
        "manifests": {k: str(v) for k, v in manifests.items()},
        "output_dir": str(tmp_path / "out"),
        "audit": {"max_reference_mismatch_fraction": 1e-9, "max_excluded_fraction": 0.5, **audit},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return load_cpg_encoder_config(p)


def test_manifest_statuses_aliases_and_conflicts(tmp_path):
    a, b = tmp_path / "a.tsv.gz", tmp_path / "b.tsv.gz"
    write_manifest(a, [
        ("chr1", 100, 102, "cg1", "CG"),
        ("chr1", 1500, 1502, "cg2", "CG"),
        ("chr1", 3000, 3002, "cg3", "CG"),     # conflicts with b
        ("NA", "NA", "NA", "rs1", "NA"),       # no coordinate
        ("chr1", 50, 52, "ch.1.1", "CH"),      # non-CpG probe
        ("chr1", 60, 99, "cg_badgeo", "CG"),   # end != beg + 2
    ])
    write_manifest(b, [
        ("chr1", 100, 102, "cg1", "CG"),                 # same id, same locus: fine
        ("chr1", 3010, 3012, "cg3", "CG"),               # disagrees
        ("chr1", 5990, 5992, "cg9_TC21", "CG"),          # replicate-suffixed
        ("chr1", 5990, 5992, "cg9_TC22", "CG"),
        ("chr2", 10, 12, "cg7_TC21", "CG"),
        ("chr2", 2500, 2502, "cg7_TC22", "CG"),          # replicates disagree -> no alias
    ])
    u = build_universe({"A": a, "B": b})
    st = dict(zip(u.table["cpg_id"].to_pylist(), u.table["status"].to_pylist()))
    assert st["cg1"] == "ok" and st["cg2"] == "ok"
    assert st["cg3"] == "conflict"
    assert st["rs1"] == "no_coordinate"
    assert st["ch.1.1"] == "non_cpg"
    assert st["cg_badgeo"] == "bad_geometry"
    assert st["cg9"] == "ok"  # alias, replicates agree
    assert "cg7" not in st  # replicates disagree: no alias
    alias = dict(zip(u.table["cpg_id"].to_pylist(), u.table["is_alias"].to_pylist()))
    assert alias["cg9"] and not alias["cg9_TC21"]
    pos = dict(zip(u.table["cpg_id"].to_pylist(), u.table["position"].to_pylist()))
    assert pos["cg1"] == 100 and pos["cg9"] == 5990  # 0-based C, as in the manifest
    assert u.stats["aliases_added"] == 1 and u.stats["alias_bases_skipped_ambiguous"] == 1


def test_audit_passes_on_consistent_build(tmp_path, genome):
    fa, cpgs = genome
    m = tmp_path / "m.tsv.gz"
    spec = [(f"cg{c}_{p}", c, p) for c, ps in cpgs.items() for p in ps]
    write_manifest(m, rows(spec))
    rep = run_coordinate_audit(make_cfg(tmp_path, fa, {"X": m}))
    assert rep["passed"], rep["problems"]
    assert rep["valid_cpg_ids"] == len(spec) and rep["reference_mismatches"] == 0
    assert rep["unique_valid_loci"] == len(spec)
    assert (tmp_path / "out" / "coordinate_audit.json").exists()
    require_passed_audit(make_cfg(tmp_path, fa, {"X": m}))  # does not raise


def test_audit_detects_one_base_shift_and_names_it(tmp_path, genome):
    fa, cpgs = genome
    m = tmp_path / "m.tsv.gz"
    # a classic bug: 1-based coordinates read as 0-based -> every CpG is off by one
    spec = [(f"cg{c}_{p}", c, p + 1) for c, ps in cpgs.items() for p in ps]
    write_manifest(m, rows(spec))
    cfg = make_cfg(tmp_path, fa, {"X": m}, max_reference_mismatch_fraction=1e-4)
    rep = run_coordinate_audit(cfg)
    assert not rep["passed"]
    assert rep["reference_mismatch_fraction"] > 0.9
    assert rep["mismatch_shift_diagnostics"]["cg_at_shift_-1"] >= len(spec) - 1
    with pytest.raises(CoordinateAuditError):
        require_passed_audit(cfg)


def test_audit_reports_duplicate_loci_missing_chromosome_and_edges(tmp_path, genome):
    fa, cpgs = genome
    m = tmp_path / "m.tsv.gz"
    write_manifest(m, [
        ("chr1", 100, 102, "cgA", "CG"),
        ("chr1", 100, 102, "cgA_TC21", "CG"),   # second id, same locus
        ("chr1", 1500, 1502, "cgB", "CG"),
        ("chr7", 10, 12, "cgC", "CG"),          # chromosome not in FASTA
        ("chr1", 5999, 6001, "cgD", "CG"),      # runs off the end of chr1 (length 6000)
    ])
    rep = run_coordinate_audit(make_cfg(tmp_path, fa, {"X": m}, max_excluded_fraction=1.0,
                                        max_reference_mismatch_fraction=1.0))
    assert rep["missing_chromosomes"] == ["chr7"]
    assert rep["duplicate_genomic_loci"] == 1
    assert rep["multiple_ids_same_locus"]["max_ids_per_locus"] >= 2
    assert not rep["passed"] or rep["missing_chromosomes"]  # missing chromosomes always a problem
    assert any("chr7" in p for p in rep["problems"])


def test_require_passed_audit_detects_changed_inputs(tmp_path, genome):
    fa, cpgs = genome
    m = tmp_path / "m.tsv.gz"
    write_manifest(m, rows([(f"cg{p}", "chr1", p) for p in cpgs["chr1"]]))
    cfg = make_cfg(tmp_path, fa, {"X": m})
    assert run_coordinate_audit(cfg)["passed"]
    write_manifest(m, rows([(f"cg{p}", "chr1", p) for p in cpgs["chr1"][:2]]))  # manifest edited
    with pytest.raises(CoordinateAuditError, match="changed"):
        require_passed_audit(cfg)

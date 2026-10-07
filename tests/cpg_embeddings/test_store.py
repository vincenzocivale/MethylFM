import json

import numpy as np
import pyarrow as pa
import pytest

from methylfm.cpg_embeddings.coordinates import TilingSpec
from methylfm.cpg_embeddings.store import (
    CpGEmbeddingStore, StoreIncompatibleError, StoreIncompleteError, StoreWriter,
)

SPEC = TilingSpec(4096, 2048)
LENGTHS = {"chr1": 20000, "chr2": 9000}


def universe(order_seed=0):
    rows = [
        # id, chrom, pos, alias, valid
        ("cgA", "chr1", 100, False, True),
        ("cgA_TC21", "chr1", 100, False, True),   # same locus as cgA
        ("cgA_TC22", "chr1", 100, False, True),
        ("cgB", "chr1", 5000, False, True),
        ("cgC", "chr1", 5001 + 1000, False, True),
        ("cgD", "chr2", 40, False, True),
        ("cgE", "chr2", 8000, False, True),
        ("cgBAD", "chr1", 7, False, False),       # failed the reference check
        ("cgNOCOORD", None, None, False, False),
    ]
    rng = np.random.default_rng(order_seed)
    rng.shuffle(rows)
    return pa.table({
        "cpg_id": [r[0] for r in rows], "chromosome": [r[1] for r in rows],
        "position": pa.array([r[2] for r in rows], pa.int64()),
        "is_alias": [r[3] for r in rows], "reference_valid": [r[4] for r in rows],
    })


def make(tmp_path, name="s", order_seed=0, d=(8, 16), dtype="float16"):
    return StoreWriter.create(
        tmp_path / name, universe=universe(order_seed), chrom_lengths=LENGTHS, spec=SPEC, dims=d,
        dtype=dtype, provenance={"source_model": "test", "reference_genome": "T"},
    )


def fill_all(w, d=(8, 16)):
    rows_of = w.window_rows()
    for wid, rows in rows_of.items():
        w.write(rows, np.tile(rows[:, None].astype(np.float32) + 1, (1, d[0])),
                np.tile(rows[:, None].astype(np.float32) + 1, (1, d[1])) * 2)
    w.mark_done(list(rows_of))


def test_rows_are_per_locus_sorted_and_ids_share_rows(tmp_path):
    w = make(tmp_path)
    fill_all(w)
    w.finalize()
    s = CpGEmbeddingStore.open(tmp_path / "s")
    assert s.num_loci == 5  # cgA/_TC21/_TC22 collapse into one locus; invalid ids have none
    rows = s.encode(["cgA", "cgA_TC21", "cgA_TC22", "cgB", "cgC", "cgD", "cgE", "cgBAD", "cgNOCOORD", "nope"])
    assert rows[0] == rows[1] == rows[2] == 0
    assert rows[3:7].tolist() == [1, 2, 3, 4]  # (chr1,100) < (chr1,5000) < (chr1,6001) < (chr2,40) < (chr2,8000)
    assert (rows[7:] == -1).all()  # invalid / unknown ids are never given an embedding
    loci = s.loci().to_pydict()
    assert list(zip(loci["chromosome"], loci["position"])) == [
        ("chr1", 100), ("chr1", 5000), ("chr1", 6001), ("chr2", 40), ("chr2", 8000)]


def test_row_order_is_deterministic_regardless_of_input_order(tmp_path):
    a, b = make(tmp_path, "a", 0), make(tmp_path, "b", 123)
    for w in (a, b):
        fill_all(w); w.finalize()
    sa, sb = (CpGEmbeddingStore.open(tmp_path / n) for n in "ab")
    assert sa.loci().equals(sb.loci())
    assert sa.index().equals(sb.index())
    assert sa.asset_id == sb.asset_id


def test_lookup_returns_exactly_the_requested_rows_from_a_memmap(tmp_path):
    w = make(tmp_path); fill_all(w); w.finalize()
    s = CpGEmbeddingStore.open(tmp_path / "s")
    assert isinstance(s._m1, np.memmap) and isinstance(s._m128, np.memmap)  # never loaded whole
    ids = np.array([[4, 0], [1, 1]])
    h1, h128 = s.lookup(ids)
    assert h1.shape == (2, 2, 8) and h128.shape == (2, 2, 16)
    assert (h1[..., 0] == ids + 1).all() and (h128[..., 0] == (ids + 1) * 2).all()
    with pytest.raises(IndexError):
        s.lookup(np.array([5]))


def test_resume_does_not_recompute_completed_windows(tmp_path):
    w = make(tmp_path)
    rows_of = w.window_rows()
    first = sorted(rows_of)[0]
    w.write(rows_of[first], np.ones((len(rows_of[first]), 8)), np.ones((len(rows_of[first]), 16)))
    w.mark_done([first])
    r = StoreWriter.resume(tmp_path / "s")
    assert first not in r.pending_window_ids() and len(r.pending_window_ids()) == len(rows_of) - 1
    with pytest.raises(StoreIncompleteError):
        r.finalize()
    with pytest.raises(StoreIncompleteError):
        CpGEmbeddingStore.open(tmp_path / "s")  # refuses partial stores


def test_nonfinite_and_overflowing_rows_are_rejected(tmp_path):
    w = make(tmp_path)
    rows = np.array([0])
    with pytest.raises(ValueError):
        w.write(rows, np.full((1, 8), np.nan), np.zeros((1, 16)))
    with pytest.raises(OverflowError):
        w.write(rows, np.full((1, 8), 1e6), np.zeros((1, 16)))


def test_manifest_incompatibility_is_detected(tmp_path):
    w = make(tmp_path); fill_all(w); w.finalize()
    s = CpGEmbeddingStore.open(tmp_path / "s")
    s.check_compatible(dim_1bp=8, dim_128bp=16, asset_id=s.asset_id)
    with pytest.raises(StoreIncompatibleError, match="dims"):
        s.check_compatible(dim_1bp=1536, dim_128bp=3072)
    with pytest.raises(StoreIncompatibleError, match="asset id"):
        s.check_compatible(dim_1bp=8, dim_128bp=16, asset_id="ag-other")
    # tampered index table
    (tmp_path / "s" / "cpg_index.parquet").write_bytes(b"corrupted")
    with pytest.raises(StoreIncompatibleError, match="checksum"):
        CpGEmbeddingStore.open(tmp_path / "s")


def test_unsupported_format_version(tmp_path):
    w = make(tmp_path); fill_all(w); w.finalize()
    m = json.loads((tmp_path / "s" / "manifest.json").read_text())
    m["format_version"] = 99
    (tmp_path / "s" / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(StoreIncompatibleError, match="format_version"):
        CpGEmbeddingStore.open(tmp_path / "s")


def test_window_assignment_stored_in_index(tmp_path):
    w = make(tmp_path); fill_all(w); w.finalize()
    s = CpGEmbeddingStore.open(tmp_path / "s")
    loci = s.loci().to_pydict()
    # chr1:100 is within `margin` of the chromosome start -> edge; chr1:5000 owned by core 2 is interior
    assert loci["edge_flag"][0] is True and loci["edge_flag"][1] is False
    assert all(loci["bp_index"][i] == loci["position"][i] - loci["window_start"][i] for i in range(5))
    idx = s.index().to_pydict()
    assert idx["row_index"][idx["cpg_id"].index("cgBAD")] == -1
    assert idx["window_id"][idx["cpg_id"].index("cgBAD")] == -1

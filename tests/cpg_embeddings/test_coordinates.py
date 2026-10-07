"""Critical: coordinate bugs would silently invalidate the whole embedding store."""
import numpy as np
import pytest

from methylfm.cpg_embeddings import coordinates as co

SPEC = co.TilingSpec(window_size=4096, core_size=2048)  # margin 1024
BIG = co.TilingSpec(window_size=1 << 20, core_size=1 << 19)


def test_spec_validation():
    assert SPEC.margin == 1024 and SPEC.num_bins == 32
    assert BIG.margin == 1 << 18
    with pytest.raises(ValueError):
        co.TilingSpec(window_size=4096, core_size=4097)
    with pytest.raises(ValueError):
        co.TilingSpec(window_size=4096, core_size=2000)  # not bin aligned
    with pytest.raises(ValueError):
        co.TilingSpec(window_size=4224, core_size=4096)  # margin 64 not bin aligned
    with pytest.raises(ValueError):
        co.TilingSpec(window_size=4096 + 1, core_size=2048)  # asymmetric


def test_zero_one_based_roundtrip():
    assert co.one_based_to_zero_based(1) == 0
    assert co.zero_based_to_one_based(0) == 1
    p = np.array([0, 5, 10**8])
    assert (co.zero_based_to_one_based(co.one_based_to_zero_based(p)) == p).all()


def test_core_ownership_boundaries():
    assert co.core_index(0, SPEC) == 0
    assert co.core_index(2047, SPEC) == 0
    assert co.core_index(2048, SPEC) == 1


def test_unclamped_window_geometry():
    L = 100 * SPEC.core_size
    k = 10
    start = co.window_start(k, L, SPEC)
    assert start == k * 2048 - 1024
    first, last = k * 2048, (k + 1) * 2048 - 1
    # first base of the core sits exactly `margin` bases into the window
    assert co.bp_index(first, start) == SPEC.margin
    assert co.bp_index(last, start) == SPEC.margin + SPEC.core_size - 1
    # 128-bp bin: same geometry, no cropping/padding offset
    assert co.bin_index(first, start, SPEC) == SPEC.margin // 128
    assert co.bin_index(last, start, SPEC) == (SPEC.margin + SPEC.core_size - 1) // 128
    assert not co.edge_flag(np.array([first, last]), start, L, SPEC).any()


def test_bin_and_bp_index_consistent_with_explicit_output_arrays():
    """Emulate the model outputs: the 1-bp output at index i describes base start+i; the
    128-bp output at bin j describes bases [start+128j, start+128j+128)."""
    rng = np.random.default_rng(0)
    L = 50 * SPEC.core_size + 777
    pos = rng.integers(0, L - 2, 5000)
    for p in pos:
        a = co.assign_windows(np.array([p]), L, SPEC)
        s = int(a["start"][0])
        out_1bp_base = s + np.arange(SPEC.window_size)
        out_128_lo = s + SPEC.bin_size * np.arange(SPEC.num_bins)
        assert out_1bp_base[a["bp"][0]] == p
        assert out_128_lo[a["bin"][0]] <= p < out_128_lo[a["bin"][0]] + SPEC.bin_size
        assert 0 <= a["bp"][0] and a["bp"][0] + 1 < SPEC.window_size  # the G is inside too


def test_chromosome_start_clamped_and_flagged():
    L = 20 * SPEC.core_size
    a = co.assign_windows(np.array([0, 10, 1023, 1024, 2047]), L, SPEC)
    assert (a["start"] == 0).all()
    assert (a["bp"] == [0, 10, 1023, 1024, 2047]).all()
    assert a["edge"].tolist() == [True, True, True, False, False]


def test_chromosome_end_clamped_and_flagged():
    L = 7 * SPEC.core_size + 500  # last core is partial
    last = L - 3  # position of a C near the very end (G at L-2)
    a = co.assign_windows(np.array([last, L - 1500]), L, SPEC)
    assert (a["start"] == L - SPEC.window_size).all()
    assert a["bp"][0] == last - (L - SPEC.window_size) and a["bp"][0] + 1 < SPEC.window_size
    assert a["edge"][0]  # fewer than `margin` bases to the right


def test_chromosome_shorter_than_window():
    L = 3000
    a = co.assign_windows(np.array([5, 2990]), L, SPEC)
    assert (a["start"] == 0).all() and (a["bp"] == [5, 2990]).all()
    assert a["edge"].all()


def test_every_position_has_exactly_one_owner_and_lies_inside_it():
    rng = np.random.default_rng(1)
    L = 30 * SPEC.core_size + 123
    pos = np.unique(rng.integers(0, L - 1, 20000))
    a = co.assign_windows(pos, L, SPEC)
    for k in np.unique(a["core"]):
        sel = a["core"] == k
        assert pos[sel].min() >= k * SPEC.core_size and pos[sel].max() < (k + 1) * SPEC.core_size
        assert len(np.unique(a["start"][sel])) == 1
    # interior CpGs always have at least `margin` bases of context on both sides
    interior = ~a["edge"]
    assert (a["bp"][interior] >= SPEC.margin).all()
    assert (SPEC.window_size - a["bp"][interior] > SPEC.margin).all()


def test_realistic_alphagenome_geometry():
    L = 248_956_422  # chr1
    p = np.array([10_000_000, 123_456_789, L - 5])
    a = co.assign_windows(p, L, BIG)
    assert (a["bp"] < BIG.window_size).all() and (a["bin"] < BIG.num_bins).all()
    assert a["edge"].tolist() == [False, False, True]


def test_chromosome_order():
    names = ["chrX", "chr10", "chr2", "chrM", "chr1", "chrY", "chrUn_x"]
    assert sorted(names, key=co.chromosome_sort_key) == ["chr1", "chr2", "chr10", "chrX", "chrY", "chrM", "chrUn_x"]

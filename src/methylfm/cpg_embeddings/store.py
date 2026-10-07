"""Persistent, memory-mappable store of frozen AlphaGenome CpG embeddings.

Layout of a store directory::

    manifest.json            provenance + geometry + integrity (see below)
    cpg_index.parquet        one row per CpG id (explicit ids and aliases)
    loci.parquet             one row per unique genomic locus (= embedding row)
    alphagenome_1bp.npy      float16 [num_loci, D1]     (.npy -> np.load(mmap_mode="r"))
    alphagenome_128bp.npy    float16 [num_loci, D128]
    progress.json            window ids already extracted (resume)

* ``row_index`` is the row of the two matrices. Several CpG ids can share a
  row (EPICv2 replicate probes, aliases): embeddings are per *locus*. The
  integer that MethylFM models consume as ``cpg_id`` is this ``row_index``;
  ids with no embedding (invalid coordinate, missing from the store) map to
  ``-1`` and are dropped by the data layer, never silently hashed.
* Rows are ordered by (chromosome in natural order, position); this makes the
  row order a pure function of the CpG universe (deterministic) and keeps
  neighbouring loci adjacent on disk.
* Matrices are plain ``.npy`` files: trivially memory-mappable, no extra
  dependency, batched fancy-index reads touch only the requested pages.

No JAX, no torch in this module.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from methylfm.cpg_embeddings.coordinates import TilingSpec, assign_windows
from methylfm.cpg_embeddings.manifest import sort_loci

FORMAT_VERSION = 1
MANIFEST = "manifest.json"
INDEX = "cpg_index.parquet"
LOCI = "loci.parquet"
F_1BP = "alphagenome_1bp.npy"
F_128BP = "alphagenome_128bp.npy"
PROGRESS = "progress.json"
_FINGERPRINT_ROWS = 4096


class StoreIncompatibleError(RuntimeError):
    """The store does not match what the caller expects (dims, asset id, ...)."""


class StoreIncompleteError(RuntimeError):
    """Extraction has not finished; refusing to use a partial store."""


def _sha256_bytes(*chunks: bytes) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- build


@dataclass
class LociTable:
    """Loci + their window geometry, in row order."""

    chromosome: np.ndarray  # object [N]
    position: np.ndarray  # int64 [N]
    core: np.ndarray
    window_start: np.ndarray
    bp: np.ndarray
    bin: np.ndarray
    edge: np.ndarray
    window_id: np.ndarray  # int64 [N]
    windows: list[tuple[str, int]]  # window_id -> (chromosome, core)


def build_loci(chromosomes, positions, chrom_lengths: dict[str, int], spec: TilingSpec) -> LociTable:
    """Unique loci sorted by (chromosome, position) with window assignment."""
    chromosomes = np.asarray(chromosomes, dtype=object)
    positions = np.asarray(positions, dtype=np.int64)
    pairs = np.unique(np.rec.fromarrays([chromosomes.astype(str), positions]))
    chrom_u = np.asarray(pairs.f0, dtype=object)
    pos_u = np.asarray(pairs.f1, dtype=np.int64)
    order = sort_loci(chrom_u, pos_u)
    chrom_u, pos_u = chrom_u[order], pos_u[order]

    n = len(pos_u)
    core = np.zeros(n, np.int64)
    start = np.zeros(n, np.int64)
    bp = np.zeros(n, np.int64)
    binx = np.zeros(n, np.int64)
    edge = np.zeros(n, bool)
    for c in dict.fromkeys(chrom_u.tolist()):
        sel = chrom_u == c
        a = assign_windows(pos_u[sel], chrom_lengths[c], spec)
        core[sel], start[sel], bp[sel], binx[sel], edge[sel] = a["core"], a["start"], a["bp"], a["bin"], a["edge"]

    # window ids in (chromosome order, core) order == first-appearance order
    keys = list(zip(chrom_u.tolist(), core.tolist()))
    windows = list(dict.fromkeys(keys))
    wid_of = {k: i for i, k in enumerate(windows)}
    window_id = np.fromiter((wid_of[k] for k in keys), dtype=np.int64, count=n)
    return LociTable(chrom_u, pos_u, core, start, bp, binx, edge, window_id, windows)


class StoreWriter:
    """Creates a store skeleton (index + empty memmaps) and fills it window by
    window with durable progress, so an interrupted extraction can resume."""

    def __init__(self, store_dir: str | Path):
        self.dir = Path(store_dir)
        self._m1: np.ndarray | None = None
        self._m128: np.ndarray | None = None
        self.manifest: dict = {}
        self.loci: LociTable | None = None
        self._done: set[int] = set()

    # -- create / reopen -----------------------------------------------------
    @classmethod
    def create(
        cls,
        store_dir,
        *,
        universe: pa.Table,
        chrom_lengths: dict[str, int],
        spec: TilingSpec,
        dims: tuple[int, int],
        dtype: str,
        provenance: dict,
    ) -> "StoreWriter":
        """``universe``: columns cpg_id, chromosome, position, is_alias,
        reference_valid (rows with reference_valid=False get row_index -1)."""
        w = cls(store_dir)
        w.dir.mkdir(parents=True, exist_ok=True)
        if (w.dir / MANIFEST).exists():
            raise FileExistsError(f"{w.dir} already contains a store; use StoreWriter.resume")

        valid = np.array(universe["reference_valid"].to_pylist(), dtype=bool)
        chrom = np.array([c or "" for c in universe["chromosome"].to_pylist()], dtype=object)
        pos = np.array([p if p is not None else -1 for p in universe["position"].to_pylist()], dtype=np.int64)
        loci = build_loci(chrom[valid], pos[valid], chrom_lengths, spec)

        # row_index per id
        row_of = {(c, int(p)): i for i, (c, p) in enumerate(zip(loci.chromosome.tolist(), loci.position.tolist()))}
        ids = universe["cpg_id"].to_pylist()
        order = np.argsort(np.array(ids, dtype=object), kind="stable")
        row = np.full(len(ids), -1, np.int64)
        wid = np.full(len(ids), -1, np.int64)
        edge = np.zeros(len(ids), bool)
        for i in np.flatnonzero(valid):
            r = row_of[(chrom[i], int(pos[i]))]
            row[i], wid[i], edge[i] = r, loci.window_id[r], loci.edge[r]
        index = pa.table(
            {
                "internal_index": pa.array(np.arange(len(ids), dtype=np.int64)),
                "cpg_id": pa.array([ids[i] for i in order], pa.string()),
                "chromosome": pa.array([chrom[i] or None for i in order], pa.string()),
                "position": pa.array([int(pos[i]) if pos[i] >= 0 else None for i in order], pa.int64()),
                "row_index": pa.array(row[order]),
                "window_id": pa.array(wid[order]),
                "edge_flag": pa.array(edge[order]),
                "reference_valid": pa.array(valid[order]),
                "is_alias": pa.array(np.array(universe["is_alias"].to_pylist(), dtype=bool)[order]),
            }
        )
        loci_t = pa.table(
            {
                "row_index": pa.array(np.arange(len(loci.position), dtype=np.int64)),
                "chromosome": pa.array(loci.chromosome.tolist(), pa.string()),
                "position": pa.array(loci.position),
                "window_id": pa.array(loci.window_id),
                "core": pa.array(loci.core),
                "window_start": pa.array(loci.window_start),
                "bp_index": pa.array(loci.bp),
                "bin_index": pa.array(loci.bin),
                "edge_flag": pa.array(loci.edge),
            }
        )
        pq.write_table(index, w.dir / INDEX)
        pq.write_table(loci_t, w.dir / LOCI)

        n = len(loci.position)
        np.lib.format.open_memmap(w.dir / F_1BP, mode="w+", dtype=dtype, shape=(n, dims[0])).flush()
        np.lib.format.open_memmap(w.dir / F_128BP, mode="w+", dtype=dtype, shape=(n, dims[1])).flush()

        w.manifest = {
            **provenance,
            "format_version": FORMAT_VERSION,
            "complete": False,
            "storage_dtype": dtype,
            "embedding_1bp_dim": dims[0],
            "embedding_128bp_dim": dims[1],
            "num_cpgs": int(valid.sum()),
            "num_cpg_ids": len(ids),
            "num_loci": n,
            "num_windows": len(loci.windows),
            "sequence_length": spec.window_size,
            "core_region_size": spec.core_size,
            "window_stride": spec.core_size,
            "chromosomes": list(dict.fromkeys(loci.chromosome.tolist())),
            "windows": [[c, int(k)] for c, k in loci.windows],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cpg_index_sha256": _file_sha256(w.dir / INDEX),
            "loci_sha256": _file_sha256(w.dir / LOCI),
        }
        _atomic_write_json(w.dir / MANIFEST, w.manifest)
        _atomic_write_json(w.dir / PROGRESS, {"completed_windows": []})
        w.loci = loci
        return w

    @classmethod
    def resume(cls, store_dir) -> "StoreWriter":
        w = cls(store_dir)
        w.manifest = json.loads((w.dir / MANIFEST).read_text())
        w._done = set(json.loads((w.dir / PROGRESS).read_text())["completed_windows"])
        return w

    # -- fill ------------------------------------------------------------------
    def _open(self) -> None:
        if self._m1 is None:
            self._m1 = np.load(self.dir / F_1BP, mmap_mode="r+")
            self._m128 = np.load(self.dir / F_128BP, mmap_mode="r+")

    def window_rows(self) -> dict[int, np.ndarray]:
        """window_id -> sorted row indices it owns."""
        t = pq.read_table(self.dir / LOCI, columns=["row_index", "window_id"])
        rows = t["row_index"].to_numpy()
        wids = t["window_id"].to_numpy()
        order = np.argsort(wids, kind="stable")
        rows, wids = rows[order], wids[order]
        cuts = np.flatnonzero(np.diff(wids)) + 1
        return {int(w[0]): r for r, w in zip(np.split(rows, cuts), np.split(wids, cuts))}

    def pending_window_ids(self) -> list[int]:
        return [w for w in range(self.manifest["num_windows"]) if w not in self._done]

    def write(self, rows: np.ndarray, h1: np.ndarray, h128: np.ndarray) -> None:
        self._open()
        dt = np.dtype(self.manifest["storage_dtype"])
        for name, h, m in (("1bp", h1, self._m1), ("128bp", h128, self._m128)):
            if not np.isfinite(h).all():
                raise ValueError(f"non-finite {name} embeddings; refusing to store")
            if dt == np.float16 and np.abs(h).max() > np.finfo(np.float16).max:
                raise OverflowError(f"{name} embeddings overflow float16 (max |h|={np.abs(h).max():.4g})")
            m[rows] = h.astype(dt)

    def mark_done(self, window_ids: list[int]) -> None:
        """Flush data to disk *then* record progress, so progress never runs ahead of data."""
        self._open()
        self._m1.flush()
        self._m128.flush()
        self._done.update(int(w) for w in window_ids)
        _atomic_write_json(self.dir / PROGRESS, {"completed_windows": sorted(self._done)})

    def finalize(self, extra: dict | None = None) -> dict:
        if self.pending_window_ids():
            raise StoreIncompleteError(f"{len(self.pending_window_ids())} windows still pending")
        self._open()
        n = self.manifest["num_loci"]
        probe = np.unique(np.linspace(0, max(0, n - 1), min(n, _FINGERPRINT_ROWS)).astype(np.int64))
        fp = _sha256_bytes(np.ascontiguousarray(self._m1[probe]).tobytes(), np.ascontiguousarray(self._m128[probe]).tobytes())
        self.manifest.update(extra or {})
        self.manifest["data_fingerprint_sha256"] = fp
        self.manifest["complete"] = True
        self.manifest["store_id"] = _compute_store_id(self.manifest)
        _atomic_write_json(self.dir / MANIFEST, self.manifest)
        return self.manifest


_ID_EXCLUDE = {"created_at", "store_id", "complete", "extraction_throughput", "validation"}


def _compute_store_id(manifest: dict) -> str:
    core = {k: v for k, v in sorted(manifest.items()) if k not in _ID_EXCLUDE}
    return "ag-" + _sha256_bytes(json.dumps(core, sort_keys=True, default=str).encode())[:20]


# --------------------------------------------------------------------------- read


class CpGEmbeddingStore:
    """Read-only, memory-mapped view of a finished store."""

    def __init__(self, path: str | Path, manifest: dict, m1: np.ndarray, m128: np.ndarray):
        self.path = Path(path)
        self.manifest = manifest
        self._m1, self._m128 = m1, m128
        self._vocab: tuple[np.ndarray, np.ndarray] | None = None

    @classmethod
    def open(cls, path: str | Path, *, require_complete: bool = True) -> "CpGEmbeddingStore":
        path = Path(path)
        if not (path / MANIFEST).exists():
            raise FileNotFoundError(f"no CpG embedding store at {path} (missing {MANIFEST})")
        manifest = json.loads((path / MANIFEST).read_text())
        if manifest.get("format_version") != FORMAT_VERSION:
            raise StoreIncompatibleError(
                f"store format_version={manifest.get('format_version')} != supported {FORMAT_VERSION}"
            )
        if require_complete and not manifest.get("complete"):
            raise StoreIncompleteError(f"store at {path} is not complete (extraction unfinished)")
        m1 = np.load(path / F_1BP, mmap_mode="r")
        m128 = np.load(path / F_128BP, mmap_mode="r")
        n = manifest["num_loci"]
        if m1.shape != (n, manifest["embedding_1bp_dim"]) or m128.shape != (n, manifest["embedding_128bp_dim"]):
            raise StoreIncompatibleError(
                f"matrix shapes {m1.shape}/{m128.shape} disagree with manifest "
                f"({n}, {manifest['embedding_1bp_dim']})/({n}, {manifest['embedding_128bp_dim']})"
            )
        if str(m1.dtype) != manifest["storage_dtype"] or str(m128.dtype) != manifest["storage_dtype"]:
            raise StoreIncompatibleError("matrix dtype disagrees with manifest storage_dtype")
        if require_complete:
            # Cheap, always-on integrity check of the small tables.
            for fname, key in ((INDEX, "cpg_index_sha256"), (LOCI, "loci_sha256")):
                if _file_sha256(path / fname) != manifest[key]:
                    raise StoreIncompatibleError(f"{fname} does not match the manifest checksum")
        return cls(path, manifest, m1, m128)

    # -- metadata ----------------------------------------------------------------
    @property
    def asset_id(self) -> str:
        return self.manifest["store_id"]

    @property
    def num_loci(self) -> int:
        return self.manifest["num_loci"]

    @property
    def dim_1bp(self) -> int:
        return self.manifest["embedding_1bp_dim"]

    @property
    def dim_128bp(self) -> int:
        return self.manifest["embedding_128bp_dim"]

    def check_compatible(self, *, dim_1bp: int, dim_128bp: int, asset_id: str | None = None) -> None:
        if (self.dim_1bp, self.dim_128bp) != (dim_1bp, dim_128bp):
            raise StoreIncompatibleError(
                f"store has embedding dims ({self.dim_1bp}, {self.dim_128bp}) but the model config "
                f"expects ({dim_1bp}, {dim_128bp})"
            )
        if asset_id is not None and asset_id != self.asset_id:
            raise StoreIncompatibleError(
                f"store asset id {self.asset_id!r} != the {asset_id!r} this model was trained with. "
                "Loading a model with different CpG embeddings would silently change its inputs."
            )

    # -- vocabulary ----------------------------------------------------------------
    def _vocabulary(self) -> tuple[np.ndarray, np.ndarray]:
        if self._vocab is None:
            t = pq.read_table(self.path / INDEX, columns=["cpg_id", "row_index"])
            self._vocab = (np.array(t["cpg_id"].to_pylist(), dtype=str), t["row_index"].to_numpy())
        return self._vocab

    def encode(self, cpg_ids) -> np.ndarray:
        """External CpG id strings -> row indices; ``-1`` for ids without an embedding."""
        names, rows = self._vocabulary()
        q = np.asarray(list(cpg_ids), dtype=str)
        if q.size == 0:
            return np.zeros(0, np.int64)
        pos = np.searchsorted(names, q)
        pos = np.minimum(pos, len(names) - 1)
        hit = names[pos] == q
        return np.where(hit, rows[pos], -1).astype(np.int64)

    def loci(self) -> pa.Table:
        return pq.read_table(self.path / LOCI)

    def index(self) -> pa.Table:
        return pq.read_table(self.path / INDEX)

    # -- lookup -------------------------------------------------------------------------
    def lookup(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fetch only the requested rows (any shape of integer indices).
        Reads are issued in sorted order so memory-mapped pages are touched
        once; the matrices are never loaded whole."""
        rows = np.asarray(rows, dtype=np.int64)
        flat = rows.reshape(-1)
        if flat.size and (flat.min() < 0 or flat.max() >= self.num_loci):
            raise IndexError(f"row index out of range [0, {self.num_loci}): {flat.min()}..{flat.max()}")
        uniq, inv = np.unique(flat, return_inverse=True)
        h1 = np.asarray(self._m1[uniq])[inv].reshape(rows.shape + (self.dim_1bp,))
        h128 = np.asarray(self._m128[uniq])[inv].reshape(rows.shape + (self.dim_128bp,))
        return h1, h128

    def lookup_unique(self, uniq_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Like `lookup` for rows already sorted and unique (no re-dedup)."""
        return np.asarray(self._m1[uniq_rows]), np.asarray(self._m128[uniq_rows])


# --------------------------------------------------------------------------- testing helper


def write_synthetic_store(
    path: str | Path, num_loci: int, dim_1bp: int = 16, dim_128bp: int = 32, seed: int = 0,
    cpg_ids: list[str] | None = None,
) -> CpGEmbeddingStore:
    """A tiny, *random* store with the real on-disk format -- for unit tests and
    pipeline smoke runs only. Carries no biology."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = cpg_ids or [f"cg{i:08d}" for i in range(num_loci)]
    ids_sorted = sorted(ids)
    index = pa.table(
        {
            "internal_index": pa.array(np.arange(len(ids_sorted), dtype=np.int64)),
            "cpg_id": pa.array(ids_sorted),
            "chromosome": pa.array(["chr1"] * len(ids_sorted)),
            "position": pa.array(np.arange(len(ids_sorted), dtype=np.int64) * 10 + 1000),
            "row_index": pa.array(np.arange(len(ids_sorted), dtype=np.int64)),
            "window_id": pa.array(np.zeros(len(ids_sorted), np.int64)),
            "edge_flag": pa.array(np.zeros(len(ids_sorted), bool)),
            "reference_valid": pa.array(np.ones(len(ids_sorted), bool)),
            "is_alias": pa.array(np.zeros(len(ids_sorted), bool)),
        }
    )
    loci = pa.table(
        {
            "row_index": index["row_index"],
            "chromosome": index["chromosome"],
            "position": index["position"],
            "window_id": index["window_id"],
            "core": pa.array(np.zeros(len(ids_sorted), np.int64)),
            "window_start": pa.array(np.zeros(len(ids_sorted), np.int64)),
            "bp_index": index["position"],
            "bin_index": pa.array(np.zeros(len(ids_sorted), np.int64)),
            "edge_flag": index["edge_flag"],
        }
    )
    pq.write_table(index, path / INDEX)
    pq.write_table(loci, path / LOCI)
    np.save(path / F_1BP, rng.standard_normal((len(ids_sorted), dim_1bp)).astype(np.float16))
    np.save(path / F_128BP, rng.standard_normal((len(ids_sorted), dim_128bp)).astype(np.float16))
    manifest = {
        "format_version": FORMAT_VERSION, "complete": True, "synthetic": True, "storage_dtype": "float16",
        "embedding_1bp_dim": dim_1bp, "embedding_128bp_dim": dim_128bp, "num_loci": len(ids_sorted),
        "num_cpgs": len(ids_sorted), "num_cpg_ids": len(ids_sorted), "num_windows": 1,
        "cpg_index_sha256": _file_sha256(path / INDEX), "loci_sha256": _file_sha256(path / LOCI),
        "source_model": "SYNTHETIC-RANDOM", "seed": seed,
    }
    manifest["store_id"] = _compute_store_id(manifest)
    _atomic_write_json(path / MANIFEST, manifest)
    return CpGEmbeddingStore.open(path)

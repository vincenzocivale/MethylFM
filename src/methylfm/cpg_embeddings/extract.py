"""One-time, resumable extraction of frozen AlphaGenome trunk embeddings for
every CpG locus of the (audited) universe.

    reference FASTA --window--> AlphaGenome (official, frozen) --> h_1bp, h_128bp
                                                                  | rows owned by the window's core
                                                                  v
                                                       cpg_embeddings store

One AlphaGenome forward serves *all* CpGs owned by a window (hundreds), never
one forward per CpG. Windows follow the deterministic tiling in
`coordinates.py`; each CpG is written exactly once, from the window that owns
it, and embeddings from different windows are never averaged.

Runs in the separate JAX environment (see docs/CPG_ENCODER.md).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from methylfm.cpg_embeddings.audit import UNIVERSE_FILENAME, require_passed_audit
from methylfm.cpg_embeddings.coordinates import TilingSpec
from methylfm.cpg_embeddings.sequence import fetch_window
from methylfm.cpg_embeddings.store import LOCI, MANIFEST, StoreWriter


def _chrom_lengths(fasta) -> dict[str, int]:
    return {k: len(fasta[k]) for k in fasta.keys()}


def _subset_universe(universe, chromosomes):
    if not chromosomes:
        return universe
    import pyarrow as pa
    import pyarrow.compute as pc

    keep = pc.is_in(universe["chromosome"], value_set=pa.array(list(chromosomes)))
    return universe.filter(pc.fill_null(keep, False))


def _open_or_create_store(cfg, universe, chrom_lengths, spec, dims, audit, software, ckpt_info):
    store_dir = Path(cfg["store_dir"])
    provenance = {
        **software,
        **ckpt_info,
        "reference_genome": cfg["reference_genome"],
        "reference_fasta_sha256": audit["reference_fasta_sha256"],
        "cpg_manifest_sha256": audit["cpg_manifest_sha256"],
        "audit_universe_sha256": audit["universe_parquet_sha256"],
        "extraction_chromosomes": cfg["extraction"]["chromosomes"],
    }
    if (store_dir / MANIFEST).exists():
        w = StoreWriter.resume(store_dir)
        m = w.manifest
        mismatches = {
            k: (m.get(k), v)
            for k, v in {**provenance, "sequence_length": spec.window_size, "core_region_size": spec.core_size,
                         "storage_dtype": cfg["extraction"]["storage_dtype"]}.items()
            if k not in {"extraction_software_commit", "jax_version"} and m.get(k) != v
        }
        if mismatches:
            raise RuntimeError(
                f"existing store at {store_dir} was created with different settings {mismatches}; "
                "use a different store_dir or delete it. Refusing to mix extraction settings."
            )
        if m.get("complete"):
            print(f"[extract] store at {store_dir} is already complete.")
        else:
            print(f"[extract] resuming: {len(w._done)}/{m['num_windows']} windows already done")
        return w
    return StoreWriter.create(
        store_dir,
        universe=universe,
        chrom_lengths=chrom_lengths,
        spec=spec,
        dims=dims,
        dtype=cfg["extraction"]["storage_dtype"],
        provenance=provenance,
    )


def run_extraction(cfg: dict, benchmark_only: bool = False) -> int:
    import jax
    from pyfaidx import Fasta

    from methylfm.cpg_embeddings.agmodel import D_128BP, D_1BP, AlphaGenomeTrunk, gather_rows, software_provenance

    audit = require_passed_audit(cfg)  # no GPU time before the coordinates are verified
    spec = TilingSpec(cfg["tiling"]["window_size"], cfg["tiling"]["core_size"])
    out_dir = Path(cfg["output_dir"])

    universe = pq.read_table(out_dir / UNIVERSE_FILENAME)
    universe = _subset_universe(universe, cfg["extraction"]["chromosomes"])
    fasta = Fasta(cfg["reference_fasta"], as_raw=True, sequence_always_upper=True)
    lengths = _chrom_lengths(fasta)

    trunk = AlphaGenomeTrunk(cfg)
    geometry = trunk.check_geometry(spec.window_size)
    print(f"[extract] official output geometry confirmed: {geometry}")
    software = {**software_provenance(), "output_geometry": geometry}

    if benchmark_only:
        return _benchmark(cfg, universe, fasta, lengths, spec, trunk)

    writer = _open_or_create_store(
        cfg, universe, lengths, spec, (D_1BP, D_128BP), audit, software, trunk.checkpoint_info
    )
    manifest = writer.manifest
    if manifest.get("complete"):
        return 0

    loci = pq.read_table(Path(cfg["store_dir"]) / LOCI).to_pydict()
    chrom_of = np.array(loci["chromosome"], dtype=object)
    bp = np.asarray(loci["bp_index"], dtype=np.int64)
    binx = np.asarray(loci["bin_index"], dtype=np.int64)
    wstart = np.asarray(loci["window_start"], dtype=np.int64)
    windows = [(c, int(k)) for c, k in manifest["windows"]]
    rows_of = writer.window_rows()

    pending = writer.pending_window_ids()
    max_w = cfg["extraction"]["max_windows"]
    if max_w:
        pending = pending[:max_w]
    batch = cfg["extraction"]["batch_windows"]
    print(f"[extract] {len(pending)} windows to do ({manifest['num_loci']} loci, window={spec.window_size}, core={spec.core_size})")

    t_start = time.time()
    done_rows = done_bp = 0
    log = []
    for b0 in range(0, len(pending), batch):
        ids = pending[b0 : b0 + batch]
        t0 = time.time()
        onehots = []
        for w in ids:
            r0 = rows_of[w][0]
            seq = fetch_window(fasta, chrom_of[r0], int(wstart[r0]), spec.window_size)
            onehots.append(trunk.encode_sequence(seq))
        e1, e128 = trunk.embed(np.stack(onehots).astype(np.float32))
        for j, w in enumerate(ids):
            rows = rows_of[w]
            h1, h128 = gather_rows(e1, e128, bp[rows], binx[rows], batch_item=j)
            writer.write(rows, h1, h128)
            done_rows += len(rows)
        writer.mark_done(ids)
        done_bp += spec.window_size * len(ids)
        dt = time.time() - t0
        n_done = b0 + len(ids)
        if (n_done // batch) % 10 == 0 or n_done == len(pending):
            elapsed = time.time() - t_start
            stats = _device_stats(jax)
            rec = {
                "windows_done": n_done, "windows_total": len(pending), "windows_per_s": n_done / elapsed,
                "bp_per_s": done_bp / elapsed, "cpgs_per_s": done_rows / elapsed,
                "gpu_peak_gib": stats, "last_batch_s": dt,
                "stored_bytes": done_rows * (D_1BP + D_128BP) * np.dtype(manifest["storage_dtype"]).itemsize,
                "eta_min": (len(pending) - n_done) / max(1e-9, n_done / elapsed) / 60,
            }
            log.append(rec)
            print("[extract] " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}))
            (out_dir / "extraction_report.json").write_text(json.dumps(log[-200:], indent=2))

    if not writer.pending_window_ids():
        elapsed = time.time() - t_start
        throughput = {
            "windows": len(pending), "seconds": elapsed, "windows_per_s": len(pending) / max(elapsed, 1e-9),
            "cpgs_per_s": done_rows / max(elapsed, 1e-9),
        }
        m = writer.finalize({"extraction_throughput": throughput})
        print(f"[extract] complete. store_id={m['store_id']}")
    else:
        print(f"[extract] stopped early (max_windows); {len(writer.pending_window_ids())} windows still pending")
    return 0


def _device_stats(jax) -> float | None:
    try:
        return round(jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**30, 2)
    except Exception:
        return None


def _benchmark(cfg, universe, fasta, lengths, spec, trunk) -> int:
    """Time real windows (sequence fetch + forward + gather), write nothing, and
    extrapolate total GPU time / storage for the full universe."""
    import jax
    import pyarrow as pa

    from methylfm.cpg_embeddings.agmodel import D_128BP, D_1BP, gather_rows
    from methylfm.cpg_embeddings.store import build_loci

    valid = np.array(universe["reference_valid"].to_pylist(), dtype=bool)
    chrom = np.array([c or "" for c in universe["chromosome"].to_pylist()], dtype=object)
    pos = np.array([p if p is not None else -1 for p in universe["position"].to_pylist()], dtype=np.int64)
    loci = build_loci(chrom[valid], pos[valid], lengths, spec)
    n_windows = len(loci.windows)
    wid = loci.window_id
    sel = np.unique(np.linspace(0, n_windows - 1, min(n_windows, 12)).astype(int))
    times = []
    for i, w in enumerate(sel):
        rows = np.flatnonzero(wid == w)
        t0 = time.time()
        seq = fetch_window(fasta, loci.chromosome[rows[0]], int(loci.window_start[rows[0]]), spec.window_size)
        e1, e128 = trunk.embed(trunk.encode_sequence(seq)[None].astype(np.float32))
        h1, h128 = gather_rows(e1, e128, loci.bp[rows], loci.bin[rows])
        times.append((time.time() - t0, len(rows)))
    steady = [t for t, _ in times[1:]] or [times[0][0]]
    per_window = float(np.mean(steady))
    itemsize = np.dtype(cfg["extraction"]["storage_dtype"]).itemsize
    report = {
        "window_size": spec.window_size, "core_size": spec.core_size,
        "num_cpg_ids": int(valid.sum()), "num_loci": int(len(loci.position)), "num_windows": n_windows,
        "benchmarked_windows": len(times), "seconds_per_window_steady": per_window,
        "windows_per_s": 1.0 / per_window, "bp_per_s": spec.window_size / per_window,
        "mean_loci_per_window": float(len(loci.position) / n_windows),
        "estimated_gpu_hours": per_window * n_windows / 3600,
        "estimated_storage_gib": len(loci.position) * (D_1BP + D_128BP) * itemsize / 2**30,
        "gpu_peak_gib": _device_stats(jax), "device": str(jax.devices()[0]),
        "per_window": [{"seconds": t, "loci": n} for t, n in times],
    }
    out = Path(cfg["output_dir"]) / "extraction_benchmark.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_window"}, indent=2))
    return 0

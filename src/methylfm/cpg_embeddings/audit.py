"""Coordinate audit: verify that the CpG universe is consistent with the
reference FASTA *before* any GPU time is spent.

For every CpG id with a coordinate we read the two reference bases at
``(p, p+1)`` (0-based, forward strand) and require ``C`` then ``G``
(case-insensitive; soft-masked repeats are lower case and are fine). A wrong
assembly, a 0/1-based slip or a strand mistake shows up as a mismatch rate far
above the tolerated one -- and the shift diagnostics (``CG`` found at p-1, p+1,
...) say which. The audit writes:

* ``coordinate_audit.json``  -- the report (also returned);
* ``cpg_universe.parquet``   -- every id with ``status`` and ``reference_valid``,
  the input of the extraction step.

Extraction refuses to start unless the audit passed and matches the config.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from methylfm.cpg_embeddings.manifest import STATUS_OK, build_universe

AUDIT_FILENAME = "coordinate_audit.json"
UNIVERSE_FILENAME = "cpg_universe.parquet"
_CHUNK = 1 << 24


class CoordinateAuditError(RuntimeError):
    pass


def sha256_file(path: str | Path, cache: bool = True) -> str:
    """SHA-256 of a (possibly huge) file, cached in a ``.sha256`` sidecar keyed
    on size+mtime so the reference FASTA is only hashed once."""
    path = Path(path)
    sidecar = path.with_name(path.name + ".sha256.cache")
    st = path.stat()
    key = f"{st.st_size}:{int(st.st_mtime)}"
    if cache and sidecar.exists():
        cached = sidecar.read_text().split()
        if len(cached) == 2 and cached[0] == key:
            return cached[1]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    digest = h.hexdigest()
    if cache:
        try:
            sidecar.write_text(f"{key} {digest}\n")
        except OSError:
            pass
    return digest


def read_chromosome(fasta, name: str) -> np.ndarray:
    """Whole chromosome as upper-cased uint8 ASCII (pyfaidx handle in)."""
    seq = str(fasta[name][:])
    arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8).copy()
    arr &= 0b11011111  # force upper case for letters
    return arr


def check_cpg_motif(seq: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, dict]:
    """Boolean ``is_cg`` per position plus shift diagnostics for the failures."""
    n = seq.shape[0]
    C, G = ord("C"), ord("G")
    in_range = (positions >= 0) & (positions + 1 < n)
    p = np.where(in_range, positions, 0)
    is_cg = in_range & (seq[p] == C) & (seq[np.minimum(p + 1, n - 1)] == G)
    diag = {}
    bad = ~is_cg & in_range
    for shift in (-2, -1, 1, 2):
        q = positions[bad] + shift
        ok = (q >= 0) & (q + 1 < n)
        q = np.where(ok, q, 0)
        hit = ok & (seq[q] == C) & (seq[np.minimum(q + 1, n - 1)] == G)
        diag[f"cg_at_shift_{shift:+d}"] = int(hit.sum())
    return is_cg, diag


def run_coordinate_audit(cfg: dict, write: bool = True) -> dict:
    from pyfaidx import Fasta

    t0 = time.time()
    out_dir = Path(cfg["output_dir"])
    universe = build_universe(cfg["manifests"])
    table = universe.table
    ids = table["cpg_id"].to_pylist()
    status = np.array(table["status"].to_pylist(), dtype=object)
    chrom = np.array([c if c is not None else "" for c in table["chromosome"].to_pylist()], dtype=object)
    pos = np.array([p if p is not None else -1 for p in table["position"].to_pylist()], dtype=np.int64)

    fasta = Fasta(str(cfg["reference_fasta"]), as_raw=True, sequence_always_upper=False)
    fasta_chroms = set(fasta.keys())
    has_coord = status == STATUS_OK
    manifest_chroms = sorted({c for c in chrom[has_coord]})
    missing = [c for c in manifest_chroms if c not in fasta_chroms]

    reference_valid = np.zeros(len(ids), dtype=bool)
    mismatch_examples: list[dict] = []
    out_of_range = 0
    shift_diag: dict[str, int] = {}
    per_chrom: dict[str, dict] = {}
    status = status.copy()
    for c in manifest_chroms:
        sel = np.flatnonzero(has_coord & (chrom == c))
        if c in missing:
            status[sel] = "missing_chromosome"
            continue
        seq = read_chromosome(fasta, c)
        is_cg, diag = check_cpg_motif(seq, pos[sel])
        reference_valid[sel] = is_cg
        oor = int(((pos[sel] < 0) | (pos[sel] + 1 >= seq.shape[0])).sum())
        out_of_range += oor
        for k, v in diag.items():
            shift_diag[k] = shift_diag.get(k, 0) + v
        n_bad = int((~is_cg).sum())
        per_chrom[c] = {"ids": int(sel.size), "mismatches": n_bad, "length": int(seq.shape[0])}
        for i in sel[~is_cg][: max(0, 5 - len(mismatch_examples))]:
            j = int(pos[i])
            mismatch_examples.append(
                {"cpg_id": ids[i], "chromosome": c, "position": j, "ref": bytes(seq[max(0, j - 2) : j + 4]).decode()}
            )
        del seq
    status[has_coord & ~reference_valid & (status == STATUS_OK)] = "reference_mismatch"

    # ---- unique-locus statistics (over reference-valid ids)
    valid_idx = np.flatnonzero(reference_valid)
    loci = {}
    for i in valid_idx:
        loci.setdefault((chrom[i], int(pos[i])), []).append(i)
    ids_per_locus = np.array([len(v) for v in loci.values()], dtype=np.int64) if loci else np.zeros(0, np.int64)
    multi = [(k, [ids[i] for i in v]) for k, v in loci.items() if len(v) > 1]
    non_alias = ~np.array(table["is_alias"].to_pylist(), dtype=bool)

    n_ids = len(ids)
    by_status = {s: int((status == s).sum()) for s in sorted(set(status.tolist()))}
    n_with_coord = int(has_coord.sum())
    n_mismatch = int((status == "reference_mismatch").sum())
    n_excluded = n_ids - int(reference_valid.sum())
    mismatch_fraction = n_mismatch / max(1, n_with_coord)
    excluded_fraction = n_excluded / max(1, n_ids)

    tol = cfg["audit"]
    problems = []
    if missing:
        problems.append(f"chromosomes in manifest but not in FASTA: {missing}")
    if mismatch_fraction > tol["max_reference_mismatch_fraction"]:
        problems.append(
            f"reference mismatch fraction {mismatch_fraction:.3e} exceeds tolerated "
            f"{tol['max_reference_mismatch_fraction']:.1e} (assembly/coordinate-convention inconsistency; "
            f"shift diagnostics: {shift_diag})"
        )
    if excluded_fraction > tol["max_excluded_fraction"]:
        problems.append(f"excluded fraction {excluded_fraction:.3e} exceeds tolerated {tol['max_excluded_fraction']:.1e}")

    report = {
        "reference_genome": cfg["reference_genome"],
        "reference_fasta": str(cfg["reference_fasta"]),
        "reference_fasta_sha256": sha256_file(cfg["reference_fasta"]),
        "manifests": {p: {"path": str(path), "sha256": sha256_file(path)} for p, path in cfg["manifests"].items()},
        "cpg_manifest_sha256": hashlib.sha256(
            "".join(sha256_file(p) for _, p in sorted(cfg["manifests"].items())).encode()
        ).hexdigest(),
        "convention": "0-based position of the forward-strand cytosine; ref[p]=='C' and ref[p+1]=='G'",
        "total_cpg_ids": n_ids,
        "total_explicit_ids": int(non_alias.sum()),
        "alias_ids": int((~non_alias).sum()),
        "ids_by_status": by_status,
        "valid_cpg_ids": int(reference_valid.sum()),
        "invalid_coordinates": by_status.get("no_coordinate", 0) + by_status.get("bad_geometry", 0)
        + by_status.get("conflict", 0) + out_of_range,
        "non_cpg_probes": by_status.get("non_cpg", 0),
        "reference_mismatches": n_mismatch,
        "reference_mismatch_fraction": mismatch_fraction,
        "mismatch_shift_diagnostics": shift_diag,
        "mismatch_examples": mismatch_examples,
        "missing_chromosomes": missing,
        "unique_valid_loci": len(loci),
        "duplicate_genomic_loci": len(multi),
        "multiple_ids_same_locus": {
            "loci_with_multiple_ids": len(multi),
            "max_ids_per_locus": int(ids_per_locus.max()) if ids_per_locus.size else 0,
            "examples": [{"locus": f"{k[0]}:{k[1]}", "ids": v[:6]} for k, v in multi[:5]],
        },
        "platform_stats": universe.stats,
        "per_chromosome": per_chrom,
        "tolerances": tol,
        "passed": not problems,
        "problems": problems,
        "seconds": round(time.time() - t0, 1),
    }

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        out = table.append_column("reference_valid", pa.array(reference_valid))
        out = out.set_column(out.schema.get_field_index("status"), "status", pa.array(status.tolist(), pa.string()))
        pq.write_table(out, out_dir / UNIVERSE_FILENAME)
        report["universe_parquet_sha256"] = sha256_file(out_dir / UNIVERSE_FILENAME, cache=False)
        (out_dir / AUDIT_FILENAME).write_text(json.dumps(report, indent=2, default=str))
    return report


def require_passed_audit(cfg: dict) -> dict:
    """Used by extraction: the audit must exist, have passed, and describe the
    same FASTA/manifests the config points at now."""
    path = Path(cfg["output_dir"]) / AUDIT_FILENAME
    if not path.exists():
        raise CoordinateAuditError(f"{path} not found: run `methylfm audit-cpg-coordinates` first.")
    report = json.loads(path.read_text())
    if not report.get("passed"):
        raise CoordinateAuditError(f"coordinate audit did not pass: {report.get('problems')}")
    if report["reference_fasta_sha256"] != sha256_file(cfg["reference_fasta"]):
        raise CoordinateAuditError("reference FASTA changed since the audit; re-run the audit")
    for p, path_ in cfg["manifests"].items():
        if report["manifests"].get(p, {}).get("sha256") != sha256_file(path_):
            raise CoordinateAuditError(f"manifest {p} changed since the audit; re-run the audit")
    uni = Path(cfg["output_dir"]) / UNIVERSE_FILENAME
    if report.get("universe_parquet_sha256") != sha256_file(uni, cache=False):
        raise CoordinateAuditError(f"{uni} does not match the audit report; re-run the audit")
    return report

"""Validation of an extracted CpG embedding store.

Checks that need no model (run anywhere): integrity, representation sanity.
Checks that re-run the official AlphaGenome (JAX environment + GPU):

* ``tile_vs_centered``  tiled-pipeline embedding vs an independent forward on a
  window centred on the CpG. Detects coordinate / window-offset bugs. Reported
  for 1-bp and 128-bp, both phase-matched (same 128-bp bin grid as the tile)
  and exactly centred (arbitrary bin phase).
* ``determinism``       re-running a stored window reproduces the stored rows.
* ``boundary``          the same CpG at several distances from the window edge and
  with shifted windows: how much does the embedding change near an input
  boundary, i.e. is the core/margin policy sufficient.
* ``precision``         bf16-compute (official, stored) vs fp32-compute, and
  fp16 storage vs the bf16 model output.
* ``reverse_complement`` AUDIT ONLY: embedding of the CpG cytosine from the
  reverse-complemented window. Never averaged into the store.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from methylfm.cpg_embeddings.coordinates import TilingSpec
from methylfm.cpg_embeddings.store import CpGEmbeddingStore


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def _summ(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size), "mean": float(x.mean()), "min": float(x.min()),
        "p05": float(np.percentile(x, 5)), "median": float(np.median(x)), "max": float(x.max()),
    }


# --------------------------------------------------------------------------- no-model checks


def integrity_checks(store: CpGEmbeddingStore, sample: int = 20000, seed: int = 0) -> dict:
    n = store.num_loci
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(n, size=min(n, sample), replace=False))
    h1, h128 = store.lookup(rows)
    norms1 = np.linalg.norm(h1.astype(np.float32), axis=1)
    norms128 = np.linalg.norm(h128.astype(np.float32), axis=1)
    index = store.index()
    ri = index["row_index"].to_numpy()
    valid = index["reference_valid"].to_numpy(zero_copy_only=False)
    loci = store.loci()
    return {
        "store_id": store.asset_id,
        "complete": bool(store.manifest.get("complete")),
        "num_loci": n,
        "sampled_rows": int(len(rows)),
        "unwritten_rows_in_sample(all-zero)": int(((norms1 == 0) | (norms128 == 0)).sum()),
        "fraction_nan_or_inf": float((~np.isfinite(h1)).mean() + (~np.isfinite(h128)).mean()),
        "row_index_dense": bool(np.array_equal(np.unique(ri[valid]), np.arange(n))),
        "valid_ids_all_have_rows": bool((ri[valid] >= 0).all() and (ri[~valid] == -1).all()),
        "loci_edge_flagged": int(loci["edge_flag"].to_numpy(zero_copy_only=False).sum()),
        "window_dtype": store.manifest["storage_dtype"],
    }


def representation_sanity(store: CpGEmbeddingStore, sample: int = 20000, num_neighbors: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = store.num_loci
    rows = np.sort(rng.choice(n, size=min(n, sample), replace=False))
    h1, h128 = store.lookup(rows)
    loci = store.loci()
    chrom = np.array(loci["chromosome"].to_pylist(), dtype=object)[rows]
    pos = loci["position"].to_numpy()[rows]
    out = {}
    for name, h in (("1bp", h1.astype(np.float32)), ("128bp", h128.astype(np.float32))):
        norms = np.linalg.norm(h, axis=1)
        var = h.var(axis=0)
        centered = h - h.mean(0, keepdims=True)
        sv = np.linalg.svd(centered[: min(len(h), 5000)], compute_uv=False)
        p = sv / sv.sum()
        eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
        # duplicates among distinct loci
        uniq = np.unique(h, axis=0).shape[0]
        # random-pair cosines
        i = rng.integers(0, len(h), 4000)
        j = rng.integers(0, len(h), 4000)
        keep = i != j
        cos_rand = _cos(h[i[keep]], h[j[keep]])
        # nearest neighbours (cosine) for a few anchors; are they nearby on the genome?
        k = min(num_neighbors, len(h))
        anchors = rng.choice(len(h), size=k, replace=False)
        hn = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
        sims = hn[anchors] @ hn.T
        sims[np.arange(k), anchors] = -2
        nn = sims.argmax(1)
        same_chrom = float((chrom[anchors] == chrom[nn]).mean())
        near_1mb = float(((chrom[anchors] == chrom[nn]) & (np.abs(pos[anchors] - pos[nn]) < 1_000_000)).mean())
        out[name] = {
            "dim": int(h.shape[1]),
            "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
            "per_dim_variance": {"min": float(var.min()), "median": float(np.median(var)), "max": float(var.max()),
                                 "dead_dims(var<1e-10)": int((var < 1e-10).sum())},
            "effective_rank": eff_rank,
            "duplicate_embedding_rate": 1.0 - uniq / len(h),
            "random_pair_cosine": _summ(cos_rand),
            "nearest_neighbor": {
                "anchors": int(k), "nn_cosine_mean": float(sims[np.arange(k), nn].mean()),
                "same_chromosome_fraction": same_chrom,
                "same_chromosome_expected_by_chance": float(
                    sum((np.mean(chrom == c)) ** 2 for c in np.unique(chrom))),
                "within_1Mb_fraction": near_1mb,
            },
        }
    return out


# --------------------------------------------------------------------------- model checks


class _Runner:
    """Single-locus re-extraction with the official model, for validation only."""

    def __init__(self, cfg: dict, store: CpGEmbeddingStore):
        from pyfaidx import Fasta

        from methylfm.cpg_embeddings.agmodel import AlphaGenomeTrunk

        self.cfg = cfg
        self.store = store
        self.spec = TilingSpec(store.manifest["sequence_length"], store.manifest["core_region_size"])
        self.fasta = Fasta(cfg["reference_fasta"], as_raw=True, sequence_always_upper=True)
        self.trunk = AlphaGenomeTrunk(cfg)
        loci = store.loci()
        self.chrom = np.array(loci["chromosome"].to_pylist(), dtype=object)
        self.pos = loci["position"].to_numpy()
        self.wstart = loci["window_start"].to_numpy()
        self.bp = loci["bp_index"].to_numpy()
        self.bin = loci["bin_index"].to_numpy()
        self.edge = loci["edge_flag"].to_numpy(zero_copy_only=False)
        self.window_id = loci["window_id"].to_numpy()
        self.L = {c: len(self.fasta[c]) for c in set(self.chrom.tolist())}

    def window(self, chrom: str, start: int, compute: str = "bf16", rc: bool = False, size: int | None = None):
        from methylfm.cpg_embeddings.sequence import fetch_window

        seq = fetch_window(self.fasta, chrom, start, size or self.spec.window_size)
        if rc:
            seq = seq[::-1].translate(str.maketrans("ACGTN", "TGCAN"))
        x = self.trunk.encode_sequence(seq)[None].astype(np.float32)
        return self.trunk.embed(x, compute=compute)

    def at(self, e1, e128, bp: int, nbin: int | None = None):
        import jax.numpy as jnp

        nb = bp // self.spec.bin_size if nbin is None else nbin
        return np.asarray(e1[0, bp].astype(jnp.float32)), np.asarray(e128[0, nb].astype(jnp.float32))


def _pick(rng, candidates: np.ndarray, k: int) -> np.ndarray:
    return rng.choice(candidates, size=min(k, len(candidates)), replace=False)


STAGES = ("tile_vs_centered", "determinism", "boundary_sensitivity", "precision", "reverse_complement_audit")


def run_model_checks(cfg: dict, store: CpGEmbeddingStore, save=lambda partial: None) -> dict:
    """Runs each stage independently; results are persisted after every stage
    (via ``save``) and a failing stage is recorded rather than losing the rest."""
    import traceback

    v = cfg["validation"]
    r = _Runner(cfg, store)
    S, M = r.spec.window_size, r.spec.margin
    interior = np.flatnonzero(~r.edge)
    report: dict = {"window_size": S, "core_size": r.spec.core_size, "margin": M, "stage_errors": {}}
    stages = {
        "tile_vs_centered": _stage_tile_vs_centered, "determinism": _stage_determinism,
        "boundary_sensitivity": _stage_boundary, "precision": _stage_precision,
        "reverse_complement_audit": _stage_rc,
    }
    for name in v.get("stages") or STAGES:
        rng = np.random.default_rng([v["seed"], STAGES.index(name)])  # stage results independent of which others ran
        try:
            report[name] = stages[name](cfg, store, r, rng, interior)
        except Exception as e:  # noqa: BLE001 - recorded, and makes validation fail
            report["stage_errors"][name] = "".join(traceback.format_exception_only(type(e), e))[-800:]
        save(report)
    return report


def _stage_tile_vs_centered(cfg, store, r, rng, interior):
    v = cfg["validation"]
    S, B = r.spec.window_size, r.spec.bin_size

    # ---- 1. tiled vs independently centred window ------------------------------------------
    rows = _pick(rng, interior, v["num_centered"])
    out = {"phase_matched": {"1bp": [], "128bp": []}, "exactly_centered": {"1bp": [], "128bp": []}}
    for i in rows:
        c, p = r.chrom[i], int(r.pos[i])
        s_tile = int(r.wstart[i])
        h1, h128 = store.lookup(np.array([i]))
        h1, h128 = h1[0].astype(np.float32), h128[0].astype(np.float32)
        # (a) centred, with the tile's 128-bp bin phase
        s_a = s_tile + B * int(round((p - S // 2 - s_tile) / B))
        s_a = int(np.clip(s_a, 0, max(0, r.L[c] - S)))
        s_a -= (s_a - s_tile) % B  # keep the same bin grid as the tile
        e1, e128 = r.window(c, s_a)
        a1, a128 = r.at(e1, e128, p - s_a)
        out["phase_matched"]["1bp"].append(_cos(h1, a1)); out["phase_matched"]["128bp"].append(_cos(h128, a128))
        # (b) exactly centred on the cytosine
        s_b = int(np.clip(p - S // 2, 0, max(0, r.L[c] - S)))
        e1, e128 = r.window(c, s_b)
        b1, b128 = r.at(e1, e128, p - s_b)
        out["exactly_centered"]["1bp"].append(_cos(h1, b1)); out["exactly_centered"]["128bp"].append(_cos(h128, b128))
    return {k: {s: _summ(np.array(x)) for s, x in d.items()} for k, d in out.items()}


def _stage_determinism(cfg, store, r, rng, interior):
    from methylfm.cpg_embeddings.agmodel import gather_rows

    wids = _pick(rng, np.unique(r.window_id), 3)
    det = {"1bp_cos": [], "128bp_cos": [], "max_abs_1bp": [], "max_abs_128bp": []}
    for w in wids:
        rows_w = np.flatnonzero(r.window_id == w)[:200]
        i0 = rows_w[0]
        e1, e128 = r.window(r.chrom[i0], int(r.wstart[i0]))
        g1, g128 = gather_rows(e1, e128, r.bp[rows_w], r.bin[rows_w])
        s1, s128 = store.lookup(rows_w)
        det["1bp_cos"].append(_cos(s1.astype(np.float32), g1)); det["128bp_cos"].append(_cos(s128.astype(np.float32), g128))
        det["max_abs_1bp"].append(np.abs(s1.astype(np.float32) - g1).max())
        det["max_abs_128bp"].append(np.abs(s128.astype(np.float32) - g128).max())
    return {
        "windows": [int(w) for w in wids],
        "cos_1bp": _summ(np.concatenate(det["1bp_cos"])), "cos_128bp": _summ(np.concatenate(det["128bp_cos"])),
        "max_abs_err_1bp": float(max(det["max_abs_1bp"])), "max_abs_err_128bp": float(max(det["max_abs_128bp"])),
        "note": "difference includes the float16 storage rounding of the stored rows",
    }



def _stage_boundary(cfg, store, r, rng, interior):
    v = cfg["validation"]
    S, B = r.spec.window_size, r.spec.bin_size
    rows = _pick(rng, interior, v["num_boundary"])
    dists = [1024, 8192, 65536]
    shifts = v["boundary_shifts"]
    b_out = {f"shift_{s:+d}": {"1bp": [], "128bp": []} for s in shifts}
    b_out.update({f"dist_to_left_edge_{d}": {"1bp": [], "128bp": []} for d in dists})
    for i in rows:
        c, p = r.chrom[i], int(r.pos[i])
        s_tile = int(r.wstart[i])
        h1, h128 = store.lookup(np.array([i]))
        h1, h128 = h1[0].astype(np.float32), h128[0].astype(np.float32)
        for s in shifts:
            st = s_tile + s
            if st < 0 or st + S > r.L[c] or not (B <= p - st < S - 2 * B):
                continue
            e1, e128 = r.window(c, st)
            a1, a128 = r.at(e1, e128, p - st)
            b_out[f"shift_{s:+d}"]["1bp"].append(_cos(h1, a1)); b_out[f"shift_{s:+d}"]["128bp"].append(_cos(h128, a128))
        for d in dists:  # CpG placed `d` bases from the window's left edge (bin-grid preserved)
            st = p - d
            st -= (st - s_tile) % B
            if st < 0 or st + S > r.L[c]:
                continue
            e1, e128 = r.window(c, st)
            a1, a128 = r.at(e1, e128, p - st)
            b_out[f"dist_to_left_edge_{d}"]["1bp"].append(_cos(h1, a1)); b_out[f"dist_to_left_edge_{d}"]["128bp"].append(_cos(h128, a128))
    return {k: {s: _summ(np.array(x)) for s, x in d.items()} for k, d in b_out.items()}


def _stage_precision(cfg, store, r, rng, interior):
    """bf16 (official compute, what was stored) vs fp32 compute, and fp16 storage vs the
    bf16 output. fp32 compute needs ~2x the memory of bf16, so this runs on a smaller
    window centred on the CpG: it measures numerical precision, not window context."""
    from methylfm.cpg_embeddings.agmodel import gather_rows

    v = cfg["validation"]
    W = int(v.get("precision_window_size", 2**18))
    prec = {"bf16_vs_fp32_compute_1bp": [], "bf16_vs_fp32_compute_128bp": [], "fp16_storage_vs_bf16_1bp": [],
            "fp16_storage_vs_bf16_128bp": []}
    errs = {"fp16_storage_max_abs_1bp": 0.0, "fp16_storage_max_abs_128bp": 0.0, "bf16_vs_fp32_max_abs_1bp": 0.0,
            "bf16_vs_fp32_max_abs_128bp": 0.0, "max_abs_value_1bp": 0.0, "max_abs_value_128bp": 0.0}
    for i in _pick(rng, interior, v["num_precision"]):
        c, p = r.chrom[i], int(r.pos[i])
        st = int(np.clip(p - W // 2, 0, max(0, r.L[c] - W)))
        st -= st % r.spec.bin_size
        # the CpG itself plus its neighbourhood (127 further bases) -> many positions per forward
        idx = np.arange(max(0, p - st - 64), min(W - 2, p - st + 64))
        bins = idx // r.spec.bin_size
        eb1, eb128 = r.window(c, st, compute="bf16", size=W)
        ef1, ef128 = r.window(c, st, compute="fp32", size=W)
        b1, b128 = gather_rows(eb1, eb128, idx, bins)
        f1, f128 = gather_rows(ef1, ef128, idx, bins)
        del eb1, eb128, ef1, ef128
        q1, q128 = b1.astype(np.float16).astype(np.float32), b128.astype(np.float16).astype(np.float32)
        prec["bf16_vs_fp32_compute_1bp"].append(_cos(b1, f1)); prec["bf16_vs_fp32_compute_128bp"].append(_cos(b128, f128))
        prec["fp16_storage_vs_bf16_1bp"].append(_cos(q1, b1)); prec["fp16_storage_vs_bf16_128bp"].append(_cos(q128, b128))
        errs["fp16_storage_max_abs_1bp"] = max(errs["fp16_storage_max_abs_1bp"], float(np.abs(q1 - b1).max()))
        errs["fp16_storage_max_abs_128bp"] = max(errs["fp16_storage_max_abs_128bp"], float(np.abs(q128 - b128).max()))
        errs["bf16_vs_fp32_max_abs_1bp"] = max(errs["bf16_vs_fp32_max_abs_1bp"], float(np.abs(b1 - f1).max()))
        errs["bf16_vs_fp32_max_abs_128bp"] = max(errs["bf16_vs_fp32_max_abs_128bp"], float(np.abs(b128 - f128).max()))
        errs["max_abs_value_1bp"] = max(errs["max_abs_value_1bp"], float(np.abs(f1).max()))
        errs["max_abs_value_128bp"] = max(errs["max_abs_value_128bp"], float(np.abs(f128).max()))
    return {"window_size": W, **{k: _summ(np.concatenate(x)) for k, x in prec.items()}, **errs}


def _stage_rc(cfg, store, r, rng, interior):
    """Audit only; never merged into the store."""
    v = cfg["validation"]
    S, M = r.spec.window_size, r.spec.margin
    rows = _pick(rng, interior, v["num_rc"])
    rc = {"same_base_1bp": [], "mirrored_bin_128bp": [], "random_pair_1bp": []}
    nb = S // r.spec.bin_size
    for i in rows:
        c = r.chrom[i]
        s_tile, bpi = int(r.wstart[i]), int(r.bp[i])
        h1, h128 = store.lookup(np.array([i]))
        h1, h128 = h1[0].astype(np.float32), h128[0].astype(np.float32)
        e1, e128 = r.window(c, s_tile, rc=True)
        # cytosine at index bp in the forward window sits at index S-2-bp in the RC window
        a1, a128 = r.at(e1, e128, S - 2 - bpi, nbin=nb - 1 - int(r.bin[i]))
        rc["same_base_1bp"].append(_cos(h1, a1)); rc["mirrored_bin_128bp"].append(_cos(h128, a128))
        rc["random_pair_1bp"].append(_cos(h1, r.at(e1, e128, int(rng.integers(M, S - M)))[0]))
    return {k: _summ(np.array(x)) for k, x in rc.items()} | {
        "note": "audit only: AlphaGenome embeddings are not guaranteed RC-equivariant; "
        "the store uses the forward genomic orientation only"}


# --------------------------------------------------------------------------- entry point


def run_validation(cfg: dict, model_checks: bool = True) -> int:
    t0 = time.time()
    store = CpGEmbeddingStore.open(cfg["store_dir"])
    report = {
        "store_id": store.asset_id,
        "manifest": {k: v for k, v in store.manifest.items() if k != "windows"},
        "integrity": integrity_checks(store, seed=cfg["validation"]["seed"]),
        "representation_sanity": representation_sanity(
            store, num_neighbors=cfg["validation"]["num_sanity_neighbors"], seed=cfg["validation"]["seed"]
        ),
    }
    failures = []
    ig = report["integrity"]
    if ig["unwritten_rows_in_sample(all-zero)"] or ig["fraction_nan_or_inf"] > 0:
        failures.append("store contains unwritten or non-finite rows")
    if not (ig["row_index_dense"] and ig["valid_ids_all_have_rows"]):
        failures.append("cpg_index/row_index inconsistency")

    out = Path(cfg["output_dir"]) / f"validation_report_{store.asset_id}.json"

    def save(partial: dict) -> None:
        report["model_checks"] = partial
        out.write_text(json.dumps(report, indent=2, default=str))

    save_empty = lambda: out.write_text(json.dumps(report, indent=2, default=str))  # noqa: E731
    save_empty()
    if model_checks:
        mc = run_model_checks(cfg, store, save=save)
        for name, err in mc["stage_errors"].items():
            failures.append(f"stage {name} crashed: {err.strip().splitlines()[-1]}")
        if "tile_vs_centered" in mc:
            t = mc["tile_vs_centered"]["phase_matched"]
            if t["1bp"]["median"] < 0.9:
                failures.append(f"tiled vs centred 1-bp cosine too low (median {t['1bp']['median']:.3f}): coordinate/window bug?")
        if "determinism" in mc and mc["determinism"]["cos_1bp"]["min"] < 0.999:
            failures.append("re-extraction does not reproduce the stored rows")
    report["failures"] = failures
    report["passed"] = not failures
    report["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("store_id", "passed", "failures", "seconds")}, indent=2))
    print(f"full report: {out}")
    return 0 if report["passed"] else 1

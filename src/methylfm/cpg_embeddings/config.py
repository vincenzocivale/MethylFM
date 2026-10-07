"""Config loading for the offline CpG-embedding pipeline (audit / extract /
validate). Plain YAML -> dict, with defaults filled in and paths resolved.
Kept free of JAX/torch so it works in both environments."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "reference_genome": "GRCh38",
    "reference_fasta": None,
    "manifests": {},
    "output_dir": "outputs/cpg_encoder",
    "store_dir": None,  # default: <output_dir>/store
    "audit": {
        # Tolerated fraction of coordinate-bearing CpG ids whose reference
        # bases are not "CG". Above this the manifest build / coordinate
        # convention is considered inconsistent and the pipeline stops.
        "max_reference_mismatch_fraction": 1.0e-4,
        # Tolerated fraction of all ids excluded for any reason.
        "max_excluded_fraction": 0.02,
    },
    "tiling": {"window_size": 1048576, "core_size": 524288},
    "model": {
        "source": "huggingface",  # huggingface | kaggle | path
        "version": "all_folds",
        "checkpoint_path": None,
        "organism": "human",
    },
    "extraction": {
        "batch_windows": 1,
        "storage_dtype": "float16",
        "chromosomes": None,  # subset for pilot runs, e.g. [chr22]
        "max_windows": None,
        "min_cpgs_per_window": 1,
    },
    "validation": {
        "num_centered": 200,
        "num_boundary": 50,
        "boundary_shifts": [-65536, -16384, 16384, 65536],
        "num_precision": 8,
        "num_rc": 50,
        "num_sanity_neighbors": 200,
        "seed": 0,
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def load_cpg_encoder_config(path: str | os.PathLike) -> dict[str, Any]:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, user)
    cfg["reference_fasta"] = os.environ.get("METHYLFM_REFERENCE_FASTA") or cfg["reference_fasta"]
    if not cfg["reference_fasta"]:
        raise ValueError(
            "reference_fasta is not set (config key or METHYLFM_REFERENCE_FASTA env var). "
            "It must be the FASTA of the assembly named in `reference_genome`."
        )
    if not cfg["manifests"]:
        raise ValueError("config `manifests` must map array platform -> manifest path")
    cfg["output_dir"] = str(Path(cfg["output_dir"]))
    cfg["store_dir"] = str(Path(cfg["store_dir"] or Path(cfg["output_dir"]) / "store"))
    return cfg

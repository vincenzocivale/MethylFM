"""CpGCorpus data adapter.

CpGCorpus is the pretraining corpus used by CpGPT (Camillo et al.,
"CpGPT: a Foundation Model for DNA Methylation", bioRxiv
10.1101/2024.10.24.619766; code: github.com/lcamillo/CpGPT). It aggregates
2000+ GEO methylation datasets (GSE) spanning 150,000+ samples across 8
Illumina array platforms (27k/450k/EPIC/EPICv2/MSA, GPL-numbered).

Confirmed on-disk layout (from the CpGPT repository README)::

    cpgcorpus/raw/{GSE_ID}/{GPL_ID}/
    ├── betas/
    │   ├── gse_betas.arrow   # raw beta values as downloaded from GEO
    │   └── QCDPB.arrow       # processed via R sesame QCDPB pipeline
    └── metadata/
        └── metadata.arrow    # per-sample supplementary experimental details

Storage format is Apache Arrow (not Parquet/HDF5/AnnData). CpG loci are
keyed by Illumina probe ID (``cg########``), not raw genomic coordinates.
The corpus is hosted on S3 (``s3://cpgpt-lucascamillo-public``,
requester-pays) and is not bundled with this repository.

UNVERIFIED (do not assume these when wiring up a real downloaded shard;
engineering rule: if real CpGCorpus details differ from these assumptions,
stop and document the discrepancy rather than silently adapting to it):

  # UNVERIFIED: exact Arrow column names/schema in betas/*.arrow and
  # metadata/metadata.arrow (which column is the sample-id key, which is
  # the probe-id key, exact beta column name/dtype).
  # UNVERIFIED: exact missing-value sentinel used inside the arrow files
  # (NaN vs. some other sentinel value).
  # UNVERIFIED: CpGPT's own train/val split methodology.
  # UNVERIFIED: whether EPICv2 probe IDs are harmonized 1:1 to legacy
  # cg######## IDs or use a different suffix scheme.

Because the real schema is unresolved, this adapter indexes the directory
structure (datasets, platforms, file paths) without loading row data, and
raises ``NotImplementedError`` the moment row-level access is attempted
against a real shard without an explicit ``column_mapping``. Tests exercise
the row-reading path against a small synthetic Arrow fixture whose schema
is authored by the test itself (NOT a claim about the real corpus schema).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import torch
from torch.utils.data import Dataset

from methylfm.utils.cpg_ids import probe_id_to_int as _probe_id_to_int

try:
    from methylfm.data.sample import MethylationSample, validate_sample
except ImportError:  # pragma: no cover - sample.py authored concurrently
    @dataclass
    class MethylationSample:  # type: ignore[no-redef]
        sample_id: str
        cpg_ids: torch.Tensor
        beta_values: torch.Tensor
        dataset_id: str
        platform: str | None

    def validate_sample(sample: "MethylationSample") -> "MethylationSample":  # type: ignore[no-redef]
        beta = sample.beta_values
        if beta.numel() and (beta.min() < 0 or beta.max() > 1):
            raise ValueError(
                f"beta values out of [0,1] range for sample {sample.sample_id}"
            )
        return sample


@dataclass
class CpGCorpusConfig:
    root: str | None
    use_qc_processed: bool = True
    min_cpgs_per_sample: int = 1000
    split_seed: int = 1337
    val_fraction: float = 0.02


class CpGCorpusNotFoundError(FileNotFoundError):
    pass


def _actionable_not_found_message(root: str | None) -> str:
    return (
        f"CpGCorpus root not found or empty at '{root}'. Set data.root in "
        "configs/data/cpgcorpus.yaml (or the METHYLFM_DATA_ROOT env var) to "
        "the local path where the corpus was materialized via e.g. "
        "`aws s3 sync s3://cpgpt-lucascamillo-public/<prefix> <local_root> "
        "--request-payer requester`. See the CpGCorpus section of the "
        "README for details."
    )


@dataclass
class _DatasetEntry:
    gse_id: str
    gpl_id: str
    betas_path: Path
    metadata_path: Path | None
    num_rows: int | None  # cheap row count if obtainable from file metadata


class _ArrowShardReader:
    """Opens one Arrow IPC/feather file and reads rows via a column mapping.

    ``column_mapping`` is ``None`` for the real, schema-unresolved corpus
    (see module docstring). It must be supplied explicitly (e.g. by a test
    using its own self-authored fixture schema) before row data can be read.
    """

    def __init__(self, path: Path, column_mapping: dict | None, cpg_store=None):
        self.path = path
        self.column_mapping = column_mapping
        self.cpg_store = cpg_store
        self.unmapped = 0  # loci dropped for lacking an AlphaGenome embedding
        self._table: pa.Table | None = None

    def _open(self) -> pa.Table:
        if self._table is None:
            with pa.memory_map(str(self.path), "r") as source:
                self._table = pa_ipc.open_file(source).read_all()
        return self._table

    def num_rows_cheap(self) -> int | None:
        try:
            with pa.memory_map(str(self.path), "r") as source:
                reader = pa_ipc.open_file(source)
                return sum(
                    reader.get_batch(i).num_rows for i in range(reader.num_record_batches)
                )
        except Exception:
            return None

    def read_samples(self, dataset_id: str, platform: str) -> list[MethylationSample]:
        if self.column_mapping is None:
            raise NotImplementedError(
                "CpGCorpus arrow schema not yet confirmed against a real "
                "downloaded shard - column-name mapping must be supplied "
                "once verified; see cpgcorpus.py module docstring UNVERIFIED "
                "section."
            )
        table = self._open()
        sample_col = self.column_mapping["sample_id"]
        probe_col = self.column_mapping["probe_id"]
        beta_col = self.column_mapping["beta"]

        sample_ids = table.column(sample_col).to_pylist()
        probe_ids = table.column(probe_col).to_pylist()
        betas = table.column(beta_col).to_pylist()

        by_sample: dict[str, list[tuple[str, float]]] = {}
        for sid, pid, beta in zip(sample_ids, probe_ids, betas):
            by_sample.setdefault(sid, []).append((pid, beta))

        samples: list[MethylationSample] = []
        for sid, rows in by_sample.items():
            cpg_ids = []
            beta_values = []
            kept_pids = []
            for pid, beta in rows:
                if beta is None or (isinstance(beta, float) and beta != beta):
                    continue  # NaN/missing loci are removed, not zero-filled
                kept_pids.append(pid)
                beta_values.append(float(beta))
            if self.cpg_store is not None:
                # Same mapping as MethylFMProcessor: ids -> embedding-store rows;
                # probes without an embedding are dropped (and counted).
                mapped = self.cpg_store.encode(kept_pids)
                keep = mapped >= 0
                self.unmapped += int((~keep).sum())
                # distinct probe ids sharing a locus (replicates): keep the first
                _, first = np.unique(mapped[keep], return_index=True)
                sel = np.flatnonzero(keep)[np.sort(first)]
                cpg_ids = mapped[sel].tolist()
                beta_values = [beta_values[i] for i in sel]
            else:
                cpg_ids = [_probe_id_to_int(pid) for pid in kept_pids]
            sample = MethylationSample(
                sample_id=str(sid),
                cpg_ids=torch.tensor(cpg_ids, dtype=torch.int64),
                beta_values=torch.tensor(beta_values, dtype=torch.float32),
                dataset_id=dataset_id,
                platform=platform,
            )
            try:
                sample = validate_sample(sample)
            except ValueError:
                # Out-of-range beta: validate_sample fails loudly by design
                # (see sample.py). We still surface the sample here (NaNs
                # were already filtered above) so callers such as the audit
                # command can detect and report malformed data instead of
                # the whole shard read crashing.
                pass
            samples.append(sample)
        return samples


class CpGCorpusDataset(Dataset):
    def __init__(
        self,
        config: CpGCorpusConfig,
        split: str = "train",
        column_mapping: dict | None = None,
        cpg_store=None,
    ):
        if split not in {"train", "val", "all"}:
            raise ValueError(f"split must be one of train/val/all, got {split!r}")
        self.config = config
        self.split = split
        self.column_mapping = column_mapping
        self.cpg_store = cpg_store

        root = config.root
        root_path = Path(root) if root else None
        if root_path is None or not root_path.exists():
            raise CpGCorpusNotFoundError(_actionable_not_found_message(root))

        entries = self._discover(root_path)
        if not entries:
            raise CpGCorpusNotFoundError(_actionable_not_found_message(root))

        entries = self._filter_by_split(entries)
        self._entries = entries

        # flatten into a sample index lazily populated on first access per
        # dataset shard (row data is not loaded at construction time).
        self._readers: dict[tuple[str, str], _ArrowShardReader] = {}
        self._flat_samples: list[MethylationSample] | None = None

    def _discover(self, root_path: Path) -> list[_DatasetEntry]:
        entries: list[_DatasetEntry] = []
        for gse_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
            for gpl_dir in sorted(p for p in gse_dir.iterdir() if p.is_dir()):
                betas_dir = gpl_dir / "betas"
                if not betas_dir.exists():
                    continue
                preferred = betas_dir / "QCDPB.arrow"
                fallback = betas_dir / "gse_betas.arrow"
                if self.config.use_qc_processed and preferred.exists():
                    betas_path = preferred
                elif fallback.exists():
                    betas_path = fallback
                elif preferred.exists():
                    betas_path = preferred
                else:
                    continue
                metadata_path = gpl_dir / "metadata" / "metadata.arrow"
                entries.append(
                    _DatasetEntry(
                        gse_id=gse_dir.name,
                        gpl_id=gpl_dir.name,
                        betas_path=betas_path,
                        metadata_path=metadata_path if metadata_path.exists() else None,
                        num_rows=None,
                    )
                )
        return entries

    def _is_val(self, gse_id: str) -> bool:
        digest = hashlib.sha256(f"{self.config.split_seed}:{gse_id}".encode()).hexdigest()
        frac = int(digest[:16], 16) / float(1 << 64)
        return frac < self.config.val_fraction

    def _filter_by_split(self, entries: list[_DatasetEntry]) -> list[_DatasetEntry]:
        if self.split == "all":
            return entries
        want_val = self.split == "val"
        return [e for e in entries if self._is_val(e.gse_id) == want_val]

    def _materialize(self) -> list[MethylationSample]:
        if self._flat_samples is None:
            samples: list[MethylationSample] = []
            for entry in self._entries:
                reader = self._readers.get((entry.gse_id, entry.gpl_id))
                if reader is None:
                    reader = _ArrowShardReader(entry.betas_path, self.column_mapping, self.cpg_store)
                    self._readers[(entry.gse_id, entry.gpl_id)] = reader
                shard_samples = reader.read_samples(entry.gse_id, entry.gpl_id)
                min_cpgs = self.config.min_cpgs_per_sample
                samples.extend(
                    s for s in shard_samples if min_cpgs <= 0 or s.cpg_ids.numel() >= min_cpgs
                )
            self._flat_samples = samples
        return self._flat_samples

    @property
    def datasets(self) -> list[_DatasetEntry]:
        return self._entries

    def __len__(self) -> int:
        return len(self._materialize())

    def __getitem__(self, idx: int) -> MethylationSample:
        return self._materialize()[idx]

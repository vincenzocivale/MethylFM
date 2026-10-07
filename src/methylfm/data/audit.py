"""Corpus-level audit for CpGCorpus.

Produces directory/platform-level statistics unconditionally (these don't
require the unresolved row-level Arrow schema, see cpgcorpus.py), and
sample-level statistics (beta range, missing values, duplicates,
malformed samples) when a usable ``column_mapping`` is available. If row
access raises ``NotImplementedError`` (real corpus, schema unresolved) or
``CpGCorpusNotFoundError``, the audit degrades gracefully to a partial
report with the issue recorded in ``blockers`` instead of crashing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from methylfm.data.cpgcorpus import (
    CpGCorpusConfig,
    CpGCorpusDataset,
    CpGCorpusNotFoundError,
)


@dataclass
class AuditReport:
    num_datasets: int
    samples_per_dataset: dict[str, int] = field(default_factory=dict)
    platform_distribution: dict[str, int] = field(default_factory=dict)
    cpg_count_distribution: dict = field(default_factory=dict)
    missing_beta_total: int = 0
    duplicate_sample_ids: list[str] = field(default_factory=list)
    duplicate_profiles_suspected: list[tuple[str, str]] = field(default_factory=list)
    beta_min: float = float("nan")
    beta_max: float = float("nan")
    malformed_samples: list[dict] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


def _percentile(sorted_vals: list[int], pct: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(round(pct * (len(sorted_vals) - 1)))))
    return float(sorted_vals[idx])


def run_audit(config: CpGCorpusConfig, column_mapping: dict | None = None) -> AuditReport:
    blockers: list[str] = []

    try:
        all_split = CpGCorpusDataset(config, split="all", column_mapping=column_mapping)
    except CpGCorpusNotFoundError as e:
        return AuditReport(num_datasets=0, blockers=[str(e)])

    num_datasets = len(all_split.datasets)
    platform_distribution: dict[str, int] = {}
    for entry in all_split.datasets:
        platform_distribution[entry.gpl_id] = platform_distribution.get(entry.gpl_id, 0) + 1

    samples_per_dataset: dict[str, int] = {}
    cpg_counts: list[int] = []
    missing_beta_total = 0
    beta_min = float("inf")
    beta_max = float("-inf")
    seen_sample_ids: dict[str, int] = {}
    duplicate_sample_ids: list[str] = []
    profile_fingerprints: dict[tuple, str] = {}
    duplicate_profiles: list[tuple[str, str]] = []
    malformed_samples: list[dict] = []

    try:
        for sample in all_split:
            try:
                samples_per_dataset[sample.dataset_id] = (
                    samples_per_dataset.get(sample.dataset_id, 0) + 1
                )
                n = sample.cpg_ids.numel()
                cpg_counts.append(n)

                beta = sample.beta_values
                missing_beta_total += int((beta != beta).sum().item())  # NaN count
                finite = beta[beta == beta]
                if finite.numel():
                    beta_min = min(beta_min, float(finite.min()))
                    beta_max = max(beta_max, float(finite.max()))
                if finite.numel() and (finite.min() < 0 or finite.max() > 1):
                    malformed_samples.append(
                        {
                            "sample_id": sample.sample_id,
                            "reason": "beta values outside [0,1] range",
                        }
                    )

                seen_sample_ids[sample.sample_id] = seen_sample_ids.get(sample.sample_id, 0) + 1
                if seen_sample_ids[sample.sample_id] == 2:
                    duplicate_sample_ids.append(sample.sample_id)

                # Cheap duplicate-profile heuristic (not exhaustive pairwise
                # comparison, which is too expensive at corpus scale): hash
                # the sorted cpg-id tuple plus a rounded mean beta.
                mean_beta = float(finite.mean()) if finite.numel() else float("nan")
                fingerprint = (
                    tuple(sorted(sample.cpg_ids.tolist())),
                    round(mean_beta, 6) if mean_beta == mean_beta else None,
                )
                fp_hash = hashlib.sha256(repr(fingerprint).encode()).hexdigest()
                if fp_hash in profile_fingerprints:
                    duplicate_profiles.append((profile_fingerprints[fp_hash], sample.sample_id))
                else:
                    profile_fingerprints[fp_hash] = sample.sample_id
            except Exception as e:  # malformed sample shouldn't abort the audit
                malformed_samples.append({"sample_id": getattr(sample, "sample_id", "?"), "reason": str(e)})
    except NotImplementedError as e:
        blockers.append(f"Row-level audit skipped: {e}")
    except CpGCorpusNotFoundError as e:
        blockers.append(str(e))

    cpg_counts_sorted = sorted(cpg_counts)
    cpg_count_distribution = {
        "min": cpg_counts_sorted[0] if cpg_counts_sorted else None,
        "max": cpg_counts_sorted[-1] if cpg_counts_sorted else None,
        "mean": (sum(cpg_counts_sorted) / len(cpg_counts_sorted)) if cpg_counts_sorted else None,
        "p50": _percentile(cpg_counts_sorted, 0.5) if cpg_counts_sorted else None,
        "p90": _percentile(cpg_counts_sorted, 0.9) if cpg_counts_sorted else None,
    }

    return AuditReport(
        num_datasets=num_datasets,
        samples_per_dataset=samples_per_dataset,
        platform_distribution=platform_distribution,
        cpg_count_distribution=cpg_count_distribution,
        missing_beta_total=missing_beta_total,
        duplicate_sample_ids=duplicate_sample_ids,
        duplicate_profiles_suspected=duplicate_profiles,
        beta_min=beta_min if beta_min != float("inf") else float("nan"),
        beta_max=beta_max if beta_max != float("-inf") else float("nan"),
        malformed_samples=malformed_samples,
        blockers=blockers,
    )


def render_report(report: AuditReport) -> str:
    lines: list[str] = []
    if report.blockers:
        lines.append("=== BLOCKERS ===")
        for b in report.blockers:
            lines.append(f"  ! {b}")
        lines.append("")

    lines.append("=== CpGCorpus Audit Report ===")
    lines.append(f"datasets: {report.num_datasets}")
    lines.append(f"platform distribution: {report.platform_distribution}")
    lines.append(f"samples per dataset (n={len(report.samples_per_dataset)}): "
                 f"{dict(list(report.samples_per_dataset.items())[:10])}"
                 + (" ..." if len(report.samples_per_dataset) > 10 else ""))
    lines.append(f"cpg count distribution: {report.cpg_count_distribution}")
    lines.append(f"missing beta total (NaN loci removed): {report.missing_beta_total}")
    lines.append(f"beta range: [{report.beta_min}, {report.beta_max}]")
    lines.append(f"duplicate sample ids: {len(report.duplicate_sample_ids)} "
                 f"{report.duplicate_sample_ids[:10]}")
    lines.append(f"duplicate profiles suspected (heuristic): "
                 f"{len(report.duplicate_profiles_suspected)} "
                 f"{report.duplicate_profiles_suspected[:10]}")
    lines.append(f"malformed samples: {len(report.malformed_samples)} "
                 f"{report.malformed_samples[:10]}")
    return "\n".join(lines)

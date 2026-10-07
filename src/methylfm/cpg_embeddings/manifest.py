"""CpG universe: probe id -> genomic locus, from Illumina annotation manifests.

Input is the Zhou-lab ``*.hg38.manifest.tsv.gz`` files (HM27, HM450, EPIC,
EPICv2, MSA). They are BED-like: ``CpG_chrm``, 0-based ``CpG_beg`` (the
cytosine), ``CpG_end == CpG_beg + 2``, ``Probe_ID``. Nothing is lifted over:
the assembly of the manifests must match the reference FASTA, and the
coordinate audit (`audit.py`) verifies that against the sequence itself.

Policies (all counted and reported by the audit, never silent):

* a probe id seen on several arrays must agree on its coordinate; ids whose
  arrays disagree are excluded (``status='conflict'``);
* records with a missing coordinate or ``CpG_end != CpG_beg + 2`` are excluded
  (``'no_coordinate'`` / ``'bad_geometry'``); probes whose ``target`` is not
  ``CG`` (e.g. ``ch.*`` CpH probes) are not CpG loci (``'non_cpg'``);
* EPICv2/MSA probe ids carry a replicate suffix (``cg00000029_TC21``). The
  suffix-less base id (``cg00000029``) is registered as an *alias* only when
  every suffixed variant agrees on the locus and the base id is not itself a
  manifest id. Whether the pretraining corpus uses suffixed or base ids for
  EPICv2 is unverified (see data/cpgcorpus.py), so both spellings resolve.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

_SUFFIX_RE = re.compile(r"^(.+?)_([TB][CO]\d+)$")

STATUS_OK = "ok"
STATUS_NO_COORD = "no_coordinate"
STATUS_BAD_GEOMETRY = "bad_geometry"
STATUS_CONFLICT = "conflict"
STATUS_NON_CPG = "non_cpg"


@dataclass
class CpGUniverse:
    """One row per CpG id. Columns of ``table``:

    ``cpg_id`` (str), ``chromosome`` (str|null), ``position`` (int64|null,
    0-based C), ``platforms`` (str, comma-joined, '' for pure aliases),
    ``status`` (str), ``is_alias`` (bool).
    """

    table: pa.Table
    stats: dict


def read_manifest(path: str | Path, platform: str) -> pa.Table:
    opts = pacsv.ConvertOptions(
        include_columns=["CpG_chrm", "CpG_beg", "CpG_end", "Probe_ID", "target"],
        null_values=["NA", ""],
        column_types={
            "CpG_chrm": pa.string(),
            "CpG_beg": pa.int64(),
            "CpG_end": pa.int64(),
            "Probe_ID": pa.string(),
            "target": pa.string(),
        },
        strings_can_be_null=True,
    )
    table = pacsv.read_csv(
        str(path),
        convert_options=opts,
        read_options=pacsv.ReadOptions(use_threads=True),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
    )
    return pa.table(
        {
            "cpg_id": table["Probe_ID"],
            "chromosome": table["CpG_chrm"],
            "position": table["CpG_beg"],
            "end": table["CpG_end"],
            "target": table["target"],
            "platform": pa.array([platform] * table.num_rows, type=pa.string()),
        }
    )


def build_universe(manifests: dict[str, str | Path]) -> CpGUniverse:
    """Merge the per-platform manifests into one id -> locus table."""
    parts = [read_manifest(path, platform) for platform, path in manifests.items()]
    records = pa.concat_tables(parts)
    n_records = records.num_rows

    has_coord = pc.and_(pc.is_valid(records["chromosome"]), pc.is_valid(records["position"]))
    good_geometry = pc.equal(pc.subtract(records["end"], records["position"]), 2)
    is_cg = pc.fill_null(pc.equal(records["target"], "CG"), False)
    usable = pc.and_(pc.and_(has_coord, pc.fill_null(good_geometry, False)), is_cg)

    # Per-id aggregation over *usable* records.
    good = records.filter(usable)
    locus_key = pc.binary_join_element_wise(good["chromosome"], pc.cast(good["position"], pa.string()), ":")
    good = good.append_column("locus", locus_key)
    grouped = good.group_by("cpg_id").aggregate(
        [("locus", "min"), ("locus", "max"), ("chromosome", "min"), ("position", "min"), ("platform", "distinct")]
    )
    gid = grouped["cpg_id"].to_pylist()
    conflict = pc.not_equal(grouped["locus_min"], grouped["locus_max"]).to_numpy(zero_copy_only=False)
    g_chrom = grouped["chromosome_min"].to_pylist()
    g_pos = grouped["position_min"].to_numpy(zero_copy_only=False)
    g_plat = [",".join(sorted(set(p))) for p in grouped["platform_distinct"].to_pylist()]

    # Every id seen in any manifest, usable or not.
    all_ids = pc.unique(records["cpg_id"]).to_pylist()
    by_id = {
        cid: (chrom, int(pos), plat, bool(conf))
        for cid, chrom, pos, plat, conf in zip(gid, g_chrom, g_pos, g_plat, conflict)
    }

    # Why unusable? Distinguish missing coordinates from bad CpG_end geometry.
    bad_geo_ids = set(
        records.filter(
            pc.and_(pc.and_(has_coord, is_cg), pc.invert(pc.fill_null(good_geometry, False)))
        )["cpg_id"].to_pylist()
    )
    non_cpg_ids = set(records.filter(pc.and_(has_coord, pc.invert(is_cg)))["cpg_id"].to_pylist())

    ids, chroms, poss, plats, statuses, aliases = [], [], [], [], [], []
    for cid in sorted(all_ids):
        rec = by_id.get(cid)
        ids.append(cid)
        aliases.append(False)
        if rec is None:
            chroms.append(None)
            poss.append(None)
            plats.append("")
            if cid in non_cpg_ids:
                statuses.append(STATUS_NON_CPG)
            elif cid in bad_geo_ids:
                statuses.append(STATUS_BAD_GEOMETRY)
            else:
                statuses.append(STATUS_NO_COORD)
        elif rec[3]:
            chroms.append(None)
            poss.append(None)
            plats.append(rec[2])
            statuses.append(STATUS_CONFLICT)
        else:
            chroms.append(rec[0])
            poss.append(rec[1])
            plats.append(rec[2])
            statuses.append(STATUS_OK)

    # Suffix-less aliases for EPICv2/MSA replicate ids.
    explicit = set(ids)
    variants: dict[str, list[int]] = {}
    for i, cid in enumerate(ids):
        m = _SUFFIX_RE.match(cid)
        if m:
            variants.setdefault(m.group(1), []).append(i)
    n_alias, n_alias_ambiguous = 0, 0
    for base in sorted(variants):
        if base in explicit:
            continue
        idxs = variants[base]
        loci = {(chroms[i], poss[i]) for i in idxs if statuses[i] == STATUS_OK}
        if any(statuses[i] != STATUS_OK for i in idxs) or len(loci) != 1:
            n_alias_ambiguous += 1
            continue
        chrom, pos = next(iter(loci))
        ids.append(base)
        chroms.append(chrom)
        poss.append(pos)
        plats.append("")
        statuses.append(STATUS_OK)
        aliases.append(True)
        n_alias += 1

    table = pa.table(
        {
            "cpg_id": pa.array(ids, pa.string()),
            "chromosome": pa.array(chroms, pa.string()),
            "position": pa.array(poss, pa.int64()),
            "platforms": pa.array(plats, pa.string()),
            "status": pa.array(statuses, pa.string()),
            "is_alias": pa.array(aliases, pa.bool_()),
        }
    )
    stat_counts = {s: statuses.count(s) for s in (STATUS_OK, STATUS_NO_COORD, STATUS_BAD_GEOMETRY, STATUS_NON_CPG, STATUS_CONFLICT)}
    return CpGUniverse(
        table=table,
        stats={
            "manifest_records": n_records,
            "unique_ids": len(all_ids),
            "ids_by_status": stat_counts,
            "ids_per_platform": {
                p: int(pc.count_distinct(r["cpg_id"]).as_py())
                for p, r in ((p, records.filter(pc.equal(records["platform"], p))) for p in manifests)
            },
            "aliases_added": n_alias,
            "alias_bases_skipped_ambiguous": n_alias_ambiguous,
        },
    )


def sort_loci(chromosomes: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Permutation sorting loci by natural chromosome order then position."""
    from methylfm.cpg_embeddings.coordinates import chromosome_sort_key

    names = sorted({str(c) for c in chromosomes}, key=chromosome_sort_key)
    rank = {n: i for i, n in enumerate(names)}
    chrom_rank = np.fromiter((rank[str(c)] for c in chromosomes), dtype=np.int64, count=len(chromosomes))
    return np.lexsort((positions, chrom_rank))

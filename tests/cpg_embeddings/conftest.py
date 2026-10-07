import gzip
from pathlib import Path

import numpy as np
import pytest

COLS = ["CpG_chrm", "CpG_beg", "CpG_end", "Probe_ID", "target"]


def write_manifest(path: Path, rows):
    with gzip.open(path, "wt") as f:
        f.write("\t".join(COLS) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def make_genome(tmp_path: Path, lengths=(6000, 5000), seed=0):
    """Random ACGT genome with all CpGs at known places (we plant them)."""
    rng = np.random.default_rng(seed)
    path = tmp_path / "ref.fa"
    seqs = {}
    with open(path, "w") as f:
        for i, n in enumerate(lengths, 1):
            s = rng.choice(list("ACT"), n)  # no G => no accidental CpGs
            seqs[f"chr{i}"] = s
    return path, seqs


def write_fasta(path: Path, seqs):
    with open(path, "w") as f:
        for name, s in seqs.items():
            f.write(f">{name}\n")
            txt = "".join(s.tolist())
            for i in range(0, len(txt), 60):
                f.write(txt[i : i + 60] + "\n")


@pytest.fixture
def genome(tmp_path):
    fa, seqs = make_genome(tmp_path)
    cpgs = {"chr1": [100, 1500, 3000, 5990], "chr2": [10, 2500]}
    for c, ps in cpgs.items():
        for p in ps:
            seqs[c][p], seqs[c][p + 1] = "C", "G"
    seqs["chr1"][700] = "c"  # soft-masked cytosine is still a valid C
    seqs["chr1"][701] = "g"
    cpgs["chr1"].append(700)
    write_fasta(fa, seqs)
    return fa, cpgs

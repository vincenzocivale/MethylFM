"""Reference-sequence access for windows (no JAX)."""
from __future__ import annotations


def fetch_window(fasta, chrom: str, start: int, size: int) -> str:
    """Upper-case reference bases ``[start, start+size)`` of ``chrom`` (0-based).

    Positions outside the chromosome are padded with ``N`` (all-zero one-hot in
    AlphaGenome's encoder). With the tiling policy this only happens for
    chromosomes shorter than a window."""
    length = len(fasta[chrom])
    lo, hi = max(0, start), min(length, start + size)
    body = str(fasta[chrom][lo:hi]).upper() if hi > lo else ""
    left = "N" * (lo - start)
    right = "N" * (size - len(left) - len(body))
    seq = left + body + right
    assert len(seq) == size
    return seq

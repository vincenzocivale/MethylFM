"""Genomic coordinate conventions and deterministic window tiling.

This module is the single source of truth for every coordinate conversion in
the CpG-embedding pipeline. It is pure NumPy (no JAX, no torch) so it can be
unit-tested in the normal environment. A bug here would silently corrupt the
whole embedding store, so every function is small and independently tested.

Conventions
-----------
* **Positions are 0-based** everywhere inside MethylFM. A CpG locus is the
  0-based coordinate ``p`` of the *cytosine on the forward strand of the
  reference*: ``ref[p] == 'C'`` and ``ref[p + 1] == 'G'``. CpG is
  self-complementary, so a probe measuring the opposite strand still has its
  C/G pair at ``(p, p + 1)`` on the forward reference; the strand of the
  probe never shifts ``p``.
* The Illumina annotation manifests we read (Zhou lab) are BED-like:
  ``CpG_beg`` is 0-based and ``CpG_end == CpG_beg + 2``. ``CpG_beg`` *is*
  ``p``. (1-based inputs must be converted with `one_based_to_zero_based`.)
* A *window* is one AlphaGenome forward input: ``window_size`` bases starting
  at 0-based ``window_start``. The model output is aligned 1:1 with the input,
  so the 1-bp embedding of base ``p`` is at index ``p - window_start`` and the
  128-bp embedding is at bin ``(p - window_start) // 128`` (the 128-bp output
  has exactly ``window_size // 128`` bins and no cropping/padding offset; this
  is asserted against the official output shapes at extraction time).

Tiling
------
A chromosome of length ``L`` is cut into consecutive, non-overlapping *cores*
of ``core_size`` bases: core ``k`` owns positions ``[k*core, (k+1)*core)``.
Each core is processed inside a window of ``window_size`` bases centred on it
(``margin = (window_size - core_size) / 2`` bases of context on each side), so
neighbouring windows overlap by ``2*margin``. A CpG is *owned* by exactly one
core, hence exactly one window; embeddings from different windows are never
averaged.

Near chromosome ends the nominal window would stick out of the chromosome. It
is then shifted (clamped) to lie inside ``[0, L)`` -- or, for chromosomes
shorter than a window, padded with ``N`` -- so the owned CpGs sit closer than
``margin`` to a window boundary. Those CpGs are flagged ``edge_flag=True``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BIN_SIZE_128 = 128


def one_based_to_zero_based(position_1based):
    return np.asarray(position_1based) - 1


def zero_based_to_one_based(position_0based):
    return np.asarray(position_0based) + 1


@dataclass(frozen=True)
class TilingSpec:
    window_size: int
    core_size: int
    bin_size: int = BIN_SIZE_128

    def __post_init__(self) -> None:
        if self.core_size <= 0 or self.window_size < self.core_size:
            raise ValueError(f"need 0 < core_size <= window_size, got {self}")
        if (self.window_size - self.core_size) % 2:
            raise ValueError("window_size - core_size must be even (symmetric margin)")
        for name, value in (
            ("window_size", self.window_size),
            ("core_size", self.core_size),
            ("margin", self.margin),
        ):
            if value % self.bin_size:
                raise ValueError(
                    f"{name} ({value}) must be a multiple of the {self.bin_size}-bp bin so "
                    "core boundaries align with output bins"
                )
        if self.margin <= 1:
            raise ValueError("margin must be > 1 so both bases of a CpG have context")

    @property
    def margin(self) -> int:
        return (self.window_size - self.core_size) // 2

    @property
    def num_bins(self) -> int:
        return self.window_size // self.bin_size


def core_index(position, spec: TilingSpec):
    """Index of the core (hence window) that owns a 0-based position."""
    return np.asarray(position) // spec.core_size


def window_start(core_idx, chrom_length: int, spec: TilingSpec):
    """0-based start of the window processing core ``core_idx``.

    Nominal start is ``core_idx*core - margin``; clamped so that the window
    lies inside ``[0, chrom_length)`` whenever the chromosome is long enough,
    otherwise 0 (the window is then N-padded on the right).
    """
    nominal = np.asarray(core_idx) * spec.core_size - spec.margin
    upper = max(0, chrom_length - spec.window_size)
    return np.clip(nominal, 0, upper)


def bp_index(position, start):
    """Index in the 1-bp AlphaGenome output (and input) of a genomic base."""
    return np.asarray(position) - np.asarray(start)


def bin_index(position, start, spec: TilingSpec):
    """Index in the 128-bp AlphaGenome output of the bin containing a base."""
    return (np.asarray(position) - np.asarray(start)) // spec.bin_size


def edge_flag(position, start, chrom_length: int, spec: TilingSpec):
    """True where fewer than ``margin`` bases of context exist on either side
    inside the window (chromosome ends / clamped windows)."""
    position = np.asarray(position)
    start = np.asarray(start)
    left = position - start
    right = np.minimum(chrom_length, start + spec.window_size) - position
    return (left < spec.margin) | (right <= spec.margin)


def assign_windows(positions: np.ndarray, chrom_length: int, spec: TilingSpec) -> dict:
    """Vectorised ownership for one chromosome.

    Returns arrays aligned with ``positions``: ``core`` (owning core index),
    ``start`` (window start), ``bp`` (1-bp output index), ``bin`` (128-bp
    output bin) and ``edge`` (boundary flag). Every position lies inside its
    window by construction; this is asserted.
    """
    positions = np.asarray(positions, dtype=np.int64)
    core = core_index(positions, spec)
    start = window_start(core, chrom_length, spec)
    bp = bp_index(positions, start)
    # +1: the guanine at p+1 must also be inside the window.
    if positions.size and (bp.min() < 0 or (bp + 1).max() >= spec.window_size):
        raise AssertionError("tiling invariant violated: CpG outside its owning window")
    return {
        "core": core,
        "start": start,
        "bp": bp,
        "bin": bin_index(positions, start, spec),
        "edge": edge_flag(positions, start, chrom_length, spec),
    }


def chromosome_sort_key(name: str) -> tuple:
    """Natural chromosome order: chr1..chr22, chrX, chrY, chrM, then others."""
    stem = name[3:] if name.lower().startswith("chr") else name
    if stem.isdigit():
        return (0, int(stem), "")
    special = {"X": 1, "Y": 2, "M": 3, "MT": 3}
    if stem.upper() in special:
        return (special[stem.upper()], 0, "")
    return (4, 0, name)

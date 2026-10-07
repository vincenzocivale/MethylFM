"""Canonical mapping from external CpG probe identifiers (e.g. Illumina
``cg########`` IDs) to the internal integer CpG id space consumed by
`CpGEncoder` implementations.

This is deliberately a single shared function: the CpGCorpus data adapter
(`data/cpgcorpus.py`) and the public `MethylFMProcessor` must produce
IDENTICAL integer ids for the same probe string, otherwise a model
pretrained against one mapping would see different ids at inference time
through the other path. This mapping is a placeholder keyed only on the
probe-id string -- it carries no biological meaning and will be revisited
once the real CpG locus encoder is chosen.
"""
from __future__ import annotations

import hashlib


def probe_id_to_int(probe_id: str) -> int:
    """Stable integer id for a probe id string (e.g. 'cg00000029').

    Falls back to a SHA-256-derived id for probe ids with no digits (e.g.
    non-standard / EPICv2-suffixed ids), so the mapping is always total.
    """
    digits = "".join(ch for ch in str(probe_id) if ch.isdigit())
    if digits:
        return int(digits)
    return int(hashlib.sha256(str(probe_id).encode()).hexdigest()[:12], 16)

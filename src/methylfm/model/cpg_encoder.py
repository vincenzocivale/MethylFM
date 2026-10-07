"""CpG locus encoders.

`CpGEncoder` is the interface (shape contract) every locus encoder satisfies.
Two implementations exist:

* `AlphaGenomeCpGEncoder` -- the production encoder: frozen, precomputed
  AlphaGenome trunk embeddings (1-bp and 128-bp scale) looked up per CpG and
  passed through a small trainable adapter. AlphaGenome itself never runs
  here; see docs/CPG_ENCODER.md and `methylfm.cpg_embeddings`.
* `MockCpGEncoder` -- a hash-embedding used exclusively by unit tests and the
  `--smoke` path. It carries no biological meaning.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from methylfm.cpg_embeddings.store import CpGEmbeddingStore


class CpGEncoder(nn.Module):
    """Abstract interface for mapping CpG loci to vector representations.

    Given integer CpG identifiers (and optional per-locus metadata), returns
    a locus representation tensor:

        l_i : [..., d_locus]

    The production implementation is `AlphaGenomeCpGEncoder`; `MockCpGEncoder`
    exists for tests only. An encoder is patient-independent: it never sees
    beta values (those enter in the tokenizer).
    """

    def __init__(self, locus_dim: int):
        super().__init__()
        self.locus_dim = locus_dim

    def forward(self, cpg_ids: Tensor, cpg_metadata: dict | None = None) -> Tensor:
        """Returns locus representations: l_i: [..., d_locus].

        Args:
            cpg_ids: integer tensor of CpG identifiers, arbitrary leading
                shape `[...]`.
            cpg_metadata: optional auxiliary per-locus metadata (format left
                unspecified; interpretation is the responsibility of a
                concrete subclass).

        Returns:
            Tensor of shape `cpg_ids.shape + (self.locus_dim,)`.
        """
        raise NotImplementedError("CpGEncoder is an interface; instantiate a concrete subclass.")

    def _validate_output(self, cpg_ids: Tensor, output: Tensor) -> None:
        """Validates that `output` has shape `cpg_ids.shape + (locus_dim,)`.

        Concrete subclasses must call this before returning from `forward`.
        """
        expected_shape = tuple(cpg_ids.shape) + (self.locus_dim,)
        assert tuple(output.shape) == expected_shape, (
            f"CpGEncoder output shape mismatch: expected {expected_shape} "
            f"(cpg_ids.shape + (locus_dim,)), got {tuple(output.shape)}"
        )


class MockCpGEncoder(CpGEncoder):
    """**FOR TESTS AND SMOKE RUNS ONLY — NOT A SCIENTIFIC CHOICE.**

    A synthetic CpG locus encoder that hashes integer CpG identifiers into a
    fixed-size embedding table. This exists purely so that the rest of the
    MethylFM pipeline (tokenizer, set encoder, backbone, losses, training
    loop) can be exercised end-to-end in unit tests and the `smoke-test` CLI
    command without committing to any biological locus representation.

    This class must NEVER be referenced by production pretraining configs
    (e.g. `configs/model/methylfm.yaml`'s default CpG encoder). It carries no
    biological meaning whatsoever.
    """

    def __init__(self, locus_dim: int, vocab_size: int = 2**20):
        super().__init__(locus_dim)
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, locus_dim)

    def forward(self, cpg_ids: Tensor, cpg_metadata: dict | None = None) -> Tensor:
        hashed_ids = torch.remainder(cpg_ids.to(torch.long), self.vocab_size)
        output = self.embedding(hashed_ids)
        self._validate_output(cpg_ids, output)
        return output


STORE_ENV_VAR = "METHYLFM_CPG_EMBEDDING_STORE"


def locate_cpg_embedding_store(
    asset_id: str | None, explicit: "CpGEmbeddingStore | str | os.PathLike | None" = None
) -> CpGEmbeddingStore:
    """Find the external CpG embedding asset.

    Order: the explicit argument, ``$METHYLFM_CPG_EMBEDDING_STORE``, then
    ``$METHYLFM_CACHE|~/.cache/methylfm / cpg_embeddings / <asset_id>``. No
    absolute path is ever stored in ``config.json``; saved configs carry only
    the asset id, which the store must match.
    """
    if isinstance(explicit, CpGEmbeddingStore):
        return explicit
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    if os.environ.get(STORE_ENV_VAR):
        candidates.append(Path(os.environ[STORE_ENV_VAR]))
    if asset_id:
        cache = Path(os.environ.get("METHYLFM_CACHE", Path.home() / ".cache" / "methylfm"))
        candidates.append(cache / "cpg_embeddings" / asset_id)
    path = next((c for c in candidates if (c / "manifest.json").exists()), None)
    if path is None:
        raise FileNotFoundError(
            "The AlphaGenome CpG embedding store was not found. Pass it explicitly "
            "(`cpg_embedding_store=...`), set $" + STORE_ENV_VAR + " to the store directory "
            f"(expected asset id: {asset_id!r}), or build it with `methylfm extract-cpg-embeddings`. "
            "Tried: " + (", ".join(str(c) for c in candidates) or "<nothing configured>")
        )
    return CpGEmbeddingStore.open(path)


def resolve_cpg_embedding_store(config, explicit=None) -> CpGEmbeddingStore:
    """`locate_cpg_embedding_store` + compatibility check against a model config."""
    asset_id = getattr(config, "cpg_embedding_asset_id", None)
    store = locate_cpg_embedding_store(asset_id, explicit)
    store.check_compatible(
        dim_1bp=config.cpg_embedding_1bp_dim, dim_128bp=config.cpg_embedding_128bp_dim, asset_id=asset_id
    )
    return store


class AlphaGenomeCpGEncoder(CpGEncoder):
    """Frozen AlphaGenome prior -> trainable, methylation-specific locus embedding.

        h  = [h_1bp ; h_128bp]                      frozen, precomputed, per locus
        u  = LN(W_in h)                             trainable projection
        d  = W_up GELU(W_down u)                    trainable residual bottleneck adapter
        l  = LN(u + d)                              the CpG locus representation

    * ``cpg_ids`` are **row indices into the embedding store** (the data layer /
      processor map ``cg########`` strings to them); negative ids are padding
      and map to zeros.
    * Everything is shared across loci: there is *no* per-CpG learnable
      parameter, so any locus with an AlphaGenome embedding is supported.
    * ``W_up`` is zero-initialised, so at the start ``l == LN(u) ~= u`` and the
      pretrained geometry is preserved.
    * The encoder is patient-independent (no beta values).

    Efficiency: ``l`` depends only on the locus, so each forward computes it
    once per *distinct* locus in the batch, then indexes. Loci are processed in
    chunks, optionally under activation checkpointing so the large frozen
    input (4608-d per locus) is never kept for backward.

    ``lookup_mode``:
      * ``"gather"``  (default) read just the needed rows from the
        memory-mapped store on CPU and move them to the device;
      * ``"resident"`` copy the whole table to the device once (needs ~2x
        ``num_loci * (D1 + D128)`` bytes of device memory for fp16 -- use on
        large GPUs where the per-step host->device traffic of ``gather`` hurts).
    """

    def __init__(
        self,
        store: CpGEmbeddingStore,
        locus_dim: int = 512,
        bottleneck_dim: int = 64,
        lookup_mode: str = "gather",
        chunk_size: int = 65536,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__(locus_dim)
        if lookup_mode not in {"gather", "resident"}:
            raise ValueError(f"lookup_mode must be 'gather' or 'resident', got {lookup_mode!r}")
        self.dim_1bp, self.dim_128bp = store.dim_1bp, store.dim_128bp
        self.bottleneck_dim = bottleneck_dim
        self.lookup_mode = lookup_mode
        self.chunk_size = chunk_size
        self.activation_checkpointing = activation_checkpointing

        self.input_proj = nn.Linear(self.dim_1bp + self.dim_128bp, locus_dim)
        self.input_norm = nn.LayerNorm(locus_dim)
        self.adapter_down = nn.Linear(locus_dim, bottleneck_dim)
        self.adapter_up = nn.Linear(bottleneck_dim, locus_dim)
        self.output_norm = nn.LayerNorm(locus_dim)
        # Re-applied after HF's blanket `_init_weights` (see modeling_methylfm).
        self.adapter_up._methylfm_zero_init = True
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        # Deliberately plain attributes (not buffers/params): the frozen store
        # must not appear in state_dict nor be broadcast by DDP.
        self._store = store
        self._resident: dict[str, tuple[Tensor, Tensor]] = {}

    @property
    def store(self) -> CpGEmbeddingStore:
        return self._store

    # -- frozen lookup --------------------------------------------------------------
    def _resident_table(self, device: torch.device) -> tuple[Tensor, Tensor]:
        key = str(device)
        if key not in self._resident:
            step = 1 << 18
            t1 = torch.empty((self._store.num_loci, self.dim_1bp), dtype=torch.float16, device=device)
            t128 = torch.empty((self._store.num_loci, self.dim_128bp), dtype=torch.float16, device=device)
            for s in range(0, self._store.num_loci, step):
                rows = np.arange(s, min(s + step, self._store.num_loci))
                h1, h128 = self._store.lookup_unique(rows)
                t1[s : s + len(rows)] = torch.from_numpy(h1).to(device)
                t128[s : s + len(rows)] = torch.from_numpy(h128).to(device)
            self._resident[key] = (t1, t128)
        return self._resident[key]

    def _fetch(self, rows: Tensor) -> Tensor:
        """Frozen ``[len(rows), D1 + D128]`` features (no grad) for sorted unique rows."""
        if self.lookup_mode == "resident":
            t1, t128 = self._resident_table(rows.device)
            return torch.cat([t1.index_select(0, rows), t128.index_select(0, rows)], dim=-1)
        h1, h128 = self._store.lookup_unique(rows.detach().cpu().numpy())
        h = torch.cat([torch.from_numpy(h1), torch.from_numpy(h128)], dim=-1)
        return h.to(rows.device, non_blocking=True)

    # -- trainable transform ----------------------------------------------------------
    def _transform(self, h: Tensor) -> Tensor:
        u = self.input_norm(self.input_proj(h.to(self.input_proj.weight.dtype)))
        delta = self.adapter_up(F.gelu(self.adapter_down(u)))
        return self.output_norm(u + delta)

    def _encode_rows(self, rows: Tensor) -> Tensor:
        return self._transform(self._fetch(rows))

    def locus_representations(self, unique_rows: Tensor) -> Tensor:
        """``l`` for sorted, unique, non-negative row indices: ``[U, locus_dim]``."""
        pieces = []
        use_ckpt = self.activation_checkpointing and torch.is_grad_enabled() and self.training
        for s in range(0, unique_rows.numel(), self.chunk_size):
            rows = unique_rows[s : s + self.chunk_size]
            if use_ckpt:
                pieces.append(checkpoint(self._encode_rows, rows, use_reentrant=False))
            else:
                pieces.append(self._encode_rows(rows))
        if not pieces:
            return unique_rows.new_zeros((0, self.locus_dim), dtype=self.input_proj.weight.dtype)
        return torch.cat(pieces, dim=0)

    def forward(self, cpg_ids: Tensor, cpg_metadata: dict | None = None) -> Tensor:
        ids = cpg_ids.to(torch.long)
        pad = ids < 0
        unique_rows, inverse = torch.unique(ids.clamp_min(0).reshape(-1), return_inverse=True)
        loci = self.locus_representations(unique_rows)
        out = loci[inverse].reshape(*ids.shape, self.locus_dim)
        out = out.masked_fill(pad.unsqueeze(-1), 0.0)
        self._validate_output(cpg_ids, out)
        return out

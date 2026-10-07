"""Public preprocessing abstraction for MethylFM.

`MethylFMProcessor` converts scientific user input (external CpG probe id
strings + raw beta values) into the padded tensors `MethylFMModel.forward`
expects. It is intentionally NOT a Hugging Face text tokenizer: the "neural
tokenizer" (FT-style CpG/beta tokenizer, see `model/tokenizer.py`) remains
part of the model. This class only prepares data.

The public call signature (`processor(cpg_ids=..., beta_values=...,
return_tensors="pt")`) is independent of the eventual CpG biological
encoder: swapping the real `CpGEncoder` implementation in later must not
require changing this class's interface.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch

from methylfm.cpg_embeddings.store import CpGEmbeddingStore
from methylfm.utils.cpg_ids import probe_id_to_int

_PREPROCESSOR_CONFIG_FILENAME = "preprocessor_config.json"


def _is_nested(cpg_ids) -> bool:
    if len(cpg_ids) == 0:
        raise ValueError("cpg_ids must not be empty")
    return isinstance(cpg_ids[0], (list, tuple))


class ProcessorOutput(dict):
    """Dict of model-ready tensors (`cpg_ids`, `beta_values`, `attention_mask`)
    so `model(**inputs)` works directly, plus a `dropped_counts` attribute
    (per-sample count of NaN/missing loci dropped) that is NOT a dict key --
    it must not leak into `model(**inputs)` as an unexpected kwarg."""

    def __init__(self, *args, dropped_counts: list[int], unmapped_counts: list[int] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropped_counts = dropped_counts
        # CpGs with no AlphaGenome embedding (unknown probe id / invalid coordinate); dropped.
        self.unmapped_counts = unmapped_counts if unmapped_counts is not None else [0] * len(dropped_counts)


class MethylFMProcessor:
    """Maps external CpG ids + beta values to model-ready tensors.

    The CpG-id -> integer mapping used here is the SAME
    `methylfm.utils.cpg_ids.probe_id_to_int` function the CpGCorpus data
    adapter uses internally, so a model sees identical ids for the same
    probe string through either path. The `% vocab_size` hash-bucketing
    applied on top is specific to today's `MockCpGEncoder` (which indexes an
    embedding table of that size) and may change once a real CpG locus
    encoder is chosen -- but this call signature will not need to change.
    """

    def __init__(self, vocab_size: int = 2**20, cpg_store: CpGEmbeddingStore | None = None) -> None:
        self.vocab_size = vocab_size
        # With a store (AlphaGenome encoder) probe ids are mapped to its row
        # indices -- the integer ids `AlphaGenomeCpGEncoder` expects; ids with
        # no embedding are dropped and counted, never hashed. Without one,
        # the legacy hash mapping used by `MockCpGEncoder` applies.
        self.cpg_store = cpg_store

    def __call__(
        self,
        cpg_ids: list[str] | list[list[str]],
        beta_values: list[float] | list[list[float]],
        return_tensors: str = "pt",
    ) -> dict:
        if return_tensors != "pt":
            raise ValueError(
                f"Unsupported return_tensors={return_tensors!r}; only 'pt' is supported today."
            )

        nested = _is_nested(cpg_ids)
        cpg_batch = cpg_ids if nested else [cpg_ids]
        beta_batch = beta_values if nested else [beta_values]

        if len(cpg_batch) != len(beta_batch):
            raise ValueError(
                f"cpg_ids batch size ({len(cpg_batch)}) does not match "
                f"beta_values batch size ({len(beta_batch)})"
            )

        store_rows = None
        if self.cpg_store is not None:
            store_rows = [self.cpg_store.encode(sample) for sample in cpg_batch]

        unmapped_counts: list[int] = []
        processed_ids: list[list[int]] = []
        processed_betas: list[list[float]] = []
        dropped_counts: list[int] = []

        for sample_idx, (sample_cpg_ids, sample_betas) in enumerate(zip(cpg_batch, beta_batch)):
            if len(sample_cpg_ids) != len(sample_betas):
                raise ValueError(
                    f"sample {sample_idx}: cpg_ids length ({len(sample_cpg_ids)}) does not "
                    f"match beta_values length ({len(sample_betas)})"
                )

            seen: dict[str, int] = {}
            kept_ids: list[int] = []
            kept_betas: list[float] = []
            seen_rows: dict[int, str] = {}
            dropped = 0
            unmapped = 0

            for pos, (raw_id, raw_beta) in enumerate(zip(sample_cpg_ids, sample_betas)):
                if not isinstance(raw_id, str) or len(raw_id) == 0:
                    raise ValueError(
                        f"sample {sample_idx}, position {pos}: CpG id must be a "
                        f"non-empty string, got {raw_id!r}"
                    )
                if raw_id in seen:
                    raise ValueError(
                        f"sample {sample_idx}: duplicate CpG id {raw_id!r} at positions "
                        f"{seen[raw_id]} and {pos}"
                    )
                seen[raw_id] = pos

                beta = float(raw_beta)
                if math.isnan(beta):
                    # Missing CpG: drop the (id, beta) pair, do not impute or
                    # zero-fill. Counted in `dropped_counts` below.
                    dropped += 1
                    continue
                if beta < 0.0 or beta > 1.0:
                    raise ValueError(
                        f"sample {sample_idx}, position {pos}: beta value {beta} for "
                        f"CpG {raw_id!r} is outside [0, 1]"
                    )

                if store_rows is not None:
                    row = int(store_rows[sample_idx][pos])
                    if row < 0:
                        unmapped += 1
                        continue
                    if row in seen_rows:
                        raise ValueError(
                            f"sample {sample_idx}: CpG ids {seen_rows[row]!r} and {raw_id!r} map to the "
                            "same genomic locus (e.g. EPICv2 replicate probes); deduplicate or average "
                            "them before calling the processor"
                        )
                    seen_rows[row] = raw_id
                    kept_ids.append(row)
                else:
                    kept_ids.append(probe_id_to_int(raw_id) % self.vocab_size)
                kept_betas.append(beta)

            processed_ids.append(kept_ids)
            processed_betas.append(kept_betas)
            dropped_counts.append(dropped)
            unmapped_counts.append(unmapped)

        batch_size = len(processed_ids)
        n_max = max((len(ids) for ids in processed_ids), default=0)

        cpg_ids_tensor = torch.zeros((batch_size, n_max), dtype=torch.long)
        beta_values_tensor = torch.zeros((batch_size, n_max), dtype=torch.float32)
        attention_mask = torch.zeros((batch_size, n_max), dtype=torch.bool)

        for i, (ids, betas) in enumerate(zip(processed_ids, processed_betas)):
            n = len(ids)
            if n == 0:
                continue
            cpg_ids_tensor[i, :n] = torch.tensor(ids, dtype=torch.long)
            beta_values_tensor[i, :n] = torch.tensor(betas, dtype=torch.float32)
            attention_mask[i, :n] = True

        return ProcessorOutput(
            {
                "cpg_ids": cpg_ids_tensor,
                "beta_values": beta_values_tensor,
                "attention_mask": attention_mask,
            },
            dropped_counts=dropped_counts,
            unmapped_counts=unmapped_counts,
        )

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        config_path = Path(save_directory) / _PREPROCESSOR_CONFIG_FILENAME
        with open(config_path, "w") as f:
            data = {"vocab_size": self.vocab_size, "processor_class": "MethylFMProcessor"}
            if self.cpg_store is not None:
                # Only the portable asset id -- never a filesystem path.
                data["cpg_encoder"] = "alphagenome"
                data["cpg_embedding_asset_id"] = self.cpg_store.asset_id
            json.dump(data, f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, cpg_embedding_store=None) -> "MethylFMProcessor":
        config_path = Path(pretrained_model_name_or_path) / _PREPROCESSOR_CONFIG_FILENAME
        if not config_path.exists():
            raise FileNotFoundError(
                f"Could not find {_PREPROCESSOR_CONFIG_FILENAME} in "
                f"{pretrained_model_name_or_path!r}. Only loading from a local "
                "directory produced by `save_pretrained` is supported today."
            )
        with open(config_path) as f:
            data = json.load(f)
        store = None
        if data.get("cpg_encoder") == "alphagenome":
            from methylfm.model.cpg_encoder import locate_cpg_embedding_store

            asset_id = data.get("cpg_embedding_asset_id")
            store = locate_cpg_embedding_store(asset_id, cpg_embedding_store)
            if asset_id is not None and store.asset_id != asset_id:
                from methylfm.cpg_embeddings.store import StoreIncompatibleError

                raise StoreIncompatibleError(
                    f"CpG embedding store {store.asset_id!r} != the {asset_id!r} this processor was saved with"
                )
        return cls(vocab_size=data["vocab_size"], cpg_store=store)

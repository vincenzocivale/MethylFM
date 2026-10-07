"""Hugging Face configuration for MethylFM.

Contains only architectural parameters needed to recreate the backbone --
no dataset paths, W&B settings, optimizer hyperparameters, or machine-
specific settings. Those live in configs/pretrain.yaml and are consumed by
`training/pretrain.py`, never by this class.
"""
from __future__ import annotations

from transformers import PretrainedConfig


class MethylFMConfig(PretrainedConfig):
    model_type = "methylfm"

    def __init__(
        self,
        hidden_size: int = 512,
        num_latents: int = 128,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        mhc_expansion_rate: int = 4,
        mhc_sinkhorn_iters: int = 20,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        initializer_range: float = 0.02,
        locus_dim: int = 512,
        cross_attention_chunk_size: int | None = None,
        # CpG locus encoder. "alphagenome" (production): frozen, precomputed
        # AlphaGenome embeddings from an external store + a trainable adapter
        # (see docs/CPG_ENCODER.md). "mock": hash embedding for unit/smoke
        # tests only. The store itself is an external asset identified by
        # `cpg_embedding_asset_id`; no filesystem path is ever stored here.
        cpg_encoder_type: str = "mock",
        cpg_encoder_vocab_size: int = 2**20,
        cpg_embedding_1bp_dim: int = 1536,
        cpg_embedding_128bp_dim: int = 3072,
        cpg_adapter_hidden_size: int = 512,
        cpg_adapter_bottleneck_size: int = 64,
        cpg_embedding_asset_id: str | None = None,
        cpg_lookup_mode: str = "gather",
        cpg_lookup_chunk_size: int = 65536,
        # LeJEPA projector (MethylFMForPreTraining only -- never used by MethylFMModel).
        projector_hidden_size: int = 1024,
        projector_output_size: int = 512,
        **kwargs,
    ) -> None:
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        if intermediate_size % hidden_size != 0:
            raise ValueError(
                f"intermediate_size ({intermediate_size}) must be a multiple of "
                f"hidden_size ({hidden_size}) -- internally this is expressed as "
                "an integer mlp_ratio."
            )
        if cpg_encoder_type not in {"mock", "alphagenome"}:
            raise ValueError(
                f"Unknown cpg_encoder_type={cpg_encoder_type!r}; expected 'alphagenome' "
                "(production) or 'mock' (tests/smoke only)."
            )
        if cpg_encoder_type == "alphagenome" and cpg_adapter_hidden_size != locus_dim:
            raise ValueError(
                f"cpg_adapter_hidden_size ({cpg_adapter_hidden_size}) must equal locus_dim ({locus_dim}): "
                "the adapter output is the locus representation l_i."
            )

        self.hidden_size = hidden_size
        self.num_latents = num_latents
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.mhc_expansion_rate = mhc_expansion_rate
        self.mhc_sinkhorn_iters = mhc_sinkhorn_iters
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.initializer_range = initializer_range
        self.locus_dim = locus_dim
        self.cross_attention_chunk_size = cross_attention_chunk_size
        self.cpg_encoder_type = cpg_encoder_type
        self.cpg_encoder_vocab_size = cpg_encoder_vocab_size
        self.cpg_embedding_1bp_dim = cpg_embedding_1bp_dim
        self.cpg_embedding_128bp_dim = cpg_embedding_128bp_dim
        self.cpg_adapter_hidden_size = cpg_adapter_hidden_size
        self.cpg_adapter_bottleneck_size = cpg_adapter_bottleneck_size
        self.cpg_embedding_asset_id = cpg_embedding_asset_id
        self.cpg_lookup_mode = cpg_lookup_mode
        self.cpg_lookup_chunk_size = cpg_lookup_chunk_size
        self.projector_hidden_size = projector_hidden_size
        self.projector_output_size = projector_output_size

        super().__init__(**kwargs)

    @property
    def mlp_ratio(self) -> int:
        return self.intermediate_size // self.hidden_size

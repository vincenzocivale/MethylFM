"""Hugging Face public model assembly layer for MethylFM.

This module wires the already-implemented, independently-tested submodules
(`model/cpg_encoder.py`, `model/tokenizer.py`, `model/cross_attention.py`,
`model/latent_block.py`, `model/pooling.py`, `pretraining/projector.py`,
`pretraining/lejepa.py`) into Hugging Face `PreTrainedModel` classes. It does
not reimplement any architecture -- see those modules for the actual
tokenizer/cross-attention/mHC/pooling logic.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from methylfm.configuration_methylfm import MethylFMConfig
from methylfm.model.cpg_encoder import (
    AlphaGenomeCpGEncoder,
    CpGEncoder,
    MockCpGEncoder,
    resolve_cpg_embedding_store,
)
from methylfm.model.cross_attention import SetToLatentCrossAttention
from methylfm.model.latent_block import MHCLatentBackbone
from methylfm.model.pooling import AttentionPool
from methylfm.model.tokenizer import FTCpGTokenizer
from methylfm.outputs import MethylFMForPreTrainingOutput, MethylFMOutput
from methylfm.pretraining.lejepa import lejepa_loss
from methylfm.pretraining.projector import LeJEPAProjector

try:
    from transformers import PreTrainedModel
except ImportError:  # pragma: no cover
    raise ImportError("MethylFM's Hugging Face API requires the `transformers` package.")


def _build_cpg_encoder(config: MethylFMConfig, cpg_embedding_store=None) -> CpGEncoder:
    if config.cpg_encoder_type == "mock":
        return MockCpGEncoder(locus_dim=config.locus_dim, vocab_size=config.cpg_encoder_vocab_size)
    if config.cpg_encoder_type == "alphagenome":
        store = resolve_cpg_embedding_store(config, cpg_embedding_store)
        # Pin the asset: a saved config then refuses embeddings from a different store.
        config.cpg_embedding_asset_id = store.asset_id
        return AlphaGenomeCpGEncoder(
            store,
            locus_dim=config.locus_dim,
            bottleneck_dim=config.cpg_adapter_bottleneck_size,
            lookup_mode=config.cpg_lookup_mode,
            chunk_size=config.cpg_lookup_chunk_size,
        )
    # Also validated in MethylFMConfig.__init__; re-checked since a config can be mutated.
    raise ValueError(f"Unknown cpg_encoder_type={config.cpg_encoder_type!r}")


class MethylFMPreTrainedModel(PreTrainedModel):
    config_class = MethylFMConfig
    base_model_prefix = "methylfm"
    supports_gradient_checkpointing = False
    _no_split_modules: list[str] = []

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if getattr(module, "_methylfm_zero_init", False):
            # Residual-adapter output projection: must start at exactly zero.
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class MethylFMModel(MethylFMPreTrainedModel):
    """The pretrained MethylFM foundation backbone.

    Pipeline: CpGEncoder -> FT-style CpG/beta tokenizer -> CpG-to-latent
    cross-attention -> mHC latent Transformer -> attention pooling.
    Deliberately does NOT contain the LeJEPA projector -- see
    `MethylFMForPreTraining` for that.
    """

    def __init__(self, config: MethylFMConfig, cpg_embedding_store=None) -> None:
        super().__init__(config)
        self.cpg_encoder = _build_cpg_encoder(config, cpg_embedding_store)
        self.tokenizer = FTCpGTokenizer(locus_dim=config.locus_dim, model_dim=config.hidden_size)
        self.cross_attention = SetToLatentCrossAttention(
            model_dim=config.hidden_size,
            num_latents=config.num_latents,
            num_heads=config.num_attention_heads,
            chunk_size=config.cross_attention_chunk_size,
            dropout=config.attention_dropout,
        )
        self.backbone = MHCLatentBackbone(
            model_dim=config.hidden_size,
            num_latents=config.num_latents,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            mlp_ratio=config.mlp_ratio,
            mhc_expansion_rate=config.mhc_expansion_rate,
            sinkhorn_iters=config.mhc_sinkhorn_iters,
        )
        self.pooler = AttentionPool(model_dim=config.hidden_size, num_heads=config.num_attention_heads)
        self.post_init()

    def forward(
        self,
        cpg_ids: torch.Tensor,
        beta_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cpg_metadata: dict | None = None,
        return_dict: bool | None = None,
    ) -> MethylFMOutput | tuple:
        """
        Args:
            cpg_ids: `[batch_size, num_cpgs]` integer CpG locus ids.
            beta_values: `[batch_size, num_cpgs]` beta values in `[0, 1]`.
            attention_mask: `[batch_size, num_cpgs]`, 1/True for an observed
                CpG, 0/False for padding. Defaults to all-ones (no padding)
                if not provided.
        """
        return_dict = return_dict if return_dict is not None else True

        if attention_mask is None:
            attention_mask = torch.ones_like(cpg_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(torch.bool)

        locus_repr = self.cpg_encoder(cpg_ids, cpg_metadata)
        tokens = self.tokenizer(locus_repr, beta_values)
        latents = self.cross_attention(tokens, attention_mask)
        last_hidden_state = self.backbone(latents)
        pooler_output = self.pooler(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooler_output)
        return MethylFMOutput(last_hidden_state=last_hidden_state, pooler_output=pooler_output)


class MethylFMForPreTraining(MethylFMPreTrainedModel):
    """MethylFMModel + LeJEPA projector + the LeJEPA loss interface.

    `MethylFMForPreTraining = MethylFMModel + LeJEPA head`. The backbone
    weights live only here (under `self.methylfm`, matching
    `base_model_prefix`); `MethylFMForPreTraining.methylfm` can be saved on
    its own (`model.methylfm.save_pretrained(path)`) to export a downstream
    checkpoint with no projector/pretraining-specific code.
    """

    def __init__(self, config: MethylFMConfig, cpg_embedding_store=None) -> None:
        super().__init__(config)
        self.methylfm = MethylFMModel(config, cpg_embedding_store=cpg_embedding_store)
        self.projector = LeJEPAProjector(
            input_dim=config.hidden_size,
            hidden_dim=config.projector_hidden_size,
            output_dim=config.projector_output_size,
        )
        self.post_init()

    def forward(
        self,
        cpg_ids: torch.Tensor,
        beta_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cpg_metadata: dict | None = None,
        return_dict: bool | None = None,
    ) -> MethylFMForPreTrainingOutput | tuple:
        """Single-view forward pass: produces the backbone outputs plus the
        projector embedding `z` for this one view. `loss`/`invariance_loss`/
        `sigreg_loss` are left `None` here -- LeJEPA's multi-view objective
        does not fit a single `forward()` call; see `compute_lejepa_loss`."""
        return_dict = return_dict if return_dict is not None else True

        backbone_out = self.methylfm(
            cpg_ids, beta_values, attention_mask, cpg_metadata, return_dict=True
        )
        z = self.projector(backbone_out.pooler_output)

        if not return_dict:
            return (backbone_out.last_hidden_state, backbone_out.pooler_output, z)
        return MethylFMForPreTrainingOutput(
            embeddings=z,
            last_hidden_state=backbone_out.last_hidden_state,
            pooler_output=backbone_out.pooler_output,
        )

    def compute_lejepa_loss(
        self,
        view_batches: list[dict],
        global_view_indices: list[int],
        sigreg_lambda: float = 0.02,
        num_slices: int = 1024,
        num_points: int = 17,
        generator: torch.Generator | None = None,
        gather_fn=None,
    ) -> MethylFMForPreTrainingOutput:
        """The LeJEPA loss interface: runs the backbone + projector over
        every view and combines them into the LeJEPA objective.

        Convenience method for single-process use (e.g. notebooks, eval
        scripts). Under DDP/Accelerate multi-process training, do NOT call
        this on an `accelerator.unwrap_model(...)`'d model -- that bypasses
        DDP's gradient-sync hooks. Instead call the (wrapped) model directly
        once per view and combine with `pretraining.lejepa.lejepa_loss`
        yourself, exactly as `training/pretrain.py` does.

        Args:
            view_batches: one dict per view, each with keys `cpg_ids`,
                `beta_values`, `attention_mask` (all `[batch_size, num_cpgs]`,
                already batched across samples for that view slot).
            global_view_indices: indices into `view_batches` identifying the
                global views (reference_view, broad_view) used for LeJEPA's
                invariance target -- see `pretraining/invariance.py`.

        Returns:
            `MethylFMForPreTrainingOutput` with `loss`, `invariance_loss`,
            `sigreg_loss` populated, and `embeddings` set to the first view's
            backbone representation `h` (for representation-health logging).
        """
        h_views, z_views = [], []
        for vb in view_batches:
            out = self.methylfm(
                vb["cpg_ids"], vb["beta_values"], vb.get("attention_mask"), return_dict=True
            )
            h_views.append(out.pooler_output)
            z_views.append(self.projector(out.pooler_output))

        losses = lejepa_loss(
            z_views,
            global_view_indices=global_view_indices,
            sigreg_lambda=sigreg_lambda,
            num_slices=num_slices,
            num_points=num_points,
            generator=generator,
            gather_fn=gather_fn,
        )
        return MethylFMForPreTrainingOutput(
            loss=losses["total"],
            invariance_loss=losses["invariance"],
            sigreg_loss=losses["sigreg"],
            embeddings=h_views[0],
        )

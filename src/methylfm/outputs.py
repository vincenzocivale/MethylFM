"""Hugging Face `ModelOutput` dataclasses for MethylFM. Avoid returning
undocumented tuples from public model forward() methods."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.utils import ModelOutput


@dataclass
class MethylFMOutput(ModelOutput):
    """Output of `MethylFMModel` -- the pretrained foundation backbone.

    Args:
        last_hidden_state: `[batch_size, num_latents, hidden_size]` -- the
            mHC backbone's latent tokens (post set-to-latent cross-attention,
            post mHC Transformer, pre-pooling).
        pooler_output: `[batch_size, hidden_size]` -- the canonical
            methylome sample embedding (learned attention pooling over
            `last_hidden_state`). This is the representation intended for
            downstream use.
    """

    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None


@dataclass
class MethylFMForPreTrainingOutput(ModelOutput):
    """Output of `MethylFMForPreTraining`.

    `loss` / `invariance_loss` / `sigreg_loss` are only populated by
    `compute_lejepa_loss(...)` (the multi-view LeJEPA objective does not fit
    a single `forward()` call over one view -- see that method's docstring).
    A plain `forward()` call over a single view leaves them `None` and
    populates `embeddings` (the projector output `z` for that view) plus the
    underlying backbone outputs.
    """

    loss: torch.FloatTensor | None = None
    invariance_loss: torch.FloatTensor | None = None
    sigreg_loss: torch.FloatTensor | None = None
    embeddings: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None

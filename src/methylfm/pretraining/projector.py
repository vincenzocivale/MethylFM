"""LeJEPA projector P_phi.

Maps the backbone representation h_s (attention-pool output) to the
projector space z_s used only for the LeJEPA pretraining loss. z_s is
discarded after pretraining -- downstream tasks must consume h_s from
the backbone, never z_s. Keeping this module separate from the backbone
enforces that separation at the type level (see methylfm.model.pooling
for h_s).
"""

import torch
import torch.nn as nn


class LeJEPAProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)

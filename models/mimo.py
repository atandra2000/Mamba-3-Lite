"""Dense cross-head mixing used after the independent SSD head outputs."""
from __future__ import annotations

import torch
import torch.nn as nn


class MIMO(nn.Module):
    """Mix all head channels while preserving the `(B, T, H, D)` interface.

    Identity initialization starts training with the familiar per-head SSD
    behavior and lets optimization learn cross-head communication gradually.
    """

    def __init__(self, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.mix = nn.Linear(n_heads * head_dim, n_heads * head_dim, bias=False)
        nn.init.eye_(self.mix.weight)
        # The model-wide initializer must not overwrite this deliberate identity.
        self.mix._identity_init = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one dense projection to flattened head channels, then restore layout."""
        B, T, H, D = x.shape
        x_flat = x.reshape(B, T, H * D)
        out = self.mix(x_flat)
        return out.reshape(B, T, H, D)

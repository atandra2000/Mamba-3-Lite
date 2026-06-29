"""MIMO head-mixing layer."""
from __future__ import annotations

import torch
import torch.nn as nn

class MIMO(nn.Module):
    """MIMO mixing layer across heads."""
    def __init__(self, d_model: int, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.mix = nn.Linear(n_heads * head_dim, n_heads * head_dim, bias=False)
        # Initialize as near-identity to start with SISO-like behavior
        nn.init.eye_(self.mix.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, H, D) -> (B, T, H, D)"""
        B, T, H, D = x.shape
        x_flat = x.reshape(B, T, H * D)
        out = self.mix(x_flat)
        return out.reshape(B, T, H, D)

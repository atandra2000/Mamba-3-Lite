"""Mamba-3 residual block: RMSNorm -> SSD Complex -> MIMO -> +Residual -> RMSNorm -> SwiGLU -> +Residual."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssd_complex import ssd_complex_chunkwise  # ponytail: ssd_chunkwise deleted (dead), ssd_naive kept in ssd.py as reference oracle
from .mimo import MIMO


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN: fused gate+up matmul -> SiLU(gate)*up -> down."""

    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.gate_up = nn.Linear(d_model, 2 * ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Mamba3Block(nn.Module):
    """One Mamba-3 layer with complex state, MIMO mixing, and no causal conv."""

    def __init__(self, cfg: dict, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = cfg["d_model"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = cfg["head_dim"]
        self.state_dim = cfg["state_dim"]
        self.chunk_size = cfg.get("chunk_size", 64)
        self.rms_norm_eps = cfg.get("rms_norm_eps", 1e-5)

        in_dim = self.n_heads * (self.head_dim + 4 * self.state_dim + 1)
        self.in_proj = nn.Linear(self.d_model, in_dim, bias=False)
        self.mimo = MIMO(self.d_model, self.n_heads, self.head_dim)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        self.A = nn.Parameter(torch.ones(self.n_heads, dtype=torch.complex64))

        # ponytail: native nn.RMSNorm replaces hand-rolled RMSNorm (torch 2.4+); eps passed explicitly.
        self.norm1 = nn.RMSNorm(self.d_model, eps=self.rms_norm_eps)
        self.norm2 = nn.RMSNorm(self.d_model, eps=self.rms_norm_eps)
        self.ffn = SwiGLUFFN(self.d_model, cfg["ffn_dim"])

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.A, -1.0)
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        # ponytail: dropped redundant no-op `self.A.fill_(-1.0)` — constant_ above already sets it.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, T, d_model)."""
        B, T, _ = x.shape
        H, D, N = self.n_heads, self.head_dim, self.state_dim

        residual = x
        h = self.norm1(x)

        proj = self.in_proj(h)
        x_ssm = proj[..., :H * D].reshape(B, T, H, D)

        # B and C are complex, each needs 2 * N real params. Total 4 * N real params.
        B_real = proj[..., H * D:H * D + H * N]
        B_imag = proj[..., H * D + H * N:H * D + 2 * H * N]
        B_t = torch.complex(B_real, B_imag).reshape(B, T, H, N)

        C_real = proj[..., H * D + 2 * H * N:H * D + 3 * H * N]
        C_imag = proj[..., H * D + 3 * H * N:H * D + 4 * H * N]
        C_t = torch.complex(C_real, C_imag).reshape(B, T, H, N)

        dt = proj[..., -H:]

        # ponytail: use_naive_ssd flag removed — never set in any config; complex path is unconditional.
        y = ssd_complex_chunkwise(x_ssm, self.A, B_t, C_t, dt, chunk_size=self.chunk_size)

        y = self.mimo(y)
        y = y.reshape(B, T, H * D)
        y = self.out_proj(y)
        x = residual + y

        residual = x
        h = self.norm2(x)
        h = self.ffn(h)
        x = residual + h

        return x
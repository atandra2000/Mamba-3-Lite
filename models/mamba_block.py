"""Mamba-3 residual block: RMSNorm -> SSD Complex -> MIMO -> +Residual -> RMSNorm -> SwiGLU -> +Residual."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssd import ssd_chunkwise, ssd_naive
from .ssd_complex import ssd_complex_chunkwise
from .mimo import MIMO


class RMSNorm(nn.Module):
    """RMSNorm — root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return (self.weight * x_normed).to(x.dtype)


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
        self.use_naive_ssd = cfg.get("use_naive_ssd", False)
        self.rms_norm_eps = cfg.get("rms_norm_eps", 1e-5)

        in_dim = self.n_heads * (self.head_dim + 4 * self.state_dim + 1)
        self.in_proj = nn.Linear(self.d_model, in_dim, bias=False)
        self.mimo = MIMO(self.d_model, self.n_heads, self.head_dim)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        self.A = nn.Parameter(torch.ones(self.n_heads, dtype=torch.complex64))

        self.norm1 = RMSNorm(self.d_model, eps=self.rms_norm_eps)
        self.norm2 = RMSNorm(self.d_model, eps=self.rms_norm_eps)
        self.ffn = SwiGLUFFN(self.d_model, cfg["ffn_dim"])

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.A, -1.0)
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.A.fill_(-1.0)

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

        if self.use_naive_ssd:
            # Fall back to real naive ssd if specified, but usually we just use complex chunkwise
            y = ssd_naive(x_ssm, self.A.real, B_t.real, C_t.real, dt)
        else:
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

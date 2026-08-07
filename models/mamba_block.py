"""Mamba-3 residual block: complex SSD mixing followed by a gated FFN.

The block uses pre-normalization and two residual paths; unlike Mamba-2 it has
no causal convolution between the input projection and the state-space scan.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssd_complex import ssd_complex_chunkwise
from .mimo import MIMO


class Mamba3Block(nn.Module):
    """One Mamba-3 layer with complex state, MIMO mixing, and no causal conv.

    The input projection emits SSD input channels, complex B/C parameters, and
    one positive-step parameter per head in that order.
    """

    def __init__(self, cfg: dict, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = cfg["d_model"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = cfg["head_dim"]
        self.state_dim = cfg["state_dim"]
        self.chunk_size = cfg.get("chunk_size", 64)
        self.ssd_dispatch = cfg.get("ssd_dispatch", "pytorch")
        self._triton_fallback_warned = False
        self.rms_norm_eps = cfg.get("rms_norm_eps", 1e-5)
        self.grad_checkpoint = cfg.get("grad_checkpoint", False)

        in_dim = self.n_heads * (self.head_dim + 4 * self.state_dim + 1)
        self.in_proj = nn.Linear(self.d_model, in_dim, bias=False)
        self.mimo = MIMO(self.n_heads, self.head_dim)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        self.A = nn.Parameter(torch.empty(self.n_heads, dtype=torch.complex64))
        nn.init.constant_(self.A, -1.0)

        self.norm1 = nn.RMSNorm(self.d_model, eps=self.rms_norm_eps)
        self.norm2 = nn.RMSNorm(self.d_model, eps=self.rms_norm_eps)
        ffn_dim = cfg["ffn_dim"]
        self.ffn_gate_up = nn.Linear(self.d_model, 2 * ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, self.d_model, bias=False)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU feed-forward path after the second RMSNorm."""
        gate, up = self.ffn_gate_up(x).chunk(2, dim=-1)
        return self.ffn_down(F.silu(gate) * up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, T, d_model)."""
        if self.grad_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D, N = self.n_heads, self.head_dim, self.state_dim

        # Keep the complex state in complex64 even when projections run under
        # BF16 autocast; this avoids losing phase information in the recurrence.

        residual = x
        h = self.norm1(x)

        proj = self.in_proj(h)
        x_ssm = proj[..., :H * D].reshape(B, T, H, D).float()

        B_real = proj[..., H * D:H * D + H * N].float()
        B_imag = proj[..., H * D + H * N:H * D + 2 * H * N].float()
        B_t = torch.complex(B_real, B_imag).reshape(B, T, H, N)

        C_real = proj[..., H * D + 2 * H * N:H * D + 3 * H * N].float()
        C_imag = proj[..., H * D + 3 * H * N:H * D + 4 * H * N].float()
        C_t = torch.complex(C_real, C_imag).reshape(B, T, H, N)

        dt = proj[..., -H:].float()

        y = self._ssd_with_dispatch(x_ssm, B_t, C_t, dt)

        y = self.mimo(y)
        y = y.reshape(B, T, H * D)
        y = self.out_proj(y)
        x = residual + y

        residual = x
        h = self.norm2(x)
        h = self._ffn(h)
        x = residual + h

        return x

    def _ssd_with_dispatch(self, x_ssm, B_t, C_t, dt):
        """Run SSD through the configured backend, warning once on Triton failure."""
        if self.ssd_dispatch != "triton":
            return ssd_complex_chunkwise(
                x_ssm, self.A, B_t, C_t, dt, chunk_size=self.chunk_size,
            )
        try:
            return ssd_complex_chunkwise(
                x_ssm, self.A, B_t, C_t, dt,
                chunk_size=self.chunk_size, ssd_dispatch="triton",
            )
        except Exception as exc:
            if not self._triton_fallback_warned:
                print(
                    f"[Mamba3Block {self.layer_idx}] ssd_dispatch='triton' "
                    f"unavailable ({type(exc).__name__}: {exc}); "
                    f"falling back to 'pytorch' for this block."
                )
                self._triton_fallback_warned = True
            return ssd_complex_chunkwise(
                x_ssm, self.A, B_t, C_t, dt, chunk_size=self.chunk_size,
            )

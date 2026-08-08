"""Fused Triton kernel for the per-chunk complex SSD compute.

Fuses L, Y_diag, and per-chunk state into a single per-(B, n_chunks, H) program.
Triton ≥ 3.x has no native complex64 pointer dtype; the parent splits each
complex tensor into contiguous float32 real/imag pairs for the kernel.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from .ssd_complex import per_chunk_ssd_pytorch

if TYPE_CHECKING:
    import triton
    import triton.language as tl

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# 256-cap on constexpr block sizes; larger dims surface a clean ValueError.
_MAX_BLOCK = 256


if HAS_TRITON:

    @triton.jit
    def _ssd_per_chunk_fwd_kernel(
        bc_re_ptr, bc_im_ptr, cc_re_ptr, cc_im_ptr, xc_re_ptr, xc_im_ptr,
        alog_re_ptr, alog_im_ptr, dst_re_ptr, dst_im_ptr,
        y_re_ptr, y_im_ptr, st_re_ptr, st_im_ptr,
        n_chunks, H, T_padded,
        BLOCK_C: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """Compute one batch/chunk/head tile, including local outputs and state."""
        b_idx = tl.program_id(0)
        c_idx = tl.program_id(1)
        h_idx = tl.program_id(2)

        c_off = tl.arange(0, BLOCK_C)
        p_off = tl.arange(0, BLOCK_P)
        n_off = tl.arange(0, BLOCK_N)
        t_base = c_idx * BLOCK_C

        bc_base = (
            b_idx * n_chunks * BLOCK_C * H * BLOCK_N
            + c_idx * BLOCK_C * H * BLOCK_N
            + c_off[:, None] * (H * BLOCK_N) + h_idx * BLOCK_N + n_off[None, :]
        )
        bc_re = tl.load(bc_re_ptr + bc_base)
        bc_im = tl.load(bc_im_ptr + bc_base)

        cc_base = (
            b_idx * n_chunks * BLOCK_C * H * BLOCK_N
            + c_idx * BLOCK_C * H * BLOCK_N
            + c_off[:, None] * (H * BLOCK_N) + h_idx * BLOCK_N + n_off[None, :]
        )
        cc_re = tl.load(cc_re_ptr + cc_base)
        cc_im = tl.load(cc_im_ptr + cc_base)

        xc_base = (
            b_idx * n_chunks * BLOCK_C * H * BLOCK_P
            + c_idx * BLOCK_C * H * BLOCK_P
            + c_off[:, None] * (H * BLOCK_P) + h_idx * BLOCK_P + p_off[None, :]
        )
        xc_re = tl.load(xc_re_ptr + xc_base)
        xc_im = tl.load(xc_im_ptr + xc_base)

        a_base = (
            b_idx * T_padded * H
            + (t_base + c_off) * H + h_idx
        )
        a_re = tl.load(alog_re_ptr + a_base)
        a_im = tl.load(alog_im_ptr + a_base)
        d_re = tl.load(dst_re_ptr + a_base)
        d_im = tl.load(dst_im_ptr + a_base)

        # Prefix differences produce all causal source-to-destination decays.
        a_re_cs = tl.cumsum(a_re, axis=0)
        a_im_cs = tl.cumsum(a_im, axis=0)
        seg_re = a_re_cs[:, None] - a_re_cs[None, :]
        seg_im = a_im_cs[:, None] - a_im_cs[None, :]
        causal = (c_off[:, None] >= c_off[None, :])
        L_re = tl.where(causal, tl.exp(seg_re) * tl.cos(seg_im), 0.0)
        L_im = tl.where(causal, tl.exp(seg_re) * tl.sin(seg_im), 0.0)

        # Form Cc @ Bc.T explicitly because Triton has no complex matmul.
        bc_t_base = (
            b_idx * n_chunks * BLOCK_C * H * BLOCK_N
            + c_idx * BLOCK_C * H * BLOCK_N
            + c_off[None, :] * (H * BLOCK_N) + h_idx * BLOCK_N + n_off[:, None]
        )
        bc_t_re = tl.load(bc_re_ptr + bc_t_base)
        bc_t_im = tl.load(bc_im_ptr + bc_t_base)
        Cb_re = tl.dot(cc_re, bc_t_re) - tl.dot(cc_im, bc_t_im)
        Cb_im = tl.dot(cc_re, bc_t_im) + tl.dot(cc_im, bc_t_re)

        M_re = L_re * Cb_re - L_im * Cb_im
        M_im = L_re * Cb_im + L_im * Cb_re

        Y_re = tl.dot(M_re, xc_re) - tl.dot(M_im, xc_im)
        Y_im = tl.dot(M_re, xc_im) + tl.dot(M_im, xc_re)

        # Accumulate the chunk-end state as Xc.T @ (decay_states * Bc).
        w_re = d_re[:, None] * bc_re - d_im[:, None] * bc_im
        w_im = d_re[:, None] * bc_im + d_im[:, None] * bc_re
        xc_t_base = (
            b_idx * n_chunks * BLOCK_C * H * BLOCK_P
            + c_idx * BLOCK_C * H * BLOCK_P
            + c_off[None, :] * (H * BLOCK_P) + h_idx * BLOCK_P + p_off[:, None]
        )
        xc_t_re = tl.load(xc_re_ptr + xc_t_base)
        xc_t_im = tl.load(xc_im_ptr + xc_t_base)
        st_re = tl.dot(xc_t_re, w_re) - tl.dot(xc_t_im, w_im)
        st_im = tl.dot(xc_t_re, w_im) + tl.dot(xc_t_im, w_re)

        y_base = (
            b_idx * n_chunks * BLOCK_C * H * BLOCK_P
            + c_idx * BLOCK_C * H * BLOCK_P
            + c_off[:, None] * (H * BLOCK_P) + h_idx * BLOCK_P + p_off[None, :]
        )
        tl.store(y_re_ptr + y_base, Y_re)
        tl.store(y_im_ptr + y_base, Y_im)
        st_base = (
            b_idx * n_chunks * H * BLOCK_P * BLOCK_N
            + c_idx * H * BLOCK_P * BLOCK_N
            + h_idx * BLOCK_P * BLOCK_N + p_off[:, None] * BLOCK_N + n_off[None, :]
        )
        tl.store(st_re_ptr + st_base, st_re)
        tl.store(st_im_ptr + st_base, st_im)


def _check_block_dims(P: int, N: int, chunk_size: int) -> None:
    """Keep compile-time Triton tiles bounded for predictable resource use."""
    for name, dim in (("P", P), ("N", N), ("chunk_size", chunk_size)):
        if dim > _MAX_BLOCK:
            raise ValueError(
                f"per_chunk_ssd_triton: {name}={dim} exceeds the {_MAX_BLOCK}-cap. "
                f"Use ssd_dispatch='pytorch' for this config."
            )


def _view_real_imag(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split complex64 into two contiguous float32 tensors for the Triton kernel.

    view_as_real gives stride-2 on the inner dim; contiguous() copies to stride-1.
    """
    if z.dtype != torch.complex64:
        raise TypeError(
            f"per_chunk_ssd_triton: expected complex64, got {z.dtype}. "
            f"This kernel is specialised for the Mamba-3 complex SSD layout."
        )
    pair = torch.view_as_real(z.contiguous())
    return pair[..., 0].contiguous(), pair[..., 1].contiguous()


def _per_chunk_ssd_triton_forward(
    Bc: torch.Tensor, Cc: torch.Tensor, Xc: torch.Tensor,
    A_log: torch.Tensor, decay_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the fused kernel and reassemble its split real/imag outputs."""
    B, n_chunks, C, H, N = Bc.shape
    P = Xc.shape[-1]
    _check_block_dims(P, N, C)

    Y_re = torch.empty((B, n_chunks, C, H, P), dtype=torch.float32, device=Bc.device)
    Y_im = torch.empty((B, n_chunks, C, H, P), dtype=torch.float32, device=Bc.device)
    S_re = torch.empty((B, n_chunks, H, P, N), dtype=torch.float32, device=Bc.device)
    S_im = torch.empty((B, n_chunks, H, P, N), dtype=torch.float32, device=Bc.device)

    Bc_re, Bc_im = _view_real_imag(Bc)
    Cc_re, Cc_im = _view_real_imag(Cc)
    Xc_re, Xc_im = _view_real_imag(Xc)
    A_log_re, A_log_im = _view_real_imag(A_log)
    dst_re, dst_im = _view_real_imag(decay_states)

    T_padded = n_chunks * C
    num_stages = int(os.environ.get("TRITON_PER_CHUNK_NUM_STAGES", "1"))
    num_warps = int(os.environ.get("TRITON_PER_CHUNK_NUM_WARPS", "4"))
    _ssd_per_chunk_fwd_kernel[(B, n_chunks, H)](
        Bc_re, Bc_im, Cc_re, Cc_im, Xc_re, Xc_im,
        A_log_re, A_log_im, dst_re, dst_im,
        Y_re, Y_im, S_re, S_im,
        n_chunks, H, T_padded,
        BLOCK_C=C, BLOCK_P=P, BLOCK_N=N,
        num_warps=num_warps, num_stages=num_stages,
    )

    Y_diag = torch.complex(Y_re, Y_im)
    state = torch.complex(S_re, S_im)
    return Y_diag, state


class _PerChunkSSDTriton(torch.autograd.Function):
    """Autograd bridge: use the fused forward and a differentiable reference backward."""

    @staticmethod
    def forward(ctx, Bc, Cc, Xc, A_log, decay_states):
        Y_diag, state = _per_chunk_ssd_triton_forward(Bc, Cc, Xc, A_log, decay_states)
        ctx.save_for_backward(Bc, Cc, Xc, A_log, decay_states)
        return Y_diag, state

    @staticmethod
    def backward(ctx, grad_y_diag, grad_state):
        Bc, Cc, Xc, A_log, decay_states = ctx.saved_tensors
        with torch.enable_grad():
            b = Bc.detach().requires_grad_(True)
            c = Cc.detach().requires_grad_(True)
            x = Xc.detach().requires_grad_(True)
            a = A_log.detach().requires_grad_(True)
            d = decay_states.detach().requires_grad_(True)
            Y_diag_ref, state_ref = per_chunk_ssd_pytorch(b, c, x, a, d)
        g_y = grad_y_diag if grad_y_diag is not None else torch.zeros_like(Y_diag_ref)
        g_s = grad_state if grad_state is not None else torch.zeros_like(state_ref)
        grads = torch.autograd.grad(
            (Y_diag_ref, state_ref), (b, c, x, a, d),
            grad_outputs=(g_y, g_s), allow_unused=True,
        )
        return grads


def per_chunk_ssd_triton(
    Bc: torch.Tensor, Cc: torch.Tensor, Xc: torch.Tensor,
    A_log: torch.Tensor, decay_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local chunk outputs and chunk-end states using the Triton backend."""
    if not HAS_TRITON:
        raise ImportError(
            "per_chunk_ssd_triton requires the `triton` package. "
            "Install with `pip install triton` (Linux + CUDA only). "
            "For CPU/Mac, set ssd_dispatch='pytorch' in the model config."
        )
    return _PerChunkSSDTriton.apply(Bc, Cc, Xc, A_log, decay_states)

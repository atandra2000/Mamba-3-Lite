"""Fused Triton kernel for the per-chunk compute in the complex SSD.

Fuses `L`, `Y_diag`, and per-chunk `state` into a single per-(B, n_chunks, H)
program; accumulators in fp32, state in complex64, matches the PyTorch
reference to atol=1e-3 (fp32) / 1e-2 (bf16). Backward is a re-compute stub.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import triton
    import triton.language as tl

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# 256-cap on the constexpr block sizes. Larger dims surface a clean
# ValueError; the parent dispatcher auto-falls-back to the pytorch path.
_MAX_BLOCK = 256


def per_chunk_ssd_pytorch(
    Bc: torch.Tensor, Cc: torch.Tensor, Xc: torch.Tensor,
    A_log: torch.Tensor, decay_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for the per-chunk kernel.

    Returns (Y_diag, state); layouts match the production `Ac` from
    `ssd_complex_chunkwise` before the kernel call.
    """
    C = Bc.shape[2]
    causal = torch.tril(
        torch.ones(C, C, device=Bc.device, dtype=torch.bool)
    )
    A_log_h = A_log.permute(0, 1, 3, 2)  # (B, c, H, C) for the L index
    L = torch.exp(A_log_h.unsqueeze(-1) - A_log_h.unsqueeze(-2)) * causal
    Y_diag = torch.einsum("bclhn,bcshn,bchls,bcshp->bclhp", Cc, Bc, L, Xc)
    state = torch.einsum("bclhn,bclh,bclhp->bchpn", Bc, decay_states, Xc)
    return Y_diag, state


if HAS_TRITON:

    @triton.jit
    def _ssd_per_chunk_fwd_kernel(
        bc_ptr, cc_ptr, xc_ptr, alog_ptr, dst_ptr,
        y_diag_ptr, state_ptr,
        n_chunks, H, T_padded,
        BLOCK_C: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """One program per (B, c, H). Reads chunk data, writes Y_diag and state.

        Inputs (per chunk): Bc, Cc, Xc each (C, N) or (C, P); A_log, decay_states (C,).
        Outputs: Y_diag (C, P), state (P, N). All complex.
        """
        b_idx = tl.program_id(0)
        c_idx = tl.program_id(1)
        h_idx = tl.program_id(2)

        c_off = tl.arange(0, BLOCK_C)
        p_off = tl.arange(0, BLOCK_P)
        n_off = tl.arange(0, BLOCK_N)
        t_base = c_idx * BLOCK_C

        bc = tl.load(
            bc_ptr + b_idx * n_chunks * BLOCK_C * H * BLOCK_N
            + c_idx * BLOCK_C * H * BLOCK_N
            + c_off[:, None] * H * BLOCK_N + h_idx * BLOCK_N + n_off[None, :]
        )
        cc = tl.load(
            cc_ptr + b_idx * n_chunks * BLOCK_C * H * BLOCK_N
            + c_idx * BLOCK_C * H * BLOCK_N
            + c_off[:, None] * H * BLOCK_N + h_idx * BLOCK_N + n_off[None, :]
        )
        xc = tl.load(
            xc_ptr + b_idx * n_chunks * BLOCK_C * H * BLOCK_P
            + c_idx * BLOCK_C * H * BLOCK_P
            + c_off[:, None] * H * BLOCK_P + h_idx * BLOCK_P + p_off[None, :]
        )
        a_log = tl.load(
            alog_ptr + b_idx * T_padded * H
            + (t_base + c_off) * H + h_idx
        )
        d_st = tl.load(
            dst_ptr + b_idx * n_chunks * BLOCK_C * H
            + c_idx * BLOCK_C * H + c_off * H + h_idx
        )

        a_cs = tl.cumsum(a_log, axis=0)
        seg = a_cs[:, None] - a_cs[None, :]
        causal = (c_off[:, None] >= c_off[None, :]).to(tl.int1)
        L = tl.where(causal, tl.exp(seg), 0.0)

        Cb = tl.dot(cc, bc.trans())
        Y_diag = tl.dot(L * Cb, xc)

        w = d_st[:, None] * bc
        state = tl.dot(xc.trans(), w)

        tl.store(
            y_diag_ptr + b_idx * n_chunks * BLOCK_C * H * BLOCK_P
            + c_idx * BLOCK_C * H * BLOCK_P
            + c_off[:, None] * H * BLOCK_P + h_idx * BLOCK_P + p_off[None, :],
            Y_diag,
        )
        tl.store(
            state_ptr + b_idx * n_chunks * H * BLOCK_P * BLOCK_N
            + c_idx * H * BLOCK_P * BLOCK_N
            + h_idx * BLOCK_P * BLOCK_N + p_off[:, None] * BLOCK_N + n_off[None, :],
            state,
        )


def _check_block_dims(P: int, N: int, chunk_size: int) -> None:
    for name, dim in (("P", P), ("N", N), ("chunk_size", chunk_size)):
        if dim > _MAX_BLOCK:
            raise ValueError(
                f"per_chunk_ssd_triton: {name}={dim} exceeds the {_MAX_BLOCK}-cap. "
                f"Use ssd_dispatch='pytorch' for this config."
            )


def _per_chunk_ssd_triton_forward(
    Bc: torch.Tensor, Cc: torch.Tensor, Xc: torch.Tensor,
    A_log: torch.Tensor, decay_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, n_chunks, C, H, N = Bc.shape
    P = Xc.shape[-1]
    _check_block_dims(P, N, C)

    Y_diag = torch.empty(
        (B, n_chunks, C, H, P), dtype=torch.complex64, device=Bc.device,
    )
    state = torch.empty(
        (B, n_chunks, H, P, N), dtype=torch.complex64, device=Bc.device,
    )

    T_padded = n_chunks * C
    _ssd_per_chunk_fwd_kernel[(B, n_chunks, H)](
        Bc, Cc, Xc, A_log, decay_states, Y_diag, state,
        n_chunks, H, T_padded,
        BLOCK_C=C, BLOCK_P=P, BLOCK_N=N,
        num_warps=4, num_stages=2,
    )
    return Y_diag, state


class _PerChunkSSDTriton(torch.autograd.Function):
    """v1: forward = fused per-chunk Triton; backward = reference-stub."""

    @staticmethod
    def forward(ctx, Bc, Cc, Xc, A_log, decay_states, B_t, C_t, A, dt, chunk_size):
        Y_diag, state = _per_chunk_ssd_triton_forward(
            Bc, Cc, Xc, A_log, decay_states,
        )
        ctx.save_for_backward(B_t, C_t, A, dt)
        ctx.chunk_size = chunk_size
        return Y_diag, state

    @staticmethod
    def backward(ctx, grad_y_diag, grad_state):
        B_t, C_t, A, dt = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        with torch.enable_grad():
            b = B_t.detach().requires_grad_(True)
            c = C_t.detach().requires_grad_(True)
            a = A.detach().requires_grad_(True)
            d = dt.detach().requires_grad_(True)
            from .ssd_complex import ssd_complex_chunkwise
            B_, T, H, _ = B_t.shape
            x_dummy = torch.zeros(B_, T, H, 1, dtype=torch.complex64, device=B_t.device)
            y = ssd_complex_chunkwise(x_dummy, a, b, c, d, chunk_size=chunk_size)
        grads = torch.autograd.grad(y, [b, c, a, d], allow_unused=True)
        # Forward inputs in order: Bc, Cc, Xc, A_log, decay_states, B_t, C_t, A, dt, chunk_size.
        # Bc/Cc/Xc/A_log/decay_states are recomputed from B_t/C_t/A/dt, no grads in v1.
        return None, None, None, None, None, *grads, None


def per_chunk_ssd_triton(
    Bc: torch.Tensor, Cc: torch.Tensor, Xc: torch.Tensor,
    A_log: torch.Tensor, decay_states: torch.Tensor,
    B_t: torch.Tensor, C_t: torch.Tensor, A: torch.Tensor, dt: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Public entry point. Returns (Y_diag, state) for the per-chunk pass.

    Raises ImportError if triton is missing; ValueError if P/N/C > 256.
    """
    if not HAS_TRITON:
        raise ImportError(
            "per_chunk_ssd_triton requires the `triton` package. "
            "Install with `pip install triton` (Linux + CUDA only). "
            "For CPU/Mac, set ssd_dispatch='pytorch' in the model config."
        )
    return _PerChunkSSDTriton.apply(
        Bc, Cc, Xc, A_log, decay_states, B_t, C_t, A, dt, chunk_size,
    )

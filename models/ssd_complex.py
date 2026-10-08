"""Complex-valued SSD sequence mixing used by each Mamba-3 block.

The chunkwise path is algebraically equivalent to the sequential reference while
exposing independent chunks for efficient PyTorch or opt-in Triton execution.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _discretise(dt: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Convert continuous decay rates into per-token complex transition factors."""
    return torch.exp(F.softplus(dt) * A)


# exp overflows to inf above ~88 in fp32 (complex64 = 2x fp32). Every causal
# decay exponent is <= 0, so clamping the positive tail at 80 cannot change a
# retained value; it only keeps the discarded region finite so `where` masking
# and its backward pass stay NaN-free. Do not tighten toward 0: a finite but
# too-small cap would change the retained decay.
_EXP_EXPONENT_CAP = 80.0


def _clamp_exp_exponent(z: torch.Tensor) -> torch.Tensor:
    """Clamp an exponent's real part so `exp` never overflows.

    The imaginary part is a phase and must pass through unclamped.
    """
    real = torch.clamp(z.real, max=_EXP_EXPONENT_CAP)
    return torch.complex(real, z.imag) if z.is_complex() else real


def ssd_naive_complex(
    x: torch.Tensor, A: torch.Tensor, B_t: torch.Tensor, C_t: torch.Tensor, dt: torch.Tensor,
) -> torch.Tensor:
    """Run the token-by-token recurrence used as the chunkwise regression oracle.

    Inputs use `(batch, time, heads, channels)` and complex state axes; the
    deliberately simple O(T) loop makes numerical mismatches easy to diagnose.
    """
    B_, T, H, D = x.shape
    N = B_t.shape[-1]
    A_bar = _discretise(dt, A)
    s = torch.zeros(B_, H, N, D, dtype=torch.complex64, device=x.device)
    ys = []
    for t in range(T):
        s = A_bar[:, t].unsqueeze(-1).unsqueeze(-1) * s             + B_t[:, t].unsqueeze(-1) * x[:, t].unsqueeze(-2)
        ys.append((C_t[:, t].unsqueeze(-1) * s).sum(dim=-2))
    return torch.stack(ys, dim=1)


def per_chunk_ssd_pytorch(
    Bc: torch.Tensor, Cc: torch.Tensor, Xc: torch.Tensor,
    A_log: torch.Tensor, decay_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-chunk outputs and chunk-end states with ordinary tensor ops.

    `A_log` is `(B, chunks, C, H)`; the causal matrix `L` contains the decay
    from source position `s` to destination `l` within one chunk. This is the
    single implementation shared by the PyTorch dispatch and the Triton
    kernel's recompute backward.
    """
    C = Bc.shape[2]
    causal = torch.tril(torch.ones(C, C, device=Bc.device, dtype=torch.bool))
    A_log_h = A_log.permute(0, 1, 3, 2)
    A_cs = A_log_h.cumsum(dim=-1)
    # Clamp the exponent before exp. The discarded non-causal region has a
    # large positive exponent whose exp overflows to inf; `inf * 0` in the mask
    # forward and the `exp` derivative backward both yield NaN. exp(z) for
    # z>88 already saturates above every causal value, so clamping is exact here.
    seg = _clamp_exp_exponent(A_cs.unsqueeze(-1) - A_cs.unsqueeze(-2))
    L = torch.where(causal, torch.exp(seg), torch.zeros((), dtype=seg.dtype, device=seg.device))
    Y_diag = torch.einsum("bclhn,bcshn,bchls,bcshp->bclhp", Cc, Bc, L, Xc)
    state = torch.einsum("bclhn,bclh,bclhp->bchpn", Bc, decay_states, Xc)
    return Y_diag, state


def ssd_complex_chunkwise(
    x: torch.Tensor, A: torch.Tensor, B_t: torch.Tensor, C_t: torch.Tensor, dt: torch.Tensor,
    chunk_size: int = 64,
    ssd_dispatch: str = "pytorch",
) -> torch.Tensor:
    """Evaluate complex SSD by separating within-chunk and cross-chunk effects.

    Padding makes the sequence divisible by `chunk_size`; outputs are unpadded
    before return. The Triton option fuses the per-chunk projections, while the
    recurrence between chunk states remains in PyTorch.
    """
    B_, T, H, D = x.shape
    N, C = B_t.shape[-1], chunk_size

    pad = (C - (T % C)) % C
    if pad > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        B_t = F.pad(B_t, (0, 0, 0, 0, 0, pad))
        C_t = F.pad(C_t, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))

    T_padded = T + pad
    n_chunks = T_padded // C

    A_log = F.softplus(dt) * A

    def _chunk(t):
        return t.reshape(B_, n_chunks, C, *t.shape[2:])

    Xc, Bc, Cc, Ac = _chunk(x).to(torch.complex64), _chunk(B_t), _chunk(C_t), _chunk(A_log)

    A_cumsum = torch.cumsum(Ac, dim=2)
    decay_states = torch.exp(A_cumsum[:, :, -1:, :] - A_cumsum)

    if ssd_dispatch == "triton":
        from .ssd_triton import per_chunk_ssd_triton
        Y_diag, states = per_chunk_ssd_triton(Bc, Cc, Xc, Ac, decay_states)
    else:
        Y_diag, states = per_chunk_ssd_pytorch(Bc, Cc, Xc, Ac, decay_states)

    chunk_decay = A_cumsum[:, :, -1, :]
    cd_perm = chunk_decay.permute(0, 2, 1).contiguous()
    cd_cumsum = torch.cumsum(cd_perm, dim=-1)
    # Prefix sums let every earlier chunk state be decayed to the current chunk
    # without an explicit sequential loop.
    cd_shift = torch.cat([torch.zeros_like(cd_cumsum[..., :1]), cd_cumsum[..., :-1]], dim=-1)
    # The strict lower triangle excludes the current chunk: its local effect is
    # already represented by Y_diag. Mask with `where` — the causal (z<=c)
    # exponent reaches +1460 and `exp` overflows, so `exp * mask` yields
    # inf * 0 = NaN.
    cd_seg = _clamp_exp_exponent(cd_shift.unsqueeze(-1) - cd_cumsum.unsqueeze(-2))
    causal_chunks = torch.tril(
        torch.ones(n_chunks, n_chunks, device=x.device, dtype=torch.bool), diagonal=-1,
    )
    decay_chunk = torch.where(
        causal_chunks, torch.exp(cd_seg),
        torch.zeros((), dtype=cd_seg.dtype, device=cd_seg.device),
    )

    states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk.to(states.dtype), states)
    # There is no external initial state: the scan starts at zero, so only
    # completed earlier chunks contribute to this cross-chunk term.

    Y_off = torch.einsum("bclhn,bchpn,bclh->bclhp", Cc.to(states.dtype), states,
                         torch.exp(A_cumsum).to(states.dtype))

    Y = Y_diag + Y_off
    Y = Y.real
    return Y.reshape(B_, T_padded, H, D)[:, :T, :, :]

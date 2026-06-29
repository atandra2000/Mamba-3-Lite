"""SSD — the Mamba-2 sequence-mixing primitive (see ../SSD.md for theory)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable causal segment-sum for the decay matrix."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_seg = x_cumsum.unsqueeze(-1) - x_cumsum.unsqueeze(-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
    return x_seg.masked_fill(~mask, float("-inf"))


def _discretise(dt: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """A_bar = exp(softplus(dt) * A)."""
    return torch.exp(torch.nn.functional.softplus(dt) * A)


def ssd_naive(
    x: torch.Tensor, A: torch.Tensor, B_t: torch.Tensor, C_t: torch.Tensor, dt: torch.Tensor,
) -> torch.Tensor:
    """O(T) sequential SSM scan — reference implementation."""
    B_, T, H, D = x.shape
    assert A.shape == (H,), f"A must be (H,), got {A.shape}"
    N = B_t.shape[-1]

    A_bar = _discretise(dt, A)
    s = torch.zeros(B_, H, N, D, dtype=x.dtype, device=x.device)
    ys = []

    for t in range(T):
        s = A_bar[:, t].unsqueeze(-1).unsqueeze(-1) * s \
            + B_t[:, t].unsqueeze(-1) * x[:, t].unsqueeze(-2)
        ys.append((C_t[:, t].unsqueeze(-1) * s).sum(dim=-2))

    return torch.stack(ys, dim=1)


def ssd_chunkwise(
    x: torch.Tensor, A: torch.Tensor, B_t: torch.Tensor, C_t: torch.Tensor, dt: torch.Tensor,
    chunk_size: int = 64, initial_states: torch.Tensor | None = None,
) -> torch.Tensor:
    """O(T·C) chunkwise SSD — production algorithm (matmul-friendly intra-chunk)."""
    B_, T, H, D = x.shape
    assert A.shape == (H,), f"A must be (H,), got {A.shape}"
    N, C = B_t.shape[-1], chunk_size

    pad = (C - (T % C)) % C
    if pad > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        B_t = F.pad(B_t, (0, 0, 0, 0, 0, pad))
        C_t = F.pad(C_t, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))

    T_padded = T + pad
    n_chunks = T_padded // C
    A_log = torch.log(_discretise(dt, A).clamp_min(1e-8))

    def _chunk(t):
        return t.reshape(B_, n_chunks, C, *t.shape[2:])

    Xc, Bc, Cc, Ac = _chunk(x), _chunk(B_t), _chunk(C_t), _chunk(A_log)

    A_cumsum = torch.cumsum(Ac, dim=2)
    L = torch.exp(segsum(Ac.permute(0, 1, 3, 2).contiguous()))
    Y_diag = torch.einsum("bclhn,bcshn,bchls,bcshp->bclhp", Cc, Bc, L, Xc)

    decay_states = torch.exp(A_cumsum[:, :, -1:, :] - A_cumsum)
    states = torch.einsum("bclhn,bclh,bclhp->bchpn", Bc, decay_states, Xc)

    if initial_states is None:
        initial_states = torch.zeros(B_, H, D, N, device=x.device, dtype=x.dtype)
    states = torch.cat([initial_states.unsqueeze(1), states], dim=1)

    chunk_decay = A_cumsum[:, :, -1, :]
    decay_chunk = torch.exp(segsum(chunk_decay.permute(0, 2, 1).contiguous()))
    states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states[:, :-1])

    Y_off = torch.einsum("bclhn,bchpn,bclh->bclhp", Cc, states, torch.exp(A_cumsum))

    return (Y_diag + Y_off).reshape(B_, T_padded, H, D)[:, :T, :, :]

"""Complex SSD — the Mamba-3 sequence-mixing primitive."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ssd_complex_chunkwise(
    x: torch.Tensor, A: torch.Tensor, B_t: torch.Tensor, C_t: torch.Tensor, dt: torch.Tensor,
    chunk_size: int = 64, initial_states: torch.Tensor | None = None,
    ssd_dispatch: str = "pytorch",
) -> torch.Tensor:
    """Complex chunkwise SSD.

    `ssd_dispatch='pytorch'`: the original 5-einsum PyTorch chain.
    `ssd_dispatch='triton'`: the per-chunk `Y_diag` and `state` passes are
    fused into a single Triton kernel via `per_chunk_ssd_triton`; the
    inter-chunk state propagation and the final `Y_off` application stay
    in PyTorch. Opt-in requires `ENABLE_TRITON_KERNELS=1` (enforced at
    the `Pretrainer` level); a kernel failure auto-falls back with a
    one-shot warning.
    """
    B_, T, H, D = x.shape
    # A is complex, B_t is complex, C_t is complex, x is real
    N, C = B_t.shape[-1], chunk_size

    pad = (C - (T % C)) % C
    if pad > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        B_t = F.pad(B_t, (0, 0, 0, 0, 0, pad))
        C_t = F.pad(C_t, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))

    T_padded = T + pad
    n_chunks = T_padded // C

    A_log = F.softplus(dt) * A  # (B, T_padded, H)

    def _chunk(t):
        return t.reshape(B_, n_chunks, C, *t.shape[2:])

    Xc, Bc, Cc, Ac = _chunk(x).to(torch.complex64), _chunk(B_t), _chunk(C_t), _chunk(A_log)

    A_cumsum = torch.cumsum(Ac, dim=2)
    decay_states = torch.exp(A_cumsum[:, :, -1:, :] - A_cumsum)

    if ssd_dispatch == "triton":
        from .ssd_triton import per_chunk_ssd_triton
        Y_diag, states = per_chunk_ssd_triton(
            Bc, Cc, Xc, Ac, decay_states, B_t, C_t, A, dt, chunk_size,
        )
    else:
        Ac_perm = Ac.permute(0, 1, 3, 2).contiguous()
        T_c = Ac_perm.size(-1)
        Ac_cumsum = torch.cumsum(Ac_perm, dim=-1)
        Ac_seg = Ac_cumsum.unsqueeze(-1) - Ac_cumsum.unsqueeze(-2)
        mask = torch.tril(torch.ones(T_c, T_c, device=x.device, dtype=torch.bool))
        L = torch.exp(Ac_seg) * mask

        Y_diag = torch.einsum("bclhn,bcshn,bchls,bcshp->bclhp", Cc, Bc, L, Xc)
        states = torch.einsum("bclhn,bclh,bclhp->bchpn", Bc, decay_states, Xc)

    if initial_states is None:
        initial_states = torch.zeros(B_, H, D, N, device=x.device, dtype=torch.complex64)
    states = torch.cat([initial_states.unsqueeze(1), states], dim=1)

    chunk_decay = A_cumsum[:, :, -1, :]
    cd_perm = chunk_decay.permute(0, 2, 1).contiguous()
    cd_cumsum = torch.cumsum(cd_perm, dim=-1)
    cd_seg = cd_cumsum.unsqueeze(-1) - cd_cumsum.unsqueeze(-2)
    decay_chunk = torch.exp(cd_seg) * torch.tril(torch.ones(n_chunks, n_chunks, device=x.device, dtype=torch.bool))

    states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states[:, :-1])

    Y_off = torch.einsum("bclhn,bchpn,bclh->bclhp", Cc, states, torch.exp(A_cumsum))

    Y = Y_diag + Y_off
    Y = Y.real
    return Y.reshape(B_, T_padded, H, D)[:, :T, :, :]


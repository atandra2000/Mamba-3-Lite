"""SSD — the Mamba-2 sequence-mixing primitive (see ../SSD.md for theory)."""
from __future__ import annotations

import torch
# ponytail: torch.nn.functional as F removed — only ssd_chunkwise (deleted) used F.pad.


def _discretise(dt: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """A_bar = exp(softplus(dt) * A)."""
    return torch.exp(torch.nn.functional.softplus(dt) * A)


def ssd_naive_complex(
    x: torch.Tensor, A: torch.Tensor, B_t: torch.Tensor, C_t: torch.Tensor, dt: torch.Tensor,
) -> torch.Tensor:
    """O(T) sequential complex SSM scan — reference oracle for ssd_complex_chunkwise."""
    B_, T, H, D = x.shape
    N = B_t.shape[-1]
    A_bar = _discretise(dt, A)
    s = torch.zeros(B_, H, N, D, dtype=torch.complex64, device=x.device)
    ys = []
    for t in range(T):
        s = A_bar[:, t].unsqueeze(-1).unsqueeze(-1) * s \
            + B_t[:, t].unsqueeze(-1) * x[:, t].unsqueeze(-2)
        ys.append((C_t[:, t].unsqueeze(-1) * s).sum(dim=-2))
    return torch.stack(ys, dim=1)


# ponytail: real-valued ssd_chunkwise deleted — superseded by ssd_complex_chunkwise
# (models/ssd_complex.py, production path). ssd_naive kept as the reference oracle.

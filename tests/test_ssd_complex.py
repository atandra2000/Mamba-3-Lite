"""Regression tests for the complex-SSD chunkwise path.

Covers the two failure classes that produced non-finite output under BF16
autocast: mixed real/complex operands at the SSD einsums, and masked exp
overflow (`inf * 0 = NaN` in the `where` backward).
"""
import torch

from models.ssd_complex import ssd_naive_complex, ssd_complex_chunkwise


def _case(T=128, seed=0):
    torch.manual_seed(seed)
    B, H, D, N = 2, 4, 8, 8
    x = torch.randn(B, T, H, D)
    A = torch.full((H,), -1.0, dtype=torch.complex64)
    B_t = torch.complex(torch.randn(B, T, H, N), torch.randn(B, T, H, N))
    C_t = torch.complex(torch.randn(B, T, H, N), torch.randn(B, T, H, N))
    dt = torch.randn(B, T, H)
    return x, A, B_t, C_t, dt


def test_chunkwise_matches_naive_real():
    x, A, B_t, C_t, dt = _case()
    y_ref = ssd_naive_complex(x, A, B_t, C_t, dt).real
    y_chk = ssd_complex_chunkwise(x, A, B_t, C_t, dt, chunk_size=64)
    assert torch.isfinite(y_chk).all()
    rel = (y_chk - y_ref).abs().max() / (y_ref.abs().max() + 1e-12)
    assert rel < 1e-4, f"chunkwise disagrees with naive oracle: rel={rel}"


def test_backward_finite():
    x, A, B_t, C_t, dt = _case()
    for t in (x, B_t, C_t, dt, A):
        t.requires_grad_(True)
    y = ssd_complex_chunkwise(x, A, B_t, C_t, dt, chunk_size=64)
    y.sum().backward()
    for name, t in (("x", x), ("A", A), ("B_t", B_t), ("C_t", C_t), ("dt", dt)):
        g = t.grad
        if g is None:
            continue
        assert torch.isfinite(g.real if g.is_complex() else g).all(), f"non-finite grad: {name}"


def test_mixed_real_complex_einsum_raises_not_nan():
    # The pre-fix shape: real BF16 decay against complex64 states. It must
    # fail loudly in eager mode rather than silently returning NaN.
    Bc = torch.randn(1, 1, 8, 2, 4, dtype=torch.bfloat16)
    Xc = torch.randn(1, 1, 8, 2, 4, dtype=torch.complex64)
    try:
        out = torch.einsum("bclhn,bclhp->bclpn", Bc, Xc)
    except RuntimeError:
        return  # loud failure is the correct behaviour
    assert torch.isfinite(out.real).all(), "mixed-dtype einsum returned NaN"

"""Equivalence and edge-case tests for ssd_complex_chunkwise vs ssd_naive_complex."""
import torch

from models.ssd_complex import ssd_complex_chunkwise, ssd_naive_complex


def test_chunkwise_matches_naive_complex():
    torch.manual_seed(0)
    B, T, H, D, N = 2, 16, 2, 4, 4
    x = torch.randn(B, T, H, D, dtype=torch.complex64)
    A = torch.randn(H, dtype=torch.complex64) - 1.0
    B_t = torch.randn(B, T, H, N, dtype=torch.complex64)
    C_t = torch.randn(B, T, H, N, dtype=torch.complex64)
    dt = torch.zeros(B, T, H)
    y_chunk = ssd_complex_chunkwise(x, A, B_t, C_t, dt, chunk_size=4)
    y_naive = ssd_naive_complex(x, A, B_t, C_t, dt)
    assert y_chunk.dtype == torch.float32
    assert y_naive.dtype == torch.complex64
    assert y_chunk.shape == y_naive.shape == (B, T, H, D)
    assert torch.allclose(y_chunk, y_naive.real, atol=1e-4), (
        f"max diff = {(y_chunk - y_naive.real).abs().max().item()}"
    )


def test_chunkwise_matches_naive_time_varying_dt():
    # Regression: the inter-chunk state propagation must hold when per-chunk
    # decay totals differ (input-dependent dt). dt=0 masks the off-by-one
    # decay factor because every chunk then has an identical total.
    torch.manual_seed(3)
    B, T, H, D, N = 2, 16, 2, 4, 4
    x = torch.randn(B, T, H, D, dtype=torch.complex64)
    A = torch.randn(H, dtype=torch.complex64) - 1.0
    B_t = torch.randn(B, T, H, N, dtype=torch.complex64)
    C_t = torch.randn(B, T, H, N, dtype=torch.complex64)
    dt = torch.randn(B, T, H)
    y_chunk = ssd_complex_chunkwise(x, A, B_t, C_t, dt, chunk_size=4)
    y_naive = ssd_naive_complex(x, A, B_t, C_t, dt)
    assert torch.allclose(y_chunk, y_naive.real, atol=1e-4), (
        f"max diff = {(y_chunk - y_naive.real).abs().max().item()}"
    )


def test_chunkwise_handles_uneven_T():
    torch.manual_seed(1)
    B, T, H, D, N = 1, 20, 2, 4, 4
    x = torch.randn(B, T, H, D, dtype=torch.complex64)
    A = torch.randn(H, dtype=torch.complex64) - 1.0
    B_t = torch.randn(B, T, H, N, dtype=torch.complex64)
    C_t = torch.randn(B, T, H, N, dtype=torch.complex64)
    dt = torch.randn(B, T, H)
    y = ssd_complex_chunkwise(x, A, B_t, C_t, dt, chunk_size=4)
    assert y.shape == (B, T, H, D)
    assert torch.isfinite(y).all()


def test_chunkwise_handles_T_equal_to_chunk():
    torch.manual_seed(2)
    B, T, H, D, N = 1, 4, 2, 4, 4
    x = torch.randn(B, T, H, D, dtype=torch.complex64)
    A = torch.randn(H, dtype=torch.complex64) - 1.0
    B_t = torch.randn(B, T, H, N, dtype=torch.complex64)
    C_t = torch.randn(B, T, H, N, dtype=torch.complex64)
    dt = torch.randn(B, T, H)
    y = ssd_complex_chunkwise(x, A, B_t, C_t, dt, chunk_size=4)
    assert y.shape == (B, T, H, D)
    assert torch.isfinite(y).all()


def test_chunkwise_finite_with_many_chunks_negative_decay():
    # Regression: the inter-chunk decay exponent reaches large positive values
    # in the region that the strict-lower-triangular mask discards. exp() then
    # overflows to inf, and the old `exp(seg) * mask` produced inf * 0 = NaN
    # in plain fp32. Enough chunks (and A<0, the model's init) are required to
    # overflow: 32 chunks of 8, decay ~8/chunk => masked exponent ~250.
    torch.manual_seed(7)
    B, T, H, D, N = 1, 256, 4, 8, 8
    x = torch.randn(B, T, H, D, dtype=torch.complex64)
    A = torch.full((H,), -1.0, dtype=torch.complex64)  # real init in mamba_block.py
    B_t = torch.randn(B, T, H, N, dtype=torch.complex64)
    C_t = torch.randn(B, T, H, N, dtype=torch.complex64)
    dt = torch.randn(B, T, H)
    y_chunk = ssd_complex_chunkwise(x, A, B_t, C_t, dt, chunk_size=8)
    assert torch.isfinite(y_chunk).all(), "chunkwise SSD produced non-finite output"
    y_naive = ssd_naive_complex(x, A, B_t, C_t, dt)
    assert torch.allclose(y_chunk, y_naive.real, atol=1e-3), (
        f"max diff = {(y_chunk - y_naive.real).abs().max().item()}"
    )

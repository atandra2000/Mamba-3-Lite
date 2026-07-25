"""MIMO mixer tests."""
import torch

from models.mimo import MIMO


def test_mimo_identity_init():
    m = MIMO(d_model=128, n_heads=4, head_dim=32)
    m.eval()
    x = torch.randn(2, 8, 4, 32)
    y = m(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6), f"max diff = {(y - x).abs().max().item()}"


def test_mimo_shape_and_finite():
    torch.manual_seed(0)
    m = MIMO(d_model=128, n_heads=4, head_dim=32)
    m.train()
    x = torch.randn(2, 8, 4, 32)
    y = m(x)
    assert y.shape == (2, 8, 4, 32)
    assert torch.isfinite(y).all()

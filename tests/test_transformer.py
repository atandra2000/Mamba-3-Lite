"""Mamba3Transformer forward smoke + config acceptance tests."""
import torch

from models.transformer import Mamba3Transformer, ModelConfig


def test_mamba3_transformer_forward():
    """Tiny config — forward shape, param count, weight tying sanity."""
    cfg = ModelConfig(
        vocab_size=100, d_model=64, n_layers=2, n_heads=4,
        head_dim=16, state_dim=8, chunk_size=4, ffn_dim=128,
        max_seq_len=32, dtype="fp32", weight_tying=True,
    )
    m = Mamba3Transformer(cfg)
    x = torch.randint(0, 100, (2, 16))
    y = m(x)
    assert y.shape == (2, 16, 100), y.shape
    n = sum(p.numel() for p in m.parameters())
    # Tiny config: ~97k params. Sanity floor: must be > embed+lm_head+2 layer weights (~80k).
    assert 80_000 < n < 5_000_000, f"unexpected param count: {n}"
    # Weight tying: embed.weight and lm_head.weight share storage.
    assert m.embed.weight.data_ptr() == m.lm_head.weight.data_ptr(), "weight tying not applied"


def test_mamba3_transformer_accepts_dict_config():
    """Mamba3Transformer(dict) must behave equivalently to Mamba3Transformer(ModelConfig)."""
    cfg_dict = {
        "vocab_size": 50, "d_model": 32, "n_layers": 2, "n_heads": 2,
        "head_dim": 16, "state_dim": 4, "chunk_size": 4,
        "ffn_dim": 64, "max_seq_len": 16, "dtype": "fp32", "weight_tying": True,
    }
    m = Mamba3Transformer(cfg_dict)
    x = torch.randint(0, 50, (2, 8))
    y = m(x)
    assert y.shape == (2, 8, 50)

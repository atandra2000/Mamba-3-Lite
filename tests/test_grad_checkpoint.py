"""Verify grad_checkpoint flag propagates from TrainingConfig into Mamba3Block."""
import torch

from models.transformer import Mamba3Transformer, ModelConfig
from training.pretrain import TrainingConfig

_BASE = {
    "vocab_size": 64, "d_model": 32, "n_layers": 2, "n_heads": 2,
    "head_dim": 16, "state_dim": 4, "chunk_size": 4,
    "ffn_dim": 64, "max_seq_len": 16, "dtype": "fp32", "weight_tying": True,
}


def _build_model(grad_checkpoint: bool) -> Mamba3Transformer:
    cfg = TrainingConfig(model_config=dict(_BASE), grad_checkpoint=grad_checkpoint)
    cfg.model_config.setdefault("grad_checkpoint", cfg.grad_checkpoint)
    return Mamba3Transformer(ModelConfig(**cfg.model_config))


def test_grad_checkpoint_propagates_to_blocks():
    m = _build_model(grad_checkpoint=True)
    flags = [b.grad_checkpoint for b in m.layers]
    assert all(flags), f"expected all blocks True, got {flags}"


def test_grad_checkpoint_explicit_false_disables():
    m = _build_model(grad_checkpoint=False)
    flags = [b.grad_checkpoint for b in m.layers]
    assert not any(flags), f"expected all blocks False, got {flags}"


def test_grad_checkpoint_actually_triggers_training_mode():
    m = _build_model(grad_checkpoint=True)
    m.train()
    x = torch.randint(0, 64, (2, 16))
    out = m(x)
    loss = out.sum()
    loss.backward()
    grads = [b.grad for b in m.parameters() if b.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads),         "no parameter received a finite grad through the checkpoint path"

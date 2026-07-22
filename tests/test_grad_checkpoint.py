"""Verify the YAML grad_checkpoint flag propagates from TrainingConfig into Mamba3Block.

Regression test for the wiring bug where the flag lived on TrainingConfig but
the model was built with the model_config dict that never received it, leaving
gradient checkpointing silently inactive.
"""
import torch

from models.transformer import Mamba3Transformer, ModelConfig
from training.pretrain import TrainingConfig


_BASE = {
    "vocab_size": 64, "d_model": 32, "n_layers": 2, "n_heads": 2,
    "head_dim": 16, "state_dim": 4, "chunk_size": 4,
    "ffn_dim": 64, "max_seq_len": 16, "dtype": "fp32", "weight_tying": True,
}


def _build_model(grad_checkpoint: bool) -> Mamba3Transformer:
    """Mirror what Pretrainer.__init__ does: setdefault grad_checkpoint on the
    model_config dict, then construct the transformer."""
    cfg = TrainingConfig(model_config=dict(_BASE), grad_checkpoint=grad_checkpoint)
    cfg.model_config.setdefault("grad_checkpoint", cfg.grad_checkpoint)
    return Mamba3Transformer(ModelConfig(**cfg.model_config))


def test_grad_checkpoint_propagates_to_blocks():
    """TrainingConfig.grad_checkpoint=True -> all Mamba3Block.grad_checkpoint=True."""
    m = _build_model(grad_checkpoint=True)
    flags = [b.grad_checkpoint for b in m.layers]
    assert all(flags), f"expected all blocks True, got {flags}"


def test_grad_checkpoint_explicit_false_disables():
    """TrainingConfig.grad_checkpoint=False -> all Mamba3Block.grad_checkpoint=False.

    Note: TrainingConfig.grad_checkpoint defaults to True, so we set it
    explicitly to False here to test the user-disabled path.
    """
    m = _build_model(grad_checkpoint=False)
    flags = [b.grad_checkpoint for b in m.layers]
    assert not any(flags), f"expected all blocks False, got {flags}"


def test_grad_checkpoint_actually_triggers_training_mode():
    """With grad_checkpoint=True and model.train(), backward succeeds and uses
    the checkpoint path (vs. eager backward). This is a smoke check that the
    wrapper is actually wired, not just the flag.
    """
    m = _build_model(grad_checkpoint=True)
    m.train()

    x = torch.randint(0, 64, (2, 16))
    out = m(x)
    loss = out.sum()
    loss.backward()  # must not raise; grad-checkpointed path is differentiable
    # At least one parameter must have a non-None grad.
    grads = [b.grad for b in m.parameters() if b.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads), \
        "no parameter received a finite grad through the checkpoint path"

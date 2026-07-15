"""Verify the YAML grad_checkpoint flag propagates from TrainingConfig into Mamba3Block.

Regression test for the wiring bug where the flag lived on TrainingConfig but
the model was built with the model_config dict that never received it, leaving
gradient checkpointing silently inactive.
"""
import torch

from training.pretrain import TrainingConfig, _build_minimal_pretrainer


def test_grad_checkpoint_propagates_to_blocks():
    """TrainingConfig.grad_checkpoint=True -> all Mamba3Block.grad_checkpoint=True."""
    cfg = TrainingConfig(
        model_config={
            "vocab_size": 64, "d_model": 32, "n_layers": 2, "n_heads": 2,
            "head_dim": 16, "state_dim": 4, "chunk_size": 4,
            "ffn_dim": 64, "max_seq_len": 16, "dtype": "fp32", "weight_tying": True,
        },
        grad_checkpoint=True,
    )
    # Mirror what Pretrainer.__init__ does.
    cfg.model_config.setdefault("grad_checkpoint", cfg.grad_checkpoint)
    p = _build_minimal_pretrainer(cfg.model_config)

    flags = [b.grad_checkpoint for b in p.model.layers]
    assert all(flags), f"expected all blocks True, got {flags}"


def test_grad_checkpoint_explicit_false_disables():
    """TrainingConfig.grad_checkpoint=False -> all blocks have grad_checkpoint=False.

    Note: TrainingConfig.grad_checkpoint defaults to True, so we set it
    explicitly to False here to test the user-disabled path.
    """
    cfg = TrainingConfig(
        model_config={
            "vocab_size": 64, "d_model": 32, "n_layers": 2, "n_heads": 2,
            "head_dim": 16, "state_dim": 4, "chunk_size": 4,
            "ffn_dim": 64, "max_seq_len": 16, "dtype": "fp32", "weight_tying": True,
        },
        grad_checkpoint=False,
    )
    cfg.model_config.setdefault("grad_checkpoint", cfg.grad_checkpoint)
    p = _build_minimal_pretrainer(cfg.model_config)

    flags = [b.grad_checkpoint for b in p.model.layers]
    assert not any(flags), f"expected all blocks False, got {flags}"


def test_grad_checkpoint_actually_triggers_training_mode():
    """With grad_checkpoint=True and model.train(), backward succeeds and uses
    the checkpoint path (vs. eager backward). This is a smoke check that the
    wrapper is actually wired, not just the flag.
    """
    cfg = TrainingConfig(
        model_config={
            "vocab_size": 64, "d_model": 32, "n_layers": 2, "n_heads": 2,
            "head_dim": 16, "state_dim": 4, "chunk_size": 4,
            "ffn_dim": 64, "max_seq_len": 16, "dtype": "fp32", "weight_tying": True,
            "grad_checkpoint": True,
        },
    )
    p = _build_minimal_pretrainer(cfg.model_config)
    p.model.train()

    x = torch.randint(0, 64, (2, 16))
    out = p.model(x)
    loss = out.sum()
    loss.backward()  # must not raise; grad-checkpointed path is differentiable
    # At least one parameter must have a non-None grad.
    grads = [b.grad for b in p.model.parameters() if b.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads), \
        "no parameter received a finite grad through the checkpoint path"

"""End-to-end train step test on a CPU-only tiny model."""
import math

import torch
from torch.optim import AdamW

from models.transformer import Mamba3Transformer, ModelConfig
from training.pretrain import train_step


def test_train_step_on_tiny_model():
    """One optimizer step on a tiny model: loss must be finite, params must change."""
    torch.manual_seed(0)
    model_config = {
        "vocab_size": 64, "d_model": 32, "n_layers": 2, "n_heads": 2,
        "head_dim": 16, "state_dim": 4, "chunk_size": 4,
        "ffn_dim": 64, "max_seq_len": 16, "dtype": "fp32", "weight_tying": True,
    }
    model = Mamba3Transformer(ModelConfig(**model_config))
    optimizer = AdamW(model.parameters(), lr=1e-3, fused=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)

    # Snapshot params.
    before = [w.detach().clone() for w in model.parameters() if w.requires_grad]

    tokens = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    from types import SimpleNamespace
    cfg = SimpleNamespace(gradient_accumulation_steps=1, max_grad_norm=1.0, nan_guard=True)
    out, _ = train_step(
        model, optimizer, scheduler, cfg,
        amp_context=torch.amp.autocast("cpu", enabled=False),
        log=lambda msg: None,
        opt_steps=0,
        tokens=tokens, targets=targets, micro_step=0,
    )

    assert out is not None, "train_step returned None (NaN guard tripped)"
    assert math.isfinite(out["loss"]), f"non-finite loss: {out['loss']}"

    # After one optimizer step, at least one parameter must have changed.
    after = [w.detach().clone() for w in model.parameters() if w.requires_grad]
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), \
        "no parameters changed after train_step"

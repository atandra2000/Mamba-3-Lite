"""End-to-end train step test on a CPU-only Pretrainer built by the minimal factory."""
import math

import torch

from training.pretrain import _build_minimal_pretrainer


def test_train_step_on_tiny_model():
    """One optimizer step on a tiny model: loss must be finite, params must change."""
    torch.manual_seed(0)
    p = _build_minimal_pretrainer({
        "vocab_size": 64, "d_model": 32, "n_layers": 2, "n_heads": 2,
        "head_dim": 16, "state_dim": 4, "chunk_size": 4,
        "ffn_dim": 64, "max_seq_len": 16, "dtype": "fp32", "weight_tying": True,
    })
    # Snapshot params.
    before = [w.detach().clone() for w in p.model.parameters() if w.requires_grad]

    tokens = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    out = p.train_step(tokens, targets, micro_step=0)

    assert out is not None, "train_step returned None (NaN guard tripped)"
    assert math.isfinite(out["loss"]), f"non-finite loss: {out['loss']}"

    # After one optimizer step, at least one parameter must have changed.
    after = [w.detach().clone() for w in p.model.parameters() if w.requires_grad]
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), \
        "no parameters changed after train_step"

"""End-to-end GPU pipeline smoke test for Mamba-3-Lite.

Run: ENABLE_TRITON_KERNELS=1 ~/.venv/bin/python tests/e2e_gpu_smoke.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("ENABLE_TRITON_KERNELS", "1")
os.environ.setdefault("TRITON_PER_CHUNK_NUM_STAGES", "1")
os.environ.setdefault("TRITON_PER_CHUNK_NUM_WARPS", "2")

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from models.transformer import Mamba3Transformer, ModelConfig
from models.ssd_complex import ssd_complex_chunkwise, ssd_naive_complex
from training.pretrain import PretrainDataset, train_step, TrainingConfig, Pretrainer
from utils.checkpoint import CheckpointManager


def _tiny_cfg(ssd_dispatch: str = "pytorch") -> ModelConfig:
    return ModelConfig(
        vocab_size=128, d_model=64, n_layers=2, n_heads=4,
        head_dim=16, state_dim=16, chunk_size=16, ffn_dim=128,
        max_seq_len=32, weight_tying=True,
        ssd_dispatch=ssd_dispatch,
    )


def _build_synthetic_shard(path: Path, n_tokens: int = 4096, vocab: int = 128):
    tokens = torch.randint(0, vocab, (n_tokens,), dtype=torch.long)
    torch.save(tokens, path)


# --------------------------------------------------------------------------- #
# 1. Environment
# --------------------------------------------------------------------------- #

def check_environment() -> torch.device:
    print("=" * 70)
    print("[1/8] Environment check")
    print("=" * 70)
    print(f"  torch       : {torch.__version__}")
    print(f"  cuda avail  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  device      : {name} (cc {cap[0]}.{cap[1]}, {total:.1f} GB)")
    try:
        import triton
        print(f"  triton      : {triton.__version__}")
    except Exception as e:
        print(f"  triton      : MISSING ({e})")
    print(f"  ENABLE_TRITON_KERNELS : {os.environ.get('ENABLE_TRITON_KERNELS', '0')}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  -> using device: {device}")
    assert device.type == "cuda", "E2E test requires CUDA"
    return device


# --------------------------------------------------------------------------- #
# 2. Data pipeline
# --------------------------------------------------------------------------- #

def check_data_pipeline(device: torch.device):
    print()
    print("=" * 70)
    print("[2/8] Data pipeline (synthetic shard -> PretrainDataset -> DataLoader -> GPU)")
    print("=" * 70)
    with tempfile.TemporaryDirectory() as tmp:
        shard_path = Path(tmp) / "shard_0000.bin"
        _build_synthetic_shard(shard_path, n_tokens=4096, vocab=128)
        ds = PretrainDataset(str(shard_path), max_seq_len=32, vocab_size=128)
        print(f"  dataset len  : {len(ds)} samples")
        assert len(ds) > 0
        inp, tgt = ds[0]
        print(f"  sample shape : input={inp.shape} target={tgt.shape} dtype={inp.dtype}")
        assert inp.shape == (32,) and tgt.shape == (32,)
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=4, drop_last=True)
        batch = next(iter(loader))
        inp_b, tgt_b = batch[0].to(device), batch[1].to(device)
        print(f"  batch shape  : input={tuple(inp_b.shape)} target={tuple(tgt_b.shape)}")
        assert inp_b.shape == (4, 32) and tgt_b.shape == (4, 32)
        assert inp_b.is_cuda
    return loader


# --------------------------------------------------------------------------- #
# 3. Model (pytorch dispatch)
# --------------------------------------------------------------------------- #

def check_model_pytorch(device: torch.device):
    print()
    print("=" * 70)
    print("[3/8] Model forward (pytorch dispatch) on GPU")
    print("=" * 70)
    torch.manual_seed(42)
    cfg = _tiny_cfg("pytorch")
    model = Mamba3Transformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,} ({n_params/1e6:.2f}M)")
    x = torch.randint(0, 128, (2, 16), device=device)
    with torch.no_grad():
        y = model(x)
    print(f"  output shape: {tuple(y.shape)} dtype={y.dtype}")
    assert y.shape == (2, 16, 128)
    assert torch.isfinite(y).all().item()
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU mem: {(total-free)/1e6:.1f} MB used / {total/1e6:.0f} MB total")
    return model


# --------------------------------------------------------------------------- #
# 4. Model (triton dispatch)
# --------------------------------------------------------------------------- #

def check_model_triton(device: torch.device):
    print()
    print("=" * 70)
    print("[4/8] Model forward (triton dispatch) on GPU")
    print("=" * 70)
    torch.manual_seed(42)
    cfg = _tiny_cfg("triton")
    model = Mamba3Transformer(cfg).to(device)
    assert all(b.ssd_dispatch == "triton" for b in model.layers)
    x = torch.randint(0, 128, (2, 16), device=device)
    with torch.no_grad():
        y = model(x)
    print(f"  output shape: {tuple(y.shape)} dtype={y.dtype}")
    assert y.shape == (2, 16, 128)
    assert torch.isfinite(y).all().item()
    for i, b in enumerate(model.layers):
        assert not b._triton_fallback_warned, f"layer {i} fell back to pytorch!"
    print("  all layers used the Triton kernel (no fallback)")
    return model


# --------------------------------------------------------------------------- #
# 5. Triton vs pytorch parity
# --------------------------------------------------------------------------- #

def check_triton_vs_pytorch(device: torch.device):
    print()
    print("=" * 70)
    print("[5/8] Triton vs pytorch dispatch parity")
    print("=" * 70)
    torch.manual_seed(0)
    B, T, H, D, N = 1, 16, 2, 16, 16
    x = torch.randn(B, T, H, D, dtype=torch.complex64, device=device)
    A = torch.randn(H, dtype=torch.complex64, device=device) - 1.0
    B_t = torch.randn(B, T, H, N, dtype=torch.complex64, device=device)
    C_t = torch.randn(B, T, H, N, dtype=torch.complex64, device=device)
    dt = torch.randn(B, T, H, device=device) * 0.1
    y_p = ssd_complex_chunkwise(x, A, B_t, C_t, dt, chunk_size=16, ssd_dispatch="pytorch")
    y_t = ssd_complex_chunkwise(x, A, B_t, C_t, dt, chunk_size=16, ssd_dispatch="triton")
    y_n = ssd_naive_complex(x, A, B_t, C_t, dt)
    diff_pt = (y_p - y_t).abs().max().item()
    diff_pn = (y_p - y_n.real).abs().max().item()
    diff_tn = (y_t - y_n.real).abs().max().item()
    print(f"  pytorch vs triton : {diff_pt:.2e}")
    print(f"  pytorch vs naive  : {diff_pn:.2e}")
    print(f"  triton  vs naive  : {diff_tn:.2e}")
    assert diff_pt < 1e-3, f"triton vs pytorch diff too large: {diff_pt}"
    assert diff_tn < 1e-3, f"triton vs naive diff too large: {diff_tn}"


# --------------------------------------------------------------------------- #
# 6. Training step
# --------------------------------------------------------------------------- #

def check_training_step(device: torch.device, loader):
    print()
    print("=" * 70)
    print("[6/8] Training step (forward + backward + AdamW) - triton dispatch")
    print("=" * 70)
    torch.manual_seed(42)
    cfg = _tiny_cfg("triton")
    model = Mamba3Transformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)
    from types import SimpleNamespace
    tcfg = SimpleNamespace(
        gradient_accumulation_steps=1, max_grad_norm=1.0, nan_guard=True,
    )
    losses = []
    it = iter(loader)
    for step in range(4):
        try:
            inp, tgt = next(it)
        except StopIteration:
            it = iter(loader)
            inp, tgt = next(it)
        inp, tgt = inp.to(device), tgt.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            metrics, _ = train_step(
                model, opt, sched, tcfg,
                torch.amp.autocast("cuda", dtype=torch.bfloat16),
                lambda msg: None, 0,
                inp, tgt, step,
            )
        if metrics is not None:
            losses.append(metrics["loss"])
            print(f"  step {step+1}/4: loss={metrics['loss']:.4f}")
    assert all(math.isfinite(l) for l in losses), f"non-finite loss: {losses}"
    assert len(losses) >= 2, "not enough successful steps"
    print(f"  losses: {losses}")
    assert losses[-1] < losses[0] * 1.5, "loss is not decreasing"
    peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"  peak VRAM: {peak:.1f} MB")
    return model


# --------------------------------------------------------------------------- #
# 7. Checkpoint round-trip
# --------------------------------------------------------------------------- #

def check_checkpoint(device: torch.device, model):
    print()
    print("=" * 70)
    print("[7/8] Checkpoint save + load round-trip")
    print("=" * 70)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_mgr = CheckpointManager(tmp)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        x = torch.randint(0, 128, (2, 16), device=device)
        model.eval()
        with torch.no_grad():
            ref = model(x).clone()
        model.train()
        ckpt_mgr.save(model, opt, step=1, extra_meta={"step": 1})
        out = Path(tmp) / "model_step_1.safetensors"
        print(f"  saved: {out.name} ({out.stat().st_size/1024:.1f} KB)")
        assert out.exists()
        torch.manual_seed(999)
        fresh = Mamba3Transformer(_tiny_cfg("triton")).to(device)
        fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=3e-4)
        meta = ckpt_mgr.load(fresh, step=1, device=str(device), optimizer=fresh_opt, strict=False)
        fresh.eval()
        with torch.no_grad():
            got = fresh(x)
        diff = (ref - got).abs().max().item()
        print(f"  loaded: step={meta.get('step')} drift={diff:.2e}")
        assert diff < 1e-4, f"checkpoint drift too large: {diff}"


# --------------------------------------------------------------------------- #
# 8. Full Pretrainer dry-run
# --------------------------------------------------------------------------- #

def check_pretrainer_dry_run(device: torch.device):
    print()
    print("=" * 70)
    print("[8/8] Full Pretrainer dry-run (2 steps, triton dispatch)")
    print("=" * 70)
    with tempfile.TemporaryDirectory() as tmp:
        shard_path = Path(tmp) / "shard_0000.bin"
        _build_synthetic_shard(shard_path, n_tokens=2048, vocab=128)
        ckpt_dir = Path(tmp) / "checkpoints"
        cfg = TrainingConfig(
            model_config=dict(
                vocab_size=128, d_model=64, n_layers=2, n_heads=4,
                head_dim=16, state_dim=16, chunk_size=16, ffn_dim=128,
                max_seq_len=32, weight_tying=True,
                ssd_dispatch="triton", grad_checkpoint=False,
            ),
            data_path=str(shard_path),
            checkpoint_dir=str(ckpt_dir),
            vocab_size=128, max_seq_len=32,
            batch_size=2, gradient_accumulation_steps=1,
            max_steps=2, warmup_steps=0,
            lr=3e-4, weight_decay=0.1,
            grad_checkpoint=False, compile_model=False,
            save_every=1000, log_every=1,
            nan_guard=True,
        )
        trainer = Pretrainer(cfg)
        trainer.train()
        print("  Pretrainer dry-run completed successfully")
        latest = trainer._find_latest_checkpoint()
        assert latest is not None, "no checkpoint saved"
        print(f"  final checkpoint: step {latest}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end GPU pipeline smoke test.")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    device = check_environment()
    loader = check_data_pipeline(device)
    check_model_pytorch(device)
    check_model_triton(device)
    check_triton_vs_pytorch(device)
    model = check_training_step(device, loader)
    check_checkpoint(device, model)
    check_pretrainer_dry_run(device)
    print()
    print("=" * 70)
    print("E2E SMOKE: ALL CHECKS PASSED")
    print(f"  device     : {device}")
    print(f"  triton     : enabled (ENABLE_TRITON_KERNELS=1)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

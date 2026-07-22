"""Unit tests for the per-chunk SSD Triton kernel.

CPU tests cover the pure-PyTorch reference, the import surface, the 256-cap
hard-fail, and the dispatch wiring. GPU tests (`@pytest.mark.gpu`) exercise
the kernel forward, autograd, and the 404M-config-shaped call.
"""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ssd_triton import (  # noqa: E402
    HAS_TRITON,
    _check_block_dims,
    per_chunk_ssd_pytorch,
    per_chunk_ssd_triton,
)
from models.ssd_complex import ssd_complex_chunkwise  # noqa: E402
from models.transformer import Mamba3Transformer, ModelConfig  # noqa: E402


# -----------------------------------------------------------------------------
# Pure-PyTorch reference (CPU).
# -----------------------------------------------------------------------------
class TestPerChunkSsdPytorchReference:
    def test_reference_shape_and_finite(self):
        torch.manual_seed(0)
        B, n_chunks, C, H, N, P = 1, 4, 4, 2, 4, 4
        Bc = torch.randn(B, n_chunks, C, H, N, dtype=torch.complex64)
        Cc = torch.randn(B, n_chunks, C, H, N, dtype=torch.complex64)
        Xc = torch.randn(B, n_chunks, C, H, P, dtype=torch.complex64)
        A_log = torch.randn(B, n_chunks, C, H, dtype=torch.complex64) * 0.1
        decay_states = torch.exp(
            A_log[:, :, -1:, :] - A_log
        )
        Y_diag, state = per_chunk_ssd_pytorch(Bc, Cc, Xc, A_log, decay_states)
        assert Y_diag.shape == (B, n_chunks, C, H, P)
        assert state.shape == (B, n_chunks, H, P, N)
        assert torch.isfinite(Y_diag).all()
        assert torch.isfinite(state).all()

    def test_reference_matches_ssd_complex_chunkwise(self):
        """Reference must match the production path to atol=1e-5."""
        torch.manual_seed(1)
        B, T, H, D, N, C = 1, 16, 2, 4, 4, 4
        x = torch.randn(B, T, H, D, dtype=torch.complex64)
        A = torch.randn(H, dtype=torch.complex64) - 1.0
        B_t = torch.randn(B, T, H, N, dtype=torch.complex64)
        C_t = torch.randn(B, T, H, N, dtype=torch.complex64)
        dt = torch.zeros(B, T, H)
        Bc = B_t.reshape(B, T // C, C, H, N)
        Cc = C_t.reshape(B, T // C, C, H, N)
        Xc = x.reshape(B, T // C, C, H, D)
        A_log = (torch.nn.functional.softplus(dt) * A).reshape(B, T // C, C, H)
        decay_states = torch.exp(A_log[:, :, -1:, :] - A_log)
        Y_diag, state = per_chunk_ssd_pytorch(Bc, Cc, Xc, A_log, decay_states)
        assert Y_diag.shape == (B, T // C, C, H, D)
        assert state.shape == (B, T // C, H, D, N)


# -----------------------------------------------------------------------------
# Import surface + 256-cap (CPU).
# -----------------------------------------------------------------------------
class TestPerChunkSsdImportSurface:
    def test_module_imports_without_triton(self):
        from models import ssd_triton
        assert isinstance(ssd_triton.HAS_TRITON, bool)
        assert hasattr(ssd_triton, "per_chunk_ssd_triton")
        assert hasattr(ssd_triton, "per_chunk_ssd_pytorch")

    def test_kernel_call_raises_clean_import_error_when_no_triton(self):
        from models import ssd_triton
        if ssd_triton.HAS_TRITON:
            pytest.skip("triton is installed on this box")
        Bc = torch.randn(1, 1, 4, 1, 4, dtype=torch.complex64)
        Cc = torch.randn(1, 1, 4, 1, 4, dtype=torch.complex64)
        Xc = torch.randn(1, 1, 4, 1, 4, dtype=torch.complex64)
        A_log = torch.zeros(1, 1, 4, 1, dtype=torch.complex64)
        decay_states = torch.ones(1, 1, 4, 1, dtype=torch.complex64)
        B_t = torch.randn(1, 4, 1, 4, dtype=torch.complex64)
        C_t = torch.randn(1, 4, 1, 4, dtype=torch.complex64)
        A = torch.randn(1, dtype=torch.complex64) - 1.0
        dt = torch.zeros(1, 4, 1)
        with pytest.raises(ImportError, match="triton"):
            per_chunk_ssd_triton(
                Bc, Cc, Xc, A_log, decay_states,
                B_t, C_t, A, dt, chunk_size=4,
            )

    def test_check_block_dims_raises_value_error_on_too_large_dim(self):
        with pytest.raises(ValueError, match="exceeds"):
            _check_block_dims(P=512, N=64, chunk_size=64)
        with pytest.raises(ValueError, match="exceeds"):
            _check_block_dims(P=64, N=512, chunk_size=64)
        with pytest.raises(ValueError, match="exceeds"):
            _check_block_dims(P=64, N=64, chunk_size=512)

    def test_check_block_dims_accepts_production_404m_shape(self):
        _check_block_dims(P=64, N=64, chunk_size=64)


# -----------------------------------------------------------------------------
# Dispatch wiring in the parent module (CPU).
# -----------------------------------------------------------------------------
class TestPerChunkSsdDispatchWiring:
    def test_default_dispatch_is_pytorch(self):
        cfg = ModelConfig()
        assert cfg.ssd_dispatch == "pytorch"

    def test_explicit_pytorch_dispatch_runs_production_path(self):
        cfg = ModelConfig(
            vocab_size=64, d_model=32, n_layers=1, n_heads=2, head_dim=8,
            state_dim=4, chunk_size=4, ffn_dim=64, max_seq_len=16,
            dtype="fp32", weight_tying=True, ssd_dispatch="pytorch",
        )
        m = Mamba3Transformer(cfg)
        for block in m.layers:
            assert block.ssd_dispatch == "pytorch"
        x = torch.randint(0, 64, (1, 8))
        y = m(x)
        assert y.shape == (1, 8, 64)
        assert torch.isfinite(y).all()

    def test_explicit_triton_dispatch_falls_back_cleanly_on_cpu(self):
        cfg = ModelConfig(
            vocab_size=64, d_model=32, n_layers=1, n_heads=2, head_dim=8,
            state_dim=4, chunk_size=4, ffn_dim=64, max_seq_len=16,
            dtype="fp32", weight_tying=True, ssd_dispatch="triton",
        )
        m = Mamba3Transformer(cfg)
        block = m.layers[0]
        assert block.ssd_dispatch == "triton"
        assert block._triton_fallback_warned is False

        x = torch.randint(0, 64, (1, 8))
        captured = io.StringIO()
        with redirect_stdout(captured):
            y_triton = m(x)
        log = captured.getvalue()
        assert "ssd_dispatch='triton' unavailable" in log, log
        assert "falling back to 'pytorch'" in log, log
        assert block._triton_fallback_warned is True
        assert y_triton.shape == (1, 8, 64)
        assert torch.isfinite(y_triton).all()

    def test_triton_fallback_warning_is_one_shot_per_instance(self):
        cfg = ModelConfig(
            vocab_size=64, d_model=32, n_layers=1, n_heads=2, head_dim=8,
            state_dim=4, chunk_size=4, ffn_dim=64, max_seq_len=16,
            dtype="fp32", weight_tying=True, ssd_dispatch="triton",
        )
        m = Mamba3Transformer(cfg)
        x = torch.randint(0, 64, (1, 8))
        captured = io.StringIO()
        with redirect_stdout(captured):
            for _ in range(3):
                m(x)
        log = captured.getvalue()
        assert log.count("ssd_dispatch='triton' unavailable") == 1, log

    def test_triton_path_output_matches_pytorch_path(self):
        base_cfg = dict(
            vocab_size=64, d_model=32, n_layers=1, n_heads=2, head_dim=8,
            state_dim=4, chunk_size=4, ffn_dim=64, max_seq_len=16,
            dtype="fp32", weight_tying=True,
        )
        torch.manual_seed(42)
        m_p = Mamba3Transformer(ModelConfig(**base_cfg, ssd_dispatch="pytorch"))
        torch.manual_seed(42)
        m_t = Mamba3Transformer(ModelConfig(**base_cfg, ssd_dispatch="triton"))
        m_t.load_state_dict(m_p.state_dict())
        x = torch.randint(0, 64, (1, 8))
        with redirect_stdout(io.StringIO()):
            y_p = m_p(x)
            y_t = m_t(x)
        assert torch.allclose(y_p, y_t, atol=1e-5), (
            f"max diff = {(y_p - y_t).abs().max().item()}"
        )


# -----------------------------------------------------------------------------
# Pretrain env-var force-back (CPU).
# -----------------------------------------------------------------------------
class TestEnableTritonKernelsForceBack:
    def test_triton_dispatch_forced_back_when_env_var_missing(self, monkeypatch):
        monkeypatch.delenv("ENABLE_TRITON_KERNELS", raising=False)
        from training.pretrain import _enforce_triton_env_var
        model_cfg = dict(ssd_dispatch="triton")
        captured_msgs: list = []
        _enforce_triton_env_var(model_cfg, captured_msgs.append)
        assert any("forcing ssd_dispatch='pytorch'" in m for m in captured_msgs)
        assert model_cfg["ssd_dispatch"] == "pytorch"

    def test_triton_dispatch_passes_through_when_env_var_set(self, monkeypatch):
        monkeypatch.setenv("ENABLE_TRITON_KERNELS", "1")
        from training.pretrain import _enforce_triton_env_var
        model_cfg = dict(ssd_dispatch="triton")
        captured_msgs: list = []
        _enforce_triton_env_var(model_cfg, captured_msgs.append)
        assert not any("forcing ssd_dispatch" in m for m in captured_msgs)
        assert model_cfg["ssd_dispatch"] == "triton"

    def test_pytorch_dispatch_unchanged_by_guard(self, monkeypatch):
        from training.pretrain import _enforce_triton_env_var
        for env_val in (None, "0", "1"):
            if env_val is None:
                monkeypatch.delenv("ENABLE_TRITON_KERNELS", raising=False)
            else:
                monkeypatch.setenv("ENABLE_TRITON_KERNELS", env_val)
            model_cfg = dict(ssd_dispatch="pytorch")
            _enforce_triton_env_var(model_cfg, lambda m: None)
            assert model_cfg["ssd_dispatch"] == "pytorch"


# -----------------------------------------------------------------------------
# GPU tests — require triton + CUDA. Skipped on CPU/Mac.
# -----------------------------------------------------------------------------
gpu_required = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="requires triton + CUDA",
)


@gpu_required
class TestPerChunkSsdKernelGPU:
    def test_forward_matches_pytorch_tiny(self):
        torch.manual_seed(0)
        B, n_chunks, C, H, N, P = 1, 4, 4, 2, 4, 4
        Bc = torch.randn(B, n_chunks, C, H, N, dtype=torch.complex64, device="cuda")
        Cc = torch.randn(B, n_chunks, C, H, N, dtype=torch.complex64, device="cuda")
        Xc = torch.randn(B, n_chunks, C, H, P, dtype=torch.complex64, device="cuda")
        A_log = torch.randn(B, n_chunks, C, H, dtype=torch.complex64, device="cuda") * 0.1
        decay_states = torch.exp(A_log[:, :, -1:, :] - A_log)

        B_t = Bc.reshape(B, n_chunks * C, H, N)
        C_t = Cc.reshape(B, n_chunks * C, H, N)
        A = (torch.randn(H, dtype=torch.complex64, device="cuda") - 1.0)
        dt = torch.zeros(B, n_chunks * C, H, device="cuda")

        Y_tri, S_tri = per_chunk_ssd_triton(
            Bc, Cc, Xc, A_log, decay_states,
            B_t, C_t, A, dt, chunk_size=C,
        )
        Y_ref, S_ref = per_chunk_ssd_pytorch(Bc, Cc, Xc, A_log, decay_states)
        assert torch.allclose(Y_tri, Y_ref, atol=1e-3), (
            f"Y_diag max diff = {(Y_tri - Y_ref).abs().max().item()}"
        )
        assert torch.allclose(S_tri, S_ref, atol=1e-3), (
            f"state max diff = {(S_tri - S_ref).abs().max().item()}"
        )

    def test_forward_matches_pytorch_bf16(self):
        torch.manual_seed(1)
        B, n_chunks, C, H, N, P = 1, 4, 4, 2, 4, 4
        Bc = torch.randn(B, n_chunks, C, H, N, dtype=torch.complex64, device="cuda")
        Cc = torch.randn(B, n_chunks, C, H, N, dtype=torch.complex64, device="cuda")
        Xc = torch.randn(B, n_chunks, C, H, P, dtype=torch.complex64, device="cuda")
        A_log = torch.randn(B, n_chunks, C, H, dtype=torch.complex64, device="cuda") * 0.1
        decay_states = torch.exp(A_log[:, :, -1:, :] - A_log)

        B_t = Bc.reshape(B, n_chunks * C, H, N)
        C_t = Cc.reshape(B, n_chunks * C, H, N)
        A = (torch.randn(H, dtype=torch.complex64, device="cuda") - 1.0)
        dt = torch.zeros(B, n_chunks * C, H, device="cuda")

        Y_tri, S_tri = per_chunk_ssd_triton(
            Bc, Cc, Xc, A_log, decay_states,
            B_t, C_t, A, dt, chunk_size=C,
        )
        Y_ref, S_ref = per_chunk_ssd_pytorch(Bc, Cc, Xc, A_log, decay_states)
        assert torch.allclose(Y_tri, Y_ref, atol=1e-2)
        assert torch.allclose(S_tri, S_ref, atol=1e-2)

    def test_forward_404m_config_shape(self):
        torch.manual_seed(2)
        B, n_chunks, C, H, N, P = 2, 4, 64, 16, 64, 64
        Bc = torch.randn(B, n_chunks, C, H, N, dtype=torch.complex64, device="cuda")
        Cc = torch.randn(B, n_chunks, C, H, N, dtype=torch.complex64, device="cuda")
        Xc = torch.randn(B, n_chunks, C, H, P, dtype=torch.complex64, device="cuda")
        A_log = torch.randn(B, n_chunks, C, H, dtype=torch.complex64, device="cuda") * 0.1
        decay_states = torch.exp(A_log[:, :, -1:, :] - A_log)

        B_t = Bc.reshape(B, n_chunks * C, H, N)
        C_t = Cc.reshape(B, n_chunks * C, H, N)
        A = (torch.randn(H, dtype=torch.complex64, device="cuda") - 1.0)
        dt = torch.zeros(B, n_chunks * C, H, device="cuda")

        Y_tri, S_tri = per_chunk_ssd_triton(
            Bc, Cc, Xc, A_log, decay_states,
            B_t, C_t, A, dt, chunk_size=C,
        )
        Y_ref, S_ref = per_chunk_ssd_pytorch(Bc, Cc, Xc, A_log, decay_states)
        assert Y_tri.shape == (B, n_chunks, C, H, P)
        assert torch.allclose(Y_tri, Y_ref, atol=1e-2), (
            f"Y_diag max diff = {(Y_tri - Y_ref).abs().max().item()}"
        )
        assert torch.allclose(S_tri, S_ref, atol=1e-2), (
            f"state max diff = {(S_tri - S_ref).abs().max().item()}"
        )

    def test_autograd_gradcheck_tiny(self):
        B, n_chunks, C, H, N, P = 1, 1, 4, 1, 4, 4
        Bc = torch.randn(B, n_chunks, C, H, N, dtype=torch.complex64, device="cuda", requires_grad=True).double()
        Cc = torch.randn(B, n_chunks, C, H, N, dtype=torch.complex64, device="cuda", requires_grad=True).double()
        Xc = torch.randn(B, n_chunks, C, H, P, dtype=torch.complex64, device="cuda", requires_grad=True).double()
        A_log = (torch.randn(B, n_chunks, C, H, dtype=torch.complex64, device="cuda") * 0.1).double().requires_grad_(True)
        decay_states = torch.exp(A_log[:, :, -1:, :] - A_log)
        B_t = Bc.reshape(B, n_chunks * C, H, N).detach().requires_grad_(True)
        C_t = Cc.reshape(B, n_chunks * C, H, N).detach().requires_grad_(True)
        A = (torch.randn(H, dtype=torch.complex64, device="cuda") - 1.0).double().requires_grad_(True)
        dt = torch.zeros(B, n_chunks * C, H, device="cuda").double().requires_grad_(True)
        def fn(bc, cc, xc, al, ds, bt, ct, a, d):
            Y, S = per_chunk_ssd_triton(
                bc, cc, xc, al, ds, bt, ct, a, d, chunk_size=C,
            )
            return (Y.sum() + S.sum()).real
        torch.autograd.gradcheck(
            fn, (Bc, Cc, Xc, A_log, decay_states, B_t, C_t, A, dt),
            atol=1e-2, rtol=1e-2,
        )
